# Qwen3-30B-A3B FP8 推理测试与 Profiling 方案

在 NVIDIA GB10（DGX Spark，121GB 统一内存）上，使用 Docker + vLLM 对 Qwen3-30B-A3B-FP8 模型进行
TTFT / TPOT 基准测试，并采集 prefill / decode 两阶段的 kernel 级 profiling 数据。

对应 task.txt 测试项：`input: 8K, output: 1K, batch size: 4, task: TTFT, TPOT; profiling data of prefill and decode`

## 1. 环境与组件

| 组件 | 版本/说明 |
|---|---|
| 硬件 | NVIDIA GB10（DGX Spark），121GB 统一内存（CPU/GPU 共享） |
| 模型 | `models/Qwen3-30B-A3B-FP8`（FP8 量化，7 个分片共 31GB） |
| 推理框架 | vLLM 0.20.1（镜像 `nvcr.io/nvidia/vllm:26.05-py3`，定制镜像 `vllm-nsys:fp8`） |
| 服务端口 | 8000（OpenAI 兼容 API） |
| GPU/TP | TP=1（单卡） |

> **统一内存注意**：GB10 的 CPU 和 GPU 共享同一 121GB 内存池，所有内存预算必须合并计算
> （详见第 5 节 OOM 防范）。

## 2. 启动推理服务

```bash
# 常规基准测试（CUDA Graph 开启，性能最优）
TP_SIZE=1 bash start_vllm.sh
```

Profiling 用启动命令（eager 模式 + torch profiler 配置，见第 4 节）：

```bash
docker run -d \
  --name vllm-prof \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $(pwd)/models:/models \
  -v $(pwd)/prof_patch:/models/prof_patch \
  -v $(pwd)/prof_traces:/models/prof_traces \
  -e HF_HOME=/models \
  -e PYTHONPATH=/models/prof_patch \
  -e VLLM_TORCH_PROFILER_DIR=/models/prof_traces \
  -e VLLM_PROF_STOP_AT_STEP=280 \
  -e VLLM_PROF_PREFIX=qwen3_fp8 \
  --entrypoint python3 \
  vllm-nsys:fp8 \
  -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B-FP8 \
  --served-model-name qwen3-30b-a3b \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.70 \
  --trust-remote-code \
  --enforce-eager \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/models/prof_traces \
  --profiler-config.torch_profiler_with_stack=false \
  --host 0.0.0.0 --port 8000
```

## 3. TTFT / TPOT 基准测试

脚本：`bench_ttft_tpot.py`（容器内运行，`docker exec vllm-prof python3 /root/bench_ttft_tpot.py`）

### 数据获取原理

- 输入：`/v1/completions` 的 `prompt` 字段直接传 **token id 数组**，保证输入长度精确为 8192 tokens
  （文本 prompt 经 tokenizer 往返会有 4% 左右的长度漂移）。
- 输出：`max_tokens=1024` + `ignore_eos=true`，强制生成满 1K tokens，避免 EOS 截断影响 TPOT。
- 并发：`asyncio.gather` 同时发出 4 个流式请求（batch size=4）。
- 计时（客户端流式统计）：
  - **TTFT** = 收到第一个 token 的时刻 − 请求发出时刻
  - **TPOT** = (最后一个 token 时刻 − 第一个 token 时刻) / (输出 token 数 − 1)
- 运行前先用小请求预热（仅生成 8 token），排除 CUDA graph / cache 冷启动影响。

### 基准结果（Qwen3-30B-A3B-FP8，8K in / 1K out / batch 4，CUDA Graph 模式）

| 指标 | 结果 |
|---|---|
| TTFT 平均 | **1515 ms**（4 请求：1510.8 ~ 1516.7 ms） |
| TPOT 平均 | **29.15 ms**（≈34.3 tok/s 每请求） |
| 每请求吞吐 | 32.7 tok/s |
| batch 总耗时 | 31.35 s |

eager 模式（`--enforce-eager`）下 TTFT ≈ 2156 ms、TPOT ≈ 34.15 ms；graph+no-compile 模式
（`--compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}'`）下
TTFT ≈ 2313 ms、TPOT ≈ 34.25 ms。

四种模式对比（8K in / 1K out / batch 4，TTFT / TPOT）：

| 模式 | FP8 | NVFP4 | 说明 |
|---|---|---|---|
| graph + compile（默认） | 1515 / 29.15 ms | **1223 / 19.47 ms** | 正式报数 |
| graph + no-compile | 2313 / 34.25 ms | 1196 / 26.76 ms | `mode=NONE` + `FULL_DECODE_ONLY` |
| eager | ~2156 / ~34.15 ms | —（trace 见 4.5 节 decode 30.8ms/步） | profiling 会话 |

FP8 结果文件：`bench_result.json` / `bench_result_graph_nocompile.json`；
NVFP4：`bench_result_nvfp4.json` / `bench_result_nvfp4_graph_nocompile.json`。

结论：
- **FP8**：性能收益几乎全部来自 torch.compile 融合（TPOT -15%、TTFT -35%），
  CUDA Graph 单独启用对 decode 几乎无增益。
- **NVFP4**：compile 依然显著（TPOT 26.76→19.47 ms，-27%）；graph 单独启用
  相比 eager decode（~30.8ms/步）有 ~4ms 收益（消除 launch 开销）。

### NVFP4 对比结果（同参数：8K in / 1K out / batch 4，graph+compile）

启动：同上命令换 `--model /models/Qwen3-30B-A3B-NVFP4`（vLLM 自动识别 ModelOpt NVFP4，
`quantization=modelopt_fp4`、KV cache `fp8_e4m3`、GEMM 走 `FlashInferCutlassNvFp4LinearKernel`）。

| 指标 | FP8 | NVFP4 | 提升 |
|---|---|---|---|
| TTFT 平均 | 1515 ms | **1223 ms** | -19% |
| TPOT 平均 | 29.15 ms | **19.47 ms** | -33% |
| 每请求吞吐 | 32.7 tok/s | 48.4 tok/s | +48% |
| batch 总耗时 | 31.35 s | 21.15 s | |

结果文件：`bench_result_nvfp4.json`；profiling 见第 4.5 节。

### 其他获取 TTFT/TPOT 的途径

```bash
# 1. vLLM 官方基准（服务端统计，含分位数）
docker exec <容器> vllm bench serve --backend openai \
  --host localhost --port 8000 --model qwen3-30b-a3b \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --num-prompts 4 --request-rate inf --percentile-metrics ttft,tpot,itl

# 2. Prometheus 指标（服务端直方图，持续累积）
curl localhost:8000/metrics | grep -E 'time_to_first_token|time_per_output_token'
```

## 4. Prefill / Decode Profiling

### 4.1 重要结论（2026-09-01 修正）：kernel 采集失败是 nsys 版本回归，2025.3.2 可用

早期结论“nsys 在 GB10 上无法采集 kernel，属平台限制”**已被推翻**。v3~v5 三轮对照
实验确认 **nsys 版本是决定性变量**：

| nsys 版本 | 环境 | kernel 结果 |
|---|---|---|
| **2025.3.2**（bind-mount 宿主机 `/opt/nvidia/nsight-systems/2025.3.2` 进容器） | vllm-nsys:fp8 容器内 | ✅ **177,561 条**，覆盖全程 229s |
| 2025.6.3（镜像自带） | 同容器 | ❌ 0~417 条（仅启动期），静默无报错 |
| 2026.1.3 | 宿主机 | ❌ 0 条 |

- 早期对 2025.3.2 的失败测试疑与 `--capture-range=cudaProfilerApi` 窗口模式有关：
  v3 实测该模式下 RUNTIME/NVTX/memcpy 均正常但 **KERNEL 表缺失**（任何版本、容器内也
  一样）；全程立即采集则 kernel 完整（v5 实测）。
- 2025.6.x+ 存在 kernel 采集回归，2025.3.2 是当前唯一验证可用版本。
- 结论：**容器内 + nsys 2025.3.2 + 全程采集**即可获得完整 kernel 数据，逐 kernel
  归因不再强依赖 torch profiler（后者仍是算子级/注解级分析的补充）。

### 4.2 方案：torch profiler + sitecustomize 自动打点

该 vLLM 构建没有 HTTP profiler 端点（`/start_profile` 404），因此通过
`PYTHONPATH` 注入 `prof_patch/sitecustomize.py`，monkey-patch Worker 的 `execute_model`：

- 第 1 次 `execute_model`：自动启动 torch profiler（覆盖 prefill 起点）
- 第 N 次（环境变量 `VLLM_PROF_STOP_AT_STEP`，默认 280）：停止并导出 chrome trace

关键启动参数：

```
--enforce-eager                                          # CUDA Graph 内的 kernel 无法逐条记录，必须 eager
--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=/models/prof_traces # trace 输出目录
--profiler-config.torch_profiler_with_stack=false        # 关闭调用栈（内存大户，防 OOM）
-e VLLM_PROF_STOP_AT_STEP=280                            # 采集窗口（步数）
```

**采集窗口注意事项**：步数是全局计数的。基准脚本预热请求也占 step，若预热生成 1024 token
就会消耗 1024 步窗口导致真正测试没被采到。因此预热只生成 8 token；窗口 280 步 ≈
prefill(4~8 chunk) + ~270 decode step。

### 4.3 采集结果分析

分析脚本：`python3 analyze_trace.py [trace文件]`
（按 `user_annotation` 步骤时长 >200ms 判定 prefill chunk，其余为 decode step）

**PREFILL**（4 个 chunk 共 2.17s ≈ TTFT 2272ms，GPU busy 66%）：

| kernel | 耗时 | 占比 |
|---|---|---|
| `fused_moe_kernel`（MoE 专家计算） | 579 ms | 40% |
| `flash_fwd_splitkv_kernel`（注意力） | 411 ms | 28% |
| `cutlass_3x_gemm_fp8_blockwise`（FP8 GEMM） | 105 ms | 7% |

**DECODE**（274 步，平均 33.8 ms/步，GPU busy 91%）：

| kernel | 耗时 | 占比 |
|---|---|---|
| `fused_moe_kernel` | 3941 ms | 37% |
| `flash_fwd_splitkv_kernel`（注意力） | 2310 ms | 22% |
| elementwise copy（eager 开销） | 1275 ms | 12% |
| `cutlass_3x_gemm_fp8_blockwise` | 1163 ms | 11% |

**洞察**：两阶段均由 MoE + 注意力主导；decode 阶段 12% 花在 elementwise copy 上，这是
eager 模式代价，CUDA Graph 模式下可基本消除（对应 TPOT 34.15 → 29.15 ms）。

**Profiler 对性能的影响**：采集期间 decode step 约 187 ms（含导出阻塞）；TTFT 在窗口内
测得 2272 ms，仅作 profiling 参考，**性能数字以无 profiler 的干净跑分为准**（第 3 节）。

### 4.4 查看 trace

trace 文件体积大（83~121MB），不进 git 仓库，统一从 GitHub Release 下载：

- Release 页面：<https://github.com/OuYangxin12/vllm-profiling/releases/tag/artifacts>
- FP8 trace：<https://github.com/OuYangxin12/vllm-profiling/releases/download/artifacts/qwen3_fp8_dp0_pp0_tp0_dcp0_ep0_rank0.1788165308338834706.pt.trace.json.gz>
- NVFP4 trace：<https://github.com/OuYangxin12/vllm-profiling/releases/download/artifacts/qwen3_fp4_dp0_pp0_tp0_dcp0_ep0_rank0.1788168168135033208.pt.trace.json.gz>
- NVFP4 kernel 汇总：<https://github.com/OuYangxin12/vllm-profiling/releases/download/artifacts/qwen3_fp4_kernel_summary.txt>

```bash
# 方式一：Chrome 浏览器打开 chrome://tracing，加载 *.pt.trace.json.gz
# 方式二：Perfetto UI（https://ui.perfetto.dev）拖入文件
# 方式三：命令行汇总
python3 analyze_trace.py prof_traces/qwen3_fp8_*.pt.trace.json.gz
```

后续新增 trace 上传：`gh release upload artifacts <新trace文件> --clobber`

### 4.5 NVFP4 profiling 结果

采集方式与 4.2 完全一致（`VLLM_PROF_PREFIX=qwen3_fp4`、窗口 280 步），
trace：`prof_traces/qwen3_fp4_*.pt.trace.json.gz`，汇总：`prof_traces/qwen3_fp4_kernel_summary.txt`。

**PREFILL**（4 chunks，GPU busy 99%）：

| kernel | 耗时 | 说明 |
|---|---|---|
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | 389 ms | 注意力 |
| `cutlass GemmUniversal GroupProblemShape` ×2 | 326 ms | FP4 分组 GEMM（MoE 专家） |
| `tensorrt_llm doActivationKernel`（fp4→bf16） | 46 ms | MoE 激活 |
| `tensorrt_llm expandInputRowsKernel` | 29 ms | MoE token 重排 |

**DECODE**（274 步，平均 30.8 ms/步，GPU busy 85%）：

| kernel | 耗时 | 占比 |
|---|---|---|
| `cutlass` FP4 分组 GEMM（MoE） | 1906 ms | 23% |
| elementwise copy（eager 开销） | 1338 ms | 16% |
| `cutlass` 分组 GEMM（另一分支） | 985 ms | 12% |
| `flashinfer::BatchPrefill`（注意力） | 873 ms | 11% |
| `cutlass_80_wmma bf16 GEMM` | 692 ms | 8% |
| `cvt_fp16_to_fp4`（量化转换） | 99 ms | 1% |

对比 FP8：MoE 从 `fused_moe_kernel`（vLLM 自研）切换到 TRT-LLM/CUTLASS FP4 分组 GEMM，
FP4 带来的带宽收益使 eager decode 从 33.8 降到 30.8 ms/步；elementwise copy 占比
依旧最高（16%），compile 融合后预计收益更大（对应干净跑分 TPOT 19.47 ms）。

### 4.6 graph+no-compile 模式的 profiling（FP8 / NVFP4）

配置：第 3 节 graph+no-compile 同款 `mode=NONE` + `FULL_DECODE_ONLY`，叠加 torch profiler（0.70 显存、关 stack、窗口 280 步），
前缀 `qwen3_fp8_nc` / `qwen3_fp4_nc`。

**重要发现**：FULL_DECODE_ONLY 图内的 decode kernel 也被 kineto（CUPTI graph trace）
完整逐条记录，并非只有回放事件，kernel 级归因仍然可用。但该模式下
`execute_context_N` 注释的时间片不再可靠，每步耗时请用 **GPU busy 总量 / 步数**：
NVFP4 ≈ 27.8 ms/步（7.63s/274，GPU busy 99%）、FP8 ≈ 40.1 ms/步（10.99s/274，94%），
与各自干净跑分 TPOT（26.76 / 34.25 ms）基本吻合。

**FP8**（`prof_traces/qwen3_fp8_nc_kernel_summary.txt`）：

| 阶段 | 主要 kernel（与 eager 模式同构） |
|---|---|
| prefill | `fused_moe_kernel` 611ms、`flash_fwd_splitkv` 405ms、`cutlass_3x_gemm_fp8_blockwise` 107ms |
| decode | `fused_moe_kernel` 4268ms(39%)、`flash_fwd_splitkv` 1933ms(18%)、elementwise copy 1355ms(12%)、`cutlass_3x_gemm_fp8_blockwise` 1161ms(11%) |

**NVFP4**（`prof_traces/qwen3_fp4_nc_kernel_summary.txt`）：

| 阶段 | 主要 kernel |
|---|---|
| prefill | `flashinfer::BatchPrefill` 391ms、CUTLASS FP4 分组 GEMM 323ms、`doActivationKernel` 47ms |
| decode | CUTLASS FP4 分组 GEMM 1511+783+911ms、elementwise copy 1356ms、`flashinfer::BatchPrefill` 842+399ms |

与 eager 模式（4.3/4.5 节）对比：kernel 构成基本一致，说明 FULL_DECODE_ONLY 图
只是消除 CPU launch 开销，GPU 侧计算图未变；收益主要来自 GPU busy 率提升
（NVFP4 decode 85%→99%）。profiling 会话中的客户端 TPOT（FP8 59ms / NVFP4 51ms）
含 profiler 开销与导出阻塞，不作性能参考。

### 4.7 nsys 原生报告（kernel + GPU metrics 时间线）

**GPU metrics（SM 吞吐/显存带宽曲线）**与 **kernel 逐条记录**（正确版本 + 全程采集）
均可采集。推荐方案（v5，已验证，即 `run_verify_nsys.sh`）：

```bash
docker run -d --name vllm-fp8-nsys-v3 --privileged --gpus all --ipc=host \
  --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/work -v $(pwd)/models:/models \
  -v /opt/nvidia/nsight-systems/2025.3.2:/opt/nvidia/nsight-systems/2025.3.2:ro \
  -e PYTHONPATH=/work/prof_patch -e VLLM_NVTX_LABEL=1 \
  --entrypoint "" vllm-nsys:fp8 \
  /opt/nvidia/nsight-systems/2025.3.2/bin/nsys profile \
    -o /work/nsys_reports/qwen3_fp8_v5 --force-overwrite=true \
    -t cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-30B-A3B-FP8 --served-model-name qwen3-30b-a3b \
    --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --trust-remote-code --host 0.0.0.0 --port 8000
# 就绪后（health 200）：docker exec vllm-fp8-nsys-v3 python3 /work/verify_bench.py
# 收尾：docker stop -t 180 vllm-fp8-nsys-v3
```

要点：
- **必须 bind-mount 2025.3.2 并显式调用**，镜像自带的 2025.6.3 采不到 kernel（4.1 节）；
- **不要用 `--capture-range=cudaProfilerApi` 窗口模式**（KERNEL 表缺失），改全程采集，
  prefill/decode 切分靠 NVTX 标签后处理；
- `--cuda-graph-trace=node` 把 decode CUDA Graph 展开为逐 kernel 记录；
- NVTX 标签（`prefill_stepN_tokensN` / `decode_stepN_tokensN`）由 prof_patch 补丁产生；
  导出 sqlite 后文本**内联在 `NVTX_EVENTS.text` 列**（不经 StringIds join）；
- decode 走 CUDA Graph 时 NVTX 区间时长只含 CPU launch（~7ms），**步耗时用相邻区间
  start 差值**（实测平均 30.39ms，与客户端 TPOT 29.78ms 吻合）；
- **收尾必须 `docker stop -t 180`**：默认 10s 超时 SIGKILL 会丢失报告（v4 教训）。

产物：`nsys_reports/qwen3_fp8_v5.nsys-rep`（32MB）：**177,561 条 kernel** 覆盖 229s 全程
（Top：`fused_moe_kernel` 3088ms、`cutlass_3x_gemm_fp8_blockwise`、`flash_fwd_splitkv`、
`act_and_mul`），NVTX prefill ×4（共 966ms）/ decode ×140。验证负载
`verify_bench.py`（warmup 64/8 + batch4 8K入/128出）TTFT 1509ms / TPOT 29.78ms，
与无 profiler 干净跑分（1515/29.15ms）一致，profiling 开销可忽略。

早期报告（`qwen3_fp8.nsys-rep`/`qwen3_fp4.nsys-rep`，47MB/37MB，5264 万/4052 万条
GPU metrics）仅 GPU metrics 可用（当时镜像为 2025.6.3，kernel 泳道为空属版本回归，
非平台限制）；采集时若需 GPU metrics 加 `--gpu-metrics-devices=all`
（需 `--privileged`，且同一 GPU 上不能与其他 nsys GPU-metrics 会话并发）。

> 分窗口采集（`qwen3_fp{8,4}_{prefill,decode}.nsys-rep`）仅剩 GPU metrics 时间窗
> 的价值，KERNEL 表在该模式下缺失，不再推荐用于 kernel 归因。

## 5. OOM 防范规范（GB10 统一内存必读）

统一内存下内存预算合并计算：**vLLM 预留 + profiler 峰值 + 系统 ≈ 121GB，不可超**。

| 措施 | 说明 |
|---|---|
| `--gpu-memory-utilization 0.70` | 预留 ~85GB（权重 30GB + KV cache ~55GB），留 ~36GB 余量；0.90 只留 ~12GB，极易 OOM |
| `--profiler-config.torch_profiler_with_stack=false` | 默认 true，调用栈累积是最大内存开销 |
| 控制采集步数 | `VLLM_PROF_STOP_AT_STEP`，不要跑满整个请求 |
| 预热请求最小化 | 预热 max_tokens=8，避免占用窗口和内存 |
| 过程监控 | 采集期间 `free -h`，available 低于 8GB 立即停止 |

KV cache 需求估算（FP8 KV）：batch 4 × (8K+1K) ≈ 36K tokens，在 0.70 配置下远小于容量，
如需更大并发/上下文可适当上调 utilization，但建议不超过 0.80。

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `start_vllm.sh` | 常规 vLLM 服务启动脚本（CUDA Graph 模式） |
| `bench_ttft_tpot.py` | TTFT/TPOT 基准脚本（8K in / 1K out / batch 4） |
| `bench_result.json` | FP8 干净基准结果（graph+compile） |
| `bench_result_graph_nocompile.json` | FP8 graph+no-compile 基准结果 |
| `bench_result_nvfp4.json` | NVFP4 干净基准结果（graph+compile） |
| `bench_result_nvfp4_graph_nocompile.json` | NVFP4 graph+no-compile 基准结果 |
| `prof_patch/sitecustomize.py` | torch profiler 自动打点 + nsys cudaProfilerApi + NVTX 阶段标注补丁（PYTHONPATH 注入） |
| `run_verify_nsys.sh` | nsys v5 采集方案启动脚本（2025.3.2 注入 + NVTX 标签 + graph 展开，见 4.7 节） |
| `verify_bench.py` | nsys 验证负载客户端（warmup + batch4 8K入/128出） |
| `nsys_reports/qwen3_fp8_v5.nsys-rep` | 已验证的 kernel+NVTX 全程报告（177,561 条 kernel，见 4.7 节） |
| `nsys_reports/*.nsys-rep` | 其余历史报告（GPU metrics 曲线 / 分窗口，见 `nsys_reports/README.md`） |
| `prof_traces/*.pt.trace.json.gz` | chrome trace（FP8 / NVFP4 的 prefill+decode kernel 级数据，不入库，见 4.4 节 Release 下载链接） |
| `prof_traces/qwen3_fp4_kernel_summary.txt` | NVFP4 eager trace kernel 汇总 |
| `prof_traces/qwen3_fp8_nc_kernel_summary.txt` | FP8 graph+no-compile trace kernel 汇总 |
| `prof_traces/qwen3_fp4_nc_kernel_summary.txt` | NVFP4 graph+no-compile trace kernel 汇总 |
| `analyze_trace.py` | trace 分析脚本（prefill/decode kernel 汇总） |
| `download_model.sh` | ModelScope 模型下载脚本 |
| `task.txt` | 测试任务清单 |
