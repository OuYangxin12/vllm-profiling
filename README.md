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

三种模式对比（8K in / 1K out / batch 4）：

| 模式 | TTFT | TPOT | 说明 |
|---|---|---|---|
| graph + compile（默认） | 1515 ms | 29.15 ms | 正式报数，见 `bench_result.json` |
| graph + no-compile | 2313 ms | 34.25 ms | 见 `bench_result_graph_nocompile.json` |
| eager | ~2156 ms | ~34.15 ms | profiling 会话，无独立 bench 文件 |

结论：性能收益主要来自 **torch.compile 算子融合**（TPOT -15%、TTFT -35%），
CUDA Graph 单独启用（无 compile）对 decode 几乎无增益，与 trace 分析中
decode 12% 时间在 elementwise copy 的结论互相印证——该开销在
FULL_DECODE_ONLY 图内依然存在，需 compile 融合才能消除。

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

### 4.1 重要结论：nsys 在 GB10 上不可用于 kernel 采集

在当前 GB10 + 驱动 580.82.09（open kernel module）环境下验证过：

- nsys 2024.2.3（CUPTI 12.5）→ `CUPTI_ERROR_INVALID_DEVICE`（不支持 CUDA 13.2 驱动）
- nsys 2025.3.2 / 2025.6.3 / 2026.1.3（CUPTI 13.1~13.3）→ CUDA API/Memcpy/Runtime 均可采集，
  但 **kernel 活动记录始终为 0**，且无任何报错（含 `--trace=cuda-sw` 软件模式）
- `NSYS_CUPTI_LIBRARY_PATH` 指向 CUDA 工具包自带的 CUPTI（`libcupti.so.2026.2.1`，
  进程内直调可正常收到 kernel 记录）→ nsys 管线下依然采不到
- 宿主机/容器、root/普通用户、`--privileged` 均排除

结论：是 nsys 采集管线与 GB10（Blackwell iGPU 硬件追踪路径）的兼容性问题，**用 torch profiler
（kineto，进程内 CUPTI）替代**，已验证可完整采集 kernel 数据。

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

```bash
# 方式一：Chrome 浏览器打开 chrome://tracing，加载 *.pt.trace.json.gz
# 方式二：Perfetto UI（https://ui.perfetto.dev）拖入文件
# 方式三：命令行汇总
python3 analyze_trace.py prof_traces/qwen3_fp8_*.pt.trace.json.gz
```

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
| `bench_result.json` | 最近一次干净基准结果 |
| `prof_patch/sitecustomize.py` | torch profiler 自动打点补丁（PYTHONPATH 注入） |
| `prof_traces/*.pt.trace.json.gz` | chrome trace（prefill + decode kernel 级数据） |
| `analyze_trace.py` | trace 分析脚本（prefill/decode kernel 汇总） |
| `download_model.sh` | ModelScope 模型下载脚本 |
| `task.txt` | 测试任务清单 |
