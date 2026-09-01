# Qwen3-30B-A3B 推理基准测试与 Profiling 方案

在 NVIDIA GB10（DGX Spark，121GB 统一内存）上，使用 Docker + vLLM（0.20.1）对
Qwen3-30B-A3B-FP8 / NVFP4 模型进行 TTFT / TPOT 基准测试，并采集 prefill / decode
两阶段的 kernel 级 profiling 数据。

对应 task.txt 测试项：`input: 8K, output: 1K, batch size: 4, task: TTFT, TPOT; profiling data of prefill and decode`

> 采集过程中踩过的坑（nsys 版本回归、CUPTI 尾部丢事件、profiler stop 陷阱等）的
> 完整记录见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)，本文档只写最终验证过的做法。

## 1. 环境与组件

| 组件 | 版本/说明 |
|---|---|
| 硬件 | NVIDIA GB10（DGX Spark），121GB 统一内存（CPU/GPU 共享） |
| 模型 | `models/Qwen3-30B-A3B-FP8`（31GB）/ `models/Qwen3-30B-A3B-NVFP4`（ModelOpt 官方 NVFP4） |
| 推理框架 | vLLM 0.20.1（基础镜像 `nvcr.io/nvidia/vllm:26.05-py3`，定制镜像 `vllm-nsys:fp8`） |
| 服务端口 | 8000（OpenAI 兼容 API） |
| GPU/TP | TP=1（单卡） |

> **统一内存注意**：GB10 的 CPU 和 GPU 共享同一 121GB 内存池，所有内存预算必须合并
> 计算（详见第 5 节 OOM 防范）。

## 2. TTFT / TPOT 基准测试

```bash
TP_SIZE=1 bash start_vllm.sh          # 常规基准服务（CUDA Graph + compile，性能最优）
python3 bench_ttft_tpot.py            # 容器内运行：docker exec vllm-prof python3 /root/bench_ttft_tpot.py
```

数据获取要点：

- 输入：`/v1/completions` 的 `prompt` 字段直接传 **token id 数组**，保证输入精确
  8192 tokens（文本 prompt 经 tokenizer 往返会有 ~4% 长度漂移）；
- 输出：`max_tokens=1024` + `ignore_eos=true`，强制生成满 1K，避免 EOS 截断影响 TPOT；
- 并发：`asyncio.gather` 同时发出 4 个流式请求（batch size=4）；
- 计时：TTFT = 首 token 时刻 − 请求发出时刻；TPOT =（末 token − 首 token）/(输出数−1)；
- 运行前先用只生成 8 token 的小请求预热，排除 CUDA Graph / cache 冷启动影响。

### 基准结果（8K in / 1K out / batch 4）

| 模式 | FP8（TTFT/TPOT ms） | NVFP4（TTFT/TPOT ms） | 启动参数 |
|---|---|---|---|
| **graph + compile（默认）** | **1515 / 29.15** | **1223 / 19.47** | `start_vllm.sh` 默认 |
| graph + no-compile | 2313 / 34.25 | 1196 / 26.76 | `--compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| eager（`--enforce-eager`） | ~2156 / ~34.15 | —（decode 30.8ms/步） | `--enforce-eager` |

结果文件：`bench_result.json`、`bench_result_graph_nocompile.json`（FP8）；
`bench_result_nvfp4.json`、`bench_result_nvfp4_graph_nocompile.json`（NVFP4）。

结论：

- **FP8**：收益几乎全部来自 torch.compile 融合（TPOT −15%、TTFT −35%），
  CUDA Graph 单独启用对 decode 几乎无增益；
- **NVFP4**：compile 依然显著（TPOT −27%）；graph 单独启用相对 eager decode
  有 ~4ms 收益（消除 launch 开销，GPU busy 85%→99%）；
- **NVFP4 vs FP8（默认模式）**：TTFT −19%、TPOT −33%、每请求吞吐 +48%。

其他获取途径：

```bash
# vLLM 官方基准（服务端统计，含分位数）
docker exec <容器> vllm bench serve --backend openai \
  --host localhost --port 8000 --model qwen3-30b-a3b \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --num-prompts 4 --request-rate inf --percentile-metrics ttft,tpot,itl
# Prometheus 指标（服务端直方图）
curl localhost:8000/metrics | grep -E 'time_to_first_token|time_per_output_token'
```

## 3. nsys Profiling（推荐方案，kernel 级）

由 `run_verify_nsys.sh` 一键完成：容器内 bind-mount 宿主机 **nsys 2025.3.2**、
窗口模式采集（cudaProfilerApi）、`--cuda-graph-trace=node` 展开 decode CUDA Graph、
NVTX 打 prefill/decode 阶段标签。

### 3.1 快速使用

```bash
# 1. 启动采集容器（窗口模式：START=9 跳过 8 步 warmup，STOP 设在负载尾段内）
V5_OUT=qwen3_fp8_v6 VLLM_CUDA_PROFILER_START_AT_STEP=9 VLLM_CUDA_PROFILER_STOP_AT_STEP=145 \
  bash run_verify_nsys.sh fp8        # 可选 fp8 | fp4 | fp8nc | fp4nc

# 2. 等待 /health 返回 200（约 3~4 分钟）后跑验证负载
docker exec vllm-nsys-fp8 env BENCH_OUTPUT_LEN=1024 python3 /work/verify_bench.py

# 3. 收尾（必须 -t 180，否则 nsys 来不及写报告）
docker stop -t 180 vllm-nsys-fp8

# 4. 导出 summary（各表合并为一个 txt）
nsys stats --force-export=true --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format table --output <前缀> <rep 路径>   # 再 cat 各分表为 <前缀>_stats.txt
```

### 3.2 参数说明

| 参数 | 说明 |
|---|---|
| 位置参数 `fp8 / fp4` | 量化版本，默认 graph+compile 模式，输出名 `qwen3_fp{8,4}_v5` |
| 位置参数 `fp8nc / fp4nc` | 对齐 graph+no-compile 基准（`mode=NONE` + `FULL_DECODE_ONLY`），输出名 `qwen3_fp{8,4}_v6_nc` |
| `V5_OUT=<名字>` | 覆盖报告输出名 |
| `VLLM_CUDA_PROFILER_START_AT_STEP` | 窗口起点（execute_model 全局步计数）；**设 8 以上跳过 warmup** |
| `VLLM_CUDA_PROFILER_STOP_AT_STEP` | 窗口终点，**必须设在负载内必然触发的步数**（见 3.3） |
| `BENCH_OUTPUT_LEN`（verify_bench.py） | 输出 token 数，默认 128；对齐 1K out 基准用 1024 |

### 3.3 正确做法要点（每条都验证过）

1. **nsys 必须用 2025.3.2**（脚本已 bind-mount `/opt/nvidia/nsight-systems/2025.3.2`）：
   2025.6.x+ 存在 kernel 采集回归，静默失败不报错；
2. **用窗口模式**（设 START/STOP 后自动加 `--capture-range=cudaProfilerApi`）：
   cudaProfilerStop 触发 buffer 强制 flush，可避免全程采集的 CUPTI 尾部丢事件，
   且报告体积小一个量级；
3. **STOP 宁小勿大**：步数 = 8（warmup）+ ~5（prefill chunk+混合步）+ 输出 token 数
   再减 2~5 步余量。STOP 一旦超过实际最后一步，引擎空闲后不再调 execute_model，
   cudaProfilerStop 永不触发，报告退化为 docker stop 收尾（尾部丢数据）；
   例：128 out → STOP=145；1024 out → STOP=1037；
4. **收尾必须 `docker stop -t 180`**：默认 10s 超时 SIGKILL 会丢失报告；
5. **每次报告先做空步检测再用**：导出 sqlite 后核对每个 decode 步的 kernel 数，
   零 kernel 步（除 warmup 首 chunk 和 stop 后 tokens0 伪步外）即 CUPTI 停采；
6. **sqlite 分析要点**：NVTX 文本内联在 `NVTX_EVENTS.text` 列（有 NULL，查询加
   `COALESCE(text,'')`，不经 StringIds join）；decode 步耗时用相邻 NVTX 区间
   start 差值取 median（剔除 >100ms 间隙），区间时长本身只含 CPU launch ~3-7ms；
   GPU busy 用 [步 start, 下一步 start) 窗口内 kernel 时长之和；
7. **混合步识别**：prefill chunk 与 decode 混调的步会被 512-token 阈值误标成
   `decode_stepN`（特征：scheduled tokens < 512 但 ~1000+ kernel、数百 ms busy），
   判相别以 kernel 数/busy 为准，不要只看标签。

### 3.4 产物与结果

| 报告（nsys_reports/） | 模式 | 内容 |
|---|---|---|
| `qwen3_fp8_v5.nsys-rep` + `_stats.txt` | 全程采集 | 224,395 条 kernel，覆盖加载/Graph 捕获/负载全时间线；尾部 ~0.8s 缺失，归因只用到 decode step116 |
| `qwen3_fp4_v5.nsys-rep` + `_stats.txt` | 全程采集 | 314,796 条 kernel，负载段完整 |
| `qwen3_fp8_v6.nsys-rep` + `_stats.txt` | 窗口（128 out） | 140,947 条 kernel，零丢失；decode 129 步、1,072 kernel/步、busy 29.68ms ≈ 周期 29.80ms（99.6% GPU busy） |
| `qwen3_fp8_v6_nc.nsys-rep` + `_stats.txt` | 窗口（1024 out，nc） | 1,171,187 条 kernel，零空步；decode 1023 步、busy 37.78ms ≈ 周期 37.79ms（99.9%） |
| `qwen3_fp4_v6_nc.nsys-rep` + `_stats.txt` | 窗口（1024 out，nc） | 1,251,219 条 kernel，零空步；busy 27.41ms ≈ 周期 27.45ms（99.9%） |

窗口报告对应的 bench 复测（TTFT/TPOT）：FP8 compile 1509.8/29.81ms、
FP8 nc 2268/37.63ms、FP4 nc 1276/27.73ms，均与干净基线吻合（profiling 开销可忽略）。

kernel 归因结论（compile 模式，`fused_moe_kernel` 38%+、`flash attention` ~21%、
`cutlass fp8/fp4 GEMM` ~11-23% 为主；NVFP4 的 MoE 切换为 TRT-LLM/CUTLASS FP4 分组
GEMM + `flashinfer::BatchPrefill`）；**no-compile 模式的标志是 `direct_copy`
elementwise 拷贝占 13.5%（FP8）/ 17.8%（FP4）GPU 时间**——torch.compile 融合收益
的直接证据；nc vs compile 的 decode busy 差：FP8 37.78 vs 29.68ms（+27%）、
FP4 27.41 vs 19.74ms（+39%）。

如需 GPU metrics 曲线（SM 吞吐/显存带宽），采集时加 `--gpu-metrics-devices=all`
（需 `--privileged`，且同一 GPU 上不能与其他 nsys GPU-metrics 会话并发）。

打开方式：`nsys-ui nsys_reports/<报告>`（宿主机 2026.1.3 可查看 2025.3.2 采集的报告），
重点看 CUDA HW kernel 泳道与 NVTX 泳道；报告背景与逐报告说明见
`nsys_reports/README.md`。

## 4. torch profiler 补充方案（算子级 trace）

nsys 覆盖不到的算子级/注解级分析可用 torch profiler。该 vLLM 构建没有 HTTP
profiler 端点，通过 `PYTHONPATH` 注入 `prof_patch/sitecustomize.py`，monkey-patch
`Worker.execute_model`：第 1 步自动启动 profiler，第 `VLLM_PROF_STOP_AT_STEP` 步
停止并导出 chrome trace。

关键启动参数：

```
--enforce-eager                                          # eager 模式采集；nc 模式换 --compilation-config
--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=/models/prof_traces # trace 输出目录
--profiler-config.torch_profiler_with_stack=false        # 关闭调用栈（内存大户，防 OOM）
-e VLLM_PROF_STOP_AT_STEP=280                            # 采集窗口（全局步计数，warmup 也占步）
-e VLLM_PROF_PREFIX=qwen3_fp8                            # trace 文件名前缀
```

窗口估算：280 步 ≈ prefill(4~8 chunk) + ~270 decode 步；预热请求只生成 8 token，
避免消耗窗口。FULL_DECODE_ONLY 模式下 CUDA Graph 内的 decode kernel 也能被
kineto（CUPTI graph trace）逐条记录，但 `execute_context_N` 注解时间片不可靠，
每步耗时用 **GPU busy 总量 / 步数**。

完整 docker run 命令：

```bash
docker run -d --name vllm-prof \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $(pwd)/models:/models -v $(pwd)/prof_patch:/models/prof_patch \
  -v $(pwd)/prof_traces:/models/prof_traces \
  -e HF_HOME=/models -e PYTHONPATH=/models/prof_patch \
  -e VLLM_TORCH_PROFILER_DIR=/models/prof_traces \
  -e VLLM_PROF_STOP_AT_STEP=280 -e VLLM_PROF_PREFIX=qwen3_fp8 \
  --entrypoint python3 vllm-nsys:fp8 \
  -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B-FP8 --served-model-name qwen3-30b-a3b \
  --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.70 --trust-remote-code \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/models/prof_traces \
  --profiler-config.torch_profiler_with_stack=false \
  --host 0.0.0.0 --port 8000
```

### 4.1 trace 分析结果（8K in / 1K out / batch 4）

分析脚本：`python3 analyze_trace.py prof_traces/<trace>.pt.trace.json.gz`。
trace 体积大（23~121MB）不入库，从 GitHub Release 下载：
<https://github.com/OuYangxin12/vllm-profiling/releases/tag/artifacts>
（新增上传：`gh release upload artifacts <trace> --clobber`）。
查看：chrome://tracing 或 <https://ui.perfetto.dev>。

**FP8**（eager / nc 的 kernel 构成基本一致，FULL_DECODE_ONLY 只消除 CPU launch 开销）：

| 阶段 | top kernel（耗时/占比） |
|---|---|
| prefill | `fused_moe_kernel` 579ms(40%)、`flash_fwd_splitkv` 411ms(28%)、`cutlass_3x_gemm_fp8_blockwise` 105ms(7%) |
| decode | `fused_moe_kernel` 3941ms(37%)、`flash_fwd_splitkv` 2310ms(22%)、elementwise copy 1275ms(12%)、`cutlass_3x_gemm_fp8_blockwise` 1163ms(11%) |

**NVFP4**（MoE 从 `fused_moe_kernel` 切换为 TRT-LLM/CUTLASS FP4 分组 GEMM）：

| 阶段 | top kernel |
|---|---|
| prefill | `flashinfer::BatchPrefillWithPagedKVCacheKernel` 389ms、CUTLASS FP4 分组 GEMM ×2 326ms、`doActivationKernel`(fp4→bf16) 46ms |
| decode | CUTLASS FP4 分组 GEMM 1906ms(23%)、elementwise copy 1338ms(16%)、`flashinfer::BatchPrefill` 873ms(11%)、bf16 wmma GEMM 692ms(8%) |

**洞察**：两阶段均由 MoE + 注意力主导；eager 模式 decode 有 12~16% 花在
elementwise copy 上，compile 融合后基本消除（对应 TPOT 34.15 → 29.15 / 30.8 → 19.47）。

profiling 会话中的客户端 TPOT（含 profiler 开销与导出阻塞，如 eager FP8 ~187ms/步）
不作性能参考，性能数字一律以第 2 节干净跑分为准。

## 5. OOM 防范规范（GB10 统一内存必读）

统一内存下内存预算合并计算：**vLLM 预留 + profiler 峰值 + 系统 ≈ 121GB，不可超**。

| 措施 | 说明 |
|---|---|
| `--gpu-memory-utilization 0.70` | 预留 ~85GB（权重 30GB + KV cache ~55GB），留 ~36GB 余量；0.90 只留 ~12GB，极易 OOM |
| `--profiler-config.torch_profiler_with_stack=false` | 默认 true，调用栈累积是最大内存开销 |
| 控制采集步数 | `VLLM_PROF_STOP_AT_STEP`，不要跑满整个请求 |
| 预热请求最小化 | 预热 max_tokens=8，避免占用窗口和内存 |
| 过程监控 | 采集期间 `free -h`，available 低于 8GB 立即停止 |

KV cache 需求估算（FP8 KV）：batch 4 × (8K+1K) ≈ 36K tokens，在 0.70 配置下远小于
容量；如需更大并发/上下文可适当上调 utilization，但建议不超过 0.80。

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `start_vllm.sh` | 常规 vLLM 服务启动脚本（CUDA Graph + compile 模式） |
| `bench_ttft_tpot.py` | TTFT/TPOT 基准脚本（8K in / 1K out / batch 4） |
| `bench_result.json` / `bench_result_graph_nocompile.json` | FP8 干净基准结果（compile / no-compile） |
| `bench_result_nvfp4.json` / `bench_result_nvfp4_graph_nocompile.json` | NVFP4 干净基准结果 |
| `bench_result_fp{8,4}_nc_profile.json` | nc 窗口采集轮的 bench 复测（TPOT 含流式尾部伪影，见 TROUBLESHOOTING） |
| `run_verify_nsys.sh` | nsys 采集脚本（`bash run_verify_nsys.sh [fp8\|fp4\|fp8nc\|fp4nc]`，见第 3 节） |
| `verify_bench.py` | 采集验证负载客户端（warmup + batch4 8K 入；`BENCH_OUTPUT_LEN` 控制输出长度） |
| `prof_patch/sitecustomize.py` | torch profiler 打点 + nsys cudaProfilerApi 窗口控制 + NVTX 阶段标注补丁 |
| `analyze_trace.py` | torch profiler trace 分析脚本（prefill/decode kernel 汇总） |
| `nsys_reports/*.nsys-rep` / `*_stats.txt` | nsys 报告与 summary（明细见 `nsys_reports/README.md`） |
| `prof_traces/*.pt.trace.json.gz` | chrome trace（不入库，GitHub Release 下载，见 4.1 节） |
| `prof_traces/*_kernel_summary.txt` | trace kernel 汇总（eager FP8/NVFP4 与 nc FP8/NVFP4） |
| `TROUBLESHOOTING.md` | 采集过程中的问题与修正记录（版本回归、尾部丢失、STOP 陷阱等） |
| `download_model.sh` | ModelScope 模型下载脚本 |
| `task.txt` | 测试任务清单 |
