# nsys 报告内容说明（采集方式与版本结论，2026-09-01 更新）

## 推荐方案（v5，已验证）：kernel + NVTX + 全程采集

由 `run_verify_nsys.sh` 生成：容器内 bind-mount 宿主机 **nsys 2025.3.2**（镜像自带
2025.6.3 采不到 kernel，见下方结论），全程立即采集（无 capture-range），
`--cuda-graph-trace=node` 展开 decode CUDA Graph，`VLLM_NVTX_LABEL=1` 补丁打
prefill/decode 阶段标签；收尾必须 `docker stop -t 180`。

| 报告 | kernel 数据 | NVTX 标签 | 覆盖时长 |
|---|---|---|---|
| qwen3_fp8_v5.nsys-rep (32MB) | **177,561 条**（Top：fused_moe_kernel 3088ms、cutlass_3x_gemm_fp8_blockwise、flash_fwd_splitkv） | prefill ×4（966ms）/ decode ×140 | 229s 全程 |

验证负载 `verify_bench.py`（warmup 64/8 + batch4 8K入/128出）：TTFT 1509ms /
TPOT 29.78ms，与干净跑分（1515/29.15ms）一致，profiling 开销可忽略。
sqlite 分析注意：NVTX 文本**内联在 `NVTX_EVENTS.text` 列**（不经 StringIds join）；
decode 步耗时用相邻 NVTX 区间 start 差值（30.39ms，区间时长本身仅含 CPU launch ~7ms）。

## 全程报告（仅 GPU metrics 可用）

由镜像自带 nsys 2025.6.3 所采，kernel 泳道为空（版本回归，非平台限制）。

| 报告 | 量化版本 | GPU metrics 样本 | kernel 数据 |
|---|---|---|---|
| qwen3_fp8.nsys-rep (47MB) | FP8 | 5264 万条 | 无 |
| qwen3_fp4.nsys-rep (37MB) | NVFP4 | 4052 万条 | 无 |
| qwen3_fp4_nvtx.nsys-rep (42MB) | NVFP4 | — | 无（KERNEL 表缺失） |

## prefill / decode 分离报告（仅 GPU metrics 时间窗）

通过 sitecustomize 补丁调 `torch.cuda.profiler.start/stop()`（cudaProfilerApi）配合
`--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown` 截取窗口。
**该模式下 KERNEL 表缺失（任何版本、容器内也一样），不再用于 kernel 归因**，
仅保留 GPU metrics 时间窗价值。

| 报告 | 窗口（步） | GPU metrics 时间跨度 | 验证 |
|---|---|---|---|
| qwen3_fp8_prefill.nsys-rep (1.4MB) | 9~13 | 1.1 s | ≈ 4 个 prefill chunk |
| qwen3_fp8_decode.nsys-rep (3.4MB) | 15~280 | 7.8 s | ≈ 274 步 × 29ms |
| qwen3_fp4_prefill.nsys-rep (1.5MB) | 9~13 | 同上结构 | |
| qwen3_fp4_decode.nsys-rep (2.8MB) | 15~280 | 同上结构 | |

## 打开方式

`nsys-ui qwen3_fp8_v5.nsys-rep`（宿主机 2026.1.3 可查看 2025.3.2 采集的报告）。
v5 报告重点看 CUDA HW kernel 泳道（逐 kernel + CUDA Graph 展开节点）与 NVTX 泳道
（prefill/decode 阶段区间）；旧报告仅看 GPU Metrics 的 SM Throughput / DRAM
Bandwidth 曲线。

## 结论（2026-09-01 修正）

- kernel 采集失败**不是 GB10 平台限制**，而是 nsys **2025.6.x+ 的版本回归**
  （2025.6.3 与宿主机 2026.1.3 均静默失败）；**2025.3.2 容器内全程采集已验证
  可得完整 kernel 数据**（详见主 README 4.1/4.7 节）。
- `--capture-range=cudaProfilerApi` 窗口模式下 KERNEL 表缺失，勿用于 kernel 采集。
- 逐 kernel 归因首选本目录 v5 方案；torch profiler trace（prof_traces/）仍是
  算子级/注解级分析的补充。
