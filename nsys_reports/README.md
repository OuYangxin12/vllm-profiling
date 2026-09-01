# nsys 报告内容说明（采集方式与版本结论，2026-09-01 更新）

## 推荐方案（v5，已验证）：kernel + NVTX + 全程采集

由 `bash run_verify_nsys.sh [fp8|fp4]` 生成：容器内 bind-mount 宿主机 **nsys 2025.3.2**
（镜像自带 2025.6.3 采不到 kernel，见下方结论），全程立即采集（无 capture-range），
`--cuda-graph-trace=node` 展开 decode CUDA Graph，`VLLM_NVTX_LABEL=1` 补丁打
prefill/decode 阶段标签；收尾必须 `docker stop -t 180`。

| 报告 | kernel 数据 | NVTX 标签 | 验证基准（TTFT/TPOT） |
|---|---|---|---|
| qwen3_fp8_v5.nsys-rep (32MB) | **177,561 条**（Top：fused_moe_kernel 3088ms、cutlass_3x_gemm_fp8_blockwise、flash_fwd_splitkv） | prefill ×4 / decode ×140 | 1509/29.78ms ≈ 干净基线 1515/29.15 |
| qwen3_fp4_v5.nsys-rep (53MB) | **314,796 条**（Top：CUTLASS FP4 分组 GEMM ~35%、flashinfer::BatchPrefill） | prefill ×4 / decode ×140 | 1227.5/19.74ms ≈ 干净基线 1223/19.47 |

summary 导出：`nsys stats --force-export=true <rep> > <rep去掉后缀>_stats.txt`
（产物 `qwen3_fp{8,4}_v5_stats.txt`，含 nvtx_sum / cuda_api_sum / cuda_gpu_kern_sum /
memops 各段，参考 openpi-nsys.txt 格式）。

sqlite 分析注意：NVTX 文本**内联在 `NVTX_EVENTS.text` 列**（不经 StringIds join）；
decode 步耗时用相邻 NVTX 区间 start 差值，且须剔除 >100ms 的间隙（warmup→bench
间隔、prefill 期间间隙）后取中位数，否则均值被拉高（FP8 稳态 30.39ms、FP4 稳态
19.72ms，均与客户端 TPOT 吻合；区间时长本身仅含 CPU launch ~7ms）。

## 已删除的历史报告（缺 kernel 数据）

早期用镜像自带 nsys 2025.6.3 所采的报告（kernel 泳道为空）与 capture-range
窗口模式报告（KERNEL 表缺失）已全部删除，不再保留：

- `qwen3_fp8.nsys-rep` / `qwen3_fp4.nsys-rep`（全程，仅 GPU metrics）
- `qwen3_fp{8,4}_prefill.nsys-rep` / `qwen3_fp{8,4}_decode.nsys-rep`（分窗口）
- `qwen3_fp{8,4}_nvtx.nsys-rep`（NVTX 试验）

如需 GPU metrics 曲线，重新采集时加 `--gpu-metrics-devices=all`（需 `--privileged`，
不能与其他 nsys GPU-metrics 会话并发）。

## 打开方式

`nsys-ui nsys_reports/qwen3_fp8_v5.nsys-rep`（宿主机 2026.1.3 可查看 2025.3.2
采集的报告）。重点看 CUDA HW kernel 泳道（逐 kernel + CUDA Graph 展开节点）与
NVTX 泳道（prefill/decode 阶段区间）。

## 结论（2026-09-01 修正）

- kernel 采集失败**不是 GB10 平台限制**，而是 nsys **2025.6.x+ 的版本回归**
  （2025.6.3 与宿主机 2026.1.3 均静默失败）；**2025.3.2 容器内全程采集已验证
  可得完整 kernel 数据**（详见主 README 4.1/4.7 节）。
- `--capture-range=cudaProfilerApi` 窗口模式下 KERNEL 表缺失，勿用于 kernel 采集。
- 逐 kernel 归因首选本目录 v5 方案；torch profiler trace（prof_traces/）仍是
  算子级/注解级分析的补充。
