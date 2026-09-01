# nsys 报告内容说明（GB10 已知限制）

两个 `.nsys-rep` 均由 `nsys profile` 直接包裹 `vllm serve` 采集全程生成
（--trace=cuda,osrt,nvtx --gpu-metrics-devices=all --stop-on-exit，容器 --privileged）。

| 报告 | 量化版本 | GPU metrics 样本 | kernel 数据 | 基准期间数据 |
|---|---|---|---|---|
| qwen3_fp8.nsys-rep (47MB) | FP8 | 5264 万条 | 无 | 有（SM/带宽曲线可见基准负载） |
| qwen3_fp4.nsys-rep (37MB) | NVFP4 | 4052 万条 | 无 | 有 |

**打开方式**：`nsys-ui qwen3_fp8.nsys-rep`（本机已装 2026.1.3），重点看 GPU Metrics
泳道的 SM Throughput / DRAM Bandwidth 曲线：模型加载段平坦，基准窗口可见两段
prefill 尖峰 + 持续 decode 负载，可对比两个量化版本的带宽利用率差异。

**已知限制**：CUPTI kernel activity 在 GB10 上无法经 nsys 管线采集（4 个版本、
宿主机/容器、root/--privileged、软/硬件追踪均静默返回 0 kernel，详见 README 4.1 节），
逐 kernel 归因请使用 torch profiler trace（prof_traces/ 目录）。
