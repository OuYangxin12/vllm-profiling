# nsys 报告内容说明（GB10 已知限制与采集方式）

## 全程报告

由 `nsys profile` 直接包裹 `vllm serve` 采集全程生成（含模型加载 + 预热 + 基准）。

| 报告 | 量化版本 | GPU metrics 样本 | kernel 数据 |
|---|---|---|---|
| qwen3_fp8.nsys-rep (47MB) | FP8 | 5264 万条 | 无 |
| qwen3_fp4.nsys-rep (37MB) | NVFP4 | 4052 万条 | 无 |

## prefill / decode 分离报告

通过 sitecustomize 补丁在 EngineCore 内调 `torch.cuda.profiler.start/stop()`
（cudaProfilerApi），配合 `nsys profile --capture-range=cudaProfilerApi
--capture-range-end=stop-shutdown` 精确截取基准窗口；需
`VLLM_ENABLE_V1_MULTIPROCESSING=0`（nsys 注入下 EngineCore 子进程无法经
PYTHONPATH 加载补丁，单进程模式可绕开）。步进控制：预热占 9 步（1 prefill +
8 decode），主 prefill 为步 9-12，decode 从步 13 起。

| 报告 | 窗口（步） | GPU metrics 时间跨度 | 验证 |
|---|---|---|---|
| qwen3_fp8_prefill.nsys-rep (1.4MB) | 9~13 | 1.1 s | ≈ 4 个 prefill chunk |
| qwen3_fp8_decode.nsys-rep (3.4MB) | 15~280 | 7.8 s | ≈ 274 步 × 29ms |
| qwen3_fp4_prefill.nsys-rep (1.5MB) | 9~13 | 同上结构 | |
| qwen3_fp4_decode.nsys-rep (2.8MB) | 15~280 | 同上结构 | |

## 打开方式

`nsys-ui qwen3_fp8_prefill.nsys-rep`（本机 2026.1.3）。重点看 GPU Metrics 泳道的
SM Throughput / DRAM Bandwidth 曲线：prefill 报告为短促高负载尖峰，
decode 报告为持续平稳负载，可直接对比 FP8 vs NVFP4 的带宽利用率差异。

## 已知限制

CUPTI kernel activity 无法经 nsys 管线在 GB10 上采集（4 个版本、宿主机/容器、
root/--privileged、软/硬件追踪均静默返回 0 kernel，见主 README 4.1 节），
逐 kernel 归因请使用 torch profiler trace（prof_traces/ 目录）。
