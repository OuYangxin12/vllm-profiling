# nsys_reports 报告说明

采集方法与正确做法见主 [README.md](../README.md) 第 3 节；问题与修正记录见
[TROUBLESHOOTING.md](../TROUBLESHOOTING.md)。

## 报告清单

| 报告 | 采集范围 | kernel 数 | 完整性 | 对应基准（TTFT/TPOT ms） |
|---|---|---|---|---|
| `qwen3_fp8_v5`（35MB） | 全程（加载+捕获+负载） | 224,395 | 尾部 ~0.8s 缺失，归因止于 decode step116 | 1513.5/30.51 ≈ 干净 1515/29.15 |
| `qwen3_fp4_v5`（53MB） | 全程 | 314,796 | 负载段完整 | 1227.5/19.74 ≈ 干净 1223/19.47 |
| `qwen3_fp8_v6`（11MB） | 窗口（128 out） | 140,947 | 零丢失，decode 129 步全量 | 1509.8/29.81 |
| `qwen3_fp8_v6_nc`（43MB） | 窗口（1024 out，no-compile） | 1,171,187 | 零空步，decode 1023 步全量 | 2268/37.63* |
| `qwen3_fp4_v6_nc`（46MB） | 窗口（1024 out，no-compile） | 1,251,219 | 零空步，decode 1023 步全量 | 1276/27.73 |

\* 客户端 TPOT 取无伪影轮的数值；窗口轮内 stop 触发后的 client TPOT 虚高不可引用。

各 `*_stats.txt` 为 `nsys stats` 导出的 summary（nvtx_sum / cuda_api_sum /
cuda_gpu_kern_sum / cuda_gpu_mem_time_sum 合并，参考 openpi-nsys.txt 格式）。

## 使用要点

- **逐 kernel 归因优先用窗口报告**（v6 / v6_nc，负载段零丢失）；全程 v5 报告用于
  加载/Graph 捕获期时间线，且使用前必须做空步检测；
- 打开方式：`nsys-ui nsys_reports/<报告>.nsys-rep`（宿主机 2026.1.3 可查看 2025.3.2
  采集的报告），重点看 CUDA HW kernel 泳道与 NVTX 泳道；
- decode 走 CUDA Graph 时 NVTX 区间时长只含 CPU launch，步耗时用相邻区间 start
  差值取 median；GPU busy 用 [步 start, 下一步 start) 窗口统计；
- 混合步（prefill chunk + decode 混调）会被标成 decode，判相以 kernel 数/busy 为准。

## 结论

- kernel 采集失败**不是 GB10 平台限制**，而是 nsys 2025.6.x+ 版本回归；
  **2025.3.2 容器内采集已验证可用**；
- **窗口模式（capture-range=cudaProfilerApi）在 2025.3.2 下完全可用**且为推荐方案；
  全程采集有 CUPTI 尾部丢事件风险且无法根治；
- 逐 kernel 归因之外，torch profiler trace（prof_traces/）仍是算子级/注解级分析的补充。
