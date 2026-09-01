# nsys 报告内容说明（采集方式与版本结论，2026-09-01 更新）

## 推荐方案（v5，已验证）：kernel + NVTX + 全程采集

由 `bash run_verify_nsys.sh [fp8|fp4]` 生成：容器内 bind-mount 宿主机 **nsys 2025.3.2**
（镜像自带 2025.6.3 采不到 kernel，见下方结论），全程立即采集（无 capture-range），
`--cuda-graph-trace=node` 展开 decode CUDA Graph，`VLLM_NVTX_LABEL=1` 补丁打
prefill/decode 阶段标签；收尾必须 `docker stop -t 180`。

| 报告 | kernel 数据 | NVTX 标签 | 验证基准（TTFT/TPOT） |
|---|---|---|---|
| qwen3_fp8_v5.nsys-rep (35MB, 全程) | 224,395 条；**尾部 ~0.8s（27 步）CUPTI 丢事件**，归因只用到 decode step116 | prefill ×4 / decode ×140 | 1513.5/30.51ms ≈ 干净基线 1515/29.15 |
| qwen3_fp4_v5.nsys-rep (53MB, 全程) | 314,796 条，无尾部丢失（间歇性，需逐报告空步检测） | prefill ×4 / decode ×140 | 1227.5/19.74ms ≈ 干净基线 1223/19.47 |
| **qwen3_fp8_v6.nsys-rep (11MB, 窗口)** | **140,947 条，负载段零丢失**：decode 129 步全部完整（1,072 kernel/步，busy 29.68ms ≈ 99.6% GPU busy） | prefill ×4 / decode ×129 | 1509.8/29.81ms ≈ 干净基线 |
| **qwen3_fp8_v6_nc.nsys-rep (43MB, 窗口)** | **1,171,187 条，零空步**：对齐 graph+no-compile 基准（1K out），decode 1023 步全量，busy 37.78ms ≈ 周期 37.79ms ≈ 99.9% GPU busy | prefill ×4 + 混合步14 / decode ×1023 | 2268/47.52ms（TPOT 含 stop 触发后 finalize 抢 CPU 的流式尾部伪影，首轮未触发 stop 复测为 37.63ms） |
| **qwen3_fp4_v6_nc.nsys-rep (46MB, 窗口)** | **1,251,219 条，零空步**：同上 1K out，busy 27.41ms ≈ 周期 27.45ms ≈ 99.9% | prefill ×4 + 混合步14 / decode ×1023 | 1285/37.01ms（同上，首轮复测 27.73ms ≈ 干净 26.76） |

**v6 窗口模式**（逐 kernel 归因首选）：`V5_OUT=qwen3_fp8_v6
VLLM_CUDA_PROFILER_START_AT_STEP=9 VLLM_CUDA_PROFILER_STOP_AT_STEP=145
bash run_verify_nsys.sh fp8`。早期“窗口模式 KERNEL 表缺失”的结论是 v3 时代
2025.6.3 版本回归的混淆——**2025.3.2 + 窗口模式完全可用**，且 cudaProfilerStop
触发 buffer 强制 flush，彻底解决全程采集的尾部丢事件问题（该问题在全程模式下
连空闲 10s + --cuda-flush-interval=1000 都无法归零，丢失锚定在最后一段 CUPTI
活动而非停止时机）。判别方法：NVTX 仍在推进而 KERNEL/RUNTIME/MEMCPY 同时归零
→ CUPTI 停采，非 GPU 空闲。

**STOP 陷阱（v6_nc 首轮教训）**：窗口 STOP 必须设在**必然在负载内触发**的步数。
nc 报告（1K out，decode 1024 步）首采 STOP=1060 > 实际最后 execute_model 步 1039，
bench 结束后引擎空闲不再调 execute_model，步计数器停在 1039，`cudaProfilerStop`
永不触发 → 报告由 docker stop 收尾，尾部又丢 26 步（~1.0s）。修复：补丁 stop
判断 `==` 改 `>=`（prof_patch/sitecustomize.py），重采 STOP=1037 后零空步。
另外 stop 触发后 nsys finalization 与 bench 尾部并发会拖慢 SSE 流式发送，
该轮客户端 TPOT 虚高不可引用，以服务端步周期为准。

summary 导出：`nsys stats --force-export=true <rep> > <rep去掉后缀>_stats.txt`
（产物 `qwen3_fp{8,4}_v5_stats.txt`，含 nvtx_sum / cuda_api_sum / cuda_gpu_kern_sum /
memops 各段，参考 openpi-nsys.txt 格式）。

sqlite 分析注意：NVTX 文本**内联在 `NVTX_EVENTS.text` 列**（不经 StringIds join）；
decode 步耗时用相邻 NVTX 区间 start 差值，且须剔除 >100ms 的间隙（warmup→bench
间隔、prefill 期间间隙）后取中位数，否则均值被拉高（FP8 稳态 30.39ms、FP4 稳态
19.72ms，均与客户端 TPOT 吻合；区间时长本身仅含 CPU launch ~7ms）。

## 已删除的历史报告（缺 kernel 数据）

早期用镜像自带 nsys 2025.6.3 所采的报告（kernel 泳道为空）与旧版分窗口报告
（已被 v6 窗口模式取代）已全部删除，不再保留：

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
- `--capture-range=cudaProfilerApi` 窗口模式在 **2025.3.2 下完全可用**（v6 验证，
  负载段零丢失）；旧“KERNEL 表缺失”结论系 2025.6.3 版本回归混淆。
- 逐 kernel 归因首选 **v6 窗口模式**（负载段零丢失）；全程 v5 报告需先做空步检测
  且归因止于 CUPTI 停采点；torch profiler trace（prof_traces/）仍是算子级/注解级
  分析的补充。
