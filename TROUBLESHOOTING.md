# Profiling 问题与修正记录

本文档记录采集方案迭代过程中的问题、排查过程与修正结论。当前验证过的正确做法见
[README.md](README.md) 第 3 节。

## 1. nsys kernel 采集失败：不是平台限制，是版本回归（2026-09-01 确认）

早期结论"nsys 在 GB10 上无法采集 kernel，属平台限制"**已被推翻**。v3~v5 三轮对照
实验确认 nsys 版本是决定性变量：

| nsys 版本 | 环境 | kernel 结果 |
|---|---|---|
| **2025.3.2**（bind-mount 宿主机 `/opt/nvidia/nsight-systems/2025.3.2` 进容器） | vllm-nsys:fp8 容器内 | ✅ 完整（17.7万~125万条，视采集范围） |
| 2025.6.3（镜像自带） | 同容器 | ❌ 0~417 条（仅启动期），**静默失败无报错** |
| 2026.1.3 | 宿主机 | ❌ 0 条 |

**教训**：升级 nsys 后必须重新验证 kernel 泳道，不能沿用旧版本结论；GB10 平台本身
没有限制。

**衍生混淆**：早期"窗口模式（`--capture-range=cudaProfilerApi`）KERNEL 表缺失"
的结论也是同一回归的产物——当时窗口实验恰好用 2025.6.3 跑的。2025.3.2 下窗口模式
完全可用，且是现在的推荐方案。

## 2. 全程采集的 CUPTI 尾部丢事件（v5 时代）

**现象**：FP8 v5 首采报告 decode step73–139（67 步，约占 decode 一半）完全无 kernel；
客户端 TPOT 全程 29.78ms 不变，证明 GPU 在跑。证据链：KERNEL/RUNTIME/MEMCPY 三张表
全部止于 229.55s，而 NVTX（CPU 侧注入，不依赖 CUPTI）持续到 231.57s。

**排查**：
- `--cuda-flush-interval=1000` 把丢失从 ~1.9s（67 步）减到 ~0.8s（27 步），但无法归零；
- bench 结束后空闲 10s 再 stop，丢失仍 ~0.82s 且锚定在负载尾段 → 与停止时机无关；
- 结论：CUPTI activity buffer 尾部未落盘/丢失，全程采集模式下无法根治。

**判别"数据缺失 vs GPU 空闲"的方法**：NVTX 仍在推进而 KERNEL/RUNTIME/MEMCPY 同时
归零 → CUPTI 停采；若 RUNTIME 仍在则仅 activity 丢失。每次报告使用前必须做
空步检测。

**根治**：窗口模式 + cudaProfilerStop 强制 flush（见下条）。

## 3. 窗口模式 cudaProfilerStop 不触发的陷阱（v6_nc 首轮）

**现象**：nc 报告（1024 out，decode 1024 步）STOP 设 1060，重采后尾部又丢 26 步
（~1.0s）——窗口模式同款丢失。

**根因**：实际最后 execute_model 步是 1039。bench 结束后引擎空闲、不再调
execute_model，步计数器冻结在 1039；补丁 stop 判断用 `==`，`cudaProfilerStop`
**永远不触发**（sqlite `CUPTI_ACTIVITY_KIND_RUNTIME` 表中 0 次 cuProfilerStop
调用可确证），报告退化为 docker stop 收尾。

**修正**：
- 补丁 stop 判断 `==` 改为 `>=`（`prof_patch/sitecustomize.py`）；
- STOP 必须设在**必然在负载内触发**的步数：步数 = 8（warmup）+ ~5（prefill chunk
  + 混合步）+ 输出 token 数，再减 2~5 步余量（128 out → 145；1024 out → 1037）。
  宁小勿大：设小了只截断尾部几步 decode，设大了直接退化成全程采集。

**验证手段**：采完后查 sqlite 中 cuProfilerStop 的 runtime 调用次数 = 1 即触发成功。

## 4. stop 触发后客户端 TPOT 虚高的观测伪影

cudaProfilerStop 在 bench 尾段触发后，nsys 立即开始 report finalization（处理
~120 万事件、写 43~46MB 报告），CPU 被占导致 SSE token 流尾部延迟送达客户端。
该轮客户端 TPOT 明显虚高（FP8 nc 47.52ms vs 服务端步周期 37.79ms；FP4 nc 37.01ms
vs 27.45ms），但服务端每步周期全程稳定、报告数据不受影响。

**做法**：采集轮的客户端 TPOT 不可引用；性能数字引用干净跑分或 stop 未在 bench 内
触发的复测轮（FP8 nc 37.63ms / FP4 nc 27.73ms）；步周期以 nsys 时间线为准。

## 5. docker stop 超时丢失报告（v4 教训）

默认 `docker stop` 10s 后 SIGKILL，nsys 来不及完成 report finalization，报告丢失
或损坏。**必须 `docker stop -t 180`**。

## 6. torch profiler 采集窗口被 warmup 消耗

窗口步数是全局计数的，warmup 请求也占 step。若预热生成 1024 token 会消耗 1024 步
窗口，导致真正的基准段没被采到。**预热只生成 8 token**；窗口 280 步 ≈
prefill(4~8 chunk) + ~270 decode 步。

## 7. NVTX 标签与 sqlite 分析的坑

- 混合步误标：prefill chunk 与 decode 混调的步 scheduled tokens < 512，被补丁的
  512-token 阈值标成 `decode_stepN`（如 step14 "tokens49" 实际 ~1300 kernel、
  400ms+ busy，是 prefill 规模）。判相以 kernel 数/busy 为准。
- NVTX 文本**内联在 `NVTX_EVENTS.text` 列**（不经 StringIds join），且部分行为
  NULL，查询需 `COALESCE(text,'')`。
- decode 步耗时：NVTX 区间时长只含 CPU launch（~3-7ms）；步周期用相邻区间 start
  差值取 **median**（mean 会被 warmup→bench 间隔、prefill 间隙的 >100ms 大间隙
  拉高，FP4 曾因此得出 29.42ms 的错误均值）。
- GPU busy：用 [步 start, 下一步 start) 窗口内 kernel 时长之和；用区间自身
  [start, end) 会得到"每步 busy 仅 ~7ms"的误导结果。

## 8. 其他已知事项

- 2026.1.3 导出器解析 2025.6.3 所采 GPU metrics 时 GPC Clock Frequency 出现异常值
  （负数/大数），SM/Tensor 吞吐类指标不受影响；
- 全程采集报告中，CUDA Graph 捕获期（模型加载后 110~160s）会记录捕获产物 kernel
  （`FillFunctor<int>`、`delayStreamKernel` 等），勿计入负载归因；
- `verify_bench.py` 的 warmup 小 prompt（64 token < 512 阈值）会被标成 decode，
  对粗粒度阶段切分无影响；
- GitHub 对 >50MB 文件有警告（<100MB 可入库），`.nsys-rep` 报告按现有惯例直接提交；
  更大的 trace 走 GitHub Release。

## 9. 已删除的历史报告

以下报告已被更优方案取代或本身缺数据，本地与 git 均已删除：

- `qwen3_fp8.nsys-rep` / `qwen3_fp4.nsys-rep`：2025.6.3 所采全程报告，kernel 泳道
  为空（仅 GPU metrics）；
- `qwen3_fp{8,4}_prefill.nsys-rep` / `qwen3_fp{8,4}_decode.nsys-rep`：旧分窗口报告
  （被 v6 窗口模式取代）；
- `qwen3_fp{8,4}_nvtx.nsys-rep`：NVTX 试验产物；
- 各中间 sqlite 导出与 `.tmp_nsys/`（约 7GB）。
