"""
vLLM Worker torch-profiler 自动触发补丁
=======================================
通过 PYTHONPATH 的 sitecustomize 机制加载。
- 第 1 次 execute_model：启动 torch profiler（覆盖 prefill）
- 第 N 次（默认 600）：停止并导出 chrome trace（覆盖 decode 阶段）
- 仅在设置 VLLM_TORCH_PROFILER_DIR 环境变量时生效
"""

import os

if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
    _STOP_AT_STEP = int(os.environ.get("VLLM_PROF_STOP_AT_STEP", "600"))
    _PREFIX = os.environ.get("VLLM_PROF_PREFIX", "qwen3_fp8")

    def _patch():
        import vllm.v1.worker.gpu_worker as gw

        _state = {"n": 0, "started": False, "stopped": False}
        _orig = gw.Worker.execute_model

        def patched(self, *args, **kwargs):
            if not _state["started"]:
                _state["started"] = True
                try:
                    self.profile(True, profile_prefix=_PREFIX)
                    print(f"[prof-patch] profiler started at step 0", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[prof-patch] start failed: {e}", flush=True)
            result = _orig(self, *args, **kwargs)
            _state["n"] += 1
            if _state["n"] == _STOP_AT_STEP and not _state["stopped"]:
                _state["stopped"] = True
                try:
                    self.profile(False)
                    print(f"[prof-patch] profiler stopped at step {_state['n']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[prof-patch] stop failed: {e}", flush=True)
            return result

        gw.Worker.execute_model = patched
        print("[prof-patch] execute_model patched", flush=True)

    try:
        _patch()
    except Exception as e:  # noqa: BLE001
        print(f"[prof-patch] patch error: {e}", flush=True)

# ---- nsys cudaProfilerApi 模式：窗口内调 torch.cuda.profiler.start/stop ----
# 配合 nsys profile --capture-range=cudaProfilerApi 使用，仅采集基准窗口
# 可用 VLLM_CUDA_PROFILER_START_AT_STEP 控制起始步（默认 0，从第一个 execute_model 开始），
# 例如预热 9 步（1 prefill + 8 decode）后：START=9/STOP=13 仅采 prefill，START=15 仅采 decode
if os.environ.get("VLLM_CUDA_PROFILER_STOP_AT_STEP"):

    def _patch_nsys():
        import torch
        import vllm.v1.worker.gpu_worker as gw

        _stop = int(os.environ["VLLM_CUDA_PROFILER_STOP_AT_STEP"])
        _start = int(os.environ.get("VLLM_CUDA_PROFILER_START_AT_STEP", "0"))
        _s = {"n": 0, "started": False, "stopped": False}
        _orig = gw.Worker.execute_model

        def patched(self, *args, **kwargs):
            if not _s["started"] and _s["n"] >= _start:
                _s["started"] = True
                try:
                    torch.cuda.profiler.start()
                    print(f"[nsys-patch] cudaProfilerStart at step {_s['n']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[nsys-patch] start failed: {e}", flush=True)
            result = _orig(self, *args, **kwargs)
            _s["n"] += 1
            if _s["n"] == _stop and not _s["stopped"]:
                _s["stopped"] = True
                try:
                    torch.cuda.profiler.stop()
                    print(f"[nsys-patch] cudaProfilerStop at step {_s['n']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[nsys-patch] stop failed: {e}", flush=True)
            return result

        gw.Worker.execute_model = patched
        print("[nsys-patch] execute_model patched (cudaProfilerApi)", flush=True)

    try:
        _patch_nsys()
    except Exception as e:  # noqa: BLE001
        print(f"[nsys-patch] patch error: {e}", flush=True)

# ---- NVTX 阶段标注模式：每个 execute_model 步包一层 NVTX range ----
# 一次全程采集即可在 nsys timeline 的 NVTX 泳道精确区分 prefill/decode，
# 阶段判定：num_scheduled_tokens >= 512 视为 prefill chunk，否则 decode step
# 仅在设置 VLLM_NVTX_LABEL=1 时生效（建议配合 VLLM_ENABLE_V1_MULTIPROCESSING=0）
if os.environ.get("VLLM_NVTX_LABEL"):

    def _patch_nvtx():
        import torch
        import vllm.v1.worker.gpu_worker as gw

        _s = {"n": 0}
        _orig = gw.Worker.execute_model

        def patched(self, *args, **kwargs):
            req = args[0] if args else kwargs.get("scheduler_output")
            ntok = getattr(req, "total_num_scheduled_tokens", None)
            if ntok is None:
                per = getattr(req, "num_scheduled_tokens", None)
                if isinstance(per, dict):
                    ntok = sum(per.values())
            phase = "prefill" if (ntok or 0) >= 512 else "decode"
            torch.cuda.nvtx.range_push(f"{phase}_step{_s['n']}_tokens{ntok}")
            try:
                return _orig(self, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
                _s["n"] += 1

        gw.Worker.execute_model = patched
        print("[nvtx-patch] execute_model nvtx-labeled", flush=True)

    try:
        _patch_nvtx()
    except Exception as e:  # noqa: BLE001
        print(f"[nvtx-patch] patch error: {e}", flush=True)
