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
if os.environ.get("VLLM_CUDA_PROFILER_STOP_AT_STEP"):

    def _patch_nsys():
        import torch
        import vllm.v1.worker.gpu_worker as gw

        _stop = int(os.environ["VLLM_CUDA_PROFILER_STOP_AT_STEP"])
        _s = {"n": 0, "started": False, "stopped": False}
        _orig = gw.Worker.execute_model

        def patched(self, *args, **kwargs):
            if not _s["started"]:
                _s["started"] = True
                try:
                    torch.cuda.profiler.start()
                    print("[nsys-patch] cudaProfilerStart", flush=True)
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
