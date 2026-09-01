#!/bin/bash
# nsys 采集方案（已验证）：容器内 nsys 2025.3.2
#   - bind-mount 宿主机 2025.3.2：镜像自带 2025.6.x+ 有 kernel 采集回归，采不到 kernel
#   - 支持全程立即采集或窗口采集（--capture-range=cudaProfilerApi，由 START/STOP 环境变量启用）
#   - --cuda-graph-trace=node：decode CUDA Graph 展开为逐 kernel
#   - VLLM_NVTX_LABEL=1：prof_patch 补丁打 prefill/decode 阶段标签
#   - --cuda-flush-interval=1000：每 1s 落盘 CUPTI activity buffer，防尾部丢事件
#   - 推荐窗口模式（v6）：设 VLLM_CUDA_PROFILER_{START,STOP}_AT_STEP 后仅采集负载段，
#     cudaProfilerStop 触发强制 flush，彻底避免尾部丢失，报告体积也小一个量级
#   - 如需 GPU metrics 曲线：加 --gpu-metrics-devices=all（不能与其他 nsys GPU-metrics 会话并发）
# 用法：
#   bash run_verify_nsys.sh [fp8|fp4|fp8nc|fp4nc]
#   nc 变体对齐 graph+no-compile 基准（mode=NONE + FULL_DECODE_ONLY，bench 用 1K 输出）
#   轮询 curl localhost:8000/health 到 200 后：
#   docker exec <容器> env BENCH_OUTPUT_LEN=<128|1024> python3 /work/verify_bench.py
#   docker stop -t 180 <容器>   # 必须 -t 180，否则 nsys 来不及写报告
set -e
# 用法: bash run_verify_nsys.sh [fp8|fp4]  (默认 fp8)
QUANT=${1:-fp8}
EXTRA_VLLM_ARGS=()
case $QUANT in
  fp8)   MODEL=/models/Qwen3-30B-A3B-FP8;   OUT=qwen3_fp8_v5 ;;
  fp4)   MODEL=/models/Qwen3-30B-A3B-NVFP4; OUT=qwen3_fp4_v5 ;;
  fp8nc) MODEL=/models/Qwen3-30B-A3B-FP8;   OUT=qwen3_fp8_v6_nc
         EXTRA_VLLM_ARGS=(--compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}') ;;
  fp4nc) MODEL=/models/Qwen3-30B-A3B-NVFP4; OUT=qwen3_fp4_v6_nc
         EXTRA_VLLM_ARGS=(--compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}') ;;
  *) echo "unknown quant: $QUANT (use fp8|fp4|fp8nc|fp4nc)"; exit 1 ;;
esac
OUT=${V5_OUT:-$OUT}   # 可用 V5_OUT=xxx 覆盖输出名
# 可选：cudaProfilerApi 窗口模式（v6 实验）：设置 START/STOP 环境变量后启用，
# 采集窗口由 prof_patch 在指定 step 调 torch.cuda.profiler.start/stop 控制，
# 配合 --flush-on-cudaprofilerstop 在窗口结束时强制 flush，避免尾部丢失
NSYS_RANGE=""
EXTRA_ENV=""
if [ -n "$VLLM_CUDA_PROFILER_START_AT_STEP" ]; then
  NSYS_RANGE="--capture-range=cudaProfilerApi --capture-range-end=stop"
  EXTRA_ENV="-e VLLM_CUDA_PROFILER_START_AT_STEP=$VLLM_CUDA_PROFILER_START_AT_STEP -e VLLM_CUDA_PROFILER_STOP_AT_STEP=$VLLM_CUDA_PROFILER_STOP_AT_STEP"
fi
NAME=vllm-nsys-$QUANT
docker rm -f $NAME >/dev/null 2>&1 || true

docker run -d --name $NAME \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host --privileged \
  -v /home/vincent/o00806383:/work \
  -v /home/vincent/o00806383/models:/models \
  -v /opt/nvidia/nsight-systems/2025.3.2:/opt/nvidia/nsight-systems/2025.3.2:ro \
  -e PYTHONPATH=/work/prof_patch \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e VLLM_NVTX_LABEL=1 \
  $EXTRA_ENV \
  --entrypoint "" \
  vllm-nsys:fp8 \
  /opt/nvidia/nsight-systems/2025.3.2/bin/nsys profile \
    -o /work/nsys_reports/$OUT \
    --force-overwrite=true \
    -t cuda,nvtx,osrt \
    --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node \
    --cuda-flush-interval=1000 \
    $NSYS_RANGE \
    python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name qwen3-30b-a3b \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    "${EXTRA_VLLM_ARGS[@]}"

echo "[nsys] 容器 $NAME ($QUANT) 已启动。等待 /health 返回 200 后："
echo "  docker exec $NAME env BENCH_OUTPUT_LEN=<128|1024> python3 /work/verify_bench.py"
echo "  docker stop -t 180 $NAME   # 报告输出: nsys_reports/${OUT}.nsys-rep"
