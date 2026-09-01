#!/bin/bash
# nsys v5 采集方案（已验证）：容器内 nsys 2025.3.2 全程采集
#   - bind-mount 宿主机 2025.3.2：镜像自带 2025.6.x+ 有 kernel 采集回归，采不到 kernel
#   - 全程立即采集（不用 --capture-range=cudaProfilerApi：窗口模式下 KERNEL 表缺失）
#   - --cuda-graph-trace=node：decode CUDA Graph 展开为逐 kernel
#   - VLLM_NVTX_LABEL=1：prof_patch 补丁打 prefill/decode 阶段标签
#   - 如需 GPU metrics 曲线：加 --gpu-metrics-devices=all（不能与其他 nsys GPU-metrics 会话并发）
# 用法：
#   bash run_verify_nsys.sh
#   轮询 curl localhost:8000/health 到 200 后：
#   docker exec vllm-fp8-nsys-v3 python3 /work/verify_bench.py
#   docker stop -t 180 vllm-fp8-nsys-v3   # 必须 -t 180，否则 nsys 来不及写报告
set -e
NAME=vllm-fp8-nsys-v3
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
  --entrypoint "" \
  vllm-nsys:fp8 \
  /opt/nvidia/nsight-systems/2025.3.2/bin/nsys profile \
    -o /work/nsys_reports/qwen3_fp8_v5 \
    --force-overwrite=true \
    -t cuda,nvtx,osrt \
    --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-30B-A3B-FP8 \
    --served-model-name qwen3-30b-a3b \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000

echo "[nsys] 容器已启动。等待 /health 返回 200 后："
echo "  docker exec $NAME python3 /work/verify_bench.py"
echo "  docker stop -t 180 $NAME   # 报告输出: nsys_reports/qwen3_fp8_v5.nsys-rep"
