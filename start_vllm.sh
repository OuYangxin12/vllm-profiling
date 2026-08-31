#!/bin/bash

# ============================================
# Qwen3-30B-A3B vLLM 推理服务启动脚本
# ============================================

# 配置参数 - 请根据实际情况修改
MODEL_PATH="${MODEL_PATH:-/home/vincent/o00806383/models/Qwen3-30B-A3B-FP8}"  # 模型路径
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-30b-a3b}"         # 服务模型名称
HOST_PORT="${HOST_PORT:-8000}"                                   # 宿主机端口
TP_SIZE="${TP_SIZE:-2}"                                          # Tensor Parallel size (GPU数量)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"                          # 最大上下文长度
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"         # GPU显存利用率

# Docker 镜像
IMAGE="nvcr.io/nvidia/vllm:26.05-py3"

# 容器名称
CONTAINER_NAME="vllm-qwen3-30b-a3b"

# ============================================
# 检查前置条件
# ============================================
echo "=========================================="
echo "Qwen3-30B-A3B vLLM 推理服务启动"
echo "=========================================="

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker 未运行或无权限访问 Docker"
    exit 1
fi

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] 模型路径不存在: $MODEL_PATH"
    echo "[INFO] 请先下载模型，例如使用 modelscope:"
    echo "       pip install modelscope"
    echo "       modelscope download --model Qwen/Qwen3-30B-A3B --local_dir $MODEL_PATH"
    echo ""
    read -p "是否继续启动容器？(y/n): " confirm
    if [ "$confirm" != "y" ]; then
        exit 0
    fi
fi

# 检查 GPU 数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "[ERROR] 未检测到 GPU"
    exit 1
fi

echo "[INFO] 检测到 $GPU_COUNT 个 GPU"
echo "[INFO] Tensor Parallel size: $TP_SIZE"

if [ "$TP_SIZE" -gt "$GPU_COUNT" ]; then
    echo "[WARN] TP_SIZE ($TP_SIZE) 大于可用 GPU 数量 ($GPU_COUNT)"
    TP_SIZE=$GPU_COUNT
    echo "[INFO] 自动调整 TP_SIZE 为 $TP_SIZE"
fi

# ============================================
# 停止已存在的容器
# ============================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] 停止已存在的容器: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME > /dev/null 2>&1
    docker rm $CONTAINER_NAME > /dev/null 2>&1
fi

# ============================================
# 启动 vLLM 服务
# ============================================
echo ""
echo "[INFO] 启动 vLLM 服务..."
echo "-------------------------------------------"
echo "  模型路径: $MODEL_PATH"
echo "  模型名称: $SERVED_MODEL_NAME"
echo "  服务端口: $HOST_PORT"
echo "  TP Size:  $TP_SIZE"
echo "  最大长度: $MAX_MODEL_LEN"
echo "-------------------------------------------"
echo ""

# 获取模型父目录用于挂载
MODEL_DIR=$(dirname "$MODEL_PATH")

docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    -v "$MODEL_DIR:/models" \
    -e HF_HOME=/models \
    --entrypoint "" \
    $IMAGE \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "/models/$(basename $MODEL_PATH)" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port $HOST_PORT

# 检查启动结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] 容器启动成功！"
    echo "=========================================="
    echo ""
    echo "查看日志: docker logs -f $CONTAINER_NAME"
    echo "停止服务: docker stop $CONTAINER_NAME"
    echo "删除容器: docker rm $CONTAINER_NAME"
    echo ""
    echo "API 端点 (等待服务就绪后):"
    echo "  - 健康检查: curl http://localhost:$HOST_PORT/health"
    echo "  - 模型列表: curl http://localhost:$HOST_PORT/v1/models"
    echo "  - Chat API: curl http://localhost:$HOST_PORT/v1/chat/completions"
    echo ""
    
    # 等待服务就绪
    echo "[INFO] 等待服务启动..."
    sleep 5
    docker logs --tail 20 $CONTAINER_NAME
else
    echo ""
    echo "[ERROR] 容器启动失败"
    docker logs $CONTAINER_NAME 2>/dev/null | tail -20
    exit 1
fi
