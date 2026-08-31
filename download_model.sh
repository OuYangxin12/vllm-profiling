#!/bin/bash

# ============================================
# Qwen3-30B-A3B 模型下载脚本
# ============================================

MODEL_DIR="/home/vincent/o00806383/models"
MODEL_NAME="Qwen3-30B-A3B-FP8"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

echo "=========================================="
echo "下载 Qwen3-30B-A3B 模型"
echo "=========================================="

# 创建模型目录
mkdir -p "$MODEL_DIR"

# 检查是否已安装 modelscope
if ! command -v modelscope &> /dev/null; then
    echo "[INFO] 安装 modelscope..."
    pip install modelscope -q
fi

echo "[INFO] 开始下载 FP8 模型到: $MODEL_PATH"
echo "[INFO] 模型大小约 30GB，请耐心等待..."
echo ""

# 下载模型
modelscope download \
    --model "Qwen/$MODEL_NAME" \
    --local_dir "$MODEL_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] 模型下载完成！"
    echo "=========================================="
    echo ""
    echo "模型路径: $MODEL_PATH"
    echo ""
    echo "现在可以启动 vLLM 服务:"
    echo "  cd /home/vincent/o00806383"
    echo "  ./start_vllm.sh"
else
    echo ""
    echo "[ERROR] 模型下载失败"
    exit 1
fi
