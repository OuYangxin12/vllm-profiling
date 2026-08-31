#!/bin/bash

# ============================================
# nvidia/Qwen3-30B-A3B-NVFP4 模型下载脚本
# 来源: https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4
# 说明: 本机 huggingface.co 不可达，使用 hf-mirror.com 镜像
# ============================================

export HF_ENDPOINT="https://hf-mirror.com"

MODEL_ID="nvidia/Qwen3-30B-A3B-NVFP4"
MODEL_DIR="/home/vincent/o00806383/models/Qwen3-30B-A3B-NVFP4"

echo "=========================================="
echo "下载 ${MODEL_ID}"
echo "镜像: ${HF_ENDPOINT}"
echo "目标目录: ${MODEL_DIR}"
echo "=========================================="

mkdir -p "$MODEL_DIR"

# 检查下载工具（新版为 hf，旧版为 huggingface-cli）
if command -v hf &> /dev/null; then
    DOWNLOADER="hf"
elif command -v huggingface-cli &> /dev/null; then
    DOWNLOADER="huggingface-cli"
else
    echo "[INFO] 未检测到 huggingface 工具，正在安装..."
    pip install -q "huggingface_hub[cli]"
    DOWNLOADER="huggingface-cli"
fi

echo "[INFO] 使用 ${DOWNLOADER} 下载，模型大小约 18GB，请耐心等待..."
echo ""

${DOWNLOADER} download \
    "$MODEL_ID" \
    --local-dir "$MODEL_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] 模型下载完成！"
    echo "=========================================="
    echo ""
    echo "模型路径: $MODEL_DIR"
    echo ""
    echo "注意: NVFP4 量化模型需要 Blackwell 架构 GPU（如 GB10/RTX 50 系/B200）"
    echo "且 vLLM 需支持 ModelOpt FP4 量化格式，启动示例:"
    echo "  vllm serve $MODEL_DIR --quantization modelopt_fp4"
else
    echo ""
    echo "[ERROR] 模型下载失败"
    exit 1
fi
