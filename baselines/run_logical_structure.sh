#!/usr/bin/env bash
set -e

# ======= Config you edit =======
ROOT_DATA_DIR="../data"          # 你的json文件目录
OUT_DIR="../data/logical"   # 输出目录（可改）
SPLIT="dev"                   # dev/test/train 等

# 你要处理的数据集列表（按文件名前缀）
DATASETS=("ProntoQA")          # 例如 ("AR-LSAT" "FOLIO" "ProofWriter")

# DeepSeek 配置：建议用环境变量注入 key
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-umzuhgrnvqjzsyagbkxnruljvrxtiqepwjejknkedamfcjsi}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-ai/DeepSeek-V3}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.siliconflow.cn/v1}"

# ======= Run =======
mkdir -p "${OUT_DIR}"

for DS in "${DATASETS[@]}"; do
  DATA_DIR="${ROOT_DATA_DIR}/${DS}"
  IN_FILE="${DATA_DIR}/${DS}_${SPLIT}_cot.json"
  echo "[RUN] ${IN_FILE}"

  python3 add_logical_structure.py \
    --inputs "${IN_FILE}" \
    --output_dir "${OUT_DIR}" \
    --skip_existing \
    --save_every 50
done

echo "[DONE] outputs in ${OUT_DIR}"
