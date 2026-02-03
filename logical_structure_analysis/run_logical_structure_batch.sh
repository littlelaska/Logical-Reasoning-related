#!/usr/bin/env bash
# 对指定数据集批量分析每个样本的 CoT 逻辑结构，调用 SiliconFlow Batch API 完成
set -e

# ======= You edit here =======
ROOT_DATA_DIR="../data"
OUT_DIR="../data/logical"
SPLIT="train"

# 要处理的数据集（对应文件前缀：{DATASET}_{SPLIT}_cot.json）
DATASETS=("AR-LSAT")
TASK="cot"   # cot/struct 分别对应生成cot或者分析cot的逻辑结构

# SiliconFlow Batch 配置
export SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:-sk-umzuhgrnvqjzsyagbkxnruljvrxtiqepwjejknkedamfcjsi}"
export SILICONFLOW_BASE_URL_V1="${SILICONFLOW_BASE_URL_V1:-https://api.siliconflow.cn/v1}"

# 文档支持的模型示例里包含 deepseek-ai/DeepSeek-R1 :contentReference[oaicite:9]{index=9}
export SILICONFLOW_MODEL="${SILICONFLOW_MODEL:-deepseek-ai/DeepSeek-V3}"
export SILICONFLOW_COMPLETION_WINDOW="${SILICONFLOW_COMPLETION_WINDOW:-24h}"

# batch 输入文件每个最多 5000 行（建议留余量）:contentReference[oaicite:10]{index=10}
BATCH_SIZE=4500
POLL_INTERVAL=30

mkdir -p "${OUT_DIR}"

for DS in "${DATASETS[@]}"; do
  DATA_DIR="${ROOT_DATA_DIR}/${DS}"
  IN_FILE="${DATA_DIR}/${DS}_${SPLIT}_cot.json"
  echo "[RUN] ${IN_FILE}"

  python3 add_logical_structure_batch.py \
    --inputs "${IN_FILE}" \
    --input_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --task "${TASK}" \
    --split "${SPLIT}" \
    --api_key "${SILICONFLOW_API_KEY}" \
    --model "${SILICONFLOW_MODEL}" \
    --skip_existing \
    --batch_size "${BATCH_SIZE}" \
    --poll_interval "${POLL_INTERVAL}" \
    --completion_window "${SILICONFLOW_COMPLETION_WINDOW}" \
    --workdir "./batch_tmp_${DS}_${SPLIT}"
done

echo "[DONE] outputs in ${OUT_DIR}"
