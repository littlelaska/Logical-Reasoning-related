#!/usr/bin/env bash
set -euo pipefail

# =========================
# GPU selection
# =========================
# Example:
# GPU_IDS="0"
# GPU_IDS="0,1"
# GPU_IDS="2,3,4"
GPU_IDS="1"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# 自动根据 GPU 数量设置 TP
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
TP="${#GPU_ARRAY[@]}"

# =========================
# Paths (edit here)
# =========================
TRAIN_ROOT="../data/logical"         # 每个源域一个目录：../data/train/AR-LSAT/...
QUERY_ROOT="../data"     # 每个目标域一个目录：../data/new_tasks/FOLIO/...
OUT_ROOT="logical_results"

# =========================
# Domains (edit here)
# =========================
# 源域：用于 demo pool（可多个）
SOURCE_DOMAINS=("AR-LSAT" "ProofWriter" "FOLIO" "gsm8k" "ProntoQA")

# 目标域：用于测试（可多个）
TARGET_DOMAINS=("LogicalDeduction")

# 测试集划分（可多个）
SPLITS=("dev")

# =========================
# Demo selection params
# =========================
K_LIST=(0 1 2 4 6 8)

SAME_TYPE_RATIO=0.75
LAMBDA_DIV=0.7
PREN=200

# =========================
# Local models (edit here)
# =========================
BGE_PATH="../llms/bge-large-en-v1.5"
QWEN_PATH="../llms/Qwen2.5-14B-Instruct"


DO_STRUCT_PREDICT=1   # 1=对 query 预测结构；0=不预测（没有则 Other）
DO_INFER=1            # 1=直接跑 ICL 推理并写 pred_answer；0=只生成 prompt/demos

# vLLM config
GPU_UTIL=0.92

# generation params
TEMP=0.0
MAX_NEW_TOKENS=256

# =========================
# File naming rules (edit if needed)
# =========================
# 训练集（demo pool）：支持 glob。你可以改成 *_cot_logical.json 之类
train_glob_for_src () {
  local SRC="$1"
  # 例：../data/train/AR-LSAT_train_cot_logical.json
  echo "${TRAIN_ROOT}/${SRC}_*_cot_logical.json"
}

# 测试集 query 文件：你可以根据你的真实命名改这里
query_file_for_tgt_split () {
  local TGT="$1"
  local SPLIT="$2"
  # 例：../data/new_tasks/FOLIO/FOLIO_dev.json
  echo "${QUERY_ROOT}/${TGT}/${SPLIT}.json"
}

# 输出文件：按 target 域分目录，文件名包含 source__target
out_file_for_src_tgt_split () {
  local SRC="$1"
  local TGT="$2"
  local SPLIT="$3"
  # 例：../data/icl_outputs/FOLIO/AR-LSAT__FOLIO_dev_icl.json
  echo "${OUT_ROOT}/${TGT}/${SRC}__${TGT}_${SPLIT}_icl.json"
}

# =========================
# Run
# =========================
mkdir -p "${OUT_ROOT}"

for TGT in "${TARGET_DOMAINS[@]}"; do
  mkdir -p "${OUT_ROOT}/${TGT}"
done

for SRC in "${SOURCE_DOMAINS[@]}"; do
  TRAIN_GLOB="$(train_glob_for_src "${SRC}")"

  # 检查源域训练集是否存在
  if ! ls ${TRAIN_GLOB} >/dev/null 2>&1; then
    echo "[WARN] No train files for SRC=${SRC} with glob: ${TRAIN_GLOB}"
    continue
  fi

  for TGT in "${TARGET_DOMAINS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
      QUERY_FILE="$(query_file_for_tgt_split "${TGT}" "${SPLIT}")"
      if [ ! -f "${QUERY_FILE}" ]; then
        echo "[SKIP] Missing query file: ${QUERY_FILE}"
        continue
      fi

      OUT_FILE="$(out_file_for_src_tgt_split "${SRC}" "${TGT}" "${SPLIT}")"
      mkdir -p "$(dirname "${OUT_FILE}")"

      echo "============================================================"
      echo "[RUN] SRC=${SRC}  ->  TGT=${TGT}  SPLIT=${SPLIT}"
      echo "      TRAIN_GLOB=${TRAIN_GLOB}"
      echo "      QUERY_FILE=${QUERY_FILE}"
      echo "      OUT_FILE=${OUT_FILE}"
      echo "============================================================"

      python3 select_icl_demos.py \
        --train_glob "${TRAIN_GLOB}" \
        --query_file "${QUERY_FILE}" \
        --out_file "${OUT_FILE}" \
        --k "${K}" \
        --same_type_ratio "${SAME_TYPE_RATIO}" \
        --lambda_div "${LAMBDA_DIV}" \
        --preN "${PREN}" \
        --embed_model_path "${BGE_PATH}" \
        --qwen_model_path "${QWEN_PATH}" \
        $( [ "${DO_STRUCT_PREDICT}" -eq 1 ] && echo "--do_struct_predict" ) \
        $( [ "${DO_INFER}" -eq 1 ] && echo "--do_infer" ) \
        --tensor_parallel_size "${TP}" \
        --gpu_memory_utilization "${GPU_UTIL}" \
        --temperature "${TEMP}" \
        --max_new_tokens "${MAX_NEW_TOKENS}"

    done
  done
done

echo "[DONE] All runs finished. Outputs at: ${OUT_ROOT}"
