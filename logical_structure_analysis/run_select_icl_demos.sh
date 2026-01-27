#!/usr/bin/env bash
set -euo pipefail

# =========================
# GPU selection
# =========================
GPU_IDS="1"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
TP="${#GPU_ARRAY[@]}"

# =========================
# Paths (edit here)
# =========================
TRAIN_ROOT="../data/logical"
QUERY_ROOT="../data"
OUT_ROOT="logical_results"

# =========================
# Domains (edit here)
# =========================
SOURCE_DOMAINS=("AR-LSAT" "ProofWriter" "FOLIO" "gsm8k" "ProntoQA")
TARGET_DOMAINS=("LogicalDeduction")
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

DO_STRUCT_PREDICT=1
DO_INFER=1

# vLLM config
GPU_UTIL=0.92

# generation params
TEMP=0.0
MAX_NEW_TOKENS=256

# =========================
# NEW: Batch sizes for vLLM
# =========================
# 结构预测输出很短，一般可以开大一些
STRUCT_BATCH=64
# 推理 batch 受 prompt 长度影响，建议从 8/16 起
INFER_BATCH=16

# =========================
# File naming rules
# =========================
train_glob_for_src () {
  local SRC="$1"
  echo "${TRAIN_ROOT}/${SRC}_*_cot_logical.json"
}

query_file_for_tgt_split () {
  local TGT="$1"
  local SPLIT="$2"
  echo "${QUERY_ROOT}/${TGT}/${SPLIT}.json"
}

out_file_for_src_tgt_split_k () {
  local SRC="$1"
  local TGT="$2"
  local SPLIT="$3"
  local K="$4"
  echo "${OUT_ROOT}/${TGT}/k${K}/${SRC}__${TGT}_${SPLIT}_k${K}_icl.json"
}

# =========================
# Run
# =========================
mkdir -p "${OUT_ROOT}"
for TGT in "${TARGET_DOMAINS[@]}"; do
  mkdir -p "${OUT_ROOT}/${TGT}"
done

for K in "${K_LIST[@]}"; do
  echo "###############################"
  echo "[K] Running K=${K}"
  echo "###############################"

  for SRC in "${SOURCE_DOMAINS[@]}"; do
    TRAIN_GLOB="$(train_glob_for_src "${SRC}")"

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

        OUT_FILE="$(out_file_for_src_tgt_split_k "${SRC}" "${TGT}" "${SPLIT}" "${K}")"
        mkdir -p "$(dirname "${OUT_FILE}")"

        echo "============================================================"
        echo "[RUN] K=${K}  SRC=${SRC}  ->  TGT=${TGT}  SPLIT=${SPLIT}"
        echo "      TRAIN_GLOB=${TRAIN_GLOB}"
        echo "      QUERY_FILE=${QUERY_FILE}"
        echo "      OUT_FILE=${OUT_FILE}"
        echo "      TP=${TP} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "      STRUCT_BATCH=${STRUCT_BATCH} INFER_BATCH=${INFER_BATCH}"
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
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --struct_batch_size "${STRUCT_BATCH}" \
          --infer_batch_size "${INFER_BATCH}"

      done
    done
  done
done

echo "[DONE] All runs finished. Outputs at: ${OUT_ROOT}"
