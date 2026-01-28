#!/usr/bin/env bash
set -euo pipefail

# =========================
# GPU selection
# =========================
GPU_IDS="7"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
TP="${#GPU_ARRAY[@]}"

# =========================
# Paths (edit here)
# =========================
TRAIN_ROOT="../data/logical"
QUERY_ROOT="../data"

DEMO_METHOD="bm25"   # logical | embed | bm25 | random
OUT_ROOT="${DEMO_METHOD}_results"

# =========================
# Domains (edit here)
# =========================
SOURCE_DOMAINS=("AR-LSAT" "ProofWriter" "FOLIO" "gsm8k" "ProntoQA")
TARGET_DOMAINS=("LogicalDeduction")

# ==============================
# 按目标域返回对应 SPLIT
# ==============================
get_split_by_target() {
  local tgt="$1"
  case "${tgt}" in
    "ProntoQA")         echo "dev" ;;
    "FOLIO")            echo "dev" ;;
    "LogicalDeduction") echo "dev" ;;  # 你没说明，这里默认 test
    *)                  echo "test" ;;  # 默认值
  esac
}

# ==============================
# 按源域返回对应 SPLIT
# ==============================
get_split_by_source() {
  local src="$1"
  case "${src}" in
    "ProntoQA")         echo "dev" ;;
    *)                  echo "train" ;;  # 默认值
  esac
}

# =========================
# Demo selection params
# =========================
K_LIST=(0 1 2 4 6 8 16)


SAME_TYPE_RATIO=0.75
LAMBDA_DIV=0.7
PREN=200   # number of candidates to consider before selection

# =========================
# Local models (edit here)
# =========================
BGE_PATH="../llms/bge-large-en-v1.5"
QWEN_PATH="../llms/Qwen2.5-14B-Instruct"

DO_STRUCT_PREDICT=1   # whether to do structure prediction
DO_INFER=1            # whether to do final inference after demo selection

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
  local SPLIT=$(get_split_by_source "${SRC}")
  echo "${TRAIN_ROOT}/${SRC}_${SPLIT}_cot_logical.json"
}

query_file_for_tgt_split () {
  local TGT="$1"
  local SPLIT=$(get_split_by_target "${TGT}")
  echo "${QUERY_ROOT}/${TGT}/${SPLIT}.json"
}


out_file_for_src_tgt_split_k () {
  local SRC="$1"
  local TGT="$2"
  local K="$3"
  local SPLIT=$(get_split_by_target "${TGT}")
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
      
      QUERY_FILE="$(query_file_for_tgt_split "${TGT}")"
      if [ ! -f "${QUERY_FILE}" ]; then
        echo "[SKIP] Missing query file: ${QUERY_FILE}"
        continue
      fi

      OUT_FILE="$(out_file_for_src_tgt_split_k "${SRC}" "${TGT}" "${K}")"
      mkdir -p "$(dirname "${OUT_FILE}")"

      echo "============================================================"
      echo "[RUN] K=${K}  SRC=${SRC}  ->  TGT=${TGT}"
      echo "      DEMO_METHOD=${DEMO_METHOD}"
      echo "      TRAIN_GLOB=${TRAIN_GLOB}"
      echo "      QUERY_FILE=${QUERY_FILE}"
      echo "      OUT_FILE=${OUT_FILE}"
      echo "      BGE_PATH=${BGE_PATH}  QWEN_PATH=${QWEN_PATH}"
      echo "      TP=${TP} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      echo "      STRUCT_BATCH=${STRUCT_BATCH} INFER_BATCH=${INFER_BATCH}"
      echo "============================================================"

      python3 select_icl_demos.py \
        --demo_method "${DEMO_METHOD}" \
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

echo "[DONE] All runs finished. Outputs at: ${OUT_ROOT}"
