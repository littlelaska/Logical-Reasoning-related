#!/usr/bin/env bash
# 统计 run_select_icl_demos.sh 生成的结果文件的准确率，可选不同的测试方式，embedding/bm25/random等
set -euo pipefail

SOURCE_DOMAINS=("AR-LSAT" "ProofWriter" "FOLIO" "gsm8k" "ProntoQA")
TARGET_DOMAINS=("LogicalDeduction")

# ✅ 可选 K 序列
K_LIST=(0 1 2 4 6 8)
DEMO_METHOD="bm25"   # logical | embed | bm25 | random
OUT_ROOT="${DEMO_METHOD}_results"

for K in "${K_LIST[@]}"; do
  echo "==============================="
  echo "[EVAL] K=${K}"
  echo "==============================="

  for SRC in "${SOURCE_DOMAINS[@]}"; do
    for TGT in "${TARGET_DOMAINS[@]}"; do
      OUT_FILE="${OUT_ROOT}/${TGT}/k${K}/${SRC}__${TGT}_dev_k${K}_icl.json"

      if [ ! -f "${OUT_FILE}" ]; then
        echo "[SKIP] Missing: ${OUT_FILE}"
        continue
      fi

      echo "--------------------------------"
      echo "[FILE] ${OUT_FILE}"
      python3 eval_accuracy.py \
        --input "${OUT_FILE}" \
        # --by_structure
    done
  done
done
