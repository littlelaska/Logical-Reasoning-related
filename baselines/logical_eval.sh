# 源域：用于 demo pool（可多个）
SOURCE_DOMAINS=("AR-LSAT" "ProofWriter" "FOLIO" "gsm8k" "ProntoQA")

# 目标域：用于测试（可多个）
TARGET_DOMAINS=("LogicalDeduction")



# # 整体结果的统计
# python3 eval_accuracy.py \
#   --input logical_results/LogicalDeduction/AR-LSAT__LogicalDeduction_dev_icl.json

for SRC in "${SOURCE_DOMAINS[@]}"; do
  for TGT in "${TARGET_DOMAINS[@]}"; do
    OUT_FILE="logical_results/${TGT}/${SRC}__${TGT}_dev_icl.json"
    
    # 按照逻辑结构进行统计
    python3 eval_accuracy.py \
      --input "${OUT_FILE}" \
      --by_structure
  done
done

