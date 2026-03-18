#!/usr/bin/env bash

# ==============================
# 固定配置（按需改）
# ==============================
MODE="RAG"                     # CoT / Direct / RAG / Logical
MODEL_NAME="qwen2-3"            # 给定模型
RAG_TOPK=30
BATCH_SIZE=32
MAX_NEW_TOKENS=8192
ZERO_SHOT=false                # 这里只跑 1–4 shot，所以不启用 zero-shot
DTYPE="bfloat16"
DB_TYPE="embedding"              # bm25  / embedding  /  random
EMBDEDDING_MODEL="../llms/bge-large-en-v1.5"   # text2vec-large-chinese/bge-large-en/bge-large-en-v1.5
REVERSE_FLAG=false
CONE_RERANK=false

# 源域（demo 来源）
SOURCE_DOMAINS=("gsm8k" "ProntoQA" "AR-LSAT" "ProofWriter" "FOLIO" "LogicalDeduction")    # "gsm8k" "ProntoQA" "AR-LSAT" "ProofWriter" "FOLIO" "LogicalDeduction"
SOURCE_DOMAINS=("commonsense")    # "gsm8k" "ProntoQA" "AR-LSAT" "ProofWriter" "FOLIO" "LogicalDeduction"
 
# 目标域（评测数据集）
TARGET_DOMAINS=("ProntoQA" "AR-LSAT" "ProofWriter" "FOLIO" "LogicalDeduction" "gsm8k")
TARGET_DOMAINS=("LogicalDeduction")

# shots: 1, 2, 3, 4
SHOTS=(0 1 2 4 8)
SHOTS=(0)

# 直接将prompt重复n次，查看是否对最终的结果有影响
REPEAT_TIMES=(1)
REPEAT_REVERSE=true
# ==============================
# 按目标域返回对应 SPLIT
# ==============================
get_split_by_target() {
  local tgt="$1"
  case "${tgt}" in
    "gsm8k")            echo "test" ;;
    "ProntoQA")         echo "dev" ;;
    "AR-LSAT")          echo "test" ;;
    "ProofWriter")      echo "test" ;;
    "FOLIO")            echo "dev" ;;
    "LogicalDeduction") echo "dev" ;;  # 你没说明，这里默认 test
    *)                  echo "test" ;;  # 默认值
  esac
}


# ==============================
# 主循环：源域 × 目标域 × 1–4 shot
# ==============================
for SRC in "${SOURCE_DOMAINS[@]}"; do
  for TGT in "${TARGET_DOMAINS[@]}"; do

    # =============== 跳过 in-domain ===============
    if [ "${SRC}" = "${TGT}" ]; then
      echo "[跳过] SRC=${SRC} == TGT=${TGT}（同域）"
      continue
    fi
    # ==============================================

    for SHOT in "${SHOTS[@]}"; do
      # 对prompt进行重复
      for REPEAT_TIME in "${REPEAT_TIMES[@]}"; do

        # 获取 split
        DEMONSTRATION_NUM=${SHOT}
        SPLIT=$(get_split_by_target "${TGT}")

        # 建立日志目录
        LOG_DIR="logs/${MODEL_NAME}/${MODE}/${SRC}__${TGT}"
        if [ "${MODE}" = "RAG" ]; then
          LOG_DIR="logs/${MODEL_NAME}/${MODE}_${DB_TYPE}/${SRC}__${TGT}"
        fi
        if [ "${REPEAT_REVERSE}" = true ]; then
          LOG_DIR="logs/${MODEL_NAME}/${MODE}_${DB_TYPE}_reverseContext/${SRC}__${TGT}"
        fi
        if [ "$CONE_RERANK" = true ] && [ "$MODE" = "RAG" ]; then
          LOG_DIR="logs/${MODEL_NAME}/${MODE}_cone_${DB_TYPE}/${SRC}__${TGT}"
        fi
        mkdir -p "${LOG_DIR}"
        LOG_FILE="${LOG_DIR}/shot${SHOT}_repeat${REPEAT_TIME}.log"

        echo "========================================"
        echo "SRC (demo 来源) : ${SRC}"
        echo "TGT (评测数据)  : ${TGT}"
        echo "SPLIT           : ${SPLIT}"
        echo "SHOT            : ${SHOT}"
        echo "日志文件        : ${LOG_FILE}"
        echo "========================================"
        
        # LANGCHAIN_CMD
        LANGCHAIN_CMD="python dataset_cons.py \
        --dataset_name ${SRC} \
        --db_name ${SRC} \
        --db_type ${DB_TYPE} \
        --top_k ${RAG_TOPK} \
        --embedding_model ${EMBDEDDING_MODEL}"
        
        if [[ "${SRC}" != "ProntoQA" && "${SRC}" != "AR-LSAT" ]]; then 
          LANGCHAIN_CMD="${LANGCHAIN_CMD} --ds_cot"
        fi
        
        # RUN_CMD & EVAL_CMD
        RUN_CMD="python llms_baseline.py \
          --model_name ${MODEL_NAME} \
          --dataset_name ${TGT} \
          --split ${SPLIT} \
          --mode ${MODE} \
          --max_new_tokens ${MAX_NEW_TOKENS} \
          --db_name ${SRC} \
          --icl_num ${DEMONSTRATION_NUM} \
          --top_k ${RAG_TOPK} \
          --batch_test \
          --batch_size ${BATCH_SIZE} \
          --use_vllm \
          --dtype ${DTYPE} \
          --db_type ${DB_TYPE} \
          --embedding_model ${EMBDEDDING_MODEL} \
          --repeat_time ${REPEAT_TIME} \
          --all_data_switch"
          
        if [ "$CONE_RERANK" = true ] && [ "$MODE" = "RAG" ]; then
          RUN_CMD="$RUN_CMD --rerank"
        fi

        EVA_CMD="python evaluation.py \
          --dataset_name ${TGT} \
          --model_name ${MODEL_NAME} \
          --split ${SPLIT} \
          --mode ${MODE} \
          --db_name ${SRC} \
          --db_type ${DB_TYPE} \
          --icl_num ${DEMONSTRATION_NUM}"
        
        if [ "$REVERSE_FLAG" = true ] && [ "$MODE" = "RAG" ] && [ "$DEMONSTRATION_NUM" > 1 ]; then
          RUN_CMD="$RUN_CMD --reverse_rag_order"
          EVA_CMD="$EVA_CMD --reverse_rag_order"
        fi
        
        # laska 2026.3.16按照论文中进行测试，将context放在question和option之后
        if [ "$REPEAT_REVERSE" = true ] && [ "$REPEAT_TIMES" = 1 ]; then
          RUN_CMD="$RUN_CMD --repeat_reverse --save_path=./results_reverse"
          EVA_CMD="$EVA_CMD --result_path=./results_reverse" 
        fi

        {
          echo "================ RUN START ================"
          echo "[SRC=${SRC}] [TGT=${TGT}] [SHOT=${SHOT}]"
          echo "[CMD] ${LANGCHAIN_CMD}"
          echo "[CMD] ${RUN_CMD}"
          echo "-------------------------------------------"
          # ${LANGCHAIN_CMD}
          echo "-------------------------------------------"
          CUDA_VISIBLE_DEVICES=6,7 ${RUN_CMD}
          
          echo "================ EVAL START ================"
          echo "[CMD] ${EVA_CMD}"
          echo "-------------------------------------------"
          ${EVA_CMD}

          echo "================== DONE ==================="
        } &> "${LOG_FILE}"   # 重定向 stdout + stderr 到日志


        echo "日志已写入：${LOG_FILE}"
        echo
      done
    done
  done
done
