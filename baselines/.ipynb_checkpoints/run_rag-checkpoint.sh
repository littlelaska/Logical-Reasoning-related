MODE="RAG"
DATASET_NAME="ProntoQA"
MODEL_NAME="qwen14"
LANGCHAIN_DB="logicaldeduction"
RAG_TOPK=5
DEMONSTRATION_NUM=1

RUN_CMD="python llms_rag.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split dev --mode $MODE --max_new_tokens 1024 --batch_test --batch_size 16 --use_vllm --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM --top_k $RAG_TOPK --all_data_switch"


EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split dev --mode $MODE --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM"

if [ "$ZERO_SHOT" = true ]; then
    RUN_CMD="$RUN_CMD --zero-shot"
    EVA_CMD="$EVA_CMD --zero-shot"
    
fi
echo "Running: $RUN_CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN_CMD
echo "Running: $EVA_CMD"
$EVA_CMD

