MODE="RAG"    # CoT/Direct/RAG/Logical
DATASET_NAME="LogicalDeduction"    # gsm8k/ProntoQA/AR-LSAT/FOLIO/ProofWriter/LogicalDeduction
MODEL_NAME="gemma3-12"   # qwen14/qwen7/qwen3-8
SPLIT="dev"
LANGCHAIN_DB="commonsense"    # gsm8k//ProntoQA/FOLIO/ProofWriter/LogicalDeduction/commonsense
DB_TYPE="embedding"   # bm25/embedding/random
RAG_TOPK=10
DEMONSTRATION_NUM=1
ZERO_SHOT=true
DTYPE="bfloat16"
REVERSE_FLAG=false
EMBDEDDING_MODEL="../llms/bge-large-en-v1.5"   # text2vec-large-chinese/bge-large-en/bge-large-en-v1.5
CONE_RERANK=false

LANGCHAIN_CMD="python dataset_cons.py --dataset_name $LANGCHAIN_DB --db_name $LANGCHAIN_DB --db_type $DB_TYPE --top_k $RAG_TOPK  --ds_cot --embedding_model $EMBDEDDING_MODEL"

RUN_CMD="python llms_baseline.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split $SPLIT --mode $MODE --max_new_tokens 8192 --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM --top_k $RAG_TOPK --db_type $DB_TYPE --dtype $DTYPE --batch_test --batch_size 16 --embedding_model $EMBDEDDING_MODEL --all_data_switch --use_vllm"

EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split $SPLIT --mode $MODE  --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM --db_type $DB_TYPE"

if [ "$ZERO_SHOT" = true ] && [ "$MODE" != "RAG" ]; then
    RUN_CMD="$RUN_CMD --zero_shot"
    EVA_CMD="$EVA_CMD --zero_shot"
    
fi

if [ "$REVERSE_FLAG" = true ] && [ "$MODE" = "RAG" ] && [ "$DEMONSTRATION_NUM" > 1 ]; then
    RUN_CMD="$RUN_CMD --reverse_rag_order" 
    EVA_CMD="$EVA_CMD --reverse_rag_order"
fi


if [ "$CONE_RERANK" = true ] && [ "$MODE" = "RAG" ]; then
    RUN_CMD="$RUN_CMD --rerank"
fi

# echo "Building the langchain_dataset, Running: $LANGCHAIN_CMD"
# $LANGCHAIN_CMD

echo "Running: $RUN_CMD"
CUDA_VISIBLE_DEVICES=1,6 $RUN_CMD

echo "Running: $EVA_CMD"
$EVA_CMD