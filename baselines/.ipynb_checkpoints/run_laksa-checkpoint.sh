MODE="CoT"
DATASET_NAME="FOLIO"
MODEL_NAME="qwen14"
ZERO_SHOT=true

RUN_CMD="python llms_baseline.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split dev --mode $MODE --max_new_tokens 1024 --all_data_switch --batch_test --batch_size 16 --use_vllm "


EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split dev --mode $MODE"

if [ "$ZERO_SHOT" = true ]; then
    RUN_CMD="$RUN_CMD --zero-shot"
    EVA_CMD="$EVA_CMD --zero-shot"
    
fi
echo "Running: $RUN_CMD"
$RUN_CMD
echo "Running: $EVA_CMD"
$EVA_CMD

