MODE="Logical"    # cot/direct/logical
DATASET_NAME="ProntoQA"
MODEL_NAME="qwen14"
ZERO_SHOT=true
SYSTEM_PROMPT_PATH="./system_prompt"
PROMPT_FILE="logical_prompt_1.txt"

RUN_CMD="python llms_baseline.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split dev --mode $MODE --max_new_tokens 1024 --system_prompt_path  $SYSTEM_PROMPT_PATH --prompt_file $PROMPT_FILE --all_data_switch --batch_test --batch_size 16 --use_vllm --zero-shot"


EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split dev --mode $MODE"

if [ "$ZERO_SHOT" = true ]; then
    RUN_CMD="$RUN_CMD --zero-shot"
    EVA_CMD="$EVA_CMD --zero-shot"
    
fi
echo "Running: $RUN_CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN_CMD
echo "Running: $EVA_CMD"
$EVA_CMD

