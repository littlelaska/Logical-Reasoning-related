cd baseline
# 运行测试代码
python llms_baseline.py \
    --model_name qwen \    # 模型可以选择llama等
    --dataset_name ProntoQA \   # 数据集 ProntoQA | ProofWriter | FOLIO | LogicalDeduction ｜ AR-LSAT
    --split dev \
    --mode Direct \    # 可选Direct COT
    --max_new_tokens 16 \ # Direct对应16 COT对应1024等更长的长度
    --zero-shot False \  # 是否为0-shot
    --all_data_switch True  # 是否对完整数据集进行测试(True)，还是测试(单条)
    --batch_test False \  # 是否进行batch测试
    --batch_size 8 \  # batch size大小


CUDA_VISIBLE_DEVICES=1,2,3,4 python llms_baseline.py --model_name qwen --dataset_name ProntoQA --split dev --mode CoT --max_new_tokens 1024 --zero-shot --all_data_switch --batch_test  --batch_size 8

# 对测试结果进行统计
python evaluation.py \
    --dataset_name ProntoQA \
    --model_name qwen \
    --split dev \
    --mode Direct 

python evaluation.py --dataset_name ProntoQA --model_name qwen --split dev --mode Direct 