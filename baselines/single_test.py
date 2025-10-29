from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import argparse

# 尝试使用vllm加速模型推理
from vllm import LLM, SamplingParams
import torch
from dataset_cons import DatasetRetriever


model_path = "../llms/Qwen2.5-14B-Instruct"


# 加载模型
def load_model(model_path, max_new_tokens=1024):
    model = LLM(model=model_path, tokenizer=model_path,tensor_parallel_size=torch.cuda.device_count(), max_model_len=32768,dtype="float32", trust_remote_code=True, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, top_p=0.95, top_k=40, n=1)
    return model, tokenizer,sampling_params

def generate_answer(model, tokenizer, sampling_params ,query, system_prompt,max_new_tokens=1024):
    messages = [{'role':"system", "content":system_prompt},
               {'role':"user", "content":query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = model.generate([text], sampling_params=sampling_params)
    print(len(outputs), type(outputs))
    response =[output.outputs[0].text for output in outputs]
#     model_inputs = tokenizer([text], return_tensors="pt")
#     generated_ids = model.generate(**model_inputs, do_sample=False, max_new_tokens=max_new_tokens)
#     generated_ids = generated_ids[:,len(model_inputs.input_ids[0]):]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__=="__main__":
    model, tokenizer,sampling_params = load_model(model_path)
    system_prompt = "You are a logical task solver. Follow the demonstrationa to solve the new question. Remember to think step by step with concise chain-of-thought, and adhere to the context related to the question. Then on a new line, output exactly: 'The correct option is: A' or 'The correct option is: B"
    icl_prompt = "Given a problem statement as contexts, the task is to answer a logical reasoning question. \n------"
    icl_context = "Context:\nQuestion:\nAn archaeologist discovered three dig sites from different periods in one area. The archaeologist dated the first dig site as 352 years more recent than the second dig site. The third dig site was dated 3700 years older than the first dig site. The fourth dig site was twice as old as the third dig site. The archaeologist studied the fourth dig site’s relics and gave the site a date of 8400 BC. What year did the archaeologist date the second dig site?\n\nReasoning:\nThe third dig site was dated from the year 8400 / 2 = <<8400/2=4200>>4200 BC.\nThus, the first dig site was dated from the year 4200 - 3700 = <<4200-3700=500>>500 BC.\nThe second dig site was 352 years older, so it was dated from the year 500 + 352 = <<500+352=852>>852 BC.\n"
    
    query = "Context:\nEvery dumpus is not shy. Each dumpus is a tumpus. Rompuses are not wooden. Tumpuses are opaque. Every tumpus is a wumpus. Wumpuses are not floral. Each wumpus is an impus. Impuses are bitter. Every impus is a vumpus. Vumpuses are small. Each vumpus is a numpus. Every numpus is wooden. Each numpus is a yumpus. Each yumpus is orange. Each yumpus is a jompus. Each jompus is amenable. Every jompus is a zumpus. Wren is a tumpus.\n\nQuestion: Is the following statement true or false? Wren is wooden.\nOptions:\nA) True\nB) False\nReasoning:\n"
    
    final_query = icl_prompt+icl_context+query
    
    pure_response = generate_answer(model, tokenizer, sampling_params, query, system_prompt)
    print(pure_response)
    print("-------------")
    rag_response = generate_answer(model, tokenizer, sampling_params,final_query, system_prompt)
    print(rag_response)
    