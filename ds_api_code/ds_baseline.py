import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
import argparse
import requests
from openai import OpenAI

import urllib.request

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.split = args.split
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode
        self.max_new_tokens= args.max_new_tokens
        self.all_data_switch = args.all_data_switch  # 是否对完整数据集进行测试
        # 调用硅基流动的ds api
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.api_keys = "sk-umzuhgrnvqjzsyagbkxnruljvrxtiqepwjejknkedamfcjsi"
        self.label_phrase = 'The correct option is:'
        self.model = "Pro/deepseek-ai/DeepSeek-R1"
        # 批处理需要client函数
        self.client = OpenAI(api_key=self.api_keys, base_url="https://api.siliconflow.cn/v1")
        # laska 新建一个max_new_tokens的判别机制
        if self.mode == "Direct":
            if self.max_new_tokens > 1000:
                self.max_new_tokens = 16
        elif self.mode == "CoT":
            if self.max_new_tokens < 500:
                self.max_new_tokens = 1024   # 避免忘记设置正确的max_new_tokens
        # laska 新增，针对zero-shot的处理
        self.zero_shot = args.zero_shot
        if self.zero_shot:
            self.prompt_creator = self.prompt_LAST_zero_shot
            self.testing_type = '0-shot'
        else:   
            self.prompt_creator = self.prompt_LSAT
            self.testing_type = 'few-shot'
        print(f"zero_shot:{self.zero_shot}, all_data_switch:{self.all_data_switch},tets_mode:{self.mode}")
        # 定义一个用于存放task id的文件
        self.task_file = os.path.join(self.data_path, "batch_queue_id.txt")
        
    def prompt_LSAT(self, in_context_example, test_example):
        full_prompt = in_context_example
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[OPTIONS]]', options)
        return full_prompt
    
    # laska 新增zero-shot的prompt逻辑
    def prompt_LAST_zero_shot(self, in_context_example, test_example):
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = f"Context: {context}\nQuestion: {question}\nOptions:\n{options}\nLet's think step by step. The correct option is:"
        return full_prompt

    def load_in_context_examples(self):
        with open(os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}.txt')) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    # laska 每次调用batch处理文件之前，先检查之前的文件
    def task_check_and_download(self):
        pending_task_list= []
        with open(self.task_file, "r") as rfile:
            new_lines = []
            lines = rfile.readlines()
            print("当前任务文件内容")
            print(lines)
            # 一行行进行处理
            for line in lines:
                dataset_mode, batch_id = line.strip().split(":\t")
                dataset, mode, testing_type, split, model_name = dataset_mode.split("_")
                pending_task_list.append(f"{dataset}_{mode}_{testing_type}_{split}_{model_name}")
                # 当前处理结果文件的原始文件路径
                ori_file = os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')

                batch_job_status = self.batch_job_check(batch_id)
                # print(batch_job_status)
                # print(type(batch_job_status))
                job_status = batch_job_status.status
                # print("Job status:", job_status)
                if job_status == "completed":
                    output_file_url = batch_job_status.output_file_id
                    # print("Output files:", output_file_url)
                    # 写入本地文件的路径
                    batch_res_filepath = os.path.join(self.save_path, f"batch_{dataset_mode}.jsonl")
                    urllib.request.urlretrieve(output_file_url, batch_res_filepath)
                    # 调用处理文件 
                    outputs = self.process_batch_results(batch_res_filepath, ori_file)
                    savefile_path = os.path.join(self.save_path, f'{mode}_{testing_type}_{dataset}_{split}_{model_name}.json')
                    with open(savefile_path, 'w') as wfile:
                        json.dump(outputs, wfile, indent=2, ensure_ascii=False)
                    print(f"结果文件保存至 {savefile_path}")
                    
                else:# 没有完成的任务也保存
                    new_lines.append(line)
        # 将没有处理过的任务重新写入文件
        print("更新任务文件")
        print(new_lines)
        with open(self.task_file, "w") as wfile:
            for newline in new_lines:
                wfile.write(newline)    
        return pending_task_list
    
    # 对batch 处理的json文件，处理成evaluate可以处理的文件
    def process_batch_results(self, batch_results_file, ori_file):
        id_outputs_dict = {}
        outputs = []
        print("读取文件内容", ori_file)
        with open(ori_file) as f:
            raw_dataset = json.load(f)
            for item in raw_dataset:
                id = item['id']
                answer = item['answer']
                cur_dict = {'id': id, 'question': item['question'], 'answer': answer}
                id_outputs_dict[id] = cur_dict
                # outputs.append({'id': id, 'question': item['question'], 'answer': answer}) 
        print("当前文件所含测试样本个数为：",len(id_outputs_dict))       
        # 处理最终的结果文件     
        with open(batch_results_file, "r") as rfile:
            line = rfile.readlines()
            for item in line:
                res_dict = json.loads(item)
                response = res_dict["response"]
                body = response["body"]
                res_content = body["choices"][0]["message"]["content"].strip()
                reasoning_content = body["choices"][0]["message"]["reasoning_content"]
                question_id = res_dict["custom_id"]
                index_id = question_id.split("_")[-1]
                res_index = int(index_id) - 1
                # 获取当前测试数据集的标准样本
                cur_dict = id_outputs_dict[question_id]
                cur_dict['predicted_reasoning'] = reasoning_content
                cur_dict['predicted_answer'] = res_content
                outputs.append(cur_dict)
        print("当前处理结果文件所含测试样本个数为：",len(outputs))       
        # print(outputs[0])
        return outputs
        
    # 将json文件转换成jsonl文件
    def file_to_jsonl(self):
        if self.mode == "Direct":
            system_prompt = "You are solving multiple-choice logical reasoning problems.\n\nRules:\n- Read the context, question, and options carefully.\n- Do NOT write reasoning or explanations.\n- Output ONLY the final answer’s option letter (e.g., A or B or C or D). No words, no punctuation, no spaces, no newlines after the letter."
        elif self.mode == "CoT":
            system_prompt = "You are solving multiple-choice logical reasoning problems.\nOutput rules:\n- Read the context, question, and options carefully.\n- Follow the demonstration, solve the problem step by step.\n- Output ONLY the final answer’s option letter (e.g., A or B or C or D). No words, no punctuation, no spaces, no newlines after the letter."
        raw_dataset = self.load_raw_dataset(self.split)
        jsonl_filepath = os.path.join(self.data_path, self.dataset_name, f'{self.split}_{self.mode}.jsonl')
        # 如果文件存在就直接返回
        # if os.path.exists(jsonl_filepath):
        #     return jsonl_filepath
        wfile = open(jsonl_filepath, 'w')
        # 加载上下文示例
        in_context_examples = self.load_in_context_examples()
        for example in raw_dataset:
            id = example['id']
            answer = example['answer']
            # 有的数据集没有explanation字段
            try:
                explanation = example['explanation']
            except:
                explanation = "NA"
            full_prompt = self.prompt_creator(in_context_examples, example)

            messages = [{"role":"system", "content":system_prompt},
                        {"role":"user", "content":full_prompt}]
            
            wfile.write(json.dumps({"custom_id": id, "method":"POST", "url":"/v1/chat/completions", "body":{"model":self.model, "messages":messages, "max_tokens":self.max_new_tokens,"thinking_budget":32768}, "answer":answer, "explanation":explanation}) + '\n')
            
            if self.all_data_switch == False:
                print("当前只测试一条数据，跳出")
                break
        wfile.close()
        return jsonl_filepath
    
    # laska 上传批处理文件
    def batch_file_upload(self, file_path):
        batch_input_file = self.client.files.create(file=open(file_path,"rb"), purpose="batch")
        # 返回上传的文件ID
        file_id = batch_input_file.data["id"]
        print("File ID:", file_id)
        return file_id

    # laska batch进行调用的主代码
    def ds_api_batch_generate(self):
        # 进行新的任务之前，先检查之前是否有任务在运行
        print(self.max_new_tokens)
        pending_task_list = self.task_check_and_download()
        # exit()
        # 判断当前执行的任务是不是已经执行过
        cur_task = self.dataset_name+"_"+self.mode+"_"+self.testing_type+"_"+self.split+"_"+self.model_name
        if cur_task in pending_task_list:
            print("当前任务已经处理过，跳过")
            return
        print("当前任务没有处理过，开始处理")
        # 对当前任务生成jsonl文件
        # exit()
        jsonl_filepath = self.file_to_jsonl()
        print(jsonl_filepath)
        file_id = self.batch_file_upload(jsonl_filepath)
        print("Uploaded File ID:", file_id)
        # 创建新的批处理任务
        print("Creating new batch job...")
        batch_job_status = self.client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description":"batch test job"}, extra_body={"replace":{"model":"deepseek-ai/DeepSeek-R1"}})
        print("Batch Job Created:", batch_job_status)
        # 里面保存了batch job的id
        print(type(batch_job_status))
        batch_job_id = batch_job_status.id
        print("Batch Job ID:", batch_job_id)
        # 用一个文件来存放
        with open(self.task_file, "a") as wfile:
            wfile.write(self.dataset_name+"_"+self.mode+"_"+self.testing_type+"_"+self.split+"_"+self.model_name+":\t"+batch_job_id + "\n")
        return batch_job_id
    
    # laska 检查批任务状态
    def batch_job_check(self, batch_job_id):
        batch_job = self.client.batches.retrieve(batch_job_id)
        print(batch_job)
        return batch_job
    
    # laska 取消批处理任务
    def batch_job_cancel(self, batch_job_id):
        canceled_job = self.client.batches.cancel(batch_job_id)
        print(canceled_job)
        return canceled_job
    
    # laska 获取所有批量推理任务列表
    def batch_job_list(self):
        batch_jobs = self.client.batches.list().data
        print(batch_jobs)
        return batch_jobs
    
    # laska 调用ds api对单条数据进行测试
    def ds_api_generate(self, full_prompt):
        if self.mode == "Direct":
            system_prompt = "You are solving multiple-choice logical reasoning problems.\n\nRules:\n- Read the context, question, and options carefully.\n- Do NOT write reasoning or explanations.\n- Output ONLY the final answer’s option letter (e.g., A or B or C or D). No words, no punctuation, no spaces, no newlines after the letter."
        elif self.mode == "CoT":
            system_prompt = "Answer the following question, let's think step by step."
      
        message_list = [{"role":"system","content": system_prompt},
                        {"role":"user","content":full_prompt}]
        # laska 测试部分
        print(type(message_list), len(message_list))
        payload = {
            "model": "Pro/deepseek-ai/DeepSeek-R1",
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0,
            "n": 1,
            "messages": message_list
        }
        headers = {
            "Authorization": "Bearer sk-umzuhgrnvqjzsyagbkxnruljvrxtiqepwjejknkedamfcjsi",
            "Content-Type": "application/json"
        }
        response = requests.request("POST", self.url, json=payload, headers=headers)
        print(response)
        # batch的处理逻辑
        # 单条的处理逻辑
        res_dict = json.loads(response.text)
        res_content= res_dict["choices"][0]["message"]["content"]
        reasoning_content = res_dict["choices"][0]["message"]["reasoning_content"]
        # print(res_content)
        # print("===================")
        # print(reasoning_content)
        return res_content

    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()
        
        outputs = []
        for example in tqdm(raw_dataset):
            question = example['question']

            # create prompt
            full_prompt = self.prompt_creator(in_context_examples, example)
            # output = self.openai_api.generate(full_prompt)
            # 调用ds的api
            # print("---------------------------")
            output = self.ds_api_generate(full_prompt)
            
            # get the answer
            label_phrase = self.label_phrase
            generated_answer = output.split(label_phrase)[-1].strip()
            generated_reasoning = output.split(label_phrase)[0].strip()

            # create output
            output = {'id': example['id'], 
                      'question': question, 
                      'answer': example['answer'], 
                      'predicted_reasoning': generated_reasoning,
                      'predicted_answer': generated_answer}
            outputs.append(output)
            break
        # save outputs        
        with open(os.path.join(self.save_path, f'{self.mode}_{self.testing_type}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator(in_context_examples, example) for example in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    # get the answer
                    dict_output = self.update_answer(sample, output)
                    outputs.append(dict_output)
            except Exception as e:
                print(e)
                exit()
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.openai_api.generate(full_prompt)
                        # get the answer
                        dict_output = self.update_answer(sample, output)
                        outputs.append(dict_output)
                    except:
                        print('Error in generating example: ', sample['id'])

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.mode}_{self.testing_type}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def update_answer(self, sample, output):
        label_phrase = self.label_phrase
        generated_answer = output.split(label_phrase)[-1].strip()
        generated_reasoning = output.split(label_phrase)[0].strip()
        dict_output = {'id': sample['id'], 
                        'question': sample['question'], 
                        'answer': sample['answer'], 
                        'predicted_reasoning': generated_reasoning,
                        'predicted_answer': generated_answer}
        return dict_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--batch_predict', default=False, action='store_true')
    parser.add_argument('--all_data_switch', help='当前是否需要对所有数据集进行测试(True)，还是测试代码功能(Fasle:只测试一条数据就可以)', default=False, action='store_true')
    # laska定义一个针对0-shot的代码
    parser.add_argument('--zero-shot', default=False, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    # exit()
    if args.batch_predict == True:
        print("进行批处理测试")
        gpt3_problem_reduction.ds_api_batch_generate()
    else:
        print("进行单条测试")
        gpt3_problem_reduction.reasoning_graph_generation()
