from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import argparse

# 尝试使用vllm加速模型推理
from vllm import LLM, SamplingParams
import torch
from dataset_cons import DatasetRetriever, RandomRetriever

import random
import numpy as np
import torch

from datasets import load_dataset, Dataset
from copy import deepcopy

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


class LLM_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode   # Direct /CoT /RAG
        
        self.all_data_switch = args.all_data_switch  # 是否对完整数据集进行测试
        self.batch_test = args.batch_test  # 是否进行batch测试
        self.batch_size = args.batch_size  # batch size大小
        self.vllm_switch = args.use_vllm  # 是否使用vllm进行加速
        self.max_new_tokens = args.max_new_tokens
        self.zero_shot = args.zero_shot   # 在非rag的模式下生效
        self.rag_result_path = args.rag_result_path
        self.system_prompt_dir = args.system_prompt_dir
        self.user_template_dir = args.user_template_dir
        self.reverse_rag_order = args.reverse_rag_order
        # self.random_rag_order = args.random_rag_order
        self.rerank = args.rerank
        self.dtype = args.dtype    # float16/32
        # 对需要的部分数据进行初始化
        self.para_init()

        self.tokenizer, self.model= self.load_model()
        if not self.vllm_switch:
            self.device = self.model.device

        self.label_phrase = 'The correct option is:'

    # 2025.11.11 separate some init code from init
    def para_init(self):
        # 模型路径初始化
        if self.model_name == "qwen7":
            # self.model_path = "../llms/Qwen2.5-7B-Instruct"
            self.model_path = "/data_a100/models/Qwen2.5-7B-Instruct"
        elif self.model_name == "qwen14":
            self.model_path = "../llms/Qwen2.5-14B-Instruct"
        elif self.model_name == "qwen3-8":
            self.model_path = "../llms/Qwen3-8B"
        elif self.model_name == "qwen3-14":
            self.model_path = "../llms/Qwen3-14B"
        elif self.model_name == "qwen3-32":
            self.model_path = "../llms/Qwen3-32B"
        elif self.model_name == "llama3-8":
            self.model_path = "../llms/llama3.1-8B-Instruct"
        elif self.model_name == "gemma3-4":
            self.model_path = "/data_a100/models/gemma-3-4b-it"
        elif self.model_name == "gemma3-12":
            self.model_path = "/data_a100/models/gemma-3-12b-it"
        elif self.model_name == "gemma3-27":
            self.model_path = "/data_a100/models/gemma-3-27b-it"
        else:
            self.model_path = "../llms/"
        
        # 针对不同mode的参数初始化
        if self.mode == "RAG":   # 说明当前是rag模式，需要加载检索库
            # RAG检索器加载部分
            self.rag_topk = args.top_k   # 检索的样例个数
            self.rag_icl_num = args.icl_num   # 用于上下文学习的展示样例个数  
            self.db_name = args.db_name 
            self.index_path = args.index_path
            self.db_type = args.db_type
            if self.db_type == "random":
            # if self.random_rag_order:
                self.dataset_retriever = RandomRetriever(self.args)
            else:
                self.dataset_retriever = DatasetRetriever(self.args)
            # rag所用的icl template文件路径，用于包装检索到的document
            self.icl_template_file =  f"{'gsm8k' if self.db_name == 'gsm8k' else 'LogicalReasoning'}_ICL_template.txt"
            self.icl_template_path = os.path.join(self.user_template_dir, self.icl_template_file)
        # 将zero-shot的逻辑也加到这里
        elif self.zero_shot:
            self.testing_type = "0-shot"
        else:
            self.testing_type = "few-shot"
        
        # role prompt 路径初始化
        if self.dataset_name == "gsm8k":
            self.prompt_file = f"{self.dataset_name}_{self.mode}{'_0shot' if self.zero_shot else ''}.txt"
        else:
            self.prompt_file = f"LogicalReasoning_{self.mode}{'_0shot' if self.zero_shot else ''}.txt"
            # 2026.3.16 prompt中context是否后置的逻辑
            if self.args.repeat_reverse:
                self.prompt_file = f"LogicalReasoning_RAG_reverse.txt"
        self.system_prompt_path = os.path.join(self.system_prompt_dir, self.prompt_file)
        print(f"system prompt file path: {self.system_prompt_path}")

        # user prompt路径初始化
        self.user_prompt_path = os.path.join(self.user_template_dir, self.prompt_file)
        print(f"user prompt file path: {self.user_prompt_path}")
        
        # 待检查判断逻辑是否正确及完善， prompt creator初始化
        if self.mode == "RAG":
            if self.rag_icl_num > 0:
#                 self.prompt_creator = self.rag_prompt_creator
                self.prompt_creator = self.prompt_LSAT
            else:
                self.prompt_creator = self.prompt_LSAT
        else:
            self.prompt_creator = self.prompt_LSAT

        # 结果存储路径初始化
        # 统一定义存储路径
        if self.mode == "RAG":
            self.save_file = os.path.join(self.save_path, f'{self.mode}{self.rag_icl_num}_{self.db_name}_{self.db_type}{"_reversed" if self.reverse_rag_order else ""}_{self.dataset_name}_{self.split}_{self.model_name}.json')
            # laska定义一个保存检索中间结果的文件
            if not os.path.exists(self.rag_result_path):
                os.makedirs(self.rag_result_path)
            self.retrieval_save_file = os.path.join(self.rag_result_path, f'retrieval_{self.db_name}_{self.db_type}_{self.dataset_name}_{self.split}.json')   # 只与文件有关
            self.retrieval_writer = open(self.retrieval_save_file, 'w') 
        else:
            self.save_file = os.path.join(self.save_path, f'{self.mode}_{self.testing_type}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        
        # 2026.3.13 proppmpt复制部分
        self.repeat_time = self.args.repeat_time
        
        
        # 打印部分参数
        print("="*16+"parameteres"+"="*16)
    
        self.print_self()
        print("="*16+"parameteres"+"="*16)

    # 打印参数
    def print_self(self):
        for k,v in self.__dict__.items():
            print(f"{k}:{v}")

    # laska system prompt加载函数
    def load_system_prompt(self):
        with open(self.system_prompt_path, 'r') as f:
            system_prompt = f.read()
        return system_prompt

    # 2025.11.11 加载user prompt的template部分，用于构建数据
    def load_user_prompt_template(self):
        with open(self.user_prompt_path, "r") as f:
            user_prompt = f.read()
        return user_prompt
    
    # 2025.11.11 增加icl_prompt部分
    def load_icl_template(self):
        with open(self.icl_template_path, "r") as f:
            icl_template = f.read()
        return icl_template
    
    # laska 模型加载部分     
    def load_model(self):
        # vllm 新增
        if self.vllm_switch:
            print("使用vllm进行模型加载和推理")
            print("loading model from:", self.model_path)
            model = LLM(model=self.model_path, tokenizer=self.model_path,tensor_parallel_size=torch.cuda.device_count(), max_model_len=32768,dtype=self.dtype, trust_remote_code=True, gpu_memory_utilization=0.8)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
#             self.sampling_params = SamplingParams(temperature=0, max_tokens=self.max_new_tokens, top_p=0.95, top_k=40, n=1)
            self.sampling_params = SamplingParams(
                temperature=0, 
                max_tokens=self.max_new_tokens, 
                top_p=1, 
                top_k=1, 
                n=1,
                # seed=42,    # 想要复现结果可以保障随机种子一致
                )
            return tokenizer, model
        else:
            print("直接加载模型进行推理")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')   # 直接从本地路径进行加载
            print("loading model from:", self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype="auto", device_map="auto")
            print("loading complete")
            return tokenizer, model

    # 20251216 新增一个cone利用条件概率重排的功能
    def cone_rerank(self, retrieved_results, test_example):
        icl_template = self.load_icl_template()   # 需要加载模板，拼接之后给模型，查看对loss是否有提升
        user_prompt_template = self.load_user_prompt_template()
        role_content = self.load_system_prompt()
        chat_template_texts = []
        mask_lengths_idxs = []
        max_length = 0
        for result in retrieved_results:
            # 将当前的检索结果拼接成
            new_icl_template = deepcopy(icl_template)
            cur_icl = new_icl_template.format(
                context=result['context'],
                question=result['question'],
                options='\n'.join([opt.strip() for opt in result.get("options", [])]),
                cot=result['cot'],
                answer=result['answer']
                )
            new_user_prompt_template = deepcopy(user_prompt_template)
            cur_full_prompt = new_user_prompt_template.replace("[[DEMONSTRATIONS]]", cur_icl)
            # 先将query等内容也进行替换 
            if self.dataset_name == "gsm8k":
                question = test_example["question"].strip()
                cur_full_prompt = cur_full_prompt.replace('[[QUESTION]]', question)
            else:
                context = test_example['context'].strip()
                question = test_example['question'].strip()
                options = '\n'.join([opt.strip() for opt in test_example['options']])
                cur_full_prompt = cur_full_prompt.replace('[[CONTEXT]]', context)
                cur_full_prompt = cur_full_prompt.replace('[[QUESTION]]', question)
                cur_full_prompt = cur_full_prompt.replace('[[OPTIONS]]', options)
            # 先替换成为messages形式，并apply_chat_template，然后计算mask_length
            cur_messages = [{"role": "system", "content": role_content},
                            {"role":"user", "content": cur_full_prompt}]
            cur_text = self.tokenizer.apply_chat_template(cur_messages, add_generation_prompt=True, tokenize=False)
            if len(cur_text) > max_length:
                max_length = len(cur_text)
            # 存入列表供后续操作
            chat_template_texts.append(cur_text)
            # 获取icl demonstration的结束位置，计算mask length
            first_context_idx = cur_text.find("Context:")
            second_context_idx = cur_text.find("Context:", first_context_idx+1)
            mask_lengths_idxs.append(second_context_idx)
        # vllm 版本需要一条条对text进行处理
        # model_input_ids = [self.tokenizer(chat_template_text, return_tensors="pt").input_ids for chat_template_text in chat_template_texts]
        input_mask_lengths = [self.tokenizer(chat_template_texts[idx][:mask_lengths_idxs[idx]], return_tensors="pt").input_ids.shape[1] for idx in range(len(mask_lengths_idxs))]
        # 需要注意。列表里面的input是一个二维数组，1*len
        # vllm cone params
        vllm_cone_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=20, detokenize=False)
        cone_batch_size =10
        ce_list = []
        n = len(chat_template_texts)
        for start in range(0, n, cone_batch_size):
            end = min(start + cone_batch_size, n)
            sub_texts = chat_template_texts[start:end]
            # sub_input_ids = model_input_ids[start:end]

            # 调用vllm模型进行logits的获取
            if self.vllm_switch == True:
                # vllm 一条条处理input ids
                sub_input_ids = [self.tokenizer(chat_text, return_tensors="pt").input_ids for chat_text in sub_texts]
                sub_mask_lengths = input_mask_lengths[start:end]
                # vLLM 的输入用原始文本；不在这里做 tokenizer.to(device)
                vllm_outputs = self.model.generate(
                    sub_texts,
                    sampling_params=vllm_cone_params,
                    use_tqdm=False,
                )
                # 你自己的 CE 计算函数：注意它内部再搬到 GPU
                sub_ce = self.ce_loss_cal(sub_input_ids, vllm_outputs, sub_mask_lengths)
                ce_list.append(sub_ce.detach().cpu())
                # 释放这一个小 batch 的中间结果
                del vllm_outputs
                del sub_ce
                torch.cuda.empty_cache()
            else:   # 针对transformers模型的处理方法   
                sub_hf_input_ids = self.tokenizer(sub_texts, padding=True, return_tensors="pt").to(self.device)
                # print("the input size are:", sub_hf_input_ids.input_ids.shape)
                sub_mask_lengths = mask_lengths_idxs[start:end]
                # 进行测试
                with torch.no_grad():
                    sub_hf_outputs = self.model(**sub_hf_input_ids)
                padding_input_id_len = sub_hf_input_ids.input_ids.shape[1]
                # 需要重新计算mask的位置
                for ind in range(len(sub_mask_lengths)):
                    ori_len = self.tokenizer(sub_texts[ind], return_tensors="pt").input_ids.shape[1]
                    context_len =self.tokenizer(sub_texts[ind][:sub_mask_lengths[ind]], return_tensors="pt").input_ids.shape[1]
                    new_mask_len = padding_input_id_len - ori_len + context_len
                    # print("the new mask len is ", new_mask_len, context_len, ori_len)
                    sub_new_mask_lengths = [new_mask_len, ] * len(sub_mask_lengths)
                    break
                # print("the new mask lengths are", sub_new_mask_lengths)

                # 计算sub_ce_loss
                shift_sub_logits = sub_hf_outputs.logits[...,:-1,:].contiguous()
                shift_sub_labels = sub_hf_input_ids["input_ids"][...,1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)

                shift_sub_logits = shift_sub_logits.view(-1, shift_sub_logits.size(-1))
                sub_loss = loss_fct(shift_sub_logits, shift_sub_labels.view(-1)).view(shift_sub_labels.size())

                # print(sub_loss.shape)
                sub_mask = torch.zeros_like(sub_loss)
                for i in range(len(sub_new_mask_lengths)):
                    sub_mask[i, sub_new_mask_lengths[i]:] =1
                sub_loss = sub_loss * sub_mask
                # print(sub_loss)
                sub_ce_loss = torch.sum(sub_loss, 1)
                # print(sub_ce_loss)
                ce_list.append(sub_ce_loss.detach().cpu())
                del sub_hf_input_ids
                del sub_hf_outputs
                del sub_loss
                del sub_mask
                torch.cuda.empty_cache()
            # 拼回完整的 ce_loss 向量
            ce_loss = torch.cat(ce_list, dim=0)
        # print(ce_loss)
        sorted_idx = torch.argsort(ce_loss)   # 默认升序
        # print(sorted_idx)
        # exit()
        return sorted_idx 

    # 定义一个ce_loss的计算函数，输入是模型inputs和outputs
    def ce_loss_cal(self, input_ids, outputs, mask_lengths):
        all_losses = []
        for ids, out in zip(input_ids, outputs):   # 这里应该和batch size一致？
            # 这里的ids是是一个[1,seq_len]的二维数组
            ids = ids.view(ids.shape[1])   # 需要将二维数组ids展开为一维数组，长度直接是seq_len
            token_losses = []
            for tok_id, lp in zip(ids, out.prompt_logprobs):
        
                if lp is None:
                    token_losses.append(0.0)
                    continue
                info = lp.get(int(tok_id), None)
                if info is None:   
                    token_losses.append(20.0)    # fallback,赋予一个大loss
                else:
                    token_losses.append(-info.logprob)
            all_losses.append(torch.tensor(token_losses))
        # 对loss进行计算
        loss = torch.nn.utils.rnn.pad_sequence(all_losses, batch_first=True, padding_value=0.0)
        
        # 按照mask的长度进行mask
        mask = torch.zeros_like(loss)
        for i in range(len(mask_lengths)):
            # print(mask_lengths[i], mask[i].shape)
            mask[i, mask_lengths[i]:input_ids[i].shape[1]] = 1
  
        # 将context部分的loss进行mask
        loss = mask * loss
        # print(loss)
        ce_loss = torch.sum(loss, 1)
        return ce_loss

    # laska 构建使用rag动态变化demonstration的prompt生成器
    def rag_prompt_creator(self, in_context_example, test_example):
        # 2025.11.11 add system prompt
        role_content = self.load_system_prompt()   # 不论是rag还是cot的system prompt都是一样的
        user_prompt_template = self.load_user_prompt_template()  # 目前这一部分的选择是不一样的
        # 首先进行检索，得到相关的demonstration
        # 所有数据集都有question域
        rag_query =test_example["question"].strip()
        retrieved_results = self.dataset_retriever.retrieve(rag_query, self.rag_topk)
        # 制定一个template 
        icl_template = self.load_icl_template()
        print(icl_template)
#         icl_template = "Context:\n{context}\nQuestion:\n{question}\nOptions:\n{options}\nReasoning:\n{cot}\nAnswer:\n{answer}\n"
        
        # 构建检索的数据集
        overall_demonstration = ""
        for result in retrieved_results[:self.rag_icl_num]:
            overall_demonstration += icl_template.format(
                context=result['context'],
                question=result['question'],
                options='\n'.join([opt.strip() for opt in result.get("options", [])]),
                cot=result['cot'],
                answer=result['answer']
            ) + "\n"
        
        full_in_context_example = user_prompt_template.replace("[[DEMONSTRATIONS]]",)
#         full_in_context_example = head_template + "\n" + overall_demonstration
        # 将需要测试的内容进行拼接
        test_template = "Context:\n{context}\nQuestion:\n{question}\nOptions:\n{options}\nReasoning:"
#         print(test_example)
#         print(test_example["context"])
        test_example_str = test_template.format(context=test_example['context'],
                                                question=test_example['question'],
                                                options='\n'.join([opt.strip() for opt in test_example['options']]))
        # 拼接成为最终给模型进行测试的样例
        full_prompt = full_in_context_example + "\n" + test_example_str
        role_content = "You are a logical task solver. Follow the demonstrationa to solve the new question. Remember to think step by step with concise chain-of-thought, and adhere to the context related to the question. Then on a new line, output exactly: 'The correct option is: A' or 'The correct option is: B"
        messages = [
            {"role":"system", "content":role_content},
            {"role":"user", "content": full_prompt}
            ]
        print(messages)
        # laska 修改，针对本地模型，返回messages
        # 每检索一条，将检索结果写入文件
        retrieval_record = {
            'context': test_example['context'],
            'question': test_example['question'],
            'retrieved_demonstrations': full_in_context_example
        }
        # 写入json文件
        self.retrieval_writer.write(json.dumps(retrieval_record, ensure_ascii=False) + '\n')
        return messages
    
    # 针对few-shot，生成prompt，该部分完成的是在单个样例之前添加few-shot的示例
    def prompt_LSAT(self, in_context_example, test_example):
        # 2025.11.11 add system prompt
        role_content = self.load_system_prompt()   # 不论是rag还是cot的system prompt都是一样的   
        user_prompt_template = self.load_user_prompt_template()  # 目前这一部分的选择是不一样的     
        # 这一部分分支逻辑待验证代码正确性
        if self.mode == "RAG":
            full_prompt = user_prompt_template
        elif self.zero_shot == True:
            full_prompt = user_prompt_template
        else:
            full_prompt = in_context_example
        # 20251202 添加打印信息标记
        if not hasattr(type(self).prompt_LSAT, "_has_run"):
            print("👉 self.prompt_LAST 被首次调用，打印提示信息")
            print("-"*36)
            print("current role_content is :")
            print(role_content)
            print("-"*16)
            print("current user template is:")
            print(user_prompt_template)
            print("-"*16)
            print("full prompt is:")
            print(full_prompt)            
            print("-"*36)
            type(self).prompt_LSAT._has_run = True

        # 2025.11.11 增加rag的prompt构造
        # 所有数据集都有question域
        if self.mode == "RAG":
#             print(test_example)
            rag_query = test_example["question"].strip()
            retrieved_results = self.dataset_retriever.retrieve(rag_query, self.rag_topk)   # 检索回来的会比实际需要的多
            # 制定一个template 
            icl_template = self.load_icl_template()
#             print(icl_template)
            # 构建检索的数据集
            overall_demonstration = ""
            
            # laska 20251216新增rerank 逻辑
            if self.rerank:
                # print(retrieved_results[0])
                sorted_idx = self.cone_rerank(retrieved_results, test_example)
                # 对retrieved_results进行重排序
                retrieved_results = [retrieved_results[idx] for idx in sorted_idx]
                # print("after sorted~")
                # print(retrieved_results[0])
                # exit()

            # 先根据需要倒序
            if self.reverse_rag_order:
                candidates = retrieved_results[:self.rag_icl_num][::-1]   # 倒序挑前 N
            else:
                candidates = retrieved_results[:self.rag_icl_num]         # 正序挑前 N

            # 用 candidates 来循环  
            for result in candidates:
                overall_demonstration += icl_template.format(
                    context=result['context'],
                    question=result['question'],
                    options='\n'.join([opt.strip() for opt in result.get("options", [])]),
                    cot=result['cot'],
                    answer=result['answer']
                ) + "\n"
#             print("before replace:\n", overall_demonstration)
            full_prompt = user_prompt_template.replace("[[DEMONSTRATIONS]]", overall_demonstration)
        
        # 针对role paly的模型，需要加上user等角色
        # 针对gsm8k的处理逻辑不一样
        if self.dataset_name == "gsm8k":
            question = test_example['question'].strip()
            # 2026.3.13 对question进行重复
            question_list = [question,] * self.repeat_time
            question = "\n".join(question_list)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
        else:
            context = test_example['context'].strip()
            question = test_example['question'].strip()
            options = '\n'.join([opt.strip() for opt in test_example['options']])
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[OPTIONS]]', options)
            # 2026.3.14 对question进行重复
            if self.repeat_time >1:
                # print("进入repeat")
                split_parts = full_prompt.split("------")
                all_context = split_parts[-1]
                all_context = all_context.replace("Reasoning:", "")
                all_context = (all_context*self.repeat_time)
                all_context += "Reasoning:\n"
                split_parts[-1] = all_context
                # print(split_parts[-1])
                # split_parts[-1] = new_question_parts
                full_prompt = "------".join(split_parts)
                # print(full_prompt)
                # exit()
        messages = [
            {"role":"system", "content":role_content},
            {"role":"user", "content": full_prompt}
            ]
        if self.mode == "RAG":
            # laska 修改，针对本地模型，返回messages
            # 每检索一条，将检索结果写入文件
            retrieval_record = {
                'context': test_example.get('context',""),
                'question': test_example['question'],
                'retrieved_demonstrations': full_prompt
            }
            # 写入json文件
            self.retrieval_writer.write(json.dumps(retrieval_record, ensure_ascii=False) + '\n')
        return messages

    # 针对zero-shot，直接生成prompt
    def prompt_LSAT_zero_shot(self, in_context_example, test_example):
        # 2025.11.11 add system prompt
        role_content = self.load_system_prompt()
        user_prompt_template = self.load_user_prompt_template()  # 目前这一部分的选择是不一样的
        full_prompt = user_prompt_template
        # 针对gsm8k的处理逻辑不一样
        if self.dataset_name == "gsm8k":
            question = test_example['question'].strip()
            # full_prompt = f"Problem: {question}\nReasoning:"
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
        else:  # 针对其他逻辑推理的数据集
            context = test_example['context'].strip()
            question = test_example['question'].strip()
            options = '\n'.join([opt.strip() for opt in test_example['options']])
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[OPTIONS]]', options)

        messages = [
            {"role":"system", "content":role_content},  
            {"role":"user", "content": full_prompt}
            ]   
        return messages
       
    # 针对few-shot的处理代码
    def load_in_context_examples(self):
        with open(os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}.txt')) as f:
            in_context_examples = f.read()
        return in_context_examples

    # laska 加载本地数据集，2025.12.09
    # 使用HuggingFace datasets库加载本地数据集
    def load_raw_dataset(self, split):
        """
        使用 HuggingFace datasets 库加载本地 JSON / JSONL 数据。
        约定：
        - gsm8k:  文件名为 {split}.jsonl
        - 其他数据集: 文件名为 {split}.json
        """
        if self.dataset_name == "gsm8k":
            file_name = f"{split}.jsonl"   # 原来就是 jsonl
        else:
            file_name = f"{split}.json"

        data_file = os.path.join(self.data_path, self.dataset_name, file_name)

        # 用 datasets 读本地 json/jsonl
        # 这里用 data_files={split: path} 的形式，方便保留 split 名字
        ds_dict = load_dataset(
            "json",
            data_files={split: data_file}
        )
        raw_dataset: Dataset = ds_dict[split]

        print(f"[datasets] Loaded {len(raw_dataset)} examples from {data_file}")
        return raw_dataset

    def load_raw_dataset_old(self, split):
        if self.dataset_name == "gsm8k":
            with open(os.path.join(self.data_path, self.dataset_name, f"{self.split}.jsonl"), 'r') as f:
                raw_dataset = [json.loads(line) for line in f]
            return raw_dataset
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    # laska 调用本地模型生成结果
    def model_generate(self, messages):
        # print(type(messages), type(messages[0]), len(messages), len(messages[0]))
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # apply之后得到的text是一个字符串，而tokenizer的输入需要是一个list，所以需要[text]
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        # model.generate返回的结果是一个[[... ...]]的二维list，单条和batch的区别在于第一维的长度
        # print("--------the final answer is !!!!---------")
        # print(generated_ids)
        # 针对单条数据，需要去掉前面input_id的部分
        generated_ids = generated_ids[:,len(model_inputs.input_ids[0]):]
        # print(generated_ids.shape)
        # print(generated_ids)
        # response的返回是列表的形式，针对单条数据的测试，需要取第1条元素
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print("++++++the response is ++++++++++")
        # print(response)
        return response
    
    # laska 定义一个针对batch数据进行解码的函数
    def model_generate_batch(self, messages_list):
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
        # texts 经过apply函数之后是str的列表
        # print(type(texts), len(texts), type(texts[0]), len(texts[0]))
        # 分为vllm和普通调用两部分
        if self.vllm_switch:
            # vllm的调用，与model generate不同，不需要进行tokenizer的encode
            outputs = self.model.generate(texts, sampling_params=self.sampling_params)
            print(outputs)
            responses = [output.outputs[0].text for output in outputs]
            # print(responses)
                
        else:
            model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            generated_ids = self.model.generate(**model_inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
            # print("--------the final answer is !!!!---------")
            # print(generated_ids)
            generated_ids = [output_ids[len(input_ids):] for output_ids, input_ids in zip(generated_ids, model_inputs.input_ids)]
            # response
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # print(responses)
        return responses  # 返回当前一个batch的结果
    
    # laska 定义一个调用入口, 分配是batch还是单条
    def generation_entrance(self):
        if self.batch_test:
            print("进行batch测试")
            self.batch_reasoning_graph_generation(batch_size=self.batch_size)
        else:
            print("进行单条测试")
            self.reasoning_graph_generation()

    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from split = {self.split}.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()
        
        outputs = []
        for example in tqdm(raw_dataset):
            question = example['question']

            # create prompt
            full_prompt = self.prompt_creator(in_context_examples, example)
#             print(full_prompt)
            # 修改这部分模型生成代码
#             output = self.openai_api.generate(full_prompt)
            # output = self.model_generate(full_prompt)
            # laska ，修改为同一个函数调用，唯一的差别是list中的元素个数
            # 此处的full_prompt是一个list，qwen的输入格式，包含system和user两个部分
            outputs = self.model_generate_batch(full_prompt)
            output = outputs[0]  # 取出单条数据的结果
            # get the answer
            label_phrase = self.label_phrase    #  self.label_phrase = 'The correct option is:'
            generated_answer = output.split(label_phrase)[-1].strip()
            generated_reasoning = output.split(label_phrase)[0].strip()

            # create output
            output = {'id': example['id'], 
                      'question': question, 
                      'answer': example['answer'], 
                      'predicted_reasoning': generated_reasoning,
                      'predicted_answer': generated_answer}
            outputs.append(output)
            # 定义一个测试的开关
            if self.all_data_switch == False:
                print(full_prompt)
                print("当前只测试一条数据，查看结果即可")
                print(output)
                break
        # save outputs        
        with open(self.save_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    # laska 定义一个batch测试的代码
    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples,针对非0-shot的场景
        if self.mode in ["CoT", "Direct"] and not self.zero_shot:    # rag形式需要自行查找context
            in_context_examples = self.load_in_context_examples()
        else:   # rag/cot-0shot
            in_context_examples = ""
            
        outputs = []
        # split dataset into chunks
        num_examples = len(raw_dataset)
#         dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        # for chunk in tqdm(dataset_chunks):
        for start in tqdm(range(0, num_examples, batch_size)):
            end = min(start + batch_size, num_examples)
            chunk = raw_dataset.select(range(start, end))
            # create prompt
            full_prompts = [self.prompt_creator(in_context_examples, example) for example in chunk]
            # 调用模型进行batch的预测
            batch_output = self.model_generate_batch(full_prompts)
            for sample, output in zip(chunk, batch_output):
                # get the answer
                dict_output = self.update_answer(sample, output)
                outputs.append(dict_output)
            # 定义一个测试的开关
            if self.all_data_switch == False:
                print(full_prompts)
                print("当前只测试一个batch数据，查看结果即可")
                print(outputs)
                break
        # save outputs        
        with open(self.save_file, 'w') as f:
            print("saving results to ", self.save_file)
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def update_answer(self, sample, output):
        # 针对gsm8k是单独的处理
        if self.dataset_name == "gsm8k":
            label_phrase = "Final answer:"
            generated_answer = output.split(label_phrase)[-1].strip().lstrip("<").rstrip(">")
            generated_reasoning = output.split(label_phrase)[0].strip()
        # 针对其他逻辑推理的数据集ProntoQA、ProofWriter等
        else:    
            if self.mode in ["Direct", "CoT", "RAG"]:
                label_phrase = self.label_phrase
            elif self.mode in ["Logical"]:
                label_phrase = "Answer:"
                
            if label_phrase not in output and label_phrase.lower() in output:
                label_phrase = label_phrase.lower()
            generated_answer = output.split(label_phrase)[-1].strip()
            if generated_answer.lower() == "true":
                generated_answer = "A"
            elif generated_answer.lower() == "false":
                generated_answer = "B"
            generated_reasoning = output.split(label_phrase)[0].strip()
        dict_output = {'id': sample['id'], 
                        'question': sample['question'], 
                        'answer': sample['answer'], 
                        'predicted_reasoning': generated_reasoning,
                        'predicted_answer': generated_answer,
                        'generation_context':output}
        return dict_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
#     parser.add_argument('--model_path', type=str, default='../llms')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str, help='Direct or CoT or logical', default='Direct')
    parser.add_argument('--max_new_tokens', type=int)
    # laska定义一个针对0-shot的代码
    parser.add_argument('--zero_shot', default=False, action='store_true')
    # laska 定义一个batch测试的开关
    parser.add_argument('--batch_test', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    # 定义一个vllm的开关
    parser.add_argument('--use_vllm', default=False, action='store_true')
    # laska 定义一个针对是否对完整数据集进行测试的开关
    parser.add_argument('--all_data_switch', help='当前是否需要对所有数据集进行测试(True)，还是测试代码功能(Fasle:只测试一条数据就可以)', default=False, action='store_true')
    # 10.27 将system prompt放在文件中进行加载
    parser.add_argument('--system_prompt_dir', type=str, default='./system_prompt', help="定义存放system prompt的文件路径")
    # parser.add_argument('--prompt_file', help="定义system prompt的文件路径", type=str, default='logical_prompt_1.txt')
    # 11.7 将rag功能直接加进来
    parser.add_argument('--db_name', type=str, default='gsm8k', help="所使用的RAG db的名字")  # 用于检索的数据库名称
    parser.add_argument('--index_path', type=str, default='../rag_db', help="RAG向量数据库的路径")  # RAG向量数据库的路径
    parser.add_argument('--icl_num', type=int, default=0, help="RAG检索后使用的示例个数")  # RAG检索后使用的示例个数
    parser.add_argument('--top_k', type=int, default=3, help="RAG检索的top k个数")  # RAG检索的top k个数
    parser.add_argument('--rag_result_path', type=str, default='./rag_results', help="RAG检索中间结果的保存路径")  # RAG检索中间结果的保存路径
    parser.add_argument("--db_type", type=str, help="可选的langchain db类型，embedding或者bm25", default="embedding")
    # 2025.11.11 user_template_dir
    parser.add_argument("--user_template_dir", type=str, default="./user_template", help="用于存放user template文件的dir路径")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument('--reverse_rag_order', default=False, action='store_true')
    parser.add_argument('--random_rag_order', default=False, action='store_true')
    parser.add_argument("--embedding_model", type=str, help="所使用的embedding模型名字", default="../llm/bge-large-en-v1.5")
    # 20251216 新增cone 的rerank功能
    parser.add_argument("--rerank", default=False, help="是否对检索的候选进行cone重排序",action='store_true')
    # 20260313 新增对prompt进行重复的测试
    parser.add_argument("--repeat_time", type=int, default=1, help="是否直接对prompt进行简单的复制")
    # 20260316 新增一个把context放在后面的测试
    parser.add_argument("--repeat_reverse", default=False, action='store_true', help="是否将context放在question之后，将会影响代码运行过程中所选择加载的user_prompt文件")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    llm_problem_reduction = LLM_Reasoning_Graph_Baseline(args)
    # 尝试全部直接调用batch的生成代码
    llm_problem_reduction.generation_entrance()
#     llm_problem_reduction.batch_reasoning_graph_generation(batch_size=10)
    # llm_problem_reduction.reasoning_graph_generation()
