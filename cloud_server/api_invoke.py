from cloud_task_client import *
import cloud_task_client
import argparse
import time
import os
import json
from tqdm import tqdm

class DatasetCotGen:
    """针对逻辑推理数据集进行 CoT 生成的类"""
    def __init__(self, dataset_name, split="dev", all_data_switch=False, save_path="./results/", api_key=None):
        self.save_path = save_path
        self.dataset_name = dataset_name
        if self.dataset_name not in ["ProntoQA", "LogicalDeduction", "FOLIO", "ProofWriter", "AR-LSAT"]:
            raise ValueError(f"不支持的数据集名称：{self.dataset_name}，请使用支持的数据集名称。")
        elif self.dataset_name == "ProntoQA":
            self.dataset_path = "../data/ProntoQA"
        elif self.dataset_name == "LogicalDeduction":
            self.dataset_path = "../data/LogicalDeduction"
        elif self.dataset_name == "FOLIO":
            self.dataset_path = "../data/FOLIO"
        elif self.dataset_name == "ProofWriter":
            self.dataset_path = "../data/ProofWriter"
        elif self.dataset_name == "AR-LSAT":
            self.dataset_path = "../data/AR-LSAT"
        self.split = split
        # 开关决定是对所有数据进行处理还是对单条数据进行测试
        self.all_data_switch = all_data_switch
        if not api_key:
            self.api_key = "TnumU6cM" #默认服务密钥
        else:
            self.api_key = api_key
        self.CLOUD_IP = "47.115.134.188"
        self.CLOUD_PORT = 13344
        self.task_instruction = "You are a careful reasoner. Think step by step with concise chain-of-thought. Then on a new line, output exactly: 'The correct option is: A' or 'The correct option is: B'"
        
    def data_loader(self):
        with open(os.path.join(self.dataset_path, f"{self.split}.json"), 'r') as f:
            data = json.load(f)
        return data
     
    # 连通性测试函数
    def reachability_test(self):
        backoff = 1.0
        while not cloud_task_client._can_reach(CLOUD_IP, CLOUD_PORT, 2.0):
            print(f"[WARN] 目标 {CLOUD_IP}:{CLOUD_PORT} 未连通，5s 后重试…")
            time.sleep(backoff)
            backoff = min(10.0, backoff * 1.7)  

    # 定义一个请求入口
    def retrieve_query_res(self, all_data_switch=None):
        query_data = self.data_loader()
        all_data_switch = all_data_switch if all_data_switch is not None else self.all_data_switch
        if not all_data_switch:
            query_data = [query_data[0]]  # 只测试第一条数据
        # 保存结果
        os.makedirs(self.save_path, exist_ok=True)
        savefile_path = os.path.join(self.dataset_path, f"{self.dataset_name}_{self.split}_cot.json")
        existing_results = []
        self.reachability_test()  # 测试联通性
        for item in tqdm(query_data):
            context = item['context'].strip()
            question = item['question'].strip()
            options = "\n".join(opt.strip() for opt in item['options'])
            request_query = self.task_instruction + f"Context:\n{context}\n\nQuestion: {question}\n\nOptions:\n {options}\n\n The correct option is:"
            
            command = "[create_conversation=true]"
            # 调用接口获得结果
            result = cloud_task_client.api(self.api_key, request_query + command)
            print("\n[收到结果]：\n{}".format(result))
            
            item["reasoning_cot"] = result  # 将结果添加到当前数据项中
            existing_results.append(item)
            with open(savefile_path, 'w') as sf:
                json.dump(existing_results, sf, indent=2, ensure_ascii=False)
        print(f"[已保存结果至 {savefile_path}")


if __name__ == "__main__":
    dataset_cot_gen = DatasetCotGen(dataset_name="ProofWriter", split="train", all_data_switch=False, save_path="./results/")
    dataset_cot_gen.retrieve_query_res(all_data_switch=True)