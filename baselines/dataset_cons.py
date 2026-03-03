from builtins import FileNotFoundError
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.retrievers import BM25Retriever
import pickle
import json
import os 
import argparse
import random

embedding_path = "../llms/text2vec-large-chinese"

# from openicl import TopkRetriever


# 将gsm8k或者其他数据集构建处理成向量库
class DatasetCons:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.data_path = "../rag_data"
        self.ds_cot = args.ds_cot    # 这个参数主要是针对gsm8k和ProntoQA数据集，这两个数据集有自己的cot，该参数设置为true时，就使用ds接口生成的cot来构建langchian数据集，为false时，使用这两个数据集默认的cot
        if self.dataset_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
            self.ds_cot = True
        elif self.dataset_name in["gsm8k",]:
            self.ds_cot = False
        # self.embedding_path = "../llms/text2vec-large-chinese"
        # self.embedding_path = "../llms/bge-large-en"
        # self.embedding_path = "../llms/bge-large-en-v1.5"
        self.embedding_path = args.embedding_model
        self.db_type = args.db_type    # 可选值有embedding bm25，默认是bm25，新加入一个random
        self.bm25_file = "bm25_index.pkl"
        self.top_k = args.top_k
        self.split = args.db_split
        self.langchain_db_dir = "../rag_db"  # 用于存放langchain dataset的路径
        self.label_phrases = ["The correct option is:", "the correct option is:", "The final answer is:", "the final answer is:"]
    
    # 20251127 添加一个对错误数据删除的逻辑
    def wrong_cot_filter(self, answer, cot):
        keep_flag = False   # 是否保留当前数据的标志位
        for label_phrase in self.label_phrases:
            if label_phrase not in cot:
                continue
            else:
                cot_answer = cot.split(label_phrase)[-1].strip()
                if cot_answer == answer:
                    keep_flag = True
        return keep_flag

    def gsm8k_load_data(self, split="train"):
        "加载gsm8k数据集"
        "split可选 train test"
        print("current loading function is gsm8k_load_data")
        # data_file = Path(self.data_path) / f"{self.dataset_name}/{split}.jsonl"
        data_file = Path(self.data_path) / f"{split}.jsonl"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # print(line)
                # print(type(line))
                data_dict = json.loads(line)
                # print(type(data_dict))
                if line:
                    question = data_dict.get("question", "")
                    answer = data_dict.get("answer", "")
                    cot = data_dict.get("cot", "")
                    
                    documents.append(Document(page_content=question, metadata={"answer": answer, "cot": cot}))
        return documents
    
    # 10.29 加载LogicalDeduction数据集
    def logicaldeduction_load_data(self, split="train"):
        "加载LogicalDeduction数据集，可选train 和dev数据集"
        print("current loading function is logicaldeduction_load_data")
        data_file = Path(self.data_path) / f"LogicalDeduction_{split}_cot.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r", encoding="utf-8") as f:
            item_list = json.load(f)
            print(type(item_list), len(item_list))
            for item in item_list:
                context = item.get("context", "")
                question = item.get("question", "")
                options = item.get("options", "")
                answer = item.get("answer", "")
                cot = item.get("reasoning_cot", "")
                # full_prompt = f"Context: {context}\nQuestion: {question}\nExplanation: {explanation}"
                full_prompt = f"Context: {context}\nQuestion: {question}\n"
                # 添加筛选数据的逻辑
                keep_flag = self.wrong_cot_filter(answer, cot)
                if not keep_flag:
                    continue
                documents.append(Document(page_content=full_prompt, metadata={"answer": answer, "cot": cot, "question": question, "context": context, "options": options}))
        return documents
    
    # 20251119 构建一个逻辑推理任务通用的load_data
    def logical_task_common_cot_load_data(self, split="train"):
        print("current loading function is logical_task_common_cot_load_data")
        data_file = Path(self.data_path) / f"{self.dataset_name}_{split}_cot.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r", encoding="utf-8") as f:
            item_list = json.load(f)
            print(type(item_list), len(item_list))
            for item in item_list:
                context = item.get("context", "")
                question = item.get("question", "")
                options = item.get("options", "")
                answer = item.get("answer", "")
                cot = item.get("reasoning_cot", "")
                # 添加筛选数据的逻辑
                keep_flag = self.wrong_cot_filter(answer, cot)
                if not keep_flag:
                    continue
                # full_prompt = f"Context: {context}\nQuestion: {question}\nExplanation: {explanation}"
                full_prompt = f"Context: {context}\nQuestion: {question}\n"
                documents.append(Document(page_content=full_prompt, metadata={"answer": answer, "cot": cot, "question": question, "context": context, "options": options}))
        return documents
    
    # 10.26 加载逻辑语言的数据集（这个数据集的explanation用的是数据本身提供的，而非ds生成cot）
    def ProntoQA_load_data(self, split="dev"):
        "加载prontoqa数据集"
        "split可选 dev"
        print("current loading function is prontoqa_load_data")
        data_file = Path(self.data_path) / f"{split}.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r") as f:
            item_list = json.load(f)
            print(type(item_list), len(item_list))
            for item in item_list:
                context = item.get("context", "")
                explanation = "\n".join(item.get("explanation", ""))
                question = item.get("question", "")
                answer = item.get("answer", "")
                options = item.get("options", "")
                # full_prompt = f"Context: {context}\nQuestion: {question}\nExplanation: {explanation}"
                full_prompt = f"Context: {context}\nQuestion: {question}\n"
                documents.append(Document(page_content=full_prompt, metadata={"answer": answer, "explanation": explanation, "question": question, "context": context, "options": options}))
        return documents

    def build_vector_store(self):
        if self.db_type == "random":
            print("the random retrieve mode dont need to build a faiss db")
            return 
        if self.dataset_name not in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO", "gsm8k", "commonsense"]:
            print(f"the wrong dataset {self.dataset_name} were provided. Ended the program!")
            return
        # 加载调用deepseek端口的cot
        if self.ds_cot==True and self.dataset_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
            load_fn = self.logical_task_common_cot_load_data
        elif self.dataset_name in ["gsm8k", "ProntoQA"]:
            load_fn = getattr(self, f"{self.dataset_name}_load_data")
        # commonsenseQA利用gsm8k的接口
        elif self.dataset_name == "commonsense":
            load_fn = self.gsm8k_load_data
        else:
            raise ValueError(f"Unknown dataset:{self.dataset_name}, unrecognized load type:{self.ds_cot}")
#         try:
#             load_fn = getattr(self, f"{self.dataset_name}_load_data")
#         except AttributeError:
#             raise ValueError(f"Unknown dataset: {self.dataset_name}")
        # documents = load_fn(self.split)
        documents = load_fn()
        print("the langchain documents num is :", len(documents))
        # exit()
        save_index_path = Path(self.langchain_db_dir) / f"{self.dataset_name}"
        
        if not os.path.exists(save_index_path):
            os.makedirs(save_index_path)
        if self.db_type== "embedding":
            print("loadding embedding model from local path:", self.embedding_path)
            self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_path)   # 直接本地加载
            vector_store = faiss.FAISS.from_documents(documents, self.embedding)
            vector_store.save_local(save_index_path)
            print(f"Embedding Vector store saved to {save_index_path}")
        elif self.db_type == "bm25":
            save_pickle_path = Path(save_index_path) / self.bm25_file
            retriever = BM25Retriever.from_documents(documents, k=self.top_k+1)
            with open(save_pickle_path, "wb") as f:
                pickle.dump(retriever, f)
            print(f"BM25 Vector store saved to {save_pickle_path}")

        # 20251208测试topk_cone算法
        elif self.db_type == "topk_cone":
            pass
        else:
            raise ValueError(f"not a valid db_type: {self.db_type}! must be embedding or bm25")

# 构建一个利用langchain数据库进行处理的类
class DatasetRetriever:
    def __init__(self, args):
        # self.embedding_path = "../llms/text2vec-large-chinese"
        # self.embedding_path = "../llms/bge-large-en"
        self.embedding_path = args.embedding_model
        self.db_name = args.db_name   # 作为外部demonstration的检索库
        self.db_type = args.db_type
        self.langchain_db_path = Path("../rag_db") / f"{self.db_name}"
        self.bm25_file = "bm25_index.pkl"
        self.top_k = args.top_k
        # 初始化向量数据库
        self.retriever_init()        
    
    # 20251128按照检索器类型，对检索器进行初始化
    def retriever_init(self):
        if self.db_type == "embedding":
            # 载入本地faiss向量数据库
            print("loadding embedding model from local path:", self.embedding_path)
            self.embedding = HuggingFaceEmbeddings(model_name = self.embedding_path)
            print('this is the faiss retriever')
            self.vector_store = faiss.FAISS.load_local(self.langchain_db_path, self.embedding, allow_dangerous_deserialization=True)
        elif self.db_type == "bm25":
            # 载入本地bm25 pickle数据库
            print('this is the bm25 retriever')
            pickle_file = Path(self.langchain_db_path) / self.bm25_file
            with open(pickle_file, "rb") as f:
                self.retriever = pickle.load(f)
        else:
            raise ValueError(f"not a valid db_type: {self.db_type}! must be embedding or bm25")

    def retrieve(self, query, top_k=10):
        if self.db_type == "embedding":
            results = self.vector_store.similarity_search(query, k=top_k)
#             print("this is inside", len(results))
        elif self.db_type == "bm25":
            results = self.retriever.invoke(query)
        retrieve_list = []
        for i, doc in enumerate(results):
            # print(f"Result {i+1}:")
            # print("Content:", doc.page_content)
            # print("Metadata:", doc.metadata)
            # print("-------------------")
            page_content = doc.page_content
            metadata = doc.metadata
            if self.db_name in ["gsm8k","commonsense"]:
                context = ""
                question = page_content
                answer = metadata.get("answer", "")
                options = ""
                cot = metadata.get("cot", "")
            # 20251119修改
            elif self.db_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
                context = metadata.get("context", "")
                question = metadata.get("question", "")
                options = metadata.get("options", "")
                answer = metadata.get("answer", "")
                cot = metadata.get("cot", "")
                if self.db_name == "ProntoQA":
                    cot = metadata.get("cot") or metadata.get("explanation", "")
            else:
                raise ValueError("the retriever method are not supported~!", self.dataset_name)
            # 如果检索的数据库和测试数据库是同一个，去掉和query相同的检索结果
            if question in query and context in query:
                # print("~~~~~~~")
                # print(question)
                # print("---------")
                # print(query)
                # print("---------")
                # print(context)
                # print("~~~~~~~")
                # print("去掉当前的检索结果！！！")
                continue
            retrieve_list.append({
                "context": context,
                "question": question,
                "options": options,
                "answer": answer,
                "cot": cot
            })
        return retrieve_list

# 构建一个随机选择的类
class RandomRetriever(DatasetCons):
    def __init__(self, args):
        self.dataset_name = args.db_name
        self.data_path = "../rag_data"
        if self.dataset_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
            self.ds_cot = True
        elif self.dataset_name in["gsm8k",]:
            self.ds_cot = False
        self.split = "train"
        self.label_phrases = ["The correct option is:", "the correct option is:", "The final answer is:", "the final answer is:"]
        self.documents = self.retriever_init()
        pass

    def retriever_init(self):
        if self.dataset_name not in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO", "gsm8k"]:
            print(f"the wrong dataset {self.dataset_name} were provided. Ended the program!")
            return
        # 加载调用deepseek端口的cot
        if self.ds_cot==True and self.dataset_name in ["AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
            load_fn = self.logical_task_common_cot_load_data
        elif self.dataset_name in ["gsm8k", "ProntoQA"]:
            load_fn = getattr(self, f"{self.dataset_name}_load_data")
        else:
            raise ValueError(f"Unknown dataset:{self.dataset_name}, unrecognized load type:{self.ds_cot}")
        documents = load_fn()
        print("the langchain documents num is :", len(documents))
        return documents
    
    def retrieve(self, query, k=10):
        results = random.sample(self.documents, k)
        retrieve_list = []
        for i, doc in enumerate(results):
            # print(f"Result {i+1}:")
            # print("Content:", doc.page_content)
            # print("Metadata:", doc.metadata)
            # print("-------------------")
            page_content = doc.page_content
            metadata = doc.metadata
            if self.dataset_name == "gsm8k":
                context = ""
                question = page_content
                answer = metadata.get("answer", "")
                options = ""
                cot = metadata.get("cot", "")
            # 20251119修改
            elif self.dataset_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
                context = metadata.get("context", "")
                question = metadata.get("question", "")
                options = metadata.get("options", "")
                answer = metadata.get("answer", "")
                cot = metadata.get("cot", "")
                if self.dataset_name == "ProntoQA":
                    cot = metadata.get("cot") or metadata.get("explanation", "")
            else:
                raise ValueError("the retriever method are not supported~!", self.dataset_name)
          
            retrieve_list.append({
                "context": context,
                "question": question,
                "options": options,
                "answer": answer,
                "cot": cot
            })
        return retrieve_list

# 添加parse处理参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_type", type=str, help="可选的langchain db类型，embedding或者bm25", default="embedding")
    parser.add_argument("--dataset_name", type=str, help="构建langchain db的数据集名字", default="gsm8k")
    parser.add_argument("--ds_cot", help="是否使用ds接口生成的cot，还是用默认的数据,gsm8k和prontoQA有两种数据形式",default=False, action="store_true")
    parser.add_argument("--db_split", type=str, help="所使用的数据集split", default="train")
    parser.add_argument("--top_k",type=int, help="检索时返回的topk样例个数", default=3)
    parser.add_argument("--db_name", type=str, help="langchain数据库的名字，主要是检索时用，区分源、目标域", default="gsm8k")
    parser.add_argument("--embedding_model", type=str, help="所使用的embedding模型名字", default="../llm/bge-large-en-v1.5")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # 构建langchain检索向量数据库
    # laska 修改使用逻辑语言来进行icl
    data_path = "../rag_data"
    dataset_cons = DatasetCons(args)
    dataset_cons.build_vector_store()

    # 利用构建好的检索向量数据库进行实验
    if args.db_type in ["bm25", "embedding"]:
        dataset_retriever = DatasetRetriever(args)
    else:
        # 测试随机检索器
        dataset_retriever = RandomRetriever(args)
        # print(dir(dataset_retriever))
        # print(type(dataset_retriever.documents))
        # res = dataset_retriever.retrieve("")
        # print(res)
    # exit()

    context = "The bear does not chase the lion. The bear visits the lion. The bear visits the tiger. The lion chases the bear. The lion chases the tiger. The lion visits the bear. The rabbit chases the tiger. The rabbit is big. The rabbit is not green. The rabbit is young. The rabbit visits the tiger. The tiger chases the rabbit. The tiger is rough. The tiger visits the bear. If something chases the bear and it sees the rabbit then the bear does not visit the rabbit. If something chases the lion then the lion is rough. If the rabbit chases the tiger and the tiger visits the rabbit then the rabbit is not big. If something chases the lion then the lion does not visit the tiger. If something chases the rabbit and it chases the tiger then the rabbit chases the bear. If something is young and it does not see the rabbit then it sees the bear. If something visits the lion then it is young. If the tiger visits the bear and the bear is big then the tiger visits the lion. Young things are big."
    question = "Based on the above information, is the following statement true, false, or unknown? The bear visits the lion."
    query = f"Context: {context}\nQuestion: {question}\n"
    results = dataset_retriever.retrieve(query,4)
    print(results)
    print(len(results))

