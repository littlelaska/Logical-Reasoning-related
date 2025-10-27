import re
import json
from tqdm import tqdm
import random
import os
import argparse

def extract_number(string):
    # Remove all characters except digits, decimal point and negative sign
    try:
        num_string = re.sub(r'[^\d.-]', '', string)
        num_string = num_string.replace('$', '')
        return float(num_string)
    except:
        try:
            return float(random.randint(0, 100))
            # return float(w2n.word_to_num(string))
        except:
            # print('Error: ', string)
            print('Error')
            return float(random.randint(0, 100))

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))
    # return prediction == truth

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_sample(prediction, gold_answers):
    em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
    f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
    return em_score, f1_score

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')
    return None

def evaluate_QA(result_file):
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    total_em = 0.0
    total_f1 = 0.0
    count = 0
    # laska新增，记录错误样例的id
    wrong_ids = []
    sample_index = 0 
    for sample in QA_results:
        gold_answer = sample['answer'].replace('(', '').replace(')', '').strip()
        answer_str = sample['predicted_answer'].strip()
        prediction = get_choice(answer_str)

        indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is']
        if prediction is None:
            for indicator in indicators:
                if answer_str.find(indicator)>=0:
                    answer_str = answer_str.split(indicator)[1].strip()
                    prediction = get_choice(answer_str)
                    break

        if prediction is None:
            print(answer_str)

        print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
        
        em_score = 1.0 if prediction == gold_answer else 0.0
        total_em += em_score
        count += 1
        if em_score != 1.0:
            wrong_ids.append(sample_index)
        sample_index += 1
    
    avg_em = total_em / count
    print(f"EM: {avg_em}")
    print(f"wrong samples are {len(wrong_ids)}, ids are:\n", wrong_ids)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--result_path', type=str, default='./results')
    # laska 新增，针对不同的icl设置，文件名会有不同
    parser.add_argument('--zero-shot', default=False, action='store_true')
    parser.add_argument('--db_name', type=str, default='gsm8k', help="所使用的RAG db的名字")  # 用于检索的数据库名称
    parser.add_argument('--icl_num', type=int, default=2, help="RAG检索后使用的示例个数")  # RAG检索后使用的示例个数
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.zero_shot:
        testing_type = '0-shot'
    else:
        testing_type = 'few-shot'
    if args.mode in ["Direct", "CoT", "Logical"]:
        result_file = os.path.join(args.result_path, f'{args.mode}_{testing_type}_{args.dataset_name}_{args.split}_{args.model_name}.json')
    elif args.mode == "RAG":
        result_file = os.path.join(args.result_path, f'{args.mode}{args.icl_num}_{args.db_name}_{args.dataset_name}_{args.split}_{args.model_name}.json')
    evaluate_QA(result_file)
    print("当前验证的文件为：", result_file)
