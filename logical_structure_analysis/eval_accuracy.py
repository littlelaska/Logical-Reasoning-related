#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 统计对不同 logical_structure_analysis 结果文件的准确率，可选按逻辑结构分类统计
# 对应的启动脚本是logical_eval.sh

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Utils
# -------------------------
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data)}")
    return data

def normalize_text(s: str) -> str:
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


def extract_choice(s: str) -> Optional[str]:
    """
    Extract A/B/C/D or True/False/Unknown
    """
    s = s.lower()

    # choice letter
    m = re.search(r"\b([abcd])\b", s)

    if m:
        return m.group(1).upper()

    # boolean
    if "true" in s:
        return "True"
    if "false" in s:
        return "False"
    if "unknown" in s:
        return "Unknown"

    return None


def extract_number(s: str) -> Optional[float]:
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[-1])  # last number is usually final answer
    except Exception:
        return None


def match_answer(pred: str, gold: str) -> Tuple[bool, str]:
    """
    Returns (is_correct, match_type)
    """
    if not pred or not gold:
        return False, "empty"

    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    # 1) choice / boolean
    p_choice = extract_choice(pred)
    g_choice = extract_choice(gold)
    if p_choice and g_choice:
        return (p_choice == g_choice), "choice"

    # 2) numeric
    p_num = extract_number(pred)
    g_num = extract_number(gold)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-6, "numeric"

    # 3) string fallback
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True, "string"

    return False, "string"

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')
    return None

def evaluate_QA(result_file, args):
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    total_em = 0.0
    total_f1 = 0.0
    count = 0
    # laska新增，记录错误样例的id
    wrong_ids = []
    sample_index = 0 
    if args.by_structure:
        by_struct = {}
    for sample in QA_results:
        # print(type(sample))
        # print(sample.keys())
        # print(QA_results.index(sample))
        gold_answer = sample[args.gold_key].replace('(', '').replace(')', '').strip()
        try:
            answer_str = sample[args.pred_key].strip()
        except AttributeError:
            answer_str = str(sample[args.pred_key]).strip()
        prediction = get_choice(answer_str)

        indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is', 'Answer:']
        if prediction is None:
            for indicator in indicators:
                if answer_str.find(indicator)>=0:
                    answer_str = answer_str.split(indicator)[1].strip()
                    print(answer_str)
                    prediction = get_choice(answer_str)
                    print(prediction)
                    break

        # if prediction is None:
        #     print(answer_str)
        # print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
        if args.by_structure:
            struct = sample.get('logical_structure', {}).get('structure_type', 'Unknown')  # 获取逻辑结构
            if struct not in by_struct:
                by_struct[struct] = [0, 0]  # [correct, total]
            by_struct[struct][1] += 1
            if prediction == gold_answer:
                by_struct[struct][0] += 1
        
        em_score = 1.0 if prediction == gold_answer else 0.0
        total_em += em_score
        count += 1
        if em_score != 1.0:
            # print(sample_index, f"prediction: {prediction} \t gold_answers: {gold_answer}")
            wrong_ids.append(sample_index)
        sample_index += 1
    
    avg_em = total_em / count
    
    print("=" * 60)
    print(f"File: {args.input}")
    print(f"Total samples     : {len(QA_results)}")
    print(f"Correct           : {int(total_em)}")
    print(f"EM: {avg_em}")
    print(f"wrong samples are {len(wrong_ids)}, ids are:\n", wrong_ids)
    # print(f"Invalid predictions: {invalid}")
    # print(f"Accuracy          : {acc:.4f}")
    print("=" * 60)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="*_icl.json result file")
    ap.add_argument("--pred_key", default="predicted_answer")
    ap.add_argument("--gold_key", default="answer")
    ap.add_argument("--by_structure", action="store_true",
                    help="Report accuracy per logical_structure")
    args = ap.parse_args()
    
    evaluate_QA(args.input, args)
    # data = load_json_list(args.input)

    # total = 0
    # correct = 0
    # invalid = 0

    # by_struct = {}  # type -> [correct, total]
    # wrong_ids = []
    # ind = 0
    # for ex in data:
    #     pred = ex.get(args.pred_key, "")
    #     gold = ex.get(args.gold_key, "")

    #     ok, mtype = match_answer(str(pred), str(gold))

    #     total += 1
    #     if ok:
    #         correct += 1
    #     else:
    #         wrong_ids.append(ind)
    #     ind += 1
    #     if mtype == "empty":
    #         invalid += 1

    #     if args.by_structure:
    #         t = ex.get("logical_structure", {}).get("structure_type", "Unknown")
    #         if t not in by_struct:
    #             by_struct[t] = [0, 0]
    #         by_struct[t][1] += 1
    #         if ok:
    #             by_struct[t][0] += 1

    # acc = correct / total if total > 0 else 0.0

    # print("=" * 60)
    # print(f"File: {args.input}")
    # print(f"Total samples     : {total}")
    # print(f"Correct           : {correct}")
    # print(f"Invalid predictions: {invalid}")
    # print(f"Accuracy          : {acc:.4f}")
    # print(f"Wrong sample IDs  : {wrong_ids}")
    # print("=" * 60)

    # if args.by_structure:
    #     print("Accuracy by logical structure:")
    #     for t, (c, n) in sorted(by_struct.items()):
    #         a = c / n if n > 0 else 0.0
    #         print(f"  {t:10s} : {a:.4f}  ({c}/{n})")


if __name__ == "__main__":
    main()
