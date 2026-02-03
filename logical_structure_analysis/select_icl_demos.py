#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
from tqdm import tqdm

# ---- Local Embedding ----
from sentence_transformers import SentenceTransformer

# ---- Local LLM (vLLM) ----
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


STRUCT_TYPES = ["Chain", "Y-shaped", "Block", "Other"]

SYSTEM_PROMPT = "You are a logical task solver. Follow the demonstrationa to solve the new question. Remember to think step by step with concise chain-of-thought, and adhere to the context related to the question. Then on a new line, output exactly: 'The correct option is: A' or 'The correct option is: B', etc., based on your reasoning."
SYSTEM_PROMPT_ = """You are a careful logical reasoning assistant.

You may be given:
- Some demonstration examples (for reference only)
- A target logical reasoning problem with multiple-choice options (A, B, C, D, E)

Your task:
1) Use concise but necessary step-by-step reasoning to solve the TARGET problem.
2) Output the final answer on a new line in the exact required format.

Strict output rules:
- If demonstrations are provided, do NOT solve them; only use them as style/format references.
- Do NOT restate the question or options.
- Do NOT add any extra headings, labels, or commentary.
- The final line MUST be exactly:
  The correct option is: <option>
- <option> MUST be a single letter: A, B, C, D, or E
- No extra text, spaces at the end, or punctuation after the letter.

If you are unsure, still choose the best option and follow the format.
"""


SYSTEM_PROMPT_STRUCT = """You are an expert in logical reasoning analysis and chain-of-thought structure inspection.
Your task is NOT to solve the problem, but to predict the reasoning STRUCTURE likely required.

Choose exactly ONE:
(1) Chain: Single linear reasoning path.
(2) Y-shaped: Two or more independent reasoning chains merged at the conclusion.
(3) Block: One or more nodes generate multiple parallel branches, aggregated at the end.
(4) Other: Does not clearly fit the above structures.

Output STRICTLY in JSON:
{
  "structure_type": "Chain | Y-shaped | Block | Other",
  "justification": "briefly explain why this problem likely needs this structure"
}
"""

USER_TEMPLATE = """[Problem]
{problem}
"""



def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data)}")
    return data


def dump_json_list(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_id(ex: Dict[str, Any], fallback_i: int) -> str:
    for k in ["example_id", "id"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"ex_{fallback_i}"


def qtext(ex: Dict[str, Any]) -> str:
    parts = []
    ctx = ex.get("context")
    if isinstance(ctx, str) and ctx.strip():
        parts.append("Context:\n" + ctx.strip())
    parts.append("Question:\n" + str(ex.get("question", "")).strip())
    opts = ex.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        parts.append("Options:\n" + "\n".join([str(o) for o in opts]))
    return "\n".join(parts).strip()


def get_structure_type(ex: Dict[str, Any]) -> Optional[str]:
    ls = ex.get("logical_structure")
    if isinstance(ls, dict):
        t = ls.get("structure_type")
        if t in STRUCT_TYPES:
            return t
    return None


def build_demo_text(ex: Dict[str, Any]) -> str:
    """
    把检索回来的示例构造成icl需要的格式
    """
    # t = get_structure_type(ex) or "Other"
    # parts = [f"[Structure]\n{t}"]
    parts = []
    ctx = ex.get("context")
    if isinstance(ctx, str) and ctx.strip():
        parts.append("Context:\n" + ctx.strip())

    parts.append("Question:\n" + str(ex.get("question", "")).strip())

    opts = ex.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        parts.append("Options:\n" + "\n".join([str(o) for o in opts]))
    rc = ex.get("reasoning_cot")
    if isinstance(rc, str) and rc.strip():
        parts.append("Reasoning:\n" + rc.strip())

    # ans = ex.get("answer")
    # if ans is not None:
    #     parts.append("Answer:\n" + str(ans).strip())
    return "\n\n".join(parts).strip()


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def cosine_sim(q_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    # q_emb: [d], cand_embs: [N,d]
    q = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    C = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)
    return C @ q  # [N]


def mmr_select(cand_idx: List[int], sim_q: np.ndarray, cand_embs: np.ndarray, k: int, lambda_div: float) -> List[int]:
    if k <= 0 or len(cand_idx) == 0:
        return []
    k = min(k, len(cand_idx))

    # precompute cand-cand cosine
    C = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)
    cc = C @ C.T

    selected_local = []
    remaining = list(range(len(cand_idx)))

    first = int(np.argmax(sim_q))
    selected_local.append(first)
    remaining.remove(first)

    while len(selected_local) < k and remaining:
        best_i, best_score = None, -1e18
        for i in remaining:
            max_sim_to_sel = max(cc[i, s] for s in selected_local)
            score = lambda_div * float(sim_q[i]) - (1.0 - lambda_div) * float(max_sim_to_sel)
            if score > best_score:
                best_score = score
                best_i = i
        selected_local.append(best_i)
        remaining.remove(best_i)

    return [cand_idx[i] for i in selected_local]

# -----------------------------
# BM25 (pure python) utilities
# -----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def bm25_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

def bm25_build_corpus_stats(docs_tokens: List[List[str]]) -> Dict[str, Any]:
    """
    Build BM25 corpus statistics:
      - df: document frequency of term
      - idf: BM25 idf
      - doc_tf: list[dict] term->tf for each doc
      - doc_len: list[int]
      - avgdl: float
    """
    N = len(docs_tokens)
    df = {}
    doc_tf = []
    doc_len = []

    for toks in docs_tokens:
        tf = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        doc_tf.append(tf)
        doc_len.append(len(toks))

        seen = set(tf.keys())
        for w in seen:
            df[w] = df.get(w, 0) + 1

    avgdl = (sum(doc_len) / N) if N > 0 else 0.0

    # standard BM25 idf with +1
    idf = {}
    for w, dfi in df.items():
        idf[w] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

    return {
        "N": N,
        "df": df,
        "idf": idf,
        "doc_tf": doc_tf,
        "doc_len": doc_len,
        "avgdl": avgdl,
    }

def bm25_score_query(
    query_tokens: List[str],
    stats: Dict[str, Any],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """
    Return BM25 scores for all docs: shape [N]
    """
    N = stats["N"]
    idf = stats["idf"]
    doc_tf = stats["doc_tf"]
    doc_len = stats["doc_len"]
    avgdl = stats["avgdl"] if stats["avgdl"] > 0 else 1.0

    scores = np.zeros(N, dtype=np.float32)

    # unique query terms (BM25 is tf-based on doc side)
    q_terms = set(query_tokens)
    for i in range(N):
        tf = doc_tf[i]
        dl = doc_len[i]
        denom_base = k1 * (1.0 - b + b * (dl / avgdl))
        s = 0.0
        for w in q_terms:
            if w not in tf:
                continue
            f = tf[w]
            w_idf = idf.get(w, 0.0)
            s += w_idf * (f * (k1 + 1.0)) / (f + denom_base)
        scores[i] = s
    return scores


def neighbor_type(t: str) -> str:
    if t == "Chain":
        return "Y-shaped"
    if t == "Y-shaped":
        return "Block"
    if t == "Block":
        return "Y-shaped"
    return "Chain"


def build_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -----------------------------
# vLLM batch generation helpers
# -----------------------------
def vllm_generate_batch(llm: LLM, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens,top_p=1.0, top_k=1, n=1)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def predict_structure_batch_with_local_qwen(
    llm: LLM,
    tokenizer,
    problem_texts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> List[Dict[str, Any]]:
    messages_list = [
        [{"role": "system", "content": SYSTEM_PROMPT_STRUCT},
         {"role": "user", "content": USER_TEMPLATE.format(problem=pt)}]
        for pt in problem_texts
    ]
    prompts = [build_chat_prompt(tokenizer, m) for m in messages_list]
    texts = vllm_generate_batch(llm, prompts, temperature=temperature, max_tokens=max_tokens)

    out = []
    for t in texts:
        obj = extract_json_obj(t)
        if not obj:
            out.append({"structure_type": "Other", "justification": "Local Qwen output not valid JSON."})
            continue
        st = obj.get("structure_type", "Other")
        if st not in STRUCT_TYPES:
            st = "Other"
        out.append({"structure_type": st, "justification": (obj.get("justification") or "").strip()})
    return out


def build_icl_prompt(demos: List[Dict[str, Any]], query: Dict[str, Any], t_hat: str) -> str:
    demo_blocks = []
    for i, d in enumerate(demos, 1):
        # demo_blocks.append(f"### Demonstration {i}\n{d['demo_text']}")
        demo_blocks.append(f"\n{d['demo_text']}")

    # q_parts = ["### Query", f"[Predicted Structure]\n{t_hat}"]
    # q_parts = ["### Query"]
    q_parts = []
    ctx = query.get("context")
    if isinstance(ctx, str) and ctx.strip():
        q_parts.append("Context:\n" + ctx.strip())
    q_parts.append("Question:\n" + str(query.get("question", "")).strip())
    opts = query.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        q_parts.append("Options:\n" + "\n".join([str(o) for o in opts]))
    q_parts.append("Reasoning:\n")
    # q_parts.append("[Answer]:\n")

    return (
        # "You will be given demonstrations that share a similar reasoning structure. "
        "Given a problem statement as contexts, the task is to answer a logical reasoning question.\n"
        # "Follow the structure to solve the query.\n\n"
        "------"
        + "\n".join(demo_blocks)
        + "------\n"
        + "\n".join(q_parts)
    )


def build_zero_shot_prompt(query: Dict[str, Any], t_hat: str) -> str:
    # q_parts = ["### Query", f"[Predicted Structure]\n{t_hat}"]
    q_parts = [] 
    ctx = query.get("context")
    if isinstance(ctx, str) and ctx.strip():
        q_parts.append("Context:\n" + ctx.strip())
    q_parts.append("Question:\n" + str(query.get("question", "")).strip())
    opts = query.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        q_parts.append("Options:\n" + "\n".join([str(o) for o in opts]))
    # q_parts.append("[Reasoning]\n")
    # q_parts.append("[Answer]\n")
    # q_parts.append("Let's think step by step. The correct option is :")
    q_parts.append("Reasoning:")
    return "Given a problem statement as contexts, the task is to answer a logical reasoning question.\n------\n\n------\n" + "\n".join(q_parts)
    return "Given a problem statement as contexts, the task is to answer a logical reasoning question.\n------\n[[DEMONSTRATIONS]]\n------\n" + "\n".join(q_parts)
    return "Solve the following problem carefully.\n\n" + "\n\n".join(q_parts)


# 按照qwen的调用形式构造messages
def build_chat_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# 将cot中答案和原始answer不一致的样本过滤掉 
def wrong_cot_filter(answer, cot):
    label_phrases= ["The correct option is:", "the correct option is:", "The final answer is:", "the final answer is:"]
    keep_flag = False   # 是否保留当前数据的标志位
    for label_phrase in label_phrases:
        if label_phrase not in cot:
            continue
        else:
            cot_answer = cot.split(label_phrase)[-1].strip()
            if cot_answer == answer:
                keep_flag = True
    return keep_flag

# 新增一个抽取结果的部分
def update_answer(output):
    pat = re.compile(r'(?i)\bthe\s+correct\s+(?:answer|option)\s+is\s*:?[\s\n]*([A-H])\b')
    m = pat.findall(output)
    choice = m[-1] if m else None
    # print(choice)
    return choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", required=True)     # 可使用的源域文件
    ap.add_argument("--query_file", required=True)     # 目标域查询文件
    ap.add_argument("--out_file", required=True)       # 输出文件
 
    ap.add_argument("--k", type=int, required=True)     # 示例数，0表示零样本
    ap.add_argument("--same_type_ratio", type=float, default=0.75)   # 保留同类型示例的比例
    ap.add_argument("--lambda_div", type=float, default=0.7)
    ap.add_argument("--preN", type=int, default=200)     # 

    # local embedding
    ap.add_argument("--embed_model_path", required=True, help="Local path/name for bge-large-en-v1.5")
    ap.add_argument("--embed_batch", type=int, default=128)

    # local qwen for structure prediction / inference
    ap.add_argument("--qwen_model_path", default="", help="Local Qwen2.5-14B path/name")
    ap.add_argument("--do_struct_predict", action="store_true")   # 是否进行结构预测，一般只针对基于逻辑结构的检索有用
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)

    # vLLM config
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_train_per_type", type=int, default=0)

    # NEW: batch sizes
    ap.add_argument("--struct_batch_size", type=int, default=64)
    ap.add_argument("--infer_batch_size", type=int, default=16)
    
    # NEW: demo retrieval method
    ap.add_argument(
        "--demo_method",
        type=str,
        default="logical",
        choices=["logical", "embed", "bm25", "random"],
        help="How to retrieve demonstrations: logical(default), embed, bm25, random",
    )
    ap.add_argument("--random_from_topN", type=int, default=0,
                    help="If >0 and demo_method=random, sample from topN candidates (by embed sim) instead of whole train.")
    ap.add_argument("--bm25_k1", type=float, default=1.5)
    ap.add_argument("--bm25_b", type=float, default=0.75)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    llm = None
    try:
        # ---- Load train ----
        train_files = sorted(glob.glob(args.train_glob))
        if not train_files:
            raise SystemExit(f"No train files matched: {args.train_glob}")

        train_all = []
        for fp in train_files:
            train_all.extend(load_json_list(fp))
        print(f"[Info] Loaded {len(train_all)} train examples from {len(train_files)} files.")
        # filter: must have reasoning_cot + logical_structure
        # the cot answer must match the final answer
        flat_train = []
        flat_type = []
        for ex in train_all:
            rc = ex.get("reasoning_cot", "")
            ans = ex.get("answer", "")
            if not wrong_cot_filter(ans, rc):
                continue
            if not isinstance(rc, str) or not rc.strip():
                continue
            t = get_structure_type(ex)
            if t not in STRUCT_TYPES:
                continue
            flat_train.append(ex)
            flat_type.append(t)
        print(f"[Info] After filtering, {len(flat_train)} train examples remain.")
        # optional cap per type
        if args.max_train_per_type and args.max_train_per_type > 0:
            by = {t: [] for t in STRUCT_TYPES}
            for ex in flat_train:
                by[get_structure_type(ex)].append(ex)
            flat_train2, flat_type2 = [], []
            for t in STRUCT_TYPES:
                pool = by[t]
                if len(pool) > args.max_train_per_type:
                    pool = random.sample(pool, args.max_train_per_type)
                for ex in pool:
                    flat_train2.append(ex)
                    flat_type2.append(t)
            flat_train, flat_type = flat_train2, flat_type2

        train_ids = [normalize_id(ex, i) for i, ex in enumerate(flat_train)]
        train_texts = [qtext(ex) for ex in flat_train]

        # ---- Local embedding model ----
        embedder = SentenceTransformer(args.embed_model_path)

        # encode train (already batch)
        train_embs = []
        for i in tqdm(range(0, len(train_texts), args.embed_batch), desc="Embedding train", ncols=100):
            batch = train_texts[i:i + args.embed_batch]
            emb = embedder.encode(batch, batch_size=len(batch), normalize_embeddings=True, show_progress_bar=False)
            train_embs.append(np.array(emb, dtype=np.float32))
        train_emb = np.vstack(train_embs)  # [N,d]
        
        # ---- BM25 stats (optional) ----
        bm25_stats = None
        if args.demo_method == "bm25":
            train_tokens = [bm25_tokenize(t) for t in train_texts]
            bm25_stats = bm25_build_corpus_stats(train_tokens)


        # ---- Local Qwen (optional) ----
        tokenizer = None
        if (args.do_struct_predict or args.do_infer) and args.qwen_model_path:
            llm = LLM(
                model=args.qwen_model_path,
                tokenizer=args.qwen_model_path,
                max_model_len=32768,
                dtype="float16",
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                # seed=args.seed,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)

        # ---- Load queries ----
        queries = load_json_list(args.query_file)
        # queries = load_json_list(args.query_file)[:10]

        # -------- batch structure predict (if enabled) --------
        if args.do_struct_predict and args.demo_method == "logical":
            need_idx = [i for i, q in enumerate(queries) if get_structure_type(q) is None]
            if need_idx:
                if llm is None:
                    for i in need_idx:
                        queries[i]["logical_structure"] = {"structure_type": "Other", "justification": "No local qwen configured."}
                else:
                    for s in tqdm(range(0, len(need_idx), args.struct_batch_size), desc="Struct batch", ncols=100):
                        ids = need_idx[s:s + args.struct_batch_size]
                        pts = [qtext(queries[i]) for i in ids]
                        preds = predict_structure_batch_with_local_qwen(llm, tokenizer, pts, temperature=0.0, max_tokens=256)
                        for i, pred in zip(ids, preds):
                            queries[i]["logical_structure"] = pred
        # else:
             
        #     for q in queries:
        #         if get_structure_type(q) is None:
        #             q["logical_structure"] = {"structure_type": "Other", "justification": "struct predict disabled."}

        # ---- Main loop: select demos + build prompts ----
        out: List[Dict[str, Any]] = []
        pending_prompts: List[str] = []
        pending_out_indices: List[int] = []
        pending_messages: List[List[Dict[str, str]]] = []

        for qex in tqdm(queries, desc="Selecting demos", ncols=100):
            t_hat = get_structure_type(qex) or "Other"
            t2 = neighbor_type(t_hat)

            # ===== k==0 => zero-shot =====
            if args.k == 0:
                qout = dict(qex)
                qout["icl_demos"] = []
                qout["icl_prompt"] = build_zero_shot_prompt(qex, t_hat)
                out.append(qout)

                if args.do_infer:
                    pending_out_indices.append(len(out) - 1)
                    pending_prompts.append(qout["icl_prompt"])
                    pending_messages.append(build_chat_prompt(tokenizer, build_chat_messages(SYSTEM_PROMPT, qout["icl_prompt"])))
                continue

            k = max(0, args.k)

            # ====== compute query embedding once (for logical/embed/random_from_topN and MMR diversity) ======
            q_emb = embedder.encode([qtext(qex)], normalize_embeddings=True, show_progress_bar=False)
            q_emb = np.array(q_emb[0], dtype=np.float32)

            # embedding similarity for all docs (used by logical/embed/random_from_topN)
            sim_embed_all = cosine_sim(q_emb, train_emb)  # [N]

            # helper: topN indices by a score vector
            def topn_by_score(score_vec: np.ndarray, idxs: List[int], n: int) -> List[int]:
                if not idxs:
                    return []
                n = min(n, len(idxs))
                scores = score_vec[idxs]
                order = np.argsort(-scores)[:n]
                return [idxs[i] for i in order]

            # helper: MMR on a candidate list using embed sim to query + embed embeddings for diversity
            def mmr_on_embed(cand_top: List[int], kpick: int, score_vec_for_query: np.ndarray) -> List[int]:
                if kpick <= 0 or not cand_top:
                    return []
                cand_embs = train_emb[cand_top]
                sim_q = score_vec_for_query[cand_top]
                return mmr_select(cand_top, sim_q, cand_embs, kpick, lambda_div=args.lambda_div)

            # ====== choose candidates & pick according to method ======
            picked: List[int] = []

            preN = max(args.preN, k * 20)

            if args.demo_method == "random":
                if args.random_from_topN and args.random_from_topN > 0:
                    cand_top = topn_by_score(sim_embed_all, list(range(len(flat_train))), args.random_from_topN)
                    picked = random.sample(cand_top, min(k, len(cand_top))) if k > 0 else []
                else:
                    all_idx = list(range(len(flat_train)))
                    picked = random.sample(all_idx, min(k, len(all_idx))) if k > 0 else []

            elif args.demo_method == "embed":
                cand_top = topn_by_score(sim_embed_all, list(range(len(flat_train))), preN)
                # 用 MMR 做多样性（和你原来一致）
                picked = mmr_on_embed(cand_top, k, sim_embed_all)

            elif args.demo_method == "bm25":
                assert bm25_stats is not None, "bm25_stats not initialized"
                q_tokens = bm25_tokenize(qtext(qex))
                sim_bm25_all = bm25_score_query(q_tokens, bm25_stats, k1=args.bm25_k1, b=args.bm25_b)

                # 先 topN 再 MMR（多样性仍用 embedding 空间）
                cand_top = topn_by_score(sim_bm25_all, list(range(len(flat_train))), preN)
                picked = mmr_on_embed(cand_top, k, sim_bm25_all)

            else:
                # args.demo_method == "logical" (default): your original behavior
                k_same = int(round(k * args.same_type_ratio))
                k_same = min(k_same, k)
                k_nei = k - k_same

                cand_same = [i for i, t in enumerate(flat_type) if t == t_hat]
                cand_nei = [i for i, t in enumerate(flat_type) if t == t2]
                if not cand_same:
                    cand_same = list(range(len(flat_train)))
                if not cand_nei:
                    cand_nei = list(range(len(flat_train)))

                cand_same_top = topn_by_score(sim_embed_all, cand_same, preN)
                cand_nei_top = topn_by_score(sim_embed_all, cand_nei, preN)

                pick_same = mmr_on_embed(cand_same_top, k_same, sim_embed_all)
                pick_nei = mmr_on_embed(cand_nei_top, k_nei, sim_embed_all)

                # merge unique
                seen = set()
                for idx in pick_same + pick_nei:
                    did = train_ids[idx]
                    if did in seen:
                        continue
                    picked.append(idx)
                    seen.add(did)
                    if len(picked) >= k:
                        break

            # safety: ensure <=k unique
            if k > 0 and len(picked) > k:
                picked = picked[:k]

            demos = []
            for idx in picked:
                demos.append({
                    "demo_id": train_ids[idx],
                    "structure_type": flat_type[idx],
                    "demo_text": build_demo_text(flat_train[idx]),
                })

            qout = dict(qex)
            qout["icl_demos"] = demos
            qout["icl_prompt"] = build_icl_prompt(demos, qex, t_hat)
            out.append(qout)

            if args.do_infer:
                pending_out_indices.append(len(out) - 1)
                pending_prompts.append(qout["icl_prompt"])
                # 构造qwen等使用的messages格式
                pending_messages.append(build_chat_prompt(tokenizer ,build_chat_messages(SYSTEM_PROMPT, qout["icl_prompt"])))

        # ---- Batch inference ----
        if args.do_infer:
            if llm is None:
                for oi in pending_out_indices:
                    out[oi]["pred_answer"] = ""
                    out[oi]["pred_error"] = "do_infer enabled but no local qwen configured."
            else:
                for s in tqdm(range(0, len(pending_prompts), args.infer_batch_size), desc="Infer batch", ncols=100):
                    pb = pending_prompts[s:s + args.infer_batch_size]
                    # print(f"Pending batch size: {len(pb)}")
                    # print(pb[0])
                    # print("*"*20)
                    # print("\n\n")
                    # print(pending_messages[s])
                    # exit()
                    pb = pending_messages[s:s + args.infer_batch_size]
                    texts = vllm_generate_batch(llm, pb, temperature=args.temperature, max_tokens=args.max_new_tokens)
                    for j, txt in enumerate(texts):
                        oi = pending_out_indices[s + j]
                        out[oi]["pred_answer"] = (txt or "").strip()
                        answer_split = update_answer(out[oi]["pred_answer"])
                        out[oi]["predicted_answer"] = answer_split

        dump_json_list(args.out_file, out)
        print(f"[OK] Wrote: {args.out_file}")

    finally:
        # ---- clean vLLM / torch distributed ----
        try:
            if llm is not None:
                del llm
        except Exception:
            pass

        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
