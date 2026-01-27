#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---- Local Embedding ----
from sentence_transformers import SentenceTransformer

# ---- Local LLM (vLLM) ----
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


STRUCT_TYPES = ["Chain", "Y-shaped", "Block", "Other"]

SYSTEM_PROMPT = """You are an expert in logical reasoning analysis and chain-of-thought structure inspection.
Your task is NOT to solve the problem, but to predict the reasoning STRUCTURE likely required.

Choose exactly ONE:
(1) Chain: mostly linear derivation.
(2) Y-shaped: must combine (at least) two independent derived facts.
(3) Block: must check/aggregate multiple constraints/cases/entities.
(4) Other: unclear/mixed.

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
        parts.append("[Context]\n" + ctx.strip())
    parts.append("[Question]\n" + str(ex.get("question", "")).strip())
    opts = ex.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        parts.append("[Options]\n" + "\n".join([str(o) for o in opts]))
    return "\n\n".join(parts).strip()


def get_structure_type(ex: Dict[str, Any]) -> Optional[str]:
    ls = ex.get("logical_structure")
    if isinstance(ls, dict):
        t = ls.get("structure_type")
        if t in STRUCT_TYPES:
            return t
    return None


def build_demo_text(ex: Dict[str, Any]) -> str:
    t = get_structure_type(ex) or "Other"
    parts = [f"[Structure]\n{t}"]

    ctx = ex.get("context")
    if isinstance(ctx, str) and ctx.strip():
        parts.append("[Context]\n" + ctx.strip())

    parts.append("[Question]\n" + str(ex.get("question", "")).strip())

    opts = ex.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        parts.append("[Options]\n" + "\n".join([str(o) for o in opts]))

    rc = ex.get("reasoning_cot")
    if isinstance(rc, str) and rc.strip():
        parts.append("[Reasoning]\n" + rc.strip())

    ans = ex.get("answer")
    if ans is not None:
        parts.append("[Answer]\n" + str(ans).strip())

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
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
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
        [{"role": "system", "content": SYSTEM_PROMPT},
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
        demo_blocks.append(f"### Demonstration {i}\n{d['demo_text']}")

    q_parts = ["### Query", f"[Predicted Structure]\n{t_hat}"]
    ctx = query.get("context")
    if isinstance(ctx, str) and ctx.strip():
        q_parts.append("[Context]\n" + ctx.strip())
    q_parts.append("[Question]\n" + str(query.get("question", "")).strip())
    opts = query.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        q_parts.append("[Options]\n" + "\n".join([str(o) for o in opts]))
    q_parts.append("[Reasoning]\n")
    q_parts.append("[Answer]\n")

    return (
        "You will be given demonstrations that share a similar reasoning structure. "
        "Follow the structure to solve the query.\n\n"
        + "\n\n".join(demo_blocks)
        + "\n\n"
        + "\n\n".join(q_parts)
    )


def build_zero_shot_prompt(query: Dict[str, Any], t_hat: str) -> str:
    q_parts = ["### Query", f"[Predicted Structure]\n{t_hat}"]
    ctx = query.get("context")
    if isinstance(ctx, str) and ctx.strip():
        q_parts.append("[Context]\n" + ctx.strip())
    q_parts.append("[Question]\n" + str(query.get("question", "")).strip())
    opts = query.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        q_parts.append("[Options]\n" + "\n".join([str(o) for o in opts]))
    q_parts.append("[Reasoning]\n")
    q_parts.append("[Answer]\n")
    return "Solve the following problem carefully.\n\n" + "\n\n".join(q_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", required=True)
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--out_file", required=True)

    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--same_type_ratio", type=float, default=0.75)
    ap.add_argument("--lambda_div", type=float, default=0.7)
    ap.add_argument("--preN", type=int, default=200)

    # local embedding
    ap.add_argument("--embed_model_path", required=True, help="Local path/name for bge-large-en-v1.5")
    ap.add_argument("--embed_batch", type=int, default=128)

    # local qwen for structure prediction / inference
    ap.add_argument("--qwen_model_path", default="", help="Local Qwen2.5-14B path/name")
    ap.add_argument("--do_struct_predict", action="store_true")
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

        # filter: must have reasoning_cot + logical_structure
        flat_train = []
        flat_type = []
        for ex in train_all:
            rc = ex.get("reasoning_cot", "")
            if not isinstance(rc, str) or not rc.strip():
                continue
            t = get_structure_type(ex)
            if t not in STRUCT_TYPES:
                continue
            flat_train.append(ex)
            flat_type.append(t)

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

        # ---- Local Qwen (optional) ----
        tokenizer = None
        if (args.do_struct_predict or args.do_infer) and args.qwen_model_path:
            llm = LLM(
                model=args.qwen_model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)

        # ---- Load queries ----
        queries = load_json_list(args.query_file)

        # -------- batch structure predict (if enabled) --------
        if args.do_struct_predict:
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
        else:
            for q in queries:
                if get_structure_type(q) is None:
                    q["logical_structure"] = {"structure_type": "Other", "justification": "struct predict disabled."}

        # ---- Main loop: select demos + build prompts ----
        out: List[Dict[str, Any]] = []
        pending_prompts: List[str] = []
        pending_out_indices: List[int] = []

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
                continue

            k = max(0, args.k)
            k_same = int(round(k * args.same_type_ratio))
            k_same = min(k_same, k)
            k_nei = k - k_same

            cand_same = [i for i, t in enumerate(flat_type) if t == t_hat]
            cand_nei = [i for i, t in enumerate(flat_type) if t == t2]
            if not cand_same:
                cand_same = list(range(len(flat_train)))
            if not cand_nei:
                cand_nei = list(range(len(flat_train)))

            # embed query (single; cheap enough)
            q_emb = embedder.encode([qtext(qex)], normalize_embeddings=True, show_progress_bar=False)
            q_emb = np.array(q_emb[0], dtype=np.float32)

            sim_all = cosine_sim(q_emb, train_emb)  # [N]

            def topn(idxs: List[int], n: int) -> List[int]:
                n = min(n, len(idxs))
                sims = sim_all[idxs]
                order = np.argsort(-sims)[:n]
                return [idxs[i] for i in order]

            preN = max(args.preN, k * 20)
            cand_same_top = topn(cand_same, preN)
            cand_nei_top = topn(cand_nei, preN)

            def mmr_on(cand_top: List[int], kpick: int) -> List[int]:
                if kpick <= 0 or not cand_top:
                    return []
                cand_embs = train_emb[cand_top]
                sim_q = sim_all[cand_top]
                return mmr_select(cand_top, sim_q, cand_embs, kpick, lambda_div=args.lambda_div)

            pick_same = mmr_on(cand_same_top, k_same)
            pick_nei = mmr_on(cand_nei_top, k_nei)

            picked = []
            seen = set()
            for idx in pick_same + pick_nei:
                did = train_ids[idx]
                if did in seen:
                    continue
                picked.append(idx)
                seen.add(did)
                if len(picked) >= k:
                    break

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

        # ---- Batch inference ----
        if args.do_infer:
            if llm is None:
                for oi in pending_out_indices:
                    out[oi]["pred_answer"] = ""
                    out[oi]["pred_error"] = "do_infer enabled but no local qwen configured."
            else:
                for s in tqdm(range(0, len(pending_prompts), args.infer_batch_size), desc="Infer batch", ncols=100):
                    pb = pending_prompts[s:s + args.infer_batch_size]
                    texts = vllm_generate_batch(llm, pb, temperature=args.temperature, max_tokens=args.max_new_tokens)
                    for j, txt in enumerate(texts):
                        oi = pending_out_indices[s + j]
                        out[oi]["pred_answer"] = (txt or "").strip()

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
