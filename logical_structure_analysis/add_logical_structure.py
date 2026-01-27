#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 对当前数据集分析每个样本的 CoT 逻辑结构，调用 DeepSeek API 完成
import argparse
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


SYSTEM_PROMPT = """You are an expert in logical reasoning analysis and chain-of-thought structure inspection.

Your task is NOT to solve the problem, but to analyze the STRUCTURE of the given chain-of-thought (CoT).

You must classify the CoT into one of four predefined structural types based on reasoning dependencies and information flow.
Focus on how conclusions are derived, not on correctness, length, or wording.

Internally identify reasoning nodes and their dependency relations before classification, but only output the final JSON.
Any output that is not valid JSON is considered incorrect.
"""


USER_PROMPT_TEMPLATE = """I will give you:
1. A logical reasoning problem
2. A chain-of-thought (CoT) produced by a language model

Your task is to classify the STRUCTURE of the CoT into exactly ONE of the following four types.

====================
CoT STRUCTURE TYPES
====================

(1) Chain
- Single linear reasoning path.

(2) Y-shaped
- Two or more independent reasoning chains merged at the conclusion.

(3) Block
- One or more nodes generate multiple parallel branches, aggregated at the end.

(4) Other
- Does not clearly fit the above structures.

====================
INPUT
====================

[Problem]
{problem}

[Chain-of-Thought]
{cot}

====================
OUTPUT FORMAT (STRICT JSON)
====================

{{
  "structure_type": "Chain | Y-shaped | Block | Other",
  "justification": "A concise explanation referring to reasoning dependencies."
}}
"""


def _safe_json_load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def _safe_json_dump(path: str, data: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _derive_output_path(in_path: str) -> str:
    # requirement: name is adding "logical" after "_cot"
    # e.g., AR-LSAT_dev_cot.json -> AR-LSAT_dev_cot_logical.json
    if in_path.endswith("_cot.json"):
        return in_path[:-9] + "_cot_logical.json"
    # fallback: append
    base, ext = os.path.splitext(in_path)
    return base + "_logical" + ext


def _build_problem_text(ex: Dict[str, Any]) -> str:
    # Keep it simple & robust across datasets
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    opts = ex.get("options", None)

    s = ""
    if ctx:
        s += f"Context:\n{ctx}\n\n"
    if q:
        s += f"Question:\n{q}\n\n"
    if isinstance(opts, list) and len(opts) > 0:
        s += "Options:\n" + "\n".join([str(o) for o in opts]) + "\n"
    return s.strip()


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    DeepSeek sometimes returns extra text; try to extract the first JSON object.
    """
    text = text.strip()
    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    block = m.group(0)
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def deepseek_chat_completion(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    base_url: str = "https://api.deepseek.com",
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"]


def classify_one_cot(
    api_key: str,
    model: str,
    problem_text: str,
    cot_text: str,
    base_url: str,
    max_retries: int = 5,
    sleep_base: float = 1.5,
) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(problem=problem_text, cot=cot_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            content = deepseek_chat_completion(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
                base_url=base_url,
            )
            obj = _extract_json_from_text(content)
            if obj is None:
                raise ValueError(f"Model output is not valid JSON: {content[:200]}")

            st = obj.get("structure_type", "")
            if st not in {"Chain", "Y-shaped", "Block", "Other"}:
                raise ValueError(f"Invalid structure_type: {st}")

            # keep only needed fields (you can simplify later)
            return {
                "structure_type": st,
                "justification": obj.get("justification", "").strip(),
                "raw_model_output": None,  # set to content if you want debug
            }

        except Exception as e:
            last_err = str(e)
            # exponential backoff
            time.sleep(sleep_base * (2 ** (attempt - 1)))

    return {
        "structure_type": "Other",
        "justification": f"Failed after retries. Last error: {last_err}",
        "raw_model_output": None,
    }


def collect_input_files(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*_cot.json"))))
        else:
            # glob support
            matched = glob.glob(p)
            if matched:
                files.extend(sorted(matched))
            else:
                files.append(p)
    # de-dup keep order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files/dirs/globs, e.g. data/ or data/*_cot.json",
    )
    ap.add_argument(
        "--output_dir",
        default="",
        help="If set, write outputs into this directory with derived names.",
    )
    ap.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner"))
    ap.add_argument("--base_url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--skip_existing", action="store_true", help="Skip items that already have logical_structure.")
    ap.add_argument("--save_every", type=int, default=50, help="Save progress every N items.")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set --api_key or env DEEPSEEK_API_KEY.")

    in_files = collect_input_files(args.inputs)
    if not in_files:
        raise SystemExit("No input files found.")

    for in_path in in_files:
        data = _safe_json_load(in_path)

        out_name = os.path.basename(_derive_output_path(os.path.basename(in_path)))
        out_path = _derive_output_path(in_path)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, os.path.basename(out_path))

        # Process
        changed = False
        pbar = tqdm(range(len(data)), desc=f"Processing {os.path.basename(in_path)}", ncols=100)
        for i in pbar:
            ex = data[i]
            if args.skip_existing and "logical_structure" in ex:
                continue

            cot = ex.get("reasoning_cot", "")
            if not isinstance(cot, str) or not cot.strip():
                ex["logical_structure"] = {
                    "structure_type": "Other",
                    "justification": "Empty or missing reasoning_cot.",
                }
                changed = True
                continue

            problem_text = _build_problem_text(ex)
            result = classify_one_cot(
                api_key=args.api_key,
                model=args.model,
                problem_text=problem_text,
                cot_text=cot,
                base_url=args.base_url,
            )
            # store
            ex["logical_structure"] = {
                "structure_type": result["structure_type"],
                "justification": result["justification"],
            }
            changed = True

            if (i + 1) % args.save_every == 0:
                _safe_json_dump(out_path, data)

        # Final save
        if changed:
            _safe_json_dump(out_path, data)
        else:
            # still write output for consistency
            _safe_json_dump(out_path, data)

        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
