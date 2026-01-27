#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 对当前数据集分析每个样本的 CoT 逻辑结构，批量调用 SiliconFlow Batch API 完成

import argparse
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
import urllib.request

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


# --------------------------
# Utils
# --------------------------
def safe_json_load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def safe_json_dump(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def derive_output_path(in_path: str) -> str:
    # AR-LSAT_dev_cot.json -> AR-LSAT_dev_cot_logical.json
    if in_path.endswith("_cot.json"):
        return in_path[:-9] + "_cot_logical.json"
    base, ext = os.path.splitext(in_path)
    return base + "_logical" + ext


def collect_input_files(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*_cot.json"))))
        else:
            matched = glob.glob(p)
            files.extend(sorted(matched) if matched else [p])
    # de-dup
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def build_problem_text(ex: Dict[str, Any]) -> str:
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


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
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


def chunk_list(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


# --------------------------
# SiliconFlow Batch API (OpenAI-compatible)
# --------------------------
def sf_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def sf_upload_file(client: OpenAI, jsonl_path: str) -> str:
    """
    POST /v1/files (multipart)
    purpose=batch

    SiliconFlow 返回的 file_id 通常在 response["data"]["id"]（文档示例也是这样）:contentReference[oaicite:1]{index=1}
    """
    
    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"), purpose="batch"
    )
    file_id = batch_input_file.data["id"]
    # 可以插入一个print
    print("the file id is:", file_id)

    return file_id



def sf_create_batch(
    client: OpenAI,
    input_file_id: str,
    model: str,
    completion_window: str = "24h",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    POST /v1/batches

    According to SiliconFlow doc:
    - endpoint must be "/v1/chat/completions"
    - completion_window supports 24h ~ 336h
    - recommend using extra_body={"replace":{"model": ...}} :contentReference[oaicite:1]{index=1}
    """
    
    print("Creating new batch job ...")

    batch_job_status = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": "cot logical structure labeling"},
        extra_body={"replace": {"model": model}},
    )
    print("Batch job created:", batch_job_status)
    batch_job_id = batch_job_status.id
    print("the batch job id is:", batch_job_id)

    return batch_job_id



def sf_get_batch(client, batch_id: str) -> Dict[str, Any]:
    batch_job = client.batches.retrieve(batch_id)
    
    return batch_job


def sf_download_file_content(file_id: str) -> str:
    """
    GET /v1/files/{file_id}/content
    returns raw content (jsonl)
    """
    
    r = requests.get(file_id, timeout=300)
    r.raise_for_status()
    # content is jsonl text
    return r.text


# --------------------------
# Build batch jsonl & parse output
# --------------------------
def build_batch_jsonl_lines(
    data: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    custom_id_prefix: str,
    model_in_body: str,
    skip_existing: bool,
) -> Tuple[List[str], Dict[str, int]]:
    """
    returns:
      lines: jsonl lines
      custom_id -> global index in `data`
    """
    lines: List[str] = []
    cid2idx: Dict[str, int] = {}

    for i in range(start_idx, end_idx):
        ex = data[i]
        if skip_existing and "logical_structure" in ex:
            continue

        cot = ex.get("reasoning_cot", "")
        if not isinstance(cot, str) or not cot.strip():
            # still mark locally without requesting
            ex["logical_structure"] = {"structure_type": "Other", "justification": "Empty or missing reasoning_cot."}
            continue

        problem_text = build_problem_text(ex)
        user_prompt = USER_PROMPT_TEMPLATE.format(problem=problem_text, cot=cot)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # custom_id must be unique per input file
        custom_id = f"{custom_id_prefix}-{i}"
        cid2idx[custom_id] = i

        # SiliconFlow doc要求：每行一个完整 API 请求 body，并包含 custom_id/method/url/body.messages :contentReference[oaicite:4]{index=4}
        line_obj = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_in_body,   # 允许占位；真正使用以 extra_body.replace.model 为准（文档说明）:contentReference[oaicite:5]{index=5}
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": False,
            },
        }
        lines.append(json.dumps(line_obj, ensure_ascii=False))

    return lines, cid2idx


def parse_batch_output_jsonl(output_text: str) -> Dict[str, Dict[str, Any]]:
    """
    output file: one json object per line.
    We'll map custom_id -> {ok, content or error}
    Typical OpenAI batch output format: includes custom_id + response or error.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        cid = obj.get("custom_id")
        if not cid:
            continue

        # success case
        if obj.get("response") and isinstance(obj["response"], dict):
            # response.body.choices[0].message.content
            body = obj["response"].get("body", {})
            try:
                content = body["choices"][0]["message"]["content"]
            except Exception:
                content = None
            out[cid] = {"ok": True, "content": content, "raw": obj}
        else:
            # error case
            out[cid] = {"ok": False, "error": obj.get("error"), "raw": obj}
    return out


def classify_with_batch_and_apply(
    base_url_v1: str,
    api_key: str,
    model: str,
    data: List[Dict[str, Any]],
    in_file_tag: str,
    batch_size: int = 4500,
    completion_window: str = "24h",
    poll_interval_sec: int = 30,
    skip_existing: bool = True,
    workdir: str = ".",
) -> None:
    """
    Split into multiple batch jobs if needed.
    """
    os.makedirs(workdir, exist_ok=True)

    n = len(data)
    ranges = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    for b, (s, e) in enumerate(ranges):
        prefix = f"{in_file_tag}-b{b}"
        jsonl_path = os.path.join(workdir, f"batch_input_{prefix}.jsonl")

        lines, cid2idx = build_batch_jsonl_lines(
            data=data,
            start_idx=s,
            end_idx=e,
            custom_id_prefix=prefix,
            model_in_body=model,  # placeholder; real chosen via extra_body.replace.model
            skip_existing=skip_existing,
        )

        if not lines:
            continue

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
        
        # 使用openai 的client
        client = OpenAI(api_key=api_key, base_url=base_url_v1)

        # 1) upload file (purpose=batch) :contentReference[oaicite:6]{index=6}
        input_file_id = sf_upload_file(client, jsonl_path)
        print(f"[BATCH] uploaded input_file_id={input_file_id} for {prefix}")

        # 2) create batch :contentReference[oaicite:7]{index=7}
        batch_id = sf_create_batch(
            client=client,
            input_file_id=input_file_id,
            model=model,
            completion_window=completion_window,
            metadata={"description": f"cot logical structure labeling: {prefix}"},
        )
        print(f"[BATCH] created batch_id={batch_id} for {prefix}")

        # 3) poll status until completed :contentReference[oaicite:8]{index=8}
        while True:
            batch_job = sf_get_batch(client, batch_id)
            job_status = batch_job.status

            print(f"[BATCH] {batch_id} status={job_status}")
            if job_status in {"completed", "failed", "expired", "cancelled"}:
                break
            time.sleep(poll_interval_sec)

        # 4) download outputs
        batch_job = sf_get_batch(client, batch_id)
        out_file_id = batch_job.output_file_id    # 这个fileid其实是去下载文件的url
        err_file_id = batch_job.error_file_id

        out_map: Dict[str, Dict[str, Any]] = {}
        err_map: Dict[str, Dict[str, Any]] = {}

        if out_file_id:
            out_text = sf_download_file_content(out_file_id)
            out_map = parse_batch_output_jsonl(out_text)

        if err_file_id:
            err_text = sf_download_file_content(err_file_id)
            err_map = parse_batch_output_jsonl(err_text)

        merged = {**out_map, **err_map}

        # 5) apply back to data
        for cid, idx in cid2idx.items():
            rec = merged.get(cid)
            if not rec:
                data[idx]["logical_structure"] = {
                    "structure_type": "Other",
                    "justification": "Missing batch output for this custom_id.",
                }
                continue

            if rec.get("ok") is True:
                obj = extract_json_from_text(rec.get("content") or "")
                if not obj:
                    data[idx]["logical_structure"] = {
                        "structure_type": "Other",
                        "justification": "Model output not valid JSON.",
                    }
                    continue

                st = obj.get("structure_type", "")
                if st not in {"Chain", "Y-shaped", "Block", "Other"}:
                    st = "Other"

                data[idx]["logical_structure"] = {
                    "structure_type": st,
                    "justification": (obj.get("justification") or "").strip(),
                }
            else:
                data[idx]["logical_structure"] = {
                    "structure_type": "Other",
                    "justification": f"Batch error: {rec.get('error')}",
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Files/dirs/globs, e.g. data/*_cot.json")
    ap.add_argument("--output_dir", default="", help="Write outputs into this dir with derived names.")
    ap.add_argument("--api_key", default=os.environ.get("SILICONFLOW_API_KEY", ""))
    ap.add_argument("--base_url_v1", default=os.environ.get("SILICONFLOW_BASE_URL_V1", "https://api.siliconflow.cn/v1"))
    ap.add_argument("--model", default=os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3"))
    ap.add_argument("--completion_window", default=os.environ.get("SILICONFLOW_COMPLETION_WINDOW", "24h"))
    ap.add_argument("--batch_size", type=int, default=4500, help="<= 5000 lines per batch input file (recommended).")
    ap.add_argument("--poll_interval", type=int, default=30)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--workdir", default="./batch_tmp", help="Where to write intermediate jsonl files.")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set --api_key or env SILICONFLOW_API_KEY.")

    in_files = collect_input_files(args.inputs)
    if not in_files:
        raise SystemExit("No input files found.")

    for in_path in in_files:
        data = safe_json_load(in_path)

        out_path = derive_output_path(in_path)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, os.path.basename(out_path))

        tag = os.path.splitext(os.path.basename(in_path))[0]

        classify_with_batch_and_apply(
            base_url_v1=args.base_url_v1,
            api_key=args.api_key,
            model=args.model,
            data=data,
            in_file_tag=tag,
            batch_size=args.batch_size,
            completion_window=args.completion_window,
            poll_interval_sec=args.poll_interval,
            skip_existing=args.skip_existing,
            workdir=args.workdir,
        )

        safe_json_dump(out_path, data)
        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
