#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


STRUCT_TYPES = ["Chain", "Y-shaped", "Block", "Other"]


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data)}")
    return data


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s\.\-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_choice(s: str) -> Optional[str]:
    s = (s or "").lower()

    # A/B/C/D (独立 token)
    m = re.search(r"\b([abcd])\b", s)
    if m:
        return m.group(1).upper()

    # True/False/Unknown
    if "unknown" in s:
        return "Unknown"
    if "true" in s:
        return "True"
    if "false" in s:
        return "False"
    return None


def extract_number(s: str) -> Optional[float]:
    s = s or ""
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[-1])  # 常见：最后一个数字是最终答案
    except Exception:
        return None


def match_answer(pred: str, gold: str) -> Tuple[bool, str]:
    """Return (correct?, match_type). match_type used for debugging/stats."""
    pred = pred or ""
    gold = gold or ""
    if not pred.strip() or not gold.strip():
        return False, "empty"

    p_choice = extract_choice(pred)
    g_choice = extract_choice(gold)
    if p_choice and g_choice:
        return (p_choice == g_choice), "choice"

    p_num = extract_number(pred)
    g_num = extract_number(gold)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-6, "numeric"

    pn = normalize_text(pred)
    gn = normalize_text(gold)
    if gn in pn or pn in gn:
        return True, "string"

    return False, "string"


def get_structure_type(ex: Dict[str, Any]) -> str:
    ls = ex.get("logical_structure")
    if isinstance(ls, dict):
        t = ls.get("structure_type")
        if t in STRUCT_TYPES:
            return t
    return "Other"


def parse_meta_from_path(path: str) -> Dict[str, Any]:
    """
    Expect path like:
    logical_results/<TGT>/k<K>/<SRC>__<TGT>_<SPLIT>_k<K>_icl.json
    """
    base = os.path.basename(path)
    d = os.path.dirname(path)
    m_kdir = re.search(r"/k(\d+)(/|$)", d.replace("\\", "/"))
    k = int(m_kdir.group(1)) if m_kdir else None

    m = re.match(r"(.+?)__(.+?)_(.+?)_k(\d+)_icl\.json$", base)
    if m:
        src, tgt, split, k2 = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if k is None:
            k = k2
        return {"SRC": src, "TGT": tgt, "SPLIT": split, "K": k}
    # fallback：旧命名
    m2 = re.match(r"(.+?)__(.+?)_(.+?)_icl\.json$", base)
    if m2:
        src, tgt, split = m2.group(1), m2.group(2), m2.group(3)
        return {"SRC": src, "TGT": tgt, "SPLIT": split, "K": k if k is not None else -1}

    return {"SRC": "UNKNOWN", "TGT": "UNKNOWN", "SPLIT": "UNKNOWN", "K": k if k is not None else -1}


def eval_one_file(path: str, pred_key: str, gold_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - long_df: per structure + overall rows
      - wide_df: single-row wide format
    """
    data = load_json_list(path)

    total = 0
    correct = 0
    invalid = 0

    by_struct = {t: {"total": 0, "correct": 0} for t in STRUCT_TYPES}

    for ex in data:
        pred = str(ex.get(pred_key, "") or "")
        gold = str(ex.get(gold_key, "") or "")
        st = get_structure_type(ex)

        ok, mtype = match_answer(pred, gold)

        total += 1
        if ok:
            correct += 1
        if mtype == "empty":
            invalid += 1

        if st not in by_struct:
            by_struct[st] = {"total": 0, "correct": 0}
        by_struct[st]["total"] += 1
        if ok:
            by_struct[st]["correct"] += 1

    meta = parse_meta_from_path(path)
    meta["FILE"] = path

    rows = []

    # overall row
    rows.append({
        **meta,
        "STRUCTURE": "ALL",
        "N": total,
        "CORRECT": correct,
        "ACC": (correct / total) if total else 0.0,
        "INVALID": invalid,
    })

    # per structure rows
    for t in STRUCT_TYPES:
        n = by_struct[t]["total"]
        c = by_struct[t]["correct"]
        rows.append({
            **meta,
            "STRUCTURE": t,
            "N": n,
            "CORRECT": c,
            "ACC": (c / n) if n else 0.0,
            "INVALID": None,
        })

    long_df = pd.DataFrame(rows)

    # wide one-row
    wide = {
        **meta,
        "N_ALL": total,
        "ACC_ALL": (correct / total) if total else 0.0,
        "INVALID": invalid,
    }
    for t in STRUCT_TYPES:
        n = by_struct[t]["total"]
        c = by_struct[t]["correct"]
        wide[f"N_{t}"] = n
        wide[f"ACC_{t}"] = (c / n) if n else 0.0

    wide_df = pd.DataFrame([wide])
    return long_df, wide_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logical_results", help="results root dir")
    ap.add_argument("--pattern", default="**/*_icl.json", help="glob pattern under root")
    ap.add_argument("--pred_key", default="pred_answer")
    ap.add_argument("--gold_key", default="answer")
    ap.add_argument("--out_prefix", default="summary", help="output prefix, e.g., summary -> summary_long.csv")
    args = ap.parse_args()

    search_pat = os.path.join(args.root, args.pattern)
    files = sorted(glob.glob(search_pat, recursive=True))
    files = [f for f in files if f.endswith(".json")]

    if not files:
        raise SystemExit(f"No result files found with: {search_pat}")

    all_long = []
    all_wide = []
    for fp in files:
        try:
            long_df, wide_df = eval_one_file(fp, args.pred_key, args.gold_key)
            all_long.append(long_df)
            all_wide.append(wide_df)
        except Exception as e:
            print(f"[WARN] Failed to eval {fp}: {e}")

    long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    wide_df = pd.concat(all_wide, ignore_index=True) if all_wide else pd.DataFrame()

    # sort for readability
    if not long_df.empty:
        long_df = long_df.sort_values(["TGT", "SPLIT", "K", "SRC", "STRUCTURE"]).reset_index(drop=True)
    if not wide_df.empty:
        wide_df = wide_df.sort_values(["TGT", "SPLIT", "K", "SRC"]).reset_index(drop=True)

    out_long = f"{args.out_prefix}_long.csv"
    out_wide = f"{args.out_prefix}_wide.csv"
    out_xlsx = f"{args.out_prefix}.xlsx"

    long_df.to_csv(out_long, index=False, encoding="utf-8")
    wide_df.to_csv(out_wide, index=False, encoding="utf-8")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        long_df.to_excel(w, index=False, sheet_name="long")
        wide_df.to_excel(w, index=False, sheet_name="wide")

    print(f"[OK] Wrote: {out_long}")
    print(f"[OK] Wrote: {out_wide}")
    print(f"[OK] Wrote: {out_xlsx}")


if __name__ == "__main__":
    main()
