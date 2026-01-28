# aggregate_runs.py

# 该代码用于统计多次运行的结果，计算平均值、标准差、最小值和最大值。
# 对应的是run_batch_n.sh中N_RUNS设置的多次运行结果。
import re, glob, os
import numpy as np

PAT = re.compile(r"EM:\s*([0-9]*\.?[0-9]+)")  # 你改成你日志里真实的字段

def extract_acc(log_path):
    txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    m = PAT.search(txt)
    return float(m.group(1)) if m else None

source_domians = ["ProntoQA", "AR-LSAT", "ProofWriter", "FOLIO", "gsm8k"]

for source in source_domians:
    print(f"Source Domain: {source}")
    root = f"logs/qwen14/RAG_embedding/{source}__LogicalDeduction"
    all_logs = sorted(glob.glob(os.path.join(root, "run*/shot*.log")))
    # print(all_logs)
    bucket = {}  # shot -> list of acc
    for p in all_logs:
        shot = re.search(r"shot(\d+)\.log$", p).group(1)
        acc = extract_acc(p)
        if acc is None:
            continue
        bucket.setdefault(int(shot), []).append(acc)

    for shot in sorted(bucket):
        arr = np.array(bucket[shot], dtype=float)
        mean, std = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
        min = arr.min()
        max = arr.max()
        print(f"shot={shot:>2}  n={len(arr):>2}  mean={mean:.4f}  std={std:.4f} min={min:.4f} max={max:.4f}")
