#!/usr/bin/env python3
"""Run MMLU-Pro + BBH for ONE model (allows model-level parallelism)."""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/yzs/yzs/LLM心智/experiments/scripts")
from frontier_api import call_frontier
from datasets import load_dataset

PARALLEL = 10
N_PER = 15

BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "logical_deduction_five_objects", "navigate", "object_counting",
    "ruin_names", "sports_understanding",
    "tracking_shuffled_objects_three_objects", "word_sorting",
]
MMLU_PRO_CATS = ["math", "physics", "chemistry", "law", "engineering",
                 "economics", "health", "psychology", "business", "history"]


def call(model, prompt, max_tokens=512):
    return call_frontier(model, prompt, temp=0.0, max_tokens=max_tokens)


def run_mmlu_pro(model):
    print(f"=== MMLU-Pro: {model} ===", flush=True)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_cat = {}
    for item in ds:
        cat = item["category"].lower()
        if cat in MMLU_PRO_CATS:
            by_cat.setdefault(cat, []).append(item)
    sampled = {c: by_cat[c][:N_PER] for c in MMLU_PRO_CATS if c in by_cat}

    def answer_one(cat, item):
        opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(item["options"]))
        q = item["question"]
        prompt = f"Question (category: {cat}):\n{q}\n\nOptions:\n{opts}\n\nThink step by step, then end with the line: Answer: <letter>"
        for _ in range(5):
            r = call(model, prompt, max_tokens=8192)
            if r and r.strip():
                # Look for "Answer: X" first, then any A-J letter
                m = re.search(r"answer\s*:\s*\(?([A-J])", r, re.IGNORECASE)
                if not m:
                    # Fallback: last letter mention
                    letters = re.findall(r"\b([A-J])\b", r.upper())
                    if letters:
                        pred = letters[-1]
                        gt = chr(65 + item["answer_index"])
                        return cat, item["question_id"], pred, gt, pred == gt
                if m:
                    pred = m.group(1).upper()
                    gt = chr(65 + item["answer_index"])
                    return cat, item["question_id"], pred, gt, pred == gt
        return cat, item["question_id"], None, chr(65 + item["answer_index"]), False

    import time
    PARTIAL = f"/tmp/heldout_partial_{model}_mmlu_pro.json"
    results = {c: {"correct": 0, "total": 0, "failures": 0} for c in MMLU_PRO_CATS}
    t_start = time.time()
    done = 0
    total_q = sum(len(v) for v in sampled.values())
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futures = []
        for c, items in sampled.items():
            for item in items:
                futures.append(ex.submit(answer_one, c, item))
        for fut in as_completed(futures):
            c, qid, pred, gt, ok = fut.result()
            if pred is None:
                results[c]["failures"] += 1
            else:
                results[c]["total"] += 1
                if ok:
                    results[c]["correct"] += 1
            done += 1
            elapsed = time.time() - t_start
            cur_correct = sum(r["correct"] for r in results.values())
            cur_total = sum(r["total"] for r in results.values())
            cur_acc = cur_correct / max(1, cur_total)
            eta = elapsed * (total_q - done) / max(1, done)
            print(f"  [{done}/{total_q}] cat={c} pred={pred} gt={gt} ok={ok} elapsed={elapsed:.0f}s acc={cur_acc:.2f} eta={eta:.0f}s", flush=True)
            if done % 10 == 0 or done == total_q:
                json.dump({"_done": done, "_total": total_q, "_elapsed_s": elapsed,
                           "results": results}, open(PARTIAL, "w"), indent=2)
    cat_accs = {c: r["correct"]/r["total"] if r["total"]>0 else 0 for c, r in results.items()}

    cat_zs = {}
    for c in MMLU_PRO_CATS:
        prompt = f"You are about to answer 15 multiple-choice questions in the category '{c}'. Estimate your accuracy on these questions as a number between 0 and 1 (e.g. 0.7). Output only the number."
        r = call(model, prompt, max_tokens=2048)
        m = re.search(r"0?\.\d+|\b1\.0\b", r or "")
        cat_zs[c] = float(m.group()) if m else 0.5

    diffs = [abs(cat_zs[c] - cat_accs[c]) for c in MMLU_PRO_CATS]
    cbf = 1 - sum(diffs) / len(diffs)
    return {
        "model": model, "per_category_accuracy": cat_accs, "per_category_zs": cat_zs,
        "per_category_results": results, "cbf": cbf,
        "mean_acc": sum(cat_accs.values()) / len(cat_accs),
        "mean_zs": sum(cat_zs.values()) / len(cat_zs),
        "n_failures": sum(r["failures"] for r in results.values()),
    }


def run_bbh(model):
    print(f"=== BBH: {model} ===", flush=True)
    by_task = {}
    for task in BBH_TASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            by_task[task] = list(ds)[:N_PER]
        except Exception as e:
            by_task[task] = []

    def answer_one(task, item):
        prompt = f"BBH task: {task}\n\n{item['input']}\n\nThink step by step, then end with the line: Answer: <your final answer>"
        for _ in range(5):
            r = call(model, prompt, max_tokens=8192)
            if r and r.strip():
                # Try to extract from "Answer: X" pattern; fallback to full text
                m = re.search(r"answer\s*:\s*(.+?)(?:\n|$)", r, re.IGNORECASE | re.DOTALL)
                pred = m.group(1).strip()[:200] if m else r.strip()[:200]
                gt = item["target"].strip()
                ok = gt.lower() in pred.lower() or pred.lower().startswith(gt.lower())
                return task, pred, gt, ok
        return task, None, item["target"], False

    results = {t: {"correct": 0, "total": 0, "failures": 0} for t in BBH_TASKS}
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futures = []
        for t, items in by_task.items():
            for item in items:
                futures.append(ex.submit(answer_one, t, item))
        for fut in as_completed(futures):
            t, pred, gt, ok = fut.result()
            if pred is None:
                results[t]["failures"] += 1
                continue
            results[t]["total"] += 1
            if ok:
                results[t]["correct"] += 1
    task_accs = {t: r["correct"]/r["total"] if r["total"]>0 else 0 for t, r in results.items()}
    task_zs = {}
    for t in BBH_TASKS:
        prompt = f"You are about to answer 15 BIG-Bench-Hard questions on the task '{t}'. Estimate your accuracy as a number between 0 and 1 (e.g. 0.7). Output only the number."
        r = call(model, prompt, max_tokens=2048)
        m = re.search(r"0?\.\d+|\b1\.0\b", r or "")
        task_zs[t] = float(m.group()) if m else 0.5
    diffs = [abs(task_zs[t] - task_accs[t]) for t in BBH_TASKS]
    cbf = 1 - sum(diffs) / len(diffs)
    return {
        "model": model, "per_task_accuracy": task_accs, "per_task_zs": task_zs,
        "per_task_results": results, "cbf": cbf,
        "mean_acc": sum(task_accs.values()) / len(task_accs),
        "mean_zs": sum(task_zs.values()) / len(task_zs),
        "n_failures": sum(r["failures"] for r in results.values()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["mmlu_pro", "bbh", "both"])
    args = ap.parse_args()

    out = {"model": args.model}
    if args.task in ("mmlu_pro", "both"):
        out["mmlu_pro"] = run_mmlu_pro(args.model)
    if args.task in ("bbh", "both"):
        out["bbh"] = run_bbh(args.model)
    safe = args.model.replace("/", "_")
    out_path = f"/home/yzs/yzs/LLM心智/experiments/results/heldout/n3_{safe}_{args.task}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    if "mmlu_pro" in out:
        print(f"  MMLU-Pro CBF: {out['mmlu_pro']['cbf']:.3f}")
    if "bbh" in out:
        print(f"  BBH CBF: {out['bbh']['cbf']:.3f}")
