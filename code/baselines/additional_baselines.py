#!/usr/bin/env python3
"""
Additional baselines requested by reviewers:
1. Self-Consistency (SC-5): 5 samples, majority vote confidence
2. Semantic Entropy (SE-5): 5 samples, entropy over answer distribution
"""
import numpy as np, json, os, sys, re, time, requests, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from dycab_core import compute_ece, compute_brier, compute_auroc, bootstrap_ci, log_to_notion
from config import NOTION_TOKEN, NOTION_DB_ID

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_physics", "computer_security",
    "conceptual_physics", "econometrics", "electrical_engineering", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_mathematics",
    "high_school_physics", "machine_learning", "moral_scenarios"
]

SEED = 42; CAL_SPLIT = 0.6; N_SAMPLES = 5

def load_mmlu(subject, n=40):
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", subject, split="test")
        return [{"question": item["question"],
                 "choices": "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"])]),
                 "correct": ["A","B","C","D"][item["answer"]],
                 "subject": subject} for item in list(ds)[:n]]
    except Exception as e:
        print(f"  Failed: {e}"); return []

def parse_answer(text):
    for c in ['A','B','C','D']:
        if f"Answer: {c}" in text or f"answer: {c}" in text.lower():
            return c
    for c in ['A','B','C','D']:
        if text.strip().startswith(c): return c
    return None

def call_vllm(url, model, prompt, temp=0.7, max_tokens=100):
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temp, "max_tokens": max_tokens,
        }, timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return str(e)

def call_dashscope(model, prompt, api_key, temp=0.7):
    try:
        resp = requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": temp, "max_tokens": 100},
            timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return str(e)

def self_consistency(answers):
    """Self-consistency: majority vote fraction as confidence."""
    if not answers: return None, 0.25
    from collections import Counter
    counts = Counter(a for a in answers if a)
    if not counts: return None, 0.25
    best = counts.most_common(1)[0]
    return best[0], best[1] / len(answers)

def semantic_entropy(answers):
    """Semantic entropy: for MCQ, entropy of answer distribution."""
    if not answers: return 0.5
    from collections import Counter
    counts = Counter(a for a in answers if a)
    total = sum(counts.values())
    if total == 0: return 0.5
    probs = np.array([c/total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(4)  # 4 choices
    # Convert entropy to confidence: low entropy = high confidence
    conf = 1.0 - (entropy / max_entropy)
    return np.clip(conf, 0.01, 0.99)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--api_model", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--n_per_subject", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output_dir", default="../results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.api_model:
        call_fn = lambda p, t=0.7: call_dashscope(args.api_model, p, args.api_key, t)
    else:
        vllm_url = f"http://localhost:{args.port}"
        call_fn = lambda p, t=0.7: call_vllm(vllm_url, args.model_path, p, t)
    
    print(f"{'='*60}")
    print(f"Additional Baselines: {args.model_name}")
    print(f"Subjects: {args.n_subjects}, N_samples: {args.n_samples}")
    print(f"{'='*60}")
    
    subjects = SUBJECTS[:args.n_subjects]
    prompt_tpl = "Answer the following multiple choice question. Just give the letter.\n\nQuestion: {q}\n{c}\n\nAnswer:"
    
    sc_data = {"confs": [], "labels": []}
    se_data = {"confs": [], "labels": []}
    greedy_data = {"confs": [], "labels": []}  # single greedy as baseline
    raw = []
    
    for si, subj in enumerate(subjects):
        print(f"[{si+1}/{len(subjects)}] {subj}", end=" ", flush=True)
        questions = load_mmlu(subj, args.n_per_subject)
        if not questions: continue
        n_cal = int(len(questions) * CAL_SPLIT)
        test_qs = questions[n_cal:]
        
        for q in test_qs:
            prompt = prompt_tpl.format(q=q["question"], c=q["choices"])
            
            # Greedy (temp=0.1)
            greedy_text = call_fn(prompt, 0.1)
            greedy_ans = parse_answer(greedy_text)
            greedy_correct = int((greedy_ans == q["correct"]) if greedy_ans else False)
            
            # N samples (temp=0.7)
            sampled_answers = []
            for _ in range(args.n_samples):
                text = call_fn(prompt, 0.7)
                ans = parse_answer(text)
                sampled_answers.append(ans)
                if args.api_model: time.sleep(0.02)
            
            # Self-Consistency
            sc_ans, sc_conf = self_consistency(sampled_answers)
            sc_correct = int((sc_ans == q["correct"]) if sc_ans else False)
            sc_data["confs"].append(sc_conf)
            sc_data["labels"].append(sc_correct)
            
            # Semantic Entropy
            se_conf = semantic_entropy(sampled_answers)
            se_data["confs"].append(se_conf)
            se_data["labels"].append(sc_correct)  # same answer as SC
            
            # Greedy baseline
            greedy_data["confs"].append(0.95)  # typical verbalized
            greedy_data["labels"].append(greedy_correct)
            
            raw.append({
                "subject": subj, "question": q["question"][:80],
                "correct": q["correct"],
                "greedy_ans": greedy_ans, "greedy_correct": greedy_correct,
                "sc_ans": sc_ans, "sc_conf": round(sc_conf, 3), "sc_correct": sc_correct,
                "se_conf": round(se_conf, 3),
                "samples": sampled_answers,
            })
            if args.api_model: time.sleep(0.02)
        
        sc_confs_subj = sc_data["confs"][-len(test_qs):]
        print(f"SC_conf={np.mean(sc_confs_subj):.2f} SE_conf={np.mean([se_data['confs'][-len(test_qs)+i] for i in range(len(test_qs))]):.2f}")
    
    # Compute metrics
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'ECE':>8} {'ECE_std':>8} {'Brier':>8} {'AUROC':>8} {'Cost':>6}")
    print("-"*70)
    
    results = {}
    for name, data, cost in [
        ("Self-Consistency-5", sc_data, "5x"),
        ("Semantic-Entropy-5", se_data, "5x"),
    ]:
        c = np.array(data["confs"]); l = np.array(data["labels"])
        ece = compute_ece(c, l); brier = compute_brier(c, l); auroc = compute_auroc(c, l)
        _, ece_s = bootstrap_ci(c, l, compute_ece)
        results[name] = {"ECE": round(ece,4), "ECE_std": round(ece_s,4),
                        "Brier": round(brier,4), "AUROC": round(auroc,3),
                        "mean_conf": round(float(np.mean(c)),3), "cost": cost}
        print(f"{name:<25} {ece:>8.4f} {ece_s:>8.4f} {brier:>8.4f} {auroc:>8.3f} {cost:>6}")
    
    output = {
        "metadata": {"model": args.model_name, "timestamp": timestamp,
                     "n_subjects": len(subjects), "n_per_subject": args.n_per_subject,
                     "n_samples": args.n_samples},
        "results": results,
    }
    
    result_file = os.path.join(args.output_dir, f"addl_baselines_{args.model_name}_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    
    raw_file = os.path.join(args.output_dir, f"addl_raw_{args.model_name}_{timestamp}.json")
    with open(raw_file, "w") as f:
        json.dump(raw, f, indent=2)
    
    print(f"\nResults: {result_file}")
    
    for name, r in results.items():
        log_to_notion(NOTION_DB_ID, NOTION_TOKEN,
                     f"Addl-{args.model_name}-{timestamp}", args.model_name, "MMLU", name,
                     r["ECE"], r["Brier"], r["AUROC"], -1, 0, len(sc_data["confs"]), result_file)
    print("Notion done")

if __name__ == "__main__":
    main()
