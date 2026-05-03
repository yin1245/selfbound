#!/usr/bin/env python3
"""
Feedback Classifier for CapBoundary-Bench.

Given user's natural-language response, decode whether it implies the agent's
previous answer was correct, wrong, or uncertain.

Three modes:
  1. RULE: regex-based (fast, deterministic, low recall)
  2. LLM: qwen-max judges (accurate, slow, API cost)
  3. HYBRID: rules first, LLM for ambiguous cases

This classifier is a REQUIRED COMPONENT of the User-Feedback PDC baseline (B3)
and is evaluated on its own accuracy against ground truth.
"""
import re, json, os, sys, requests

# === Rule patterns ===
# NEGATIVE: user explicitly says the answer was wrong
NEGATIVE_PATTERNS = [
    r"\b(that'?s?|it'?s?) (not|incorrect|wrong)\b",
    r"\bactually,?\s+(it|that|the answer)\b",
    r"\b(afraid|sorry) (that'?s?|it'?s?) not\b",
    r"\b(no,?\s*)(?:that'?s?)? (not|wrong|incorrect)\b",
    r"\bi (don'?t (think|believe)|disagree)\b",
    r"\byou (got it wrong|are wrong|are incorrect|are mistaken)\b",
    r"\bthat'?s? not (quite|exactly|right|correct|accurate|true)\b",
    r"\b(but|however),?\s+the (correct|right|actual) answer\b",
    r"\bthe (correct|right|actual) answer (is|should be|would be)\b",
    r"\bincorrect\b",
    r"\byou'?re? (wrong|incorrect|mistaken)\b",
    r"\byou missed\b",
    r"\btry again\b",
    r"\bis incorrect\b",
]

# POSITIVE: user explicitly confirms
POSITIVE_PATTERNS = [
    r"\bthat'?s? (correct|right|exactly right)\b",
    r"\byou'?re? (correct|right)\b",
    r"\b(exactly|precisely|correct)[!.]",
    r"\b(yes,?\s*)(?:that'?s?)? (correct|right)\b",
    r"\bwell said\b",
    r"\bgood answer\b",
]

# UNCERTAIN: user expresses doubt without explicitly correcting
UNCERTAIN_PATTERNS = [
    r"\bare you sure\b",
    r"\bhmm,? (that|this)\b",
    r"\b(doesn'?t|does not) sound (quite )?right\b",
    r"\bi'?m not sure\b",
    r"\bcould you (double[- ]check|verify|confirm)\b",
    r"\bthat seems off\b",
    r"\bi thought it (was|might be)\b",
]

NEG_RE = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]
POS_RE = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]
UNC_RE = [re.compile(p, re.IGNORECASE) for p in UNCERTAIN_PATTERNS]


def rule_classify(user_text):
    """Returns ('negative'|'positive'|'uncertain'|'neutral', matched_pattern)."""
    if not user_text or not user_text.strip():
        return "neutral", None
    text = user_text.strip()

    neg_hits = [p.pattern for p in NEG_RE if p.search(text)]
    pos_hits = [p.pattern for p in POS_RE if p.search(text)]
    unc_hits = [p.pattern for p in UNC_RE if p.search(text)]

    # Priority: negative > uncertain > positive > neutral
    if neg_hits:
        return "negative", neg_hits[0]
    if unc_hits:
        return "uncertain", unc_hits[0]
    if pos_hits:
        return "positive", pos_hits[0]
    return "neutral", None


# === LLM-based classifier (more accurate for ambiguous cases) ===
DASHSCOPE_KEY = "sk-REDACTED"

JUDGE_PROMPT = """A user is responding to an AI assistant's answer. Classify whether the user's response implies the assistant's answer was CORRECT, WRONG, or UNCLEAR.

User's response: "{user_text}"

Rules:
- CORRECT: user explicitly confirms (e.g., "That's right", "Exactly")
- WRONG: user explicitly corrects or contradicts (e.g., "Actually...", "I'm afraid that's not right")
- UNCLEAR: user asks follow-up, thanks without confirming, expresses vague uncertainty, or moves on to unrelated topic

Reply with exactly one word: CORRECT, WRONG, or UNCLEAR."""

def llm_classify(user_text):
    """Use qwen-max to classify the feedback. Returns ('negative'|'positive'|'neutral', None)."""
    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {DASHSCOPE_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": "qwen-max",
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(user_text=user_text)}],
                "temperature": 0.0,
                "max_tokens": 10,
            }, timeout=20)
        reply = resp.json()["choices"][0]["message"]["content"].strip().upper()
        if "CORRECT" in reply and "IN" not in reply:  # avoid "INCORRECT"
            return "positive", "llm"
        if "WRONG" in reply or "INCORRECT" in reply:
            return "negative", "llm"
        return "neutral", "llm"  # "UNCLEAR"
    except Exception as e:
        return "neutral", f"error:{e}"


def hybrid_classify(user_text):
    """Try rules first; fall back to LLM if neutral or ambiguous."""
    label, pat = rule_classify(user_text)
    if label in ("negative", "positive"):
        return label, pat  # rules are confident
    # Rules couldn't decide → ask LLM
    llm_label, llm_note = llm_classify(user_text)
    return llm_label, f"rule:{pat}|llm:{llm_note}"


# === Evaluation: classifier fidelity vs GT ===
def evaluate_on_session_file(path, mode="hybrid"):
    """Load session data, run classifier, compare to gt_correct."""
    with open(path) as f:
        d = json.load(f)

    classifier_fn = {"rule": rule_classify, "llm": llm_classify, "hybrid": hybrid_classify}[mode]

    # Metrics
    from collections import Counter
    confusion = Counter()  # (label, gt_label) -> count
    per_scenario = {
        "in_exp": Counter(),
        "out_exp": Counter(),
    }

    n = 0
    for sess in d["sessions"]:
        for t in sess["log"]:
            n += 1
            gt = "positive" if t["gt_correct"] else "negative"  # binary
            pred, _ = classifier_fn(t["user_response"])

            # Map "uncertain" to "negative" for binary eval (uncertain = likely wrong)
            pred_binary = pred
            if pred == "uncertain":
                pred_binary = "negative"
            if pred == "neutral":
                pred_binary = "unknown"

            confusion[(pred_binary, gt)] += 1
            scenario = "in_exp" if t["in_user_expertise"] else "out_exp"
            per_scenario[scenario][(pred_binary, gt)] += 1

    # Compute precision/recall/accuracy
    def metrics(cnt):
        tp = cnt.get(("positive", "positive"), 0)
        tn = cnt.get(("negative", "negative"), 0)
        fp = cnt.get(("positive", "negative"), 0)
        fn = cnt.get(("negative", "positive"), 0)
        unk = cnt.get(("unknown", "positive"), 0) + cnt.get(("unknown", "negative"), 0)
        total = tp + tn + fp + fn + unk
        return {
            "total": total,
            "accuracy_known": (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0,
            "coverage": (total-unk)/total if total>0 else 0,
            "precision_correct": tp/(tp+fp) if (tp+fp)>0 else 0,
            "recall_correct": tp/(tp+fn) if (tp+fn)>0 else 0,
            "precision_wrong": tn/(tn+fn) if (tn+fn)>0 else 0,
            "recall_wrong": tn/(tn+fp) if (tn+fp)>0 else 0,
        }

    print(f"\n=== Classifier Evaluation ({mode}) ===")
    print(f"Total turns: {n}\n")

    overall = metrics(confusion)
    print(f"OVERALL:")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.3f}")
        else:
            print(f"  {k:<20}: {v}")

    for scenario in ["in_exp", "out_exp"]:
        print(f"\n{scenario.upper()}:")
        m = metrics(per_scenario[scenario])
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<20}: {v:.3f}")
            else:
                print(f"  {k:<20}: {v}")

    print(f"\nConfusion matrix (pred, gt):")
    for k in sorted(confusion.keys()):
        print(f"  {k}: {confusion[k]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_file", required=True)
    ap.add_argument("--mode", default="rule", choices=["rule", "llm", "hybrid"])
    args = ap.parse_args()
    evaluate_on_session_file(args.session_file, mode=args.mode)
