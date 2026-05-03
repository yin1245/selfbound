#!/usr/bin/env python3
"""
CapBoundary-Bench Session Runner — API Agent version.

Identical to capbound_session.py but uses DashScope API for the agent
(qwen-turbo / qwen-max) instead of vLLM.

Usage:
  python capbound_session_api.py --agent_name qwen-turbo --n_sessions 10
  python capbound_session_api.py --agent_name qwen-max --n_sessions 10
"""
import json, os, sys, re, time, requests, random, argparse
from datetime import datetime
from collections import defaultdict

DASHSCOPE_KEY = "sk-REDACTED"
PROFILE_PATH = "./capbound-bench/experiments/results/model_domain_profile.json"

# === Question loading (same as capbound_session.py) ===
def load_question_pool(agent_profile, per_domain=25):
    from datasets import load_dataset
    pool = []
    all_mmlu = (agent_profile["strong_domains"] +
                agent_profile["mid_domains"] +
                agent_profile["weak_domains"])
    tier_of = {}
    for d in agent_profile["strong_domains"]: tier_of[d] = "strong"
    for d in agent_profile["mid_domains"]: tier_of[d] = "mid"
    for d in agent_profile["weak_domains"]: tier_of[d] = "weak"

    for subj in all_mmlu:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for item in list(ds)[40:40+per_domain]:
                pool.append({
                    "source": "mmlu",
                    "domain": subj,
                    "tier": tier_of[subj],
                    "question": item["question"],
                    "choices": [f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"])],
                    "gt_answer": ["A","B","C","D"][item["answer"]],
                    "format": "mcq",
                })
        except Exception as e:
            print(f"  skip {subj}: {e}")
    return pool, tier_of


PERSONA_TEMPLATES = [
    (["college_biology", "high_school_biology", "anatomy"], "academic", 0.2, 0.9),
    (["abstract_algebra", "high_school_mathematics", "formal_logic"], "formal", 0.3, 0.8),
    (["astronomy", "conceptual_physics", "college_physics"], "curious", 0.35, 0.85),
    (["business_ethics", "moral_scenarios", "global_facts"], "thoughtful", 0.4, 0.7),
    (["computer_security", "machine_learning", "electrical_engineering"], "technical", 0.25, 0.9),
    (["clinical_knowledge", "college_biology", "anatomy"], "practical", 0.3, 0.8),
    (["college_chemistry", "high_school_chemistry", "conceptual_physics"], "precise", 0.2, 0.9),
    (["econometrics", "business_ethics", "global_facts"], "analytical", 0.4, 0.75),
    (["formal_logic", "abstract_algebra", "college_physics"], "rigorous", 0.25, 0.9),
    (["high_school_biology", "clinical_knowledge", "anatomy"], "medical_student", 0.3, 0.85),
]

def build_persona(i):
    e, style, tol, expl = PERSONA_TEMPLATES[i % len(PERSONA_TEMPLATES)]
    return {"id": f"persona_{i}", "expertise": e, "style": style,
            "tolerance": tol, "explicitness": expl}


# Frontier model routing via proxy
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from frontier_api import call_frontier as _call_frontier


def call_api(model, messages, temp=0.3, max_tokens=200):
    # Claude/GPT models go through proxy
    if model.startswith("claude-") or model.startswith("gpt-"):
        # Convert messages format → single prompt + system
        system = None
        prompt_parts = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                prompt_parts.append(m["content"])
        prompt = "\n\n".join(prompt_parts)
        try:
            return _call_frontier(model, prompt, temp=temp, max_tokens=max_tokens, system=system)
        except Exception as e:
            return f"[ERROR: {e}]"

    # DashScope models (Qwen/GLM/DeepSeek)
    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {DASHSCOPE_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": temp, "max_tokens": max_tokens},
            timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


def evaluate_correctness(q, ans):
    if ans is None: return False
    if q["format"] == "mcq":
        return str(ans).strip().upper().startswith(q["gt_answer"].upper()[:1])
    return False


def no_memory_agent(agent_model, q):
    prompt = f"""Answer the question and give your confidence 0-100%.

Question: {q['question']}
Choices:
{chr(10).join(q['choices'])}

Format your response exactly as:
Answer: [A/B/C/D]
Confidence: [0-100]"""
    response = call_api(agent_model, [{"role": "user", "content": prompt}])
    m = re.search(r'Answer:\s*([A-D])', response)
    answer = m.group(1) if m else None
    mc = re.search(r'Confidence:\s*(\d+)', response)
    conf = int(mc.group(1)) / 100.0 if mc else 0.5
    return {"answer": answer, "confidence": conf, "abstain": False,
            "raw_response": response[:300]}


USER_SYSTEM = """You are role-playing a user in conversation with an AI assistant.

Your expertise: you know these academic subjects well: {expertise_str}.
In these subjects you can reliably tell if an answer is correct or wrong.
Outside these subjects, you have only vague intuition.

Your style is {style}. Your responses should be 1-2 sentences, natural, not robotic.
Do not mention being an AI, do not use bullet points, do not quote the assistant verbatim."""

def user_respond(q, agent_out, persona, is_correct):
    in_exp = q["domain"] in persona["expertise"]
    if in_exp:
        if is_correct:
            scenario = "The assistant answered CORRECTLY in your area of expertise. Respond with brief positive acknowledgment."
        else:
            scenario = f"The assistant answered WRONG in your area of expertise. The correct answer is {q['gt_answer']}. Politely correct them."
    else:
        if is_correct:
            scenario = "The assistant's answer sounds plausible but you can't verify it. Accept it, maybe ask for elaboration."
        else:
            r = random.random()
            if r < persona["tolerance"]:
                scenario = "You can't verify but something feels off. Express mild uncertainty without correcting."
            else:
                scenario = "You can't verify the answer. Accept it as if it's correct (even though it isn't)."

    system = USER_SYSTEM.format(
        expertise_str=", ".join(s.replace("_", " ") for s in persona["expertise"]),
        style=persona["style"])
    user_prompt = f"""Your previous question: {q['question']}

The assistant's answer: {agent_out['answer']}

[Stage direction - do NOT quote]
{scenario}

Now write your natural next response (1-2 sentences):"""

    return call_api(args_user_model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt}
    ], temp=0.7, max_tokens=150).strip()


def run_session(session_id, persona, pool, session_length, agent_model, seed=42):
    rng = random.Random(seed + session_id)
    strong_q = [q for q in pool if q["tier"] == "strong"]
    mid_q = [q for q in pool if q["tier"] == "mid"]
    weak_q = [q for q in pool if q["tier"] == "weak"]

    n_strong = int(session_length * 0.4)
    n_mid = int(session_length * 0.2)
    n_weak = session_length - n_strong - n_mid

    rng.shuffle(strong_q); rng.shuffle(mid_q); rng.shuffle(weak_q)
    seq = strong_q[:n_strong] + mid_q[:n_mid] + weak_q[:n_weak]
    rng.shuffle(seq)

    log = []
    for turn_idx, q in enumerate(seq, 1):
        out = no_memory_agent(agent_model, q)
        is_correct = evaluate_correctness(q, out["answer"])
        user_resp = user_respond(q, out, persona, is_correct)

        log.append({
            "turn": turn_idx, "domain": q["domain"], "tier": q["tier"],
            "question": q["question"][:200], "gt_answer": q["gt_answer"],
            "agent_answer": out["answer"], "agent_confidence": out["confidence"],
            "agent_abstain": False, "gt_correct": is_correct,
            "user_response": user_resp,
            "in_user_expertise": q["domain"] in persona["expertise"],
        })
        mark = "✓" if is_correct else "✗"
        exp = "E" if q["domain"] in persona["expertise"] else " "
        print(f"    T{turn_idx:02d}[{q['tier'][0]}{exp}] {q['domain'][:20]:<20} {mark}", flush=True)

    return {"session_id": session_id, "persona": persona, "n_turns": len(log),
            "agent_model": agent_model, "user_model": args_user_model, "log": log}


args_user_model = "qwen-max"

def main():
    global args_user_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_name", required=True, help="qwen-turbo or qwen-max")
    ap.add_argument("--n_sessions", type=int, default=10)
    ap.add_argument("--user_model", default="qwen-max", help="Model for user simulator")
    ap.add_argument("--session_length", type=int, default=50)
    ap.add_argument("--n_workers", type=int, default=1, help="Concurrent session workers")
    ap.add_argument("--output_dir", default="./capbound-bench/experiments/results")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(PROFILE_PATH) as f:
        all_profiles = json.load(f)
    profile = all_profiles[args.agent_name]

    print(f"="*70)
    print(f"CapBoundary-Bench API: {args.agent_name}")
    print(f"Sessions: {args.n_sessions}, length: {args.session_length}")
    print(f"Strong: {profile['strong_domains'][:4]}")
    print(f"Weak: {profile['weak_domains'][:4]}")
    print(f"="*70)

    args_user_model = args.user_model
    pool, _ = load_question_pool(profile, per_domain=25)
    print(f"\nQuestion pool: {len(pool)}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(i):
        persona = build_persona(i)
        print(f"  [start] Session {i+1}/{args.n_sessions}: {persona['id']}", flush=True)
        sess = run_session(i, persona, pool, args.session_length, args.agent_name)
        correct = sum(1 for t in sess["log"] if t["gt_correct"])
        print(f"  [done ] Session {i+1}/{args.n_sessions}: acc={correct}/{args.session_length} ({correct/args.session_length*100:.0f}%)", flush=True)
        return i, sess

    all_sessions = [None] * args.n_sessions
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [ex.submit(_one, i) for i in range(args.n_sessions)]
        for fut in as_completed(futs):
            try:
                i, sess = fut.result()
                all_sessions[i] = sess
            except Exception as e:
                print(f"  [ERROR] session failed: {e}", flush=True)
    all_sessions = [s for s in all_sessions if s is not None]

    # Skip the per-session prints that follow (they assumed sequential)
    if False:
        sess = None
        i = 0
        persona = None
        correct = sum(1 for t in sess["log"] if t["gt_correct"])
        pass  # handled in _one


    out_path = os.path.join(args.output_dir, f"capbound_sessions_{args.agent_name}_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "metadata": {"timestamp": ts, "agent_model": args.agent_name,
                         "user_model": args.user_model, "n_sessions": args.n_sessions,
                         "session_length": args.session_length,
                         "agent_profile": profile},
            "sessions": all_sessions,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}")

    # Summary
    total = [t for s in all_sessions for t in s["log"]]
    correct = sum(1 for t in total if t["gt_correct"])
    hr = sum(1 for t in total if t["agent_confidence"] > 0.7 and not t["gt_correct"])
    print(f"\n  Aggregate: accuracy={correct/len(total)*100:.1f}%, HR={hr/len(total)*100:.1f}%")


if __name__ == "__main__":
    main()
