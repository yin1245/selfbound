#!/usr/bin/env python3
"""
CapBoundary-Bench Baselines v2: 6 full baselines on pilot session data.

Baselines:
  B1: No-Memory (raw verbalized confidence)
  B2: Oracle PDC (uses ground truth — reference ceiling)
  B3: User-Feedback PDC (rule-based classifier decodes feedback)
  B4a: Episodic Memory (Oracle) — kNN over all (q, gt_correct)
  B4b: Episodic Memory (Feedback) — kNN over (q, rule-decoded correct)
  B5: MMC-style Capability Memory — domain-level memory with similarity judgment
  B6: LLM In-Context Self-Assessment — prompts the agent to reason over history

All baselines run as offline replays of a session dump.
"""
import json, os, sys, re, math, argparse, requests, time
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feedback_classifier import rule_classify

# ========= Metrics =========
def compute_ece(confs, labs, n_bins=10):
    if not confs: return 0.0
    c, l = np.array(confs), np.array(labs)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (c >= bins[i]) & (c < bins[i+1])
        if m.sum() == 0: continue
        e += m.sum() * abs(l[m].mean() - c[m].mean())
    return e / len(c)

def compute_brier(c, l):
    return float(np.mean((np.array(c) - np.array(l))**2)) if c else 0.0

def compute_auroc(c, l):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(l, c) if len(set(l)) > 1 else 0.5
    except:
        return 0.5


# ========= Base class =========
SELF_ASSESS_PROMPT = """You are reflecting on your own performance in the domain: "{domain}".
Below is a summary of your interaction history in this domain:

{context}

Based on this information, estimate what percentage of questions you answered correctly in this domain.
Reply with ONLY a single integer between 0 and 100."""


class Baseline:
    name = "base"
    _agent_fn = None
    _sa_cache = {}  # shared prompt->result cache across all baselines

    def reset(self):
        pass

    def predict_confidence(self, turn):
        raise NotImplementedError

    def self_assess(self, domain):
        return 0.5

    def observe(self, turn):
        pass

    def set_agent_fn(self, fn):
        self._agent_fn = fn

    @classmethod
    def clear_cache(cls):
        cls._sa_cache.clear()

    _warn_count = 0

    def _llm_self_assess(self, domain, context_text):
        if self._agent_fn is None:
            return 0.5
        prompt = SELF_ASSESS_PROMPT.format(domain=domain.replace("_", " "),
                                           context=context_text)
        if prompt in Baseline._sa_cache:
            return Baseline._sa_cache[prompt]
        try:
            resp = self._agent_fn(prompt)
            m = re.search(r'(\d+)', resp)
            if m:
                val = float(np.clip(int(m.group(1)) / 100.0, 0.01, 0.99))
                Baseline._sa_cache[prompt] = val
                return val
            if Baseline._warn_count < 3:
                print(f"    [WARN] self_assess({domain}): no number in response: {resp[:80]}")
                Baseline._warn_count += 1
        except Exception as e:
            if Baseline._warn_count < 3:
                print(f"    [WARN] self_assess({domain}) API failed: {e}")
                Baseline._warn_count += 1
        return 0.5


# ========= B1: No-Memory =========
class NoMemoryBaseline(Baseline):
    name = "NoMemory"

    def predict_confidence(self, turn):
        return turn["agent_confidence"], False

    def self_assess(self, domain):
        return 0.5


# ========= B2: Oracle PDC =========
class OraclePDC(Baseline):
    name = "OraclePDC"

    def reset(self):
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    def predict_confidence(self, turn):
        d = turn["domain"]
        if self.total[d] == 0:
            return turn["agent_confidence"], False
        return self.correct[d] / self.total[d], False

    def self_assess(self, d):
        return self.correct[d] / self.total[d] if self.total[d] > 0 else 0.5

    def observe(self, turn):
        d = turn["domain"]
        self.total[d] += 1
        if turn["gt_correct"]:
            self.correct[d] += 1


# ========= B3: User-Feedback PDC =========
class UserFeedbackPDC(Baseline):
    name = "UF-PDC"

    def reset(self):
        self.correct = defaultdict(float)
        self.total = defaultdict(float)

    def predict_confidence(self, turn):
        d = turn["domain"]
        if self.total[d] == 0:
            return turn["agent_confidence"], False
        return self.correct[d] / self.total[d], False

    def self_assess(self, d):
        # FIXED v4: return UF-PDC's actual decoded-label per-domain accuracy,
        # not an LLM verbal re-estimate.
        if self.total[d] == 0:
            return 0.5
        return float(self.correct[d] / self.total[d])

    def observe(self, turn):
        label, _ = rule_classify(turn["user_response"])
        d = turn["domain"]
        if label == "positive":
            self.correct[d] += 1
            self.total[d] += 1
        elif label in ("negative", "uncertain"):
            self.total[d] += 1


# ========= B4a/b: Episodic Memory =========
class EpisodicMemory(Baseline):
    def __init__(self, encoder, k=5, use_gt=False):
        self.encoder = encoder
        self.k = k
        self.use_gt = use_gt
        self.name = f"Episodic-{'Oracle' if use_gt else 'Feedback'}"
        self.reset()

    def reset(self):
        self.embeddings = []
        self.labels = []  # 0/1
        self.domains = []
        self.questions = []
        self.feedback_labels = []  # "positive"/"negative"/"uncertain"

    def predict_confidence(self, turn):
        if len(self.embeddings) < self.k:
            return turn["agent_confidence"], False
        # Encode query
        q_emb = self.encoder.encode([turn["question"]], show_progress_bar=False)[0]
        # Cosine similarity
        mem = np.array(self.embeddings)
        sims = mem @ q_emb / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        # Top-k
        top_k = np.argsort(-sims)[:self.k]
        neighbor_labels = [self.labels[i] for i in top_k]
        return float(np.mean(neighbor_labels)), False

    def self_assess(self, domain):
        if self.use_gt:
            labels_in_domain = [l for l, d in zip(self.labels, self.domains) if d == domain]
            if not labels_in_domain:
                return 0.5
            return float(np.mean(labels_in_domain))
        # FIXED v4: feedback variant uses decoded labels (1=positive, 0=negative)
        # as ground truth proxy; return mean over per-domain decoded labels.
        labels_in_domain = [l for l, d in zip(self.labels, self.domains) if d == domain]
        if not labels_in_domain:
            return 0.5
        return float(np.mean(labels_in_domain))

    def _build_self_assess_context(self, domain):
        entries = [(q, fl) for q, fl, d in
                   zip(self.questions, self.feedback_labels, self.domains) if d == domain]
        if not entries:
            return None
        lines = []
        for i, (q, fl) in enumerate(entries, 1):
            reaction = {"positive": "positive (correct)", "negative": "negative (wrong)",
                        "uncertain": "uncertain"}.get(fl, fl)
            lines.append(f"{i}. Question: \"{q[:80]}...\" → User reaction: {reaction}")
        return "Your past interactions in this domain:\n" + "\n".join(lines)

    def observe(self, turn):
        emb = self.encoder.encode([turn["question"]], show_progress_bar=False)[0]
        if self.use_gt:
            label = int(turn["gt_correct"])
            fl = "positive" if label else "negative"
        else:
            lbl, _ = rule_classify(turn["user_response"])
            if lbl == "positive":
                label = 1
            elif lbl in ("negative", "uncertain"):
                label = 0
            else:
                return  # skip neutral
            fl = lbl
        self.embeddings.append(emb)
        self.labels.append(label)
        self.domains.append(turn["domain"])
        self.questions.append(turn["question"])
        self.feedback_labels.append(fl)


# ========= B5: MMC-style Capability Memory =========
class MMCCapMem(Baseline):
    """MMC-style: uses running per-domain accuracy from feedback + query-level
    similarity to past errors via semantic kNN retrieval (no LLM call, just encoder).

    Simplified version of MMC v3's similarity-based calibration.
    """
    name = "MMC-CapMem"

    def __init__(self, encoder, k=3):
        self.encoder = encoder
        self.k = k
        self.reset()

    def reset(self):
        self.domain_correct = defaultdict(float)
        self.domain_total = defaultdict(float)
        self.error_embs = []
        self.error_domains = []
        self.error_questions = []

    def predict_confidence(self, turn):
        d = turn["domain"]
        base_rate = self.domain_correct[d] / self.domain_total[d] if self.domain_total[d] > 0 else 0.5

        if len(self.error_embs) < self.k:
            return base_rate, False

        # Similarity to past errors in SAME domain first, then cross-domain
        q_emb = self.encoder.encode([turn["question"]], show_progress_bar=False)[0]
        # Only use errors from same domain (otherwise cross-domain noise)
        same_domain_errors = [i for i, ed in enumerate(self.error_domains) if ed == d]
        if len(same_domain_errors) < 1:
            return base_rate, False

        mem = np.array([self.error_embs[i] for i in same_domain_errors])
        sims = mem @ q_emb / (np.linalg.norm(mem, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        max_sim = float(np.max(sims))
        conf = 0.7 * base_rate + 0.3 * (1.0 - max_sim)
        return float(np.clip(conf, 0.01, 0.99)), False

    def self_assess(self, d):
        # FIXED v4: return MMC's decoded-feedback per-domain accuracy.
        if self.domain_total[d] == 0:
            return 0.5
        return float(self.domain_correct[d] / self.domain_total[d])

    def _build_self_assess_context(self, domain):
        pos = self.domain_correct[domain]
        total = self.domain_total[domain]
        if total == 0:
            return None
        neg = total - pos
        ctx = (f"User feedback summary:\n"
               f"- Positive reactions: {int(pos)}\n"
               f"- Negative/uncertain reactions: {int(neg)}\n"
               f"- Total: {int(total)}")
        errs = [q for q, d in zip(self.error_questions, self.error_domains) if d == domain]
        if errs:
            ctx += "\n\nQuestions where users indicated you were wrong:\n"
            for i, q in enumerate(errs[:5], 1):
                ctx += f"  {i}. \"{q[:80]}...\"\n"
        return ctx

    def observe(self, turn):
        label, _ = rule_classify(turn["user_response"])
        d = turn["domain"]
        if label == "positive":
            self.domain_correct[d] += 1
            self.domain_total[d] += 1
        elif label in ("negative", "uncertain"):
            self.domain_total[d] += 1
            # Store as error case
            emb = self.encoder.encode([turn["question"]], show_progress_bar=False)[0]
            self.error_embs.append(emb)
            self.error_domains.append(d)
            self.error_questions.append(turn["question"])


# ========= B11: Zero-Shot Self-Knowledge =========
class ZeroShotSelfKnow(Baseline):
    """Ask the model to estimate its own accuracy in a domain with NO interaction
    history at all. Tests pure prior self-knowledge."""
    name = "ZeroShot-SK"

    def predict_confidence(self, turn):
        return turn["agent_confidence"], False

    def self_assess(self, domain):
        if self._agent_fn is None:
            return 0.5
        prompt = (
            f'You are an AI assistant. Without any specific interaction history, '
            f'how accurate do you think you are at answering questions in the domain: '
            f'"{domain.replace("_", " ")}"?\n'
            f'Reply with ONLY a single integer between 0 and 100.'
        )
        try:
            resp = self._agent_fn(prompt)
            m = re.search(r'(\d+)', resp)
            if m:
                return float(np.clip(int(m.group(1)) / 100.0, 0.01, 0.99))
        except Exception:
            pass
        return 0.5


# ========= B12: GT-Aware Self-Stats =========
class GTSelfStats(Baseline):
    """Give the model its actual correct/incorrect counts (from ground truth)
    and ask it to report its accuracy. Tests whether the model faithfully
    reports given statistics or introduces bias."""
    name = "GT-SelfStats"

    def reset(self):
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    def predict_confidence(self, turn):
        return turn["agent_confidence"], False

    def self_assess(self, domain):
        if self.total[domain] == 0:
            return 0.5
        if self._agent_fn is None:
            return self.correct[domain] / self.total[domain]
        c = self.correct[domain]
        t = self.total[domain]
        prompt = (
            f'You are reflecting on your own performance in the domain: '
            f'"{domain.replace("_", " ")}".\n\n'
            f'Here are your actual results: you answered {c} out of {t} questions '
            f'correctly in this domain.\n\n'
            f'Based on this, what is your accuracy percentage in this domain? '
            f'Reply with ONLY a single integer between 0 and 100.'
        )
        try:
            resp = self._agent_fn(prompt)
            m = re.search(r'(\d+)', resp)
            if m:
                return float(np.clip(int(m.group(1)) / 100.0, 0.01, 0.99))
        except Exception:
            pass
        return self.correct[domain] / self.total[domain]

    def observe(self, turn):
        d = turn["domain"]
        self.total[d] += 1
        if turn["gt_correct"]:
            self.correct[d] += 1


# ========= B6: LLM In-Context Self-Assessment =========
class LLMSelfAssess(Baseline):
    """Ask the agent itself to estimate its confidence based on recent history
    provided in-context. No structured memory — pure prompt-based reasoning.
    """
    name = "LLMSelfAssess"

    def __init__(self, agent_url="http://localhost:8100", agent_model_path="./HOME/models/llama3-8b-instruct", history_window=10):
        self.agent_url = agent_url
        self.agent_model_path = agent_model_path
        self.history_window = history_window
        self.reset()

    def reset(self):
        self.history = []  # list of (q_truncated, user_feedback_truncated, domain)

    def predict_confidence(self, turn):
        if len(self.history) < 3:
            return turn["agent_confidence"], False

        recent = self.history[-self.history_window:]
        history_text = "\n".join([
            f"- (Domain: {d}) Question: {q[:80]}... User reaction: {f[:100]}"
            for q, f, d in recent
        ])

        prompt = f"""Based on the following recent interaction history with the user, estimate your likely accuracy on the next question.

Recent history:
{history_text}

The next question is in domain: {turn['domain']}
Next question: {turn['question'][:200]}

Based on how the user reacted to your previous answers in similar domains, what is your estimated probability of answering correctly? Reply with ONLY a number between 0 and 100, nothing else."""

        try:
            resp = requests.post(f"{self.agent_url}/v1/chat/completions", json={
                "model": self.agent_model_path,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            }, timeout=20)
            text = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'(\d+)', text)
            if m:
                return float(np.clip(int(m.group(1)) / 100.0, 0.01, 0.99)), False
        except Exception:
            pass
        return turn["agent_confidence"], False

    def self_assess(self, domain):
        ctx = self._build_self_assess_context(domain)
        if ctx is None:
            return 0.5
        return self._llm_self_assess(domain, ctx)

    def _build_self_assess_context(self, domain):
        same_domain = [(q, f) for q, f, d in self.history if d == domain]
        if len(same_domain) < 2:
            return None
        lines = []
        for i, (q, f) in enumerate(same_domain, 1):
            lines.append(f"{i}. Your question: \"{q[:80]}...\"\n   User response: \"{f[:120]}\"")
        return "Raw interaction history in this domain:\n" + "\n".join(lines)

    def observe(self, turn):
        self.history.append((turn["question"], turn["user_response"], turn["domain"]))


# ========= Runner =========
def compute_session_metrics(baseline, session):
    baseline.reset()
    confs, labels, abstains = [], [], []

    for idx, turn in enumerate(session["log"]):
        c, a = baseline.predict_confidence(turn)
        confs.append(c)
        labels.append(int(turn["gt_correct"]))
        abstains.append(a)
        baseline.observe(turn)

    nc = [(c, l) for c, l, a in zip(confs, labels, abstains) if not a]
    if not nc: return None
    cc = [x[0] for x in nc]; ll = [x[1] for x in nc]

    ece = compute_ece(cc, ll)
    brier = compute_brier(cc, ll)
    auroc = compute_auroc(cc, ll)
    hc = [(c, l) for c, l in zip(cc, ll) if c > 0.7]
    hr = sum(1 for c, l in hc if l == 0) / len(hc) if hc else 0

    # CBF: only compute once at the end
    true_acc = defaultdict(lambda: [0, 0])
    for t in session["log"]:
        true_acc[t["domain"]][1] += 1
        if t["gt_correct"]:
            true_acc[t["domain"]][0] += 1
    errors = []
    for dom, (cor, tot) in true_acc.items():
        if tot >= 2:
            errors.append(abs(baseline.self_assess(dom) - cor / tot))
    final_cbf = 1 - float(np.mean(errors)) if errors else 0.0

    return {
        "ECE": round(ece, 4),
        "Brier": round(brier, 4),
        "AUROC": round(auroc, 3),
        "HR": round(hr, 4),
        "CBF": round(final_cbf, 3),
        "n_turns": len(confs),
    }


def make_agent_fn(agent_url, agent_model):
    """Create a callable that sends a prompt to the agent and returns the response text."""
    def fn(prompt):
        resp = requests.post(f"{agent_url}/v1/chat/completions", json={
            "model": agent_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 20,
        }, timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_file", required=True)
    ap.add_argument("--skip_llm_self_assess", action="store_true",
                    help="Skip B6 (requires LLM calls, slow)")
    ap.add_argument("--agent_url", default="http://localhost:8100")
    ap.add_argument("--agent_model", default="./HOME/models/llama3-8b-instruct")
    ap.add_argument("--output_dir", default="./capbound-bench/experiments/results")
    args = ap.parse_args()

    with open(args.session_file) as f:
        data = json.load(f)

    print(f"\n{'='*85}")
    print(f"CapBoundary-Bench FULL baseline evaluation")
    print(f"Sessions: {len(data['sessions'])}, turns/session: {len(data['sessions'][0]['log'])}")
    print(f"Agent: {data['metadata']['agent_model']}")
    print(f"{'='*85}\n")

    # Load encoder
    print("Loading sentence encoder...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create agent function for LLM-based self_assess
    agent_fn = make_agent_fn(args.agent_url, args.agent_model)

    baselines = [
        NoMemoryBaseline(),
        OraclePDC(),
        UserFeedbackPDC(),
        EpisodicMemory(encoder, k=5, use_gt=True),
        EpisodicMemory(encoder, k=5, use_gt=False),
        MMCCapMem(encoder, k=3),
    ]
    if not args.skip_llm_self_assess:
        baselines.append(LLMSelfAssess(args.agent_url, args.agent_model))

    for b in baselines:
        b.set_agent_fn(agent_fn)

    per_session = defaultdict(list)
    for si, sess in enumerate(data["sessions"]):
        print(f"Session {si+1}/{len(data['sessions'])}...", end=" ", flush=True)
        for b in baselines:
            m = compute_session_metrics(b, sess)
            if m: per_session[b.name].append(m)
        print("done")

    print(f"\n{'Baseline':<22} {'ECE':>7} {'Brier':>7} {'AUROC':>7} {'HR':>7} {'CBF':>7}")
    print("-" * 65)
    agg = {}
    for b in baselines:
        rs = per_session[b.name]
        if not rs: continue
        def m(k):
            vs = [r[k] for r in rs if r.get(k) is not None]
            return np.mean(vs) if vs else None
        agg[b.name] = {
            "ECE": m("ECE"), "Brier": m("Brier"), "AUROC": m("AUROC"),
            "HR": m("HR"), "CBF": m("CBF"),
        }
        print(f"{b.name:<22} {agg[b.name]['ECE']:>7.4f} {agg[b.name]['Brier']:>7.4f} "
              f"{agg[b.name]['AUROC']:>7.3f} {agg[b.name]['HR']:>7.4f} "
              f"{agg[b.name]['CBF']:>7.3f}")

    ts = os.path.basename(args.session_file).replace("capbound_sessions_", "").replace(".json", "")
    out = {
        "session_file": args.session_file,
        "agent_model": data["metadata"]["agent_model"],
        "n_sessions": len(data["sessions"]),
        "aggregate": {k: {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()}
                      for k, v in agg.items()},
    }
    out_path = os.path.join(args.output_dir, f"capbound_baselines_v2_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
