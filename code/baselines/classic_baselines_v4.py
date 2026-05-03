#!/usr/bin/env python3
"""
Classic calibration baselines for CapBoundary-Bench (B7, B8, B9, B10).

These are reference implementations of well-known methods adapted to the
online, noisy-feedback setting of CapBoundary-Bench. All four baselines
share the same Baseline interface as `capbound_baselines_v2.py`:

    reset()                        -- reset state
    predict_confidence(turn)       -- returns (conf, abstain)
    self_assess(domain)            -- per-domain accuracy estimate (for CBF)
    observe(turn)                  -- update state from turn

Baselines:
  B7 Platt-Online (Platt 1999)              -- online sigmoid recalibration via SGD
  B8 HistBin-Online (Zadrozny & Elkan 2001) -- online per-bin running accuracy
  B9 Bayes-Domain                           -- per-domain Beta(1,1) posterior
  B10 EM-Joint                              -- EM-based joint estimation of
                                               p*(d) and r(expertise)

Reproducibility:
  python3 classic_baselines.py --session_file <path-to-capbound_sessions_*.json>

Writes JSON with per-baseline aggregate metrics to --output_dir (default:
experiments/results/classic_baselines_rerun_<timestamp>.json).
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feedback_classifier import rule_classify


# ============================================================
# Shared metric helpers
# ============================================================
def compute_ece(confidences, labels, n_bins=10):
    if not confidences:
        return 0.0
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bounds = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    ece = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        conf_m = confidences[mask].mean()
        acc_m = labels[mask].mean()
        ece += (mask.sum() / total) * abs(conf_m - acc_m)
    return float(ece)


def compute_brier(confidences, labels):
    if not confidences:
        return 0.0
    return float(np.mean((np.asarray(confidences, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def compute_auroc(confidences, labels):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return 0.5
        return float(roc_auc_score(labels, confidences))
    except Exception:
        return 0.5


def decode_feedback_label(user_response):
    """Return +1 (correct), 0 (wrong), or None (ambiguous/neutral)."""
    lbl, _ = rule_classify(user_response)
    if lbl == "positive":
        return 1
    if lbl == "negative":
        return 0
    return None


# ============================================================
# Baseline base class (duck-typed with capbound_baselines_v2.Baseline)
# ============================================================
SELF_ASSESS_PROMPT = """You are reflecting on your own performance in the domain: "{domain}".
Below is a summary of your interaction history in this domain:

{context}

Based on this information, estimate what percentage of questions you answered correctly in this domain.
Reply with ONLY a single integer between 0 and 100."""


class Baseline:
    name = "base"
    _agent_fn = None
    _sa_cache = {}

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


# ============================================================
# B7: Platt-Online (Platt 1999), online SGD
# ============================================================
class PlattOnline(Baseline):
    """Online Platt scaling: confidence -> sigmoid(w*c + b).

    Parameters (w, b) are updated with SGD on decoded-feedback labels. We use
    logistic-regression cross-entropy with L2 regularization, as in the
    canonical Platt (1999) / Lin-Lin-Weng (2007) formulation.
    """
    name = "Platt-Online"

    def __init__(self, lr=0.1, l2=1e-3):
        self.lr = lr
        self.l2 = l2
        self.reset()

    def reset(self):
        # Identity start: sigmoid(1*c + 0) ~ c
        self.w = 1.0
        self.b = 0.0
        # Per-domain running accuracy (for self_assess/CBF)
        self.dom_n = defaultdict(int)
        self.dom_c = defaultdict(int)

    def _sigmoid(self, z):
        # Numerically stable sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_confidence(self, turn):
        c = float(turn["agent_confidence"])
        p = self._sigmoid(self.w * c + self.b)
        return float(np.clip(p, 1e-3, 1 - 1e-3)), False

    def self_assess(self, domain):
        # FIXED v4: return Platt's per-domain posterior from decoded feedback,
        # not an LLM verbal re-estimate. Decoded labels are 0/1 from feedback.
        n = self.dom_n.get(domain, 0)
        if n == 0:
            return 0.5
        return float(self.dom_c[domain] / n)

    def observe(self, turn):
        label = decode_feedback_label(turn["user_response"])
        if label is None:
            return
        c = float(turn["agent_confidence"])
        z = self.w * c + self.b
        p = self._sigmoid(z)
        # Gradient of binary cross-entropy + L2
        g = p - label
        self.w -= self.lr * (g * c + self.l2 * self.w)
        self.b -= self.lr * g
        # Track per-domain accuracy for CBF
        d = turn["domain"]
        self.dom_n[d] += 1
        self.dom_c[d] += label


# ============================================================
# B8: HistBin-Online (Zadrozny & Elkan 2001)
# ============================================================
class HistBinOnline(Baseline):
    """Online histogram binning: 10 equal-width bins over [0,1].

    Each bin maintains a running mean of decoded-feedback labels. Predicted
    confidence for a new turn is the recalibrated accuracy of the bin that
    its raw model confidence falls into.
    """
    name = "HistBin-Online"

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        self.bin_n = np.zeros(self.n_bins, dtype=float)
        self.bin_c = np.zeros(self.n_bins, dtype=float)
        self.dom_n = defaultdict(int)
        self.dom_c = defaultdict(int)

    def _bin_index(self, c):
        idx = int(min(max(c, 0.0), 1.0 - 1e-9) * self.n_bins)
        return max(0, min(self.n_bins - 1, idx))

    def predict_confidence(self, turn):
        c = float(turn["agent_confidence"])
        b = self._bin_index(c)
        if self.bin_n[b] < 1:
            return float(np.clip(c, 1e-3, 1 - 1e-3)), False
        return float(np.clip(self.bin_c[b] / self.bin_n[b], 1e-3, 1 - 1e-3)), False

    def self_assess(self, domain):
        # FIXED v4: return per-domain feedback posterior, not LLM verbal estimate.
        n = self.dom_n.get(domain, 0)
        if n == 0:
            return 0.5
        return float(self.dom_c[domain] / n)

    def observe(self, turn):
        label = decode_feedback_label(turn["user_response"])
        if label is None:
            return
        c = float(turn["agent_confidence"])
        b = self._bin_index(c)
        self.bin_n[b] += 1
        self.bin_c[b] += label
        d = turn["domain"]
        self.dom_n[d] += 1
        self.dom_c[d] += label


# ============================================================
# B9: Bayes-Domain (Beta conjugate per domain)
# ============================================================
class BayesDomain(Baseline):
    """Per-domain Beta(alpha, beta) posterior, initialized with Beta(1, 1).

    On decoded positive feedback: alpha += 1.
    On decoded negative feedback: beta += 1.
    Predicted confidence / self-assess = posterior mean alpha / (alpha + beta).
    """
    name = "Bayes-Domain"

    def __init__(self, alpha0=1.0, beta0=1.0):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.reset()

    def reset(self):
        self.alpha = defaultdict(lambda: self.alpha0)
        self.beta = defaultdict(lambda: self.beta0)

    def predict_confidence(self, turn):
        d = turn["domain"]
        a, b = self.alpha[d], self.beta[d]
        return float(a / (a + b)), False

    def self_assess(self, domain):
        # FIXED v4: return Beta posterior mean alpha/(alpha+beta), not LLM verbal estimate.
        a, b = self.alpha[domain], self.beta[domain]
        return float(a / (a + b))

    def observe(self, turn):
        label = decode_feedback_label(turn["user_response"])
        if label is None:
            return
        d = turn["domain"]
        if label == 1:
            self.alpha[d] += 1
        else:
            self.beta[d] += 1


# ============================================================
# B10: EM-Joint (Expectation-Maximization for p*(d) and r(expertise))
# ============================================================
class EMJoint(Baseline):
    """EM-based joint estimator of per-domain accuracy p*(d) and per-expertise
    user reliability r(in), r(out).

    Model:
      - Observation: decoded label y_t in {0, 1, missing} (from rule_classify).
      - Latent: correctness z_t in {0, 1}.
      - Emission: P(y_t = 1 | z_t = 1, expertise_t) = r(expertise_t);
                  P(y_t = 1 | z_t = 0, expertise_t) = 1 - r(expertise_t).
      - Prior on z_t: p*(d_t).
      - Expertise tag expertise_t = "in" / "out" is observed from turn.

    E-step: posterior P(z_t = 1 | y_t, expertise_t, p*, r).
    M-step: weighted MLE update of p*(d) and r(expertise).

    At test time, EM is run to convergence on the buffered labeled turns,
    and predictions use the current p*(d). Self-assess returns p*(d).

    We do ONE batch EM pass at the end of the session to produce final
    estimates (used for self_assess / CBF). During the session the
    predict_confidence uses a running point estimate of p*(d) derived from
    reliability-weighted feedback, so ECE/Brier reflect online behavior.
    """
    name = "EM-Joint"

    def __init__(self, em_iters=30, em_tol=1e-4, r_init=(0.7, 0.5)):
        self.em_iters = em_iters
        self.em_tol = em_tol
        self.r_in_init, self.r_out_init = r_init
        self.reset()

    def reset(self):
        # Online point estimates
        self.p_star = defaultdict(lambda: 0.5)
        self.r_in = self.r_in_init
        self.r_out = self.r_out_init
        # Buffer for final batch EM
        self.buffer = []  # list of (domain, expertise, decoded_label)
        # Running per-domain feedback counters (weighted by current r estimate)
        self.weighted_c = defaultdict(float)
        self.weighted_n = defaultdict(float)

    def _posterior_correct(self, p, r, y):
        """P(z = 1 | y, r, p) for decoded label y in {0, 1}."""
        # P(y | z=1) = r; P(y | z=0) = 1-r  if y=1
        # P(y | z=1) = 1-r; P(y | z=0) = r  if y=0
        if y == 1:
            like1 = r
            like0 = 1 - r
        else:
            like1 = 1 - r
            like0 = r
        num = like1 * p
        den = like1 * p + like0 * (1 - p) + 1e-12
        return num / den

    def predict_confidence(self, turn):
        d = turn["domain"]
        return float(np.clip(self.p_star[d], 1e-3, 1 - 1e-3)), False

    def self_assess(self, domain):
        # FIXED v4: return EM's joint estimate p_star[d], not LLM verbal estimate.
        # This is the algorithmic posterior after EM iteration.
        return float(np.clip(self.p_star.get(domain, 0.5), 1e-3, 1 - 1e-3))

    def observe(self, turn):
        label = decode_feedback_label(turn["user_response"])
        d = turn["domain"]
        expertise = "in" if turn.get("in_user_expertise", False) else "out"
        if label is None:
            return
        # Buffer for final batch EM
        self.buffer.append((d, expertise, label))
        # Online update: use current r estimate to weight the feedback
        r = self.r_in if expertise == "in" else self.r_out
        p = self.p_star[d]
        w1 = self._posterior_correct(p, r, label)
        self.weighted_c[d] += w1
        self.weighted_n[d] += 1.0
        # Refresh online point estimate with smoothing (Beta(1,1) prior)
        self.p_star[d] = (self.weighted_c[d] + 1.0) / (self.weighted_n[d] + 2.0)

    def run_batch_em(self):
        """Run batch EM on the full buffer to produce final estimates."""
        if not self.buffer:
            return
        # Initialize from current online estimates
        p_star = dict(self.p_star)
        r_in, r_out = self.r_in, self.r_out
        domains = set(d for d, _, _ in self.buffer)
        for d in domains:
            p_star.setdefault(d, 0.5)
        for _ in range(self.em_iters):
            # E-step: posterior per turn
            posteriors = []
            for d, exp, y in self.buffer:
                r = r_in if exp == "in" else r_out
                posteriors.append(self._posterior_correct(p_star[d], r, y))
            # M-step: update p_star per domain (weighted)
            num_p = defaultdict(float)
            den_p = defaultdict(float)
            num_rin = den_rin = 0.0
            num_rout = den_rout = 0.0
            for (d, exp, y), w in zip(self.buffer, posteriors):
                num_p[d] += w
                den_p[d] += 1.0
                # For r: P(y | z, expertise)
                # If y == 1 and z == 1 -> count as r hit (weight w)
                # If y == 0 and z == 0 -> count as r hit (weight 1-w)
                if exp == "in":
                    if y == 1:
                        num_rin += w
                    else:
                        num_rin += (1 - w)
                    den_rin += 1.0
                else:
                    if y == 1:
                        num_rout += w
                    else:
                        num_rout += (1 - w)
                    den_rout += 1.0
            p_star_new = {d: (num_p[d] + 1.0) / (den_p[d] + 2.0) for d in domains}
            r_in_new = (num_rin + 1.0) / (den_rin + 2.0) if den_rin > 0 else r_in
            r_out_new = (num_rout + 1.0) / (den_rout + 2.0) if den_rout > 0 else r_out
            # Convergence
            delta = max(
                max(abs(p_star_new[d] - p_star[d]) for d in domains),
                abs(r_in_new - r_in),
                abs(r_out_new - r_out),
            )
            p_star, r_in, r_out = p_star_new, r_in_new, r_out_new
            if delta < self.em_tol:
                break
        # Commit final estimates
        self.p_star = defaultdict(lambda: 0.5)
        for d, v in p_star.items():
            self.p_star[d] = v
        self.r_in = r_in
        self.r_out = r_out


# ============================================================
# Session metric runner (kept identical to capbound_baselines_v2)
# ============================================================
def compute_session_metrics(baseline, session):
    baseline.reset()
    confs, labels, abstains = [], [], []

    for idx, turn in enumerate(session["log"]):
        c, a = baseline.predict_confidence(turn)
        confs.append(c)
        labels.append(int(turn["gt_correct"]))
        abstains.append(a)
        baseline.observe(turn)

    # For EM-Joint, run final batch EM after all turns are observed
    if isinstance(baseline, EMJoint):
        baseline.run_batch_em()

    nc = [(c, l) for c, l, ab in zip(confs, labels, abstains) if not ab]
    if not nc:
        return None
    cc = [x[0] for x in nc]
    ll = [x[1] for x in nc]

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
    import requests
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
    ap = argparse.ArgumentParser(description="Classic baselines (B7-B10) for CapBoundary-Bench")
    ap.add_argument("--session_file", required=True)
    ap.add_argument("--agent_url", default="http://localhost:8100")
    ap.add_argument("--agent_model", default="./HOME/models/llama3-8b-instruct")
    ap.add_argument(
        "--output_dir",
        default="./HOME/yzs/LLM\u5fc3\u667a/experiments/results",
    )
    ap.add_argument("--tag", default="", help="Optional tag for output filename")
    args = ap.parse_args()

    with open(args.session_file) as f:
        data = json.load(f)

    print(f"\n{'=' * 85}")
    print("CapBoundary-Bench classic baselines (B7-B10)")
    print(f"Session file: {args.session_file}")
    print(f"Sessions: {len(data['sessions'])}, turns/session: {len(data['sessions'][0]['log'])}")
    print(f"{'=' * 85}\n")

    baselines = [PlattOnline(), HistBinOnline(), BayesDomain(), EMJoint()]

    agent_fn = make_agent_fn(args.agent_url, args.agent_model)
    for b in baselines:
        b.set_agent_fn(agent_fn)

    per_session = defaultdict(list)
    for si, sess in enumerate(data["sessions"]):
        print(f"Session {si + 1}/{len(data['sessions'])}...", end=" ", flush=True)
        for b in baselines:
            m = compute_session_metrics(b, sess)
            if m:
                per_session[b.name].append(m)
        print("done")

    print(f"\n{'Baseline':<20} {'ECE':>7} {'Brier':>7} {'AUROC':>7} {'HR':>7} {'CBF':>7}")
    print("-" * 60)
    agg = {}
    for b in baselines:
        rs = per_session[b.name]
        if not rs:
            continue

        def m(k, rs=rs):
            vs = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(vs)) if vs else None

        agg[b.name] = {
            "ECE": m("ECE"),
            "Brier": m("Brier"),
            "AUROC": m("AUROC"),
            "HR": m("HR"),
            "CBF": m("CBF"),
        }
        print(
            f"{b.name:<20} {agg[b.name]['ECE']:>7.4f} {agg[b.name]['Brier']:>7.4f} "
            f"{agg[b.name]['AUROC']:>7.3f} {agg[b.name]['HR']:>7.4f} "
            f"{agg[b.name]['CBF']:>7.3f}"
        )

    # Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    src = os.path.basename(args.session_file).replace(".json", "")
    out = {
        "session_file": args.session_file,
        "agent_model": data["metadata"].get("agent_model", "unknown"),
        "n_sessions": len(data["sessions"]),
        "aggregate": {
            k: {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()}
            for k, v in agg.items()
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"classic_baselines_{src}{tag}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
