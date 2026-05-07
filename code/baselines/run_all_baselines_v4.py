#!/usr/bin/env python3
"""
Unified runner for all 10 CapBoundary-Bench baselines (B1-B10).

Supports automatic API routing for 13 models across 3 endpoints:
  - Local vLLM (localhost:8100): Llama-3-8B, Mistral-7B, qwen2.5-7b-instruct
  - DashScope (百炼): qwen-max, qwen-plus, qwen-turbo, qwen3.5-27b,
                      deepseek-v3.2, glm-5
  - Proxy (中转站): gpt-5.4, gpt-5.4-mini, gpt-5.5, claude-opus-4-6, claude-opus-4-7, claude-sonnet-4-6

Usage:
    # Run single model
    python3 run_all_baselines.py --session_file <path>

    # Batch run all 13 models (auto-detect from results dir)
    python3 run_all_baselines.py --batch

    # Skip slow baselines
    python3 run_all_baselines.py --session_file <path> --skip_episodic
"""
import argparse
import glob
import json
import os
import re
import subprocess
import signal
import sys
import time
import requests
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capbound_baselines_v4 import (
    NoMemoryBaseline,
    OraclePDC,
    UserFeedbackPDC,
    EpisodicMemory,
    MMCCapMem,
    LLMSelfAssess,
    ZeroShotSelfKnow,
    GTSelfStats,
    compute_session_metrics as compute_v2,
)
from classic_baselines_v4 import (
    PlattOnline,
    HistBinOnline,
    BayesDomain,
    EMJoint,
    compute_session_metrics as compute_classic,
)

# ============================================================
# API routing config
# ============================================================
# All provider URLs are env-overridable so reviewers can substitute their own
# gateway/proxy without code changes. Defaults are the official provider endpoints.
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-REDACTED")
DASHSCOPE_URL = os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com") + "/compatible-mode/v1/chat/completions"

# OpenAI route (used for GPT-5.x family)
PROXY_KEY = os.environ.get("OPENAI_API_KEY", "sk-REDACTED")
PROXY_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com") + "/v1/chat/completions"
LUCEN_KEY = PROXY_KEY
LUCEN_URL = PROXY_URL

# Anthropic route (used for Claude family)
ANTHROPIC_PROXY_URL = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com") + "/v1/messages"
LINKAPI_KEY = os.environ.get("ANTHROPIC_API_KEY", "sk-REDACTED")
LINKAPI_URL = ANTHROPIC_PROXY_URL

# Zhipu / bigmodel.cn (GLM family official endpoint)
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "sk-REDACTED")
ZHIPU_URL = os.environ.get("ZHIPU_API_BASE", "https://open.bigmodel.cn") + "/api/paas/v4/chat/completions"

# DeepSeek official endpoint (separate from DashScope's re-host)
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-REDACTED")
DEEPSEEK_URL = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com") + "/v1/chat/completions"

LOCAL_URL = "http://localhost:8100/v1/chat/completions"

# Local models: model_name -> (model_path, port)
LOCAL_MODELS = {
    "Llama-3-8B": ("./HOME/models/llama3-8b-instruct", 8100),
    "Mistral-7B":  ("./HOME/models/mistral-7b-instruct", 8101),
}

LOCAL_URL_8101 = "http://localhost:8101/v1/chat/completions"

# model_name -> (api_url, api_key, model_id_for_api)
MODEL_ROUTES = {
    # Local vLLM (auto-started if needed)
    "Llama-3-8B":          ("http://localhost:8100/v1/chat/completions", None, "./HOME/models/llama3-8b-instruct"),
    "Mistral-7B":          ("http://localhost:8101/v1/chat/completions", None, "./HOME/models/mistral-7b-instruct"),
    # DashScope 百炼
    "qwen2.5-7b-instruct": (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen2.5-7b-instruct"),
    "qwen-max":      (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen-max"),
    "qwen-plus":     (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen-plus"),
    "qwen-turbo":    (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen-turbo"),
    "qwen3.5-27b":   (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen3.5-27b"),
    "deepseek-v3.2": (DASHSCOPE_URL, DASHSCOPE_KEY, "deepseek-v3"),
    "glm-5":         (DASHSCOPE_URL, DASHSCOPE_KEY, "glm-5"),
    # Proxy 中转站 (OpenAI format)
    "gpt-5.4":          (LUCEN_URL, LUCEN_KEY, "gpt-5.4"),
    "gpt-5.4-mini":     (LUCEN_URL, LUCEN_KEY, "gpt-5.4-mini"),
    "gpt-5.5":          (LUCEN_URL, LUCEN_KEY, "gpt-5.5"),
    # Proxy 中转站 (Anthropic format) - marked with special prefix
    "claude-opus-4-6":  ("ANTHROPIC:" + LINKAPI_URL, LINKAPI_KEY, "claude-opus-4-6"),
    "claude-sonnet-4-6":("ANTHROPIC:" + LINKAPI_URL, LINKAPI_KEY, "claude-sonnet-4-6"),
    "claude-opus-4-7": ("ANTHROPIC:" + LINKAPI_URL, LINKAPI_KEY, "claude-opus-4-7"),
    # New frontier additions (matched 1:1 with the latest OpenAI/Anthropic releases)
    "qwen3.6-max-preview": (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen3.6-max-preview"),
    "qwen3.6-plus":     (DASHSCOPE_URL, DASHSCOPE_KEY, "qwen3.6-plus"),
    "glm-5.1":          (ZHIPU_URL, ZHIPU_KEY, "glm-5.1"),
    "deepseek-v4-pro":  (DEEPSEEK_URL, DEEPSEEK_KEY, "deepseek-v4-pro"),
}

_vllm_procs = {}  # port -> Popen


def ensure_vllm(model_name):
    """Auto-start vLLM if model_name is a local model and server isn't running."""
    if model_name not in LOCAL_MODELS:
        return
    model_path, port = LOCAL_MODELS[model_name]
    url = f"http://localhost:{port}/v1/models"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            print(f"  vLLM already running on port {port}")
            return
    except Exception:
        pass

    print(f"  Starting vLLM for {model_name} on port {port}...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", model_path,
         "--port", str(port),
         "--gpu-memory-utilization", "0.4"],
        stdout=open(f"/tmp/vllm_{model_name}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    _vllm_procs[port] = proc

    for i in range(120):
        time.sleep(2)
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"  vLLM ready (took {(i+1)*2}s)")
                return
        except Exception:
            pass
    print(f"  WARNING: vLLM failed to start within 240s, check /tmp/vllm_{model_name}.log")


def stop_vllm():
    """Stop all vLLM servers started by this script."""
    for port, proc in _vllm_procs.items():
        print(f"  Stopping vLLM on port {port} (pid={proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _vllm_procs.clear()


def make_agent_fn(model_name):
    """Create agent_fn that routes to the correct API for self_assess calls."""
    if model_name not in MODEL_ROUTES:
        print(f"  WARNING: unknown model '{model_name}', self_assess will return 0.5")
        return None

    api_url, api_key, model_id = MODEL_ROUTES[model_name]

    # Anthropic Messages API (for Claude models via proxy)
    if api_url.startswith("ANTHROPIC:"):
        real_url = api_url[len("ANTHROPIC:"):]

        def fn(prompt):
            for attempt in range(4):
                try:
                    resp = requests.post(real_url, headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }, json={
                        "model": model_id,
                        "max_tokens": 200,
                        "messages": [{"role": "user", "content": prompt}],
                    }, timeout=90)
                    return resp.json()["content"][0]["text"]
                except Exception:
                    if attempt < 3:
                        time.sleep(5 * (attempt + 1))
            raise ConnectionError(f"Failed after 4 retries")

        return fn

    # OpenAI-compatible API (DashScope, proxy GPT, local vLLM)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def fn(prompt):
        resp = requests.post(api_url, headers=headers, json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200,
        }, timeout=60)
        return resp.json()["choices"][0]["message"]["content"]

    return fn


def run_one(session_file, output_dir, skip_episodic=False, encoder=None):
    """Run all baselines on a single session file."""
    with open(session_file) as f:
        data = json.load(f)

    model_name = data["metadata"].get("agent_model", "unknown")
    n_sess = len(data["sessions"])
    n_turns = len(data["sessions"][0]["log"])

    print(f"\n{'=' * 70}")
    print(f"  Model: {model_name}  |  {n_sess} sessions × {n_turns} turns")
    print(f"  File:  {os.path.basename(session_file)}")
    print(f"{'=' * 70}")

    ensure_vllm(model_name)
    # B6 always needs Llama vLLM on port 8100 for predict_confidence
    ensure_vllm("Llama-3-8B")
    agent_fn = make_agent_fn(model_name)

    # Clear self_assess cache from previous model
    from capbound_baselines_v4 import Baseline as BV2
    from classic_baselines_v4 import Baseline as BCL
    BV2.clear_cache()
    BCL.clear_cache()

    # B1-B3
    baselines = [
        (NoMemoryBaseline(), compute_v2),
        (OraclePDC(), compute_v2),
        (UserFeedbackPDC(), compute_v2),
    ]

    # B4a/B4b/B5
    if not skip_episodic and encoder is not None:
        baselines.extend([
            (EpisodicMemory(encoder, k=5, use_gt=True), compute_v2),
            (EpisodicMemory(encoder, k=5, use_gt=False), compute_v2),
            (MMCCapMem(encoder, k=3), compute_v2),
        ])

    # B6
    if agent_fn is not None:
        baselines.append((LLMSelfAssess(), compute_v2))

    # B11-B12
    baselines.extend([
        (ZeroShotSelfKnow(), compute_v2),
        (GTSelfStats(), compute_v2),
    ])

    # B7-B10
    baselines.extend([
        (PlattOnline(), compute_classic),
        (HistBinOnline(), compute_classic),
        (BayesDomain(), compute_classic),
        (EMJoint(), compute_classic),
    ])

    # Inject agent_fn for LLM-based self_assess
    for b, _ in baselines:
        b.set_agent_fn(agent_fn)

    # Checkpoint support
    src = os.path.basename(session_file).replace(".json", "")
    ckpt_path = os.path.join(output_dir, f".checkpoint_{src}.json")
    per_session = defaultdict(list)
    start_si = 0

    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        per_session = defaultdict(list, {k: v for k, v in ckpt["per_session"].items()})
        start_si = ckpt["completed_sessions"]
        print(f"  Resuming from checkpoint: {start_si}/{n_sess} sessions done")

    # Run
    for si in range(start_si, n_sess):
        sess = data["sessions"][si]
        print(f"  Session {si + 1}/{n_sess}...", end=" ", flush=True)
        for b, runner in baselines:
            m = runner(b, sess)
            if m:
                per_session[b.name].append(m)
        print("done")

        # Save checkpoint after each session
        with open(ckpt_path, "w") as f:
            json.dump({"completed_sessions": si + 1,
                        "per_session": dict(per_session)}, f)

    # Aggregate
    print(f"\n  {'Baseline':<22} {'ECE':>7} {'Brier':>7} {'AUROC':>7} {'HR':>7} {'CBF':>7}")
    print(f"  {'-' * 60}")
    agg = {}
    for b, _ in baselines:
        rs = per_session[b.name]
        if not rs:
            continue

        def m(k, rs=rs):
            vs = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(vs)) if vs else None

        agg[b.name] = {
            "ECE": m("ECE"), "Brier": m("Brier"), "AUROC": m("AUROC"),
            "HR": m("HR"), "CBF": m("CBF"),
        }
        print(
            f"  {b.name:<22} {agg[b.name]['ECE']:>7.4f} {agg[b.name]['Brier']:>7.4f} "
            f"{agg[b.name]['AUROC']:>7.3f} {agg[b.name]['HR']:>7.4f} "
            f"{agg[b.name]['CBF']:>7.3f}"
        )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "session_file": session_file,
        "agent_model": model_name,
        "n_sessions": n_sess,
        "aggregate": {
            k: {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()}
            for k, v in agg.items()
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"all_baselines_{src}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # Remove checkpoint on success
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"  Saved: {out_path}")
    return out_path


def find_latest_sessions(results_dir):
    """For each model, find the latest session file (prefer v2 if exists)."""
    all_files = sorted(glob.glob(os.path.join(results_dir, "capbound_sessions_*.json")))
    model_files = {}
    for f in all_files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            model = d["metadata"].get("agent_model", "unknown")
            model_files[model] = f
        except Exception:
            continue
    return model_files


def main():
    ap = argparse.ArgumentParser(description="Run all 10 CapBoundary-Bench baselines")
    ap.add_argument("--session_file", help="Single session file to process")
    ap.add_argument("--batch", action="store_true",
                    help="Auto-detect and run all 13 models from results dir")
    ap.add_argument("--skip_episodic", action="store_true",
                    help="Skip B4a/B4b/B5 (avoid sentence-transformers)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if results already exist")
    ap.add_argument("--output_dir",
                    default="./HOME/yzs/LLM\u5fc3\u667a/experiments/results")
    args = ap.parse_args()

    if not args.session_file and not args.batch:
        ap.error("Provide --session_file or --batch")

    # Load encoder once
    encoder = None
    if not args.skip_episodic:
        print("Loading sentence encoder...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if args.batch:
        model_files = find_latest_sessions(args.output_dir)
        print(f"\nFound {len(model_files)} models:")
        for model, fpath in sorted(model_files.items()):
            print(f"  {model:<25} {os.path.basename(fpath)}")

        results = []
        for model, fpath in sorted(model_files.items()):
            # Skip if result already exists for this session file
            src = os.path.basename(fpath).replace(".json", "")
            existing = glob.glob(os.path.join(args.output_dir, f"all_baselines_{src}_*.json"))
            if existing and not args.force:
                print(f"  SKIP {model} (result exists: {os.path.basename(existing[-1])})")
                continue
            try:
                out = run_one(fpath, args.output_dir, args.skip_episodic, encoder)
                results.append(out)
            except Exception as e:
                print(f"  ERROR on {model}: {e}")
                import traceback; traceback.print_exc()

        print(f"\n{'=' * 70}")
        print(f"Batch complete: {len(results)}/{len(model_files)} models succeeded")
        print(f"{'=' * 70}")
    else:
        run_one(args.session_file, args.output_dir, args.skip_episodic, encoder)

    # Cleanup any vLLM servers we started
    if _vllm_procs:
        stop_vllm()


if __name__ == "__main__":
    main()
