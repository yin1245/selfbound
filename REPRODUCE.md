# Reproducing CapBoundary-Bench

End-to-end reproduction of the headline `50×50` MMLU corpus on 15 models.

## Quick Start: Recompute aggregate metrics from released sessions

The fastest path is to start from the released session JSONLs (no API calls needed) and recompute the L0–L4 baseline metrics:

```bash
# Install dependencies
pip install numpy scipy scikit-learn sentence-transformers

# For a single model — recomputes 10 baselines on the existing session file
python code/scripts/run_all_baselines_v4.py \
    --session_file data/sessions/capbound_sessions_gpt-5.5_20260502_201014.json \
    --output_dir /tmp/

# For all 15 models in batch
python code/scripts/run_all_baselines_v4.py \
    --batch \
    --output_dir /tmp/
```

Cost: **0 USD** (no API calls). Reproduces the L0/L1/L2 baselines exactly. L3/L4 oracles are recomputed from the latent ground truth fields in each session.

## Full pipeline: Regenerate sessions from scratch

This requires API access to the model providers. Estimated cost: **~$200–400 USD** at 2026 prices for the 13 commercial-API models. Two open-weights models (Mistral-7B, Llama-3-8B) require local GPU compute (~30 min each on a single A6000).

### 1. Generate session traces

```bash
# API models (Qwen, GLM, DeepSeek via DashScope; Claude/GPT via proxy)
for model in qwen-turbo qwen-max qwen-plus qwen3.5-27b glm-5 deepseek-v3.2 \
             gpt-5.4 gpt-5.4-mini gpt-5.5 \
             claude-sonnet-4-6 claude-opus-4-6 claude-opus-4-7; do
    python code/scripts/capbound_session_api.py \
        --agent_name $model \
        --n_sessions 50 \
        --n_workers 4 \
        --output_dir data/sessions/
done

# Open-weights models (require vLLM server)
# Start vLLM server first: vllm serve /path/to/llama3-8b-instruct --port 8100
python code/scripts/capbound_session.py --n_sessions 50  # Llama-3-8B
# Edit AGENT_URL/AGENT_MODEL_PATH in capbound_session.py for Mistral-7B
```

### 2. Compute baseline metrics

```bash
python code/scripts/run_all_baselines_v4.py --batch --output_dir data/aggregates/
```

### 3. Held-out contamination control (n=6 frontier)

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
python code/scripts/eval_heldout_n6.py
```

## Configuration

API endpoints and credentials are configured via environment variables in `code/scripts/capbound_session_api.py`. The released code has all keys redacted (`sk-REDACTED`); supply your own:

- `DASHSCOPE_KEY` for Qwen/GLM/DeepSeek
- Proxy endpoint for Claude/GPT via the `claude-zhongzhuan.cloud` (Claude) and `lucen.cc` (GPT) gateways

## Versioning

| Version | Date | Notes |
|---------|------|-------|
| 1.0.0   | 2026-04-15 | Initial submission bundle (10×50 protocol on frontier, 50×50 on others) |
| 1.1.0   | 2026-05-04 | Uniform 50×50 across all 15 models; v18b LSA/ZS rerun for frontier |
| 1.1.2   | 2026-05-04 | Paper revision: H3 wording unified; "Self-Boundary Collapse" phenomenon name; benchmark-artifact paragraph; "in our protocol" qualifiers |

## Cost breakdown (2026-04-15 prices, USD)

| Model | Sessions × turns | Approx cost |
|-------|------------------|-------------|
| Qwen-Turbo | 50×50 | $5 |
| Qwen-Max | 50×50 | $25 |
| Qwen-Plus | 50×50 | $15 |
| Qwen3.5-27B | 50×50 | $20 |
| Qwen2.5-7B | 50×50 | $5 |
| GLM-5 | 50×50 | $20 |
| DeepSeek-V3.2 | 50×50 | $30 |
| GPT-5.4 | 50×50 | $40 |
| GPT-5.4-mini | 50×50 | $10 |
| GPT-5.5 | 50×50 | $50 |
| Claude-Sonnet-4-6 | 50×50 | $30 |
| Claude-Opus-4-6 | 50×50 | $60 |
| Claude-Opus-4-7 | 50×50 | $70 |
| Llama-3-8B (vLLM) | 50×50 | GPU only |
| Mistral-7B (vLLM) | 50×50 | GPU only |
| **Extension suite** (GSM8K + HumanEval + BFCL × 13 API models) | 1-shot | $50 |
| **Total** | ~47K interactions | **~$430** |

Lite v2 protocol (10×50 instead of 50×50) reproduces the headline ordering at ~1/5 of the cost.

## Contact

Anonymous repository for NeurIPS 2026 D&B Track double-blind review. Authors will be revealed in the camera-ready release.
