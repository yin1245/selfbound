# CapBoundary-Bench v1.0.0

> **Anonymous repository for NeurIPS 2026 Datasets & Benchmarks Track double-blind review.**

A diagnostic benchmark for LLM-agent self-knowledge under noisy interactive feedback.
CapBoundary-Bench operationalizes three competing hypotheses about where agent self-knowledge
fails — H1 output channel, H2 prior metacognition, H3 noise-robust online inference — through
a five-level information ladder (L0 raw confidence → L1 zero-shot prior → L2 noisy-feedback
learners → L3 binary-label oracle → L4 full-statistic oracle).

## Headline numbers

- **15 models from 7 families**, **47K+ interactions** on a 50×50 protocol (50 sessions × 50 turns)
- Cross-family LLM-simulated user with **75% in-expertise / 46% out-of-expertise** reliability
- Headline metric: **Capability Boundary Fidelity (CBF)** — L1 distance between estimated and empirical per-domain accuracy
- **H1 does not bind** (GT-SelfStats ≥ 0.99 on every model)
- **H2 splits by family** — zero-shot prior wins on GPT (frontier mean 0.640) but falls below NoMemory on all 3 Claude models
- **H3 partially refuted** — second-order Dawid–Skene-asym (0.755), GLAD (0.712, 6/6 frontier wins, p=0.031), and in-context LSA (0.748) substantially exceed L1 prior

## Repository layout

```
capbound-bench/
├── croissant.json            # Validated Croissant 1.0 metadata
├── MANIFEST.txt              # Bundle stats
├── LICENSE-CODE              # MIT (covers code/)
├── LICENSE-DATA              # CC-BY-4.0 (covers data/)
├── README.md                 # this file
├── REPRODUCE.md              # End-to-end reproduction instructions
│
├── data/
│   ├── sessions/             # 15 raw 50×50 session JSONL files (one per model)
│   ├── aggregates/           # Per-(model, baseline) aggregate metric files
│   └── heldout_n6_extension.json   # MMLU-Pro & BBH contamination control on n=6 frontier
│
├── code/
│   ├── baselines/            # 10 baseline implementations + L2 self_assess patch
│   └── scripts/              # Evaluation drivers, contamination-control runners
│
└── paper/
    ├── CapBoundary-Bench.pdf # Anonymized submission PDF
    ├── main.tex              # LaTeX source
    ├── references.bib        # Bibliography
    └── figures/              # 10 figures
```

## Quickstart

### 1. Inspect a single session

```python
import json
with open("data/sessions/capbound_sessions_gpt-5.5_20260502_201014.json") as f:
    d = json.load(f)
print(d["metadata"]["agent_model"], "—", d["metadata"]["n_sessions"], "sessions")
turn = d["sessions"][0]["log"][0]
print(turn["domain"], turn["question"][:80], "→ agent:", turn["agent_answer"], "conf:", turn["agent_confidence"])
```

### 2. Recompute CBF for a (model, baseline) pair

```python
import json, numpy as np

with open("data/aggregates/all_baselines_capbound_sessions_gpt-5.5_20260430_061851_20260501_031654.json") as f:
    agg = json.load(f)
print("ZeroShot-SK CBF:", agg["aggregate"]["ZeroShot-SK"]["CBF"])
print("LLMSelfAssess CBF:", agg["aggregate"]["LLMSelfAssess"]["CBF"])
```

### 3. Run the held-out contamination control

```bash
# Requires ANTHROPIC_API_KEY and OPENAI_API_KEY env vars
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
python code/scripts/eval_heldout_n6.py
```

## Citing

```bibtex
@inproceedings{capboundary2026,
  title = {CapBoundary-Bench: A Diagnostic Benchmark for LLM Agent Self-Knowledge under Noisy Feedback},
  author = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year = {2026}
}
```

## Licenses

- **Code** under `code/`: MIT — see `LICENSE-CODE`
- **Data** under `data/`: CC-BY-4.0 — see `LICENSE-DATA`
- **MMLU**: original MMLU license (Hendrycks et al. 2021)
- **MMLU-Pro**: TIGER-Lab MMLU-Pro license (Wang et al. 2024)
- **BBH**: BIG-Bench license (Suzgun et al. 2022)

## Reproducibility

End-to-end reproduction of the headline 50×50 corpus on 15 models takes
**~$200–400 USD** in commercial-API calls plus open-weights compute on a single
A100 / RTX A6000 workstation. See `REPRODUCE.md` for exact commands and
estimated costs per model.

API model versions are pinned with response-time access dates; two open-weights
models (Mistral-7B, Llama-3-8B) are pinned to HuggingFace snapshot commits and
serve as the time-stable replication anchor.

## Anonymization

All references to authors, institutions, API endpoints, and personal paths
have been removed from this bundle to comply with NeurIPS double-blind policy.
