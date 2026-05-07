# LADDER data/

This directory contains the released LADDER benchmark data.

## Files

```
data/
├── sessions/                         # 15 files, raw 50×50 turn JSONL per agent_model
├── aggregates/                       # 15 files, per-baseline aggregate metrics per agent_model
├── heldout_n6_extension.json         # MMLU-Pro + BBH controls, original 6 GPT/Claude frontier
└── heldout_n3_new_frontier.json      # MMLU-Pro + BBH controls, 3 new frontier extensions
```

## Held-out contamination control (MMLU-Pro + BBH)

The two held-out files together cover the **n=9 frontier subset** used in the
contamination control study (paper Tables 6 and 7):

| File | Models | Tasks |
|---|---|---|
| `heldout_n6_extension.json` | GPT-5.5, Claude-Opus-4-7 (representative bundled rows from the original GPT/Claude n=6 set) | MMLU-Pro (10 cats × 15 q) + BBH (10 tasks × 15 q) |
| `heldout_n3_new_frontier.json` | DeepSeek-V4-Pro, Qwen3.6-Max-Preview, Qwen3.6-Plus | MMLU-Pro (10 cats × 15 q) + BBH (10 tasks × 15 q) |

Per-model aggregate per-category accuracies and ZS-SK self-estimates for the full
6 GPT/Claude subset (GPT-5.4, GPT-5.4-mini, GPT-5.5, Claude-Opus-4-6,
Claude-Sonnet-4-6, Claude-Opus-4-7) appear in the paper Tables 6 and 7; the bundled
JSON contains the two representative rows above plus the 3 new frontier extensions.

Each held-out entry has the schema:
```
<model_id>: {
  "mmlu_pro": {
    "per_cat": {<cat>: {"acc": 0.93, "correct": 14, "n": 15}},
    "zs":      {<cat>: 0.85},
    "cbf":     0.937,
    "n_cats_matched": 10,
    "failures": 0
  },
  "bbh": {
    "per_task": {<task>: {"acc": ..., "correct": ..., "n": ...}},
    "zs":       {<task>: ...},
    "cbf":      0.901,
    "n_tasks_matched": 10,
    "failures": 0
  }
}
```

## Sessions

Each `data/sessions/capbound_sessions_<model_id>_<timestamp>.json` contains 50
sessions × 50 turns for one (agent_model). Schema for each turn:

| Field | Type | Description |
|---|---|---|
| `turn` | int | Turn index 1..50 |
| `domain` | str | MMLU subject |
| `tier` | str | strong / mid / weak (per-model stratification) |
| `question` | str | MMLU question text |
| `gt_answer` | str | A/B/C/D ground truth |
| `agent_answer` | str | Agent's selected letter |
| `agent_confidence` | float | Self-reported [0,1] |
| `agent_abstain` | bool | |
| `gt_correct` | bool | gt_answer == agent_answer |
| `user_response` | str | Cross-family LLM simulated user feedback |
| `in_user_expertise` | bool | True iff domain ∈ user persona expertise (75% reliability) else False (46%) |

## Aggregates

Each `data/aggregates/all_baselines_capbound_sessions_<model>_*_<computed_at>.json`
holds per-baseline scalar metrics (CBF, ECE, Brier, AUROC, HR) for one
(agent_model). Twelve baselines per file: NoMemory, ZeroShot-SK, LLMSelfAssess,
UF-PDC, Platt-Online, HistBin-Online, Bayes-Domain, EM-Joint, OraclePDC,
GT-SelfStats, DS-asym, GLAD.
