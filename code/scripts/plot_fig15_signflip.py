"""Generate fig15_signflip.pdf with 19 models.
4 cols: MMLU(estim), GSM8K, HumanEval, BFCL — confidence-accuracy gap (pp)
Star: model with sign-flip across 3 same-protocol extension tasks (GSM8K/HumanEval/BFCL).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Order: weak → strong (matching prior fig design)
MODELS = [
    "Mistral-7B", "Llama-3-8B", "GLM-5", "DeepSeek-V3", "Qwen-Turbo",
    "Qwen2.5-7B", "Qwen-Max", "Qwen-Plus", "Qwen3.5-27B",
    "GPT-5.4-mini", "GPT-5.4", "GPT-5.5",
    "Claude-Opus-4-7", "Claude-Sonnet-4-6", "Claude-Opus-4-6",
    "GLM-5.1", "DeepSeek-V4-Pro", "Qwen3.6-Max-Preview", "Qwen3.6-Plus",
]

# MMLU Gap (pp) — confidence - accuracy from 50x50 session data (mean over all turns)
MMLU_GAP = {
    "Mistral-7B": 62, "Llama-3-8B": 41, "GLM-5": 20, "DeepSeek-V3": 14,
    "Qwen-Turbo": 20, "Qwen2.5-7B": 26, "Qwen-Max": 25, "Qwen-Plus": 18,
    "Qwen3.5-27B": 19, "GPT-5.4-mini": 20, "GPT-5.4": 13, "GPT-5.5": 3,
    "Claude-Opus-4-7": 27, "Claude-Sonnet-4-6": 31, "Claude-Opus-4-6": 31,
    "GLM-5.1": 21, "DeepSeek-V4-Pro": 6, "Qwen3.6-Max-Preview": 22, "Qwen3.6-Plus": 9,
}
# GSM8K / HumanEval / BFCL Gap (pp) — verbatim from paper Tab 8 (tab:fourtask_overview)
GSM_GAP = {
    "Mistral-7B": 31, "Llama-3-8B": -3, "GLM-5": 1, "DeepSeek-V3": -21,
    "Qwen-Turbo": -33, "Qwen2.5-7B": -18, "Qwen-Max": -18, "Qwen-Plus": -11,
    "Qwen3.5-27B": -1, "GPT-5.4-mini": -2, "GPT-5.4": 0.3, "GPT-5.5": -1,
    "Claude-Opus-4-7": -10, "Claude-Sonnet-4-6": -4, "Claude-Opus-4-6": -5,
    "GLM-5.1": -46, "DeepSeek-V4-Pro": -47, "Qwen3.6-Max-Preview": 20, "Qwen3.6-Plus": 18,
}
HE_GAP = {
    "Mistral-7B": 52, "Llama-3-8B": 44, "GLM-5": -49, "DeepSeek-V3": -30,
    "Qwen-Turbo": -1, "Qwen2.5-7B": -4, "Qwen-Max": -31, "Qwen-Plus": -48,
    "Qwen3.5-27B": -35, "GPT-5.4-mini": -25, "GPT-5.4": -36, "GPT-5.5": -31,
    "Claude-Opus-4-7": -34, "Claude-Sonnet-4-6": -40, "Claude-Opus-4-6": -17,
    "GLM-5.1": -29, "DeepSeek-V4-Pro": -44, "Qwen3.6-Max-Preview": -23, "Qwen3.6-Plus": 15,
}
BFCL_GAP = {
    "Mistral-7B": 15, "Llama-3-8B": 36, "GLM-5": 32, "DeepSeek-V3": 5,
    "Qwen-Turbo": 22, "Qwen2.5-7B": 18, "Qwen-Max": 17, "Qwen-Plus": 10,
    "Qwen3.5-27B": 24, "GPT-5.4-mini": 15, "GPT-5.4": 25, "GPT-5.5": -4,
    "Claude-Opus-4-7": -9, "Claude-Sonnet-4-6": 17, "Claude-Opus-4-6": 9,
    "GLM-5.1": -17, "DeepSeek-V4-Pro": -20, "Qwen3.6-Max-Preview": 24, "Qwen3.6-Plus": 27,
}

COLS = ["MMLU\n(estim)", "GSM8K", "HumanEval", "BFCL"]
data = np.array([
    [MMLU_GAP[m], GSM_GAP[m], HE_GAP[m], BFCL_GAP[m]] for m in MODELS
], dtype=float)

# Sign-flip: a model is starred iff it has a sign reversal across the 3 protocol-controlled
# extension tasks (GSM8K, HumanEval, BFCL). Treat values within ±1pp as borderline non-flip.
def has_signflip(row3):
    signs = [(1 if v > 1 else (-1 if v < -1 else 0)) for v in row3]
    return any(s == 1 for s in signs) and any(s == -1 for s in signs)
sign_flip = [has_signflip([GSM_GAP[m], HE_GAP[m], BFCL_GAP[m]]) for m in MODELS]
n_flip = sum(sign_flip)
print(f"Sign-flip count: {n_flip}/{len(MODELS)}")

# Diverging colormap — red high (overconfident), blue low (conservative)
cmap = LinearSegmentedColormap.from_list(
    "rwb", [(0, "#2166AC"), (0.5, "#FFFFFF"), (1, "#B2182B")], N=256)
vmin, vmax = -50, 50

fig, ax = plt.subplots(figsize=(5.6, 6.8))
im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Cell text
for i in range(len(MODELS)):
    for j in range(len(COLS)):
        v = data[i, j]
        # Choose text color based on cell value (white if very intense)
        text_color = "white" if abs(v) > 30 else "black"
        ax.text(j, i, f"{v:+.0f}".replace("+", "+") if v != 0 else "0",
                ha="center", va="center", fontsize=7.5, color=text_color)

# Star markers for sign-flip
for i, sf in enumerate(sign_flip):
    if sf:
        ax.text(len(COLS) + 0.05, i, "$\\bigstar$", ha="left", va="center",
                fontsize=11, color="#B2182B")

ax.set_xticks(range(len(COLS)))
ax.set_xticklabels(COLS, fontsize=8.5)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS, fontsize=8)
ax.set_xlim(-0.5, len(COLS) - 0.5 + 0.6)
ax.set_title(f"Cross-task Gap sign matrix\n($\\bigstar$ = sign flip in protocol-controlled subset, {n_flip}/{len(MODELS)})",
             fontsize=10, pad=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.06, shrink=0.8)
cbar.set_label("Confidence $-$ Accuracy (pp)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Hide spines on heatmap edges
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(top=False, right=False, length=0)

plt.tight_layout()
plt.savefig("/tmp/fig15_signflip_n19.pdf", bbox_inches="tight", dpi=300)
plt.savefig("/tmp/fig15_signflip_n19.png", bbox_inches="tight", dpi=200)
print(f"Saved /tmp/fig15_signflip_n19.pdf  (sign-flip = {n_flip}/{len(MODELS)})")
