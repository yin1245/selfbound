"""Fig 4 (fig9_spectrum.pdf): 19-model capability spectrum.
Bars only, NO scatter markers above bars; family color in bars + small color legend.
Bottom panel keeps EM/LSA/ZS winning-baseline labels.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.4, "lines.linewidth": 0.9, "patch.linewidth": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "/tmp/ladder_n19_polish"
os.makedirs(OUT, exist_ok=True)

BK = "#1a1a1a"
GRID = "#E0E0E0"

# Family color (no marker shapes)
FAM_COLOR = {
    "Mistral":   "#C0392B",
    "Meta":      "#2874A6",
    "GLM":       "#229954",
    "DeepSeek":  "#7D3C98",
    "Qwen":      "#B7950B",
    "OpenAI":    "#117A65",
    "Anthropic": "#D35400",
}

MODELS = [
    "Mistral-7B", "Llama-3-8B", "GLM-5", "DeepSeek-V3", "Qwen-Turbo",
    "Qwen2.5-7B", "Qwen-Max", "Qwen-Plus", "Qwen3.5-27B", "GLM-5.1",
    "GPT-5.4-mini", "GPT-5.4", "GPT-5.5", "Claude-Opus-4-7",
    "Claude-Sonnet-4-6", "Claude-Opus-4-6", "DeepSeek-V4-Pro",
    "Qwen3.6-Max-Preview", "Qwen3.6-Plus",
]
MMLU_ACC = [42, 55, 55, 62, 64, 65, 71, 79, 79, 76,
            72, 82, 94, 69, 86, 90, 90, 92, 93]
FAM = ["Mistral", "Meta", "GLM", "DeepSeek", "Qwen",
       "Qwen", "Qwen", "Qwen", "Qwen", "GLM",
       "OpenAI", "OpenAI", "OpenAI", "Anthropic", "Anthropic",
       "Anthropic", "DeepSeek", "Qwen", "Qwen"]

NOMEM = [0.729, 0.720, 0.700, 0.681, 0.708, 0.744, 0.704, 0.683, 0.731, 0.600,
         0.687, 0.649, 0.570, 0.557, 0.603, 0.594, 0.575, 0.580, 0.617]
ZS    = [0.729, 0.711, 0.700, 0.673, 0.730, 0.663, 0.766, 0.735, 0.731, 0.600,
         0.769, 0.795, 0.859, 0.482, 0.472, 0.460, 0.849, 0.580, 0.617]
LSA   = [0.729, 0.549, 0.700, 0.721, 0.703, 0.701, 0.781, 0.435, 0.731, 0.600,
         0.763, 0.781, 0.902, 0.641, 0.720, 0.680, 0.675, 0.580, 0.617]
UF    = [0.720, 0.715, 0.689, 0.681, 0.697, 0.738, 0.660, 0.686, 0.712, 0.605,
         0.687, 0.668, 0.601, 0.593, 0.640, 0.618, 0.593, 0.586, 0.626]
PLATT = [0.717, 0.724, 0.696, 0.683, 0.712, 0.745, 0.684, 0.695, 0.720, 0.609,
         0.701, 0.675, 0.612, 0.577, 0.617, 0.597, 0.609, 0.599, 0.636]
HBIN  = PLATT[:]
BAYES = [0.731, 0.730, 0.711, 0.696, 0.724, 0.755, 0.702, 0.700, 0.740, 0.613,
         0.706, 0.674, 0.603, 0.571, 0.613, 0.603, 0.602, 0.598, 0.638]
EM    = [0.732, 0.731, 0.714, 0.699, 0.728, 0.757, 0.705, 0.701, 0.746, 0.615,
         0.706, 0.677, 0.605, 0.570, 0.609, 0.604, 0.605, 0.601, 0.642]

BLABELS = ["NoMem", "ZS", "LSA", "UF", "Platt", "HistBin", "Bayes", "EM"]
def best_label(i):
    cols = [NOMEM[i], ZS[i], LSA[i], UF[i], PLATT[i], HBIN[i], BAYES[i], EM[i]]
    return BLABELS[int(np.argmax(cols))]
BEST = [best_label(i) for i in range(19)]

order = np.argsort(MMLU_ACC)
ms_o   = [MODELS[i]   for i in order]
acc_o  = [MMLU_ACC[i] for i in order]
fam_o  = [FAM[i]      for i in order]
best_cbf = [max(NOMEM[i], ZS[i], LSA[i], UF[i], PLATT[i], HBIN[i],
                BAYES[i], EM[i]) for i in range(19)]
best_cbf_o = [best_cbf[i] for i in order]
best_lbl_o = [BEST[i]     for i in order]
bar_colors = [FAM_COLOR[f] for f in fam_o]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 4.4), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 1.0],
                                           "hspace": 0.06})
x = np.arange(19)

# Top panel: MMLU bars (color only, NO scatter markers above)
a1.bar(x, acc_o, width=0.66, color=bar_colors, edgecolor="white", lw=0.5, zorder=3)
for i, a in enumerate(acc_o):
    a1.text(i, a - 4, str(a), ha="center", fontsize=6, color="white", fontweight="bold")
a1.set_ylabel("MMLU Accuracy (%)")
a1.set_ylim(30, 100)
a1.yaxis.set_major_locator(plt.MultipleLocator(20))

# Family color legend (color patches, no marker shapes)
legend_handles = [Patch(facecolor=col, edgecolor="white", label=fam)
                  for fam, col in FAM_COLOR.items()]
a1.legend(handles=legend_handles, fontsize=6, ncol=4, loc="upper left",
          columnspacing=0.8, handletextpad=0.4, frameon=True, edgecolor="#cccccc")
a1.text(-0.05, 1.05, "(a)", transform=a1.transAxes,
        fontsize=10, fontweight="bold", va="top", ha="right", color=BK)

# Bottom panel: best CBF bars + best-baseline label above bar
a2.bar(x, best_cbf_o, width=0.66, color=bar_colors, edgecolor="white", lw=0.5, zorder=3)
for i, (v, bl) in enumerate(zip(best_cbf_o, best_lbl_o)):
    a2.text(i, v - 0.04, f"{v:.2f}", ha="center", fontsize=5.5,
            color="white", fontweight="bold")
    a2.text(i, v + 0.012, bl, ha="center", fontsize=5.4, color=BK, style="italic")
a2.set_ylabel("Best CBF (non-oracle)")
a2.set_ylim(0.40, 1.00)
a2.set_xticks(x)
a2.set_xticklabels(ms_o, rotation=42, ha="right", fontsize=6.0)
nomem_mean = float(np.mean(NOMEM))
a2.axhline(nomem_mean, color=BK, linestyle=":", linewidth=0.5, alpha=0.6)
a2.text(0.2, nomem_mean + 0.012, f"L0 NoMem mean (n=19, {nomem_mean:.3f})",
        fontsize=6, color=BK)
a2.text(-0.05, 1.05, "(b)", transform=a2.transAxes,
        fontsize=10, fontweight="bold", va="top", ha="right", color=BK)

plt.tight_layout()
plt.savefig(f"{OUT}/fig9_spectrum.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT}/fig9_spectrum.png", bbox_inches="tight", dpi=200)
print(f"  Saved {OUT}/fig9_spectrum.pdf  (no scatter markers; n=19; NoMem mean={nomem_mean:.4f})")
