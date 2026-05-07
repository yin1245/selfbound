"""Fig 9 (fig14_second_order.pdf): DS-asym + GLAD vs L1 ZS-SK across n=9 frontier.
Updated to n=9 (added DSv4-Pro, Q3.6-Max, Q3.6-Plus); legend overlap with headline fixed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "/tmp/ladder_n19_polish"
os.makedirs(OUT, exist_ok=True)

GRID = "#CCCCCC"
C_L0 = "#A8A8A8"
C_L1 = "#4C72B0"
C_L2_GLAD = "#CCB974"
C_L2_DS  = "#DD8452"

# n=9 frontier (data verified from glad_ds_all_models_20260506_171852.json + agg files)
models = ["GPT-5.4-mini", "GPT-5.4", "GPT-5.5",
          "Cl-Sonnet-4-6", "Cl-Opus-4-6", "Cl-Opus-4-7",
          "DSv4-Pro", "Q3.6-Max", "Q3.6-Plus"]
nomem_vals = [0.687, 0.649, 0.570, 0.603, 0.594, 0.557, 0.575, 0.580, 0.617]
zs_vals    = [0.769, 0.795, 0.859, 0.472, 0.460, 0.482, 0.849, 0.580, 0.617]
glad_vals  = [0.797, 0.843, 0.909, 0.565, 0.557, 0.600, 0.908, 0.644, 0.857]
ds_vals    = [0.797, 0.815, 0.834, 0.671, 0.734, 0.676, 0.867, 0.678, 0.835]

zs_mean   = float(np.mean(zs_vals))
ds_mean   = float(np.mean(ds_vals))
glad_mean = float(np.mean(glad_vals))
print(f"n=9 frontier means: ZS={zs_mean:.3f}, DS-asym={ds_mean:.3f}, GLAD={glad_mean:.3f}")
n_glad_wins = sum(1 for g, z in zip(glad_vals, zs_vals) if g > z)
print(f"GLAD vs ZS: {n_glad_wins}/{len(models)}")

fig, ax = plt.subplots(figsize=(9.0, 4.6))

x = np.arange(len(models))
w = 0.20
ax.bar(x - 1.5*w, nomem_vals, w, label="L0 NoMem",            color=C_L0,      edgecolor="black", linewidth=0.4)
ax.bar(x - 0.5*w, zs_vals,    w, label="L1 ZS-SK",            color=C_L1,      edgecolor="black", linewidth=0.4)
ax.bar(x + 0.5*w, glad_vals,  w, label="L2 GLAD (2nd-order)", color=C_L2_GLAD, edgecolor="black", linewidth=0.4)
ax.bar(x + 1.5*w, ds_vals,    w, label="L2 DS-asym (2nd-order)", color=C_L2_DS, edgecolor="black", linewidth=0.4)

# Reference lines (just lines, NO label in legend — label them inline)
ax.axhline(zs_mean, color=C_L1, linestyle="--", linewidth=0.7, alpha=0.7)
ax.axhline(ds_mean, color=C_L2_DS, linestyle="--", linewidth=0.7, alpha=0.7)
# Inline annotations on the right side of each ref line, vertically separated
ax.text(len(models) - 0.45, zs_mean - 0.020, f"L1 ZS frontier mean ({zs_mean:.3f})",
        fontsize=7.5, color=C_L1, ha="right", va="top")
ax.text(len(models) - 0.45, ds_mean + 0.005, f"DS-asym frontier mean ({ds_mean:.3f})",
        fontsize=7.5, color=C_L2_DS, ha="right", va="bottom")

# Legend in upper-left INSIDE the plot area (no overlap with headline)
ax.legend(loc="upper left", ncol=2, fontsize=8.5,
          frameon=True, edgecolor="#cccccc", framealpha=0.95,
          bbox_to_anchor=(0.005, 0.99), borderaxespad=0.3)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("CBF")
ax.set_ylim(0.40, 1.00)
ax.set_yticks(np.arange(0.4, 1.01, 0.1))
ax.set_axisbelow(True)
ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

# Headline ABOVE plot, well clear of legend
ax.text(0.5, 1.06,
        f"GLAD beats L1 ZS-SK on $\\mathbf{{{n_glad_wins}/{len(models)}}}$ frontier models  "
        "(one-sided binomial $p{=}0.002$)",
        transform=ax.transAxes,
        fontsize=10.5, ha="center", color="black",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF8DC",
                  edgecolor="gray", linewidth=0.5))

plt.tight_layout()
plt.savefig(f"{OUT}/fig14_second_order.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT}/fig14_second_order.png", bbox_inches="tight", dpi=180)
print(f"  Saved {OUT}/fig14_second_order.pdf")
