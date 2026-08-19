"""Top-1 movement e19->e30, and the DTU-stratified full-vs-atac NLL gap."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

COLOR = {"full": "#1b6ca8", "atac": "#d1495b"}
d = {}
for c in ("full", "atac"):
    for line in open(f"outputs/analysis/top1_{c}.jsonl"):
        r = json.loads(line); d[(c, r["epoch"])] = r

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.6))

# --- panel 1: top-1 at e19 vs e30, four definitions ---
keys = [("macro", "ont_top1_acc_macro"), ("count-weighted", "ont_top1_acc_count_weighted"),
        ("tie-aware", "ont_top1_acc_tie_aware"), ("unique-winner", "ont_unique_top1_acc")]
x = np.arange(len(keys)); w = 0.2
for i, (cond, ep, hatch) in enumerate([("full",19,""),("full",30,"///"),("atac",19,""),("atac",30,"///")]):
    vals = [d[(cond, ep)][k] for _, k in keys]
    ax1.bar(x + (i-1.5)*w, vals, w, label=f"{cond} e{ep}", color=COLOR[cond],
            alpha=0.55 if ep == 19 else 1.0, hatch=hatch, edgecolor="white", lw=0.6)
for i, (_, k) in enumerate(keys):
    for j, cond in enumerate(("full", "atac")):
        delta = d[(cond,30)][k] - d[(cond,19)][k]
        ax1.annotate(f"{delta:+.4f}", (x[i] + (j*2-1.5+0.5)*w, max(d[(cond,19)][k], d[(cond,30)][k])),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5,
                     color="#1a7f37" if delta > 0 else "#c8102e")
ax1.set_xticks(x); ax1.set_xticklabels([n for n, _ in keys])
ax1.set_ylabel("ONT top-1 accuracy"); ax1.set_ylim(0.45, 0.60)
ax1.set_title("ONT top-1 accuracy rises from e19 to e30 while KL worsens  (green = improved)", fontsize=11.5)
ax1.legend(ncol=4, fontsize=9); ax1.grid(alpha=0.25, axis="y")

# --- panel 2: DTU-decile gap, macro vs molecule-weighted ---
dtu = pd.read_excel("data/DTU_result_sorted.xlsx")[["gene_id", "DTU_score"]]
f = pd.read_csv("outputs/analysis/per_gene_full_e30.tsv", sep="\t")
a = pd.read_csv("outputs/analysis/per_gene_atac_e30.tsv", sep="\t")
m = f[f.nll_molecule_mass > 0].merge(a, on="gene_id", suffixes=("_f", "_a")).merge(dtu, on="gene_id")
m["delta"] = m.nll_weighted_sum_a / m.nll_molecule_mass_a - m.nll_weighted_sum_f / m.nll_molecule_mass_f
m["decile"] = pd.qcut(m.DTU_score, 10, labels=False, duplicates="drop")
g = m.groupby("decile")
macro = g["delta"].mean()
weighted = g.apply(lambda t: (t.nll_weighted_sum_a.sum() - t.nll_weighted_sum_f.sum())
                   / t.nll_molecule_mass_f.sum(), include_groups=False)
dec = np.arange(1, len(macro) + 1)
ax2.bar(dec - 0.19, macro.values, 0.38, label="macro (per-gene equal weight)", color="#4a7c59")
ax2.bar(dec + 0.19, weighted.values, 0.38, label="molecule-weighted", color="#e08a3c")
ax2.axhline(0, color="black", lw=0.9)
ax2.set_xticks(dec); ax2.set_xlabel("DTU decile  (1 = most stable, 10 = most switching)")
ax2.set_ylabel("NLL gap: atac − full\n(positive = full better)")
ax2.set_title("No monotone DTU trend — and the two weightings disagree in sign (e30)", fontsize=11.5)
ax2.legend(fontsize=9); ax2.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig("outputs/analysis/top1_dtu.png", dpi=150)
print("outputs/analysis/top1_dtu.png")
