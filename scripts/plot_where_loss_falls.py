"""Where the training loss actually improved between e19 and e30 (full model)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

dtu = pd.read_excel("data/DTU_result_sorted.xlsx")[["gene_id","DTU_score"]]
def load(ep):
    d = pd.read_csv(f"outputs/analysis/per_gene_full_e{ep}.tsv", sep="\t")
    d = d[d.nll_molecule_mass > 0].copy()
    d["nll"] = d.nll_weighted_sum / d.nll_molecule_mass
    return d
m = load(19).merge(load(30), on="gene_id", suffixes=("_19","_30")).merge(dtu, on="gene_id")
m["improve"] = m.nll_19 - m.nll_30

bins = [("DTU = 0", m.DTU_score==0), ("0–0.1", (m.DTU_score>0)&(m.DTU_score<=0.1)),
        ("0.1–0.3", (m.DTU_score>0.1)&(m.DTU_score<=0.3)),
        ("0.3–0.6", (m.DTU_score>0.3)&(m.DTU_score<=0.6)), ("> 0.6", m.DTU_score>0.6)]
lab, mac, wei, share = [], [], [], []
for name, mask in bins:
    g = m[mask]; lab.append(name)
    mac.append(g.improve.mean())
    wei.append((g.nll_weighted_sum_19.sum()-g.nll_weighted_sum_30.sum())/g.nll_molecule_mass_19.sum())
    share.append(g.nll_molecule_mass_19.sum()/m.nll_molecule_mass_19.sum()*100)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios":[2.1,1]})
x = np.arange(len(lab))
ax.bar(x-0.2, mac, 0.4, label="macro (per-gene equal weight)", color="#4a7c59")
ax.bar(x+0.2, wei, 0.4, label="molecule-weighted (= the training objective)", color="#e08a3c")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(lab)
ax.set_xlabel("DTU score  (left = stable, right = switching)")
ax.set_ylabel("NLL improvement e19 → e30\n(positive = got better)")
ax.set_title("The loss only improved on stable genes — switching genes got worse", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
for i,(v,s) in enumerate(zip(mac, share)):
    ax.annotate(f"{s:.1f}% of\nmolecules", (i, min(v, wei[i])), textcoords="offset points",
                xytext=(0,-30), ha="center", fontsize=7.5, color="#555")

ax2.bar(["gradient\nweight", "gene\ncount"], [1.2, 16.4], color=["#c8102e", "#4a7c59"])
ax2.set_ylabel("share of total (%)")
ax2.set_title("top_DTU genes:\n16% of genes, 1.2% of the objective", fontsize=11)
ax2.grid(alpha=0.25, axis="y")
for i,v in enumerate([1.2, 16.4]):
    ax2.annotate(f"{v}%", (i, v), textcoords="offset points", xytext=(0,4), ha="center", fontsize=10)
fig.tight_layout(); fig.savefig("outputs/analysis/where_loss_falls.png", dpi=150)
print("outputs/analysis/where_loss_falls.png")
