"""Progress curves for the R005 reliability_dtu_macro (omp16) paired runs.

Reads whatever epochs exist in both arms' history.tsv and renders the three
loss aggregations side by side, with the variant-2 gene_macro runs as a
matched-seed reference on the axes they share.  Rerun any time; it always
reflects the current state of the runs.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "full": "runs/fabric_v2_full_reliability_dtu_macro_lrhold10_omp16_seed1103",
    "atac": "runs/fabric_v2_atac_reliability_dtu_macro_lrhold10_omp16_seed1103",
}
REF = {
    "full": "runs/fabric_v2_full_macro_seed1103",
    "atac": "runs/fabric_v2_atac_macro_seed1103",
}
COLOR = {"full": "#1b6ca8", "atac": "#d1495b"}

def read(run_dir, columns):
    with open(f"{run_dir}/history.tsv") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        column: [(int(r["epoch"]), float(r[column])) for r in rows if r.get(column)]
        for column in columns
    }

new = {c: read(d, ("validation_compatible_path_nll_reliability_dtu_macro",
                   "validation_compatible_path_nll_gene_macro",
                   "validation_compatible_path_nll")) for c, d in RUNS.items()}
ref = {c: read(d, ("validation_compatible_path_nll_gene_macro",
                   "validation_compatible_path_nll")) for c, d in REF.items()}

PANELS = [
    ("validation_compatible_path_nll_reliability_dtu_macro",
     "selection: reliability/DTU-weighted macro NLL", False),
    ("validation_compatible_path_nll_gene_macro",
     "unweighted gene-macro NLL (+ variant-2 ref, dotted)", True),
    ("validation_compatible_path_nll",
     "molecule-weighted micro NLL (+ variant-2 ref, dotted)", True),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (column, title, with_ref) in zip(axes, PANELS):
    for c in ("full", "atac"):
        series = new[c][column]
        ax.plot([e for e, _ in series], [v for _, v in series], "o-",
                color=COLOR[c], lw=1.8, ms=4.5, label=f"{c} (R005)")
        if with_ref:
            rseries = [(e, v) for e, v in ref[c][column] if e <= 10]
            ax.plot([e for e, _ in rseries], [v for _, v in rseries], ":",
                    color=COLOR[c], lw=1.5, alpha=0.7, label=f"{c} (variant 2)")
    ax.axvspan(0.5, 10.5, color="#f0e6c8", alpha=0.45, zorder=0)
    ax.set_xlabel("epoch")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[0].annotate("LR hold: 5e-5 through e10", (0.03, 0.03),
                 xycoords="axes fraction", fontsize=8, color="#8a7a30")
max_epoch = max(e for c in new for e, _ in new[c][PANELS[0][0]])
fig.suptitle(
    f"R005 reliability_dtu_macro (omp16, seed 1103) — progress through e{max_epoch}",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("outputs/analysis/rdtu_progress.png", dpi=150)
print("outputs/analysis/rdtu_progress.png")
