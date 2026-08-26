"""Plot complete validation-loss trajectories for the continued resident run."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


RUNS = {
    "Full": Path(
        "runs/fabric_v2_full_rdtu_a0_shared6g2x_resident_continue100_seed1103"
    ),
    "ATAC": Path(
        "runs/fabric_v2_atac_rdtu_a0_shared6g2x_resident_continue100_seed1103"
    ),
    "RBP": Path(
        "runs/fabric_v2_rbp_rdtu_a0_shared6g2x_resident_continue100_seed1103"
    ),
}
COLORS = {"Full": "#0072B2", "ATAC": "#D55E00", "RBP": "#009E73"}
METRICS = (
    (
        "validation_compatible_path_nll_reliability_dtu_macro",
        "Checkpoint-selection loss",
        "Reliability-weighted gene-macro NLL",
    ),
    (
        "validation_compatible_path_nll_gene_macro",
        "Unweighted gene-macro loss",
        "Gene-macro NLL",
    ),
    (
        "validation_compatible_path_nll",
        "Molecule-weighted loss",
        "Compatible-path NLL",
    ),
)
CONTINUATION_EPOCH = 30
OUTPUT = Path("outputs/analysis/shared6g2x_resident_continue100_loss.png")
PDF_OUTPUT = OUTPUT.with_suffix(".pdf")


def read_history(run_dir: Path) -> list[dict[str, str]]:
    history_path = run_dir / "history.tsv"
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"training history has no completed epochs: {history_path}")
    return rows


def main() -> None:
    histories = {condition: read_history(run_dir) for condition, run_dir in RUNS.items()}
    completed = {
        condition: int(rows[-1]["epoch"]) for condition, rows in histories.items()
    }
    statuses = {
        condition: (
            "finished" if (run_dir / "metrics.tsv").is_file() else "running"
        )
        for condition, run_dir in RUNS.items()
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.2),
        constrained_layout=True,
        sharex=True,
    )

    for ax, (metric, title, ylabel) in zip(axes, METRICS, strict=True):
        for condition, rows in histories.items():
            epochs = [int(row["epoch"]) for row in rows]
            values = [float(row[metric]) for row in rows]
            color = COLORS[condition]
            ax.plot(
                epochs,
                values,
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.9,
                label=f"{condition} (e{epochs[-1]}, {statuses[condition]})",
            )
            if metric == METRICS[0][0]:
                best_index = min(range(len(values)), key=values.__getitem__)
                ax.scatter(
                    epochs[best_index],
                    values[best_index],
                    marker="*",
                    s=125,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
                ax.annotate(
                    f"e{epochs[best_index]}\n{values[best_index]:.4f}",
                    xy=(epochs[best_index], values[best_index]),
                    xytext=(5, 7),
                    textcoords="offset points",
                    color=color,
                    fontsize=7.5,
                )

        ax.axvline(
            CONTINUATION_EPOCH,
            color="#666666",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Completed epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, fontsize=9)
    axes[0].text(
        0.02,
        0.02,
        "Stars mark validation-best checkpoints\n"
        "Dashed line: continuation after e30\n(lower is better)",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="#4D4D4D",
    )
    progress = ", ".join(f"{name} e{epoch}" for name, epoch in completed.items())
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    fig.suptitle(
        "FABRIC resident three-arm validation loss, continued from epoch 30 — seed 1103\n"
        f"Snapshot {timestamp} | {progress}",
        fontsize=13,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)
    print(PDF_OUTPUT)


if __name__ == "__main__":
    main()
