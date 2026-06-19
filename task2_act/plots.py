from __future__ import annotations

from pathlib import Path


def plot_metrics(output_dir: Path, metrics_by_model: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(metrics_by_model)
    overall = [metrics_by_model[name]["overall_l1"] for name in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, overall)
    plt.ylabel("Action L1")
    plt.title("Zero-shot Action L1 on CALVIN D")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_l1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    for name, metrics in metrics_by_model.items():
        xs = [item["horizon"] for item in metrics["per_horizon_l1"]]
        ys = [item["l1"] for item in metrics["per_horizon_l1"]]
        plt.plot(xs, ys, label=name)
    plt.xlabel("Prediction Horizon")
    plt.ylabel("Action L1")
    plt.title("Action L1 by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_horizon_l1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    dim_names = [item["name"] for item in next(iter(metrics_by_model.values()))["per_dim_l1"]]
    width = 0.8 / max(1, len(names))
    base = list(range(len(dim_names)))
    for model_idx, name in enumerate(names):
        ys = [item["l1"] for item in metrics_by_model[name]["per_dim_l1"]]
        xs = [value + model_idx * width for value in base]
        plt.bar(xs, ys, width=width, label=name)
    center = [value + width * (len(names) - 1) / 2 for value in base]
    plt.xticks(center, dim_names, rotation=30, ha="right")
    plt.ylabel("Action L1")
    plt.title("Action L1 by Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_dim_l1.png", dpi=200)
    plt.close()
