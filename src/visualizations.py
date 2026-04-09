"""
Visualisation module for AML Detection
Produces 6 publication-quality plots saved to outputs/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

STYLE = {
    "normal":    "#4C72B0",
    "fraud":     "#DD8452",
    "precision": "#55A868",
    "recall":    "#C44E52",
    "fbeta":     "#8172B2",
    "ae":        "#CCB974",
    "gnn":       "#64B5CD",
    "bg":        "#F8F8F8",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   STYLE["bg"],
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ── 1. Dataset overview ───────────────────────────────────────────────────────

def plot_dataset_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("AML Dataset Overview", fontsize=14, fontweight="bold", y=1.02)

    # (a) Pattern distribution
    ax = axes[0]
    counts = df["pattern"].value_counts()
    colors = [STYLE["fraud"] if p != "legit" else STYLE["normal"] for p in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Transaction count")
    ax.set_title("Transactions by pattern")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=9)

    # (b) Amount distribution
    ax = axes[1]
    legit  = df[df["is_laundering"] == 0]["amount"]
    fraud  = df[df["is_laundering"] == 1]["amount"]
    bins   = np.logspace(0, 6, 50)
    ax.hist(legit, bins=bins, alpha=0.6, label=f"Legit (n={len(legit):,})",  color=STYLE["normal"])
    ax.hist(fraud, bins=bins, alpha=0.6, label=f"Fraud (n={len(fraud):,})",  color=STYLE["fraud"])
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount ($)")
    ax.set_ylabel("Count")
    ax.set_title("Amount distribution")
    ax.legend(fontsize=8)

    # (c) Transactions over time
    ax = axes[2]
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")
    monthly = df.groupby(["month", "is_laundering"]).size().unstack(fill_value=0)
    months = [str(m) for m in monthly.index]
    ax.bar(months, monthly.get(0, 0), label="Legit",      color=STYLE["normal"], alpha=0.8)
    ax.bar(months, monthly.get(1, 0), bottom=monthly.get(0, 0),
           label="Laundering", color=STYLE["fraud"], alpha=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Monthly transaction volume")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    path = OUT_DIR / "01_dataset_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.name}")


# ── 2. Precision-recall threshold sweep ──────────────────────────────────────

def plot_threshold_sweep(if_results: dict, ae_results: dict,
                         gnn_results: dict = None, beta: float = 2.0):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Threshold Analysis (F-β with β={beta} — recall-weighted)",
                 fontsize=13, fontweight="bold", y=1.02)

    model_results = [
        ("Isolation Forest", if_results, STYLE["normal"]),
        ("Autoencoder",      ae_results, STYLE["ae"]),
    ]
    if gnn_results:
        model_results.append(("GNN (node-level)", gnn_results, STYLE["gnn"]))

    for ax, (name, res, color) in zip(axes, model_results):
        thresholds = res["thresholds"]
        precisions = res["precisions"][:-1]
        recalls    = res["recalls"][:-1]
        f_betas    = res["f_betas"][:-1]

        ax.plot(thresholds, precisions, color=STYLE["precision"], lw=1.5, label="Precision")
        ax.plot(thresholds, recalls,    color=STYLE["recall"],    lw=1.5, label="Recall")
        ax.plot(thresholds, f_betas,    color=STYLE["fbeta"],     lw=2.0, label=f"F-{beta}")

        best_t = res["threshold"]
        ax.axvline(best_t, color="black", lw=1.2, linestyle=":", alpha=0.7)
        ax.annotate(f"P={res['precision']:.2f}\nR={res['recall']:.2f}",
                    xy=(best_t, res["recall"]),
                    xytext=(best_t + 0.04, 0.55),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{name}\nROC-AUC={res['roc_auc']:.3f}  AP={res['avg_precision']:.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "02_threshold_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.name}")


# ── 3. Precision-recall curves ────────────────────────────────────────────────

def plot_pr_curves(if_results: dict, ae_results: dict, gnn_results: dict = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=12, fontweight="bold")

    models = [
        ("Isolation Forest", if_results, STYLE["normal"]),
        ("Autoencoder",      ae_results, STYLE["ae"]),
    ]
    if gnn_results:
        models.append(("GNN", gnn_results, STYLE["gnn"]))

    for name, res, color in models:
        ax.plot(res["recalls"], res["precisions"], lw=2, color=color,
                label=f"{name} (AP={res['avg_precision']:.3f})")
        best_idx = np.argmax(res["f_betas"][:-1])
        ax.scatter(res["recalls"][best_idx], res["precisions"][best_idx],
                   s=80, color=color, zorder=5, edgecolors="white", lw=1.5)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.annotate("● = optimal threshold (max F-β)", xy=(0.55, 0.05), fontsize=8, color="gray")

    plt.tight_layout()
    path = OUT_DIR / "03_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.name}")


# ── 4. Anomaly score distributions ───────────────────────────────────────────

def plot_score_distributions(if_results: dict, ae_results: dict,
                              y: np.ndarray, gnn_results: dict = None,
                              gnn_y: np.ndarray = None):
    n_plots = 3 if gnn_results else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    fig.suptitle("Anomaly Score Distributions", fontsize=13, fontweight="bold")

    plot_data = [
        ("Isolation Forest", if_results["scores"], y),
        ("Autoencoder",      ae_results["scores"], y),
    ]
    if gnn_results and gnn_y is not None:
        plot_data.append(("GNN", gnn_results["scores"], gnn_y))

    for ax, (name, scores, labels) in zip(axes, plot_data):
        ax.hist(scores[labels == 0], bins=40, alpha=0.6, density=True,
                color=STYLE["normal"], label="Legit")
        ax.hist(scores[labels == 1], bins=40, alpha=0.6, density=True,
                color=STYLE["fraud"],  label="Laundering")

        # Optimal threshold line
        if name == "Isolation Forest":
            t = if_results["threshold"]
        elif name == "Autoencoder":
            t = ae_results["threshold"]
        else:
            t = gnn_results["threshold"]

        ax.axvline(t, color="black", lw=1.5, linestyle="--",
                   label=f"Threshold={t:.2f}")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "04_score_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.name}")


# ── 5. Training loss curves ───────────────────────────────────────────────────

def plot_training_curves(ae_losses: list, gnn_losses: list = None):
    fig, axes = plt.subplots(1, 2 if gnn_losses else 1, figsize=(12 if gnn_losses else 6, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    fig.suptitle("Training Loss Curves", fontsize=12, fontweight="bold")

    axes[0].plot(ae_losses, color=STYLE["ae"], lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Reconstruction Loss")
    axes[0].set_title("Autoencoder")

    if gnn_losses and len(axes) > 1:
        axes[1].plot(gnn_losses, color=STYLE["gnn"], lw=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Binary Cross-Entropy")
        axes[1].set_title("GNN Classifier")

    plt.tight_layout()
    path = OUT_DIR / "05_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.name}")


# ── 6. Transaction graph (subgraph of flagged accounts) ───────────────────────

def plot_transaction_subgraph(df: pd.DataFrame, max_nodes: int = 60):
    fraud_df = df[df["is_laundering"] == 1].copy()
    # Take the most active fraud accounts
    top_senders = fraud_df["sender"].value_counts().head(max_nodes // 2).index
    sub_df = fraud_df[fraud_df["sender"].isin(top_senders)].head(200)

    G = nx.DiGraph()
    for _, row in sub_df.iterrows():
        G.add_edge(row["sender"], row["receiver"],
                   weight=np.log1p(row["amount"]), pattern=row["pattern"])

    # Colour nodes by pattern involvement
    pattern_colors = {
        "smurfing":    "#E8A838",
        "layering":    "#C44E52",
        "round_trip":  "#8172B2",
        "structuring": "#55A868",
    }
    node_colors = []
    for node in G.nodes():
        patterns_found = sub_df[(sub_df["sender"] == node) |
                                 (sub_df["receiver"] == node)]["pattern"].unique()
        color = pattern_colors.get(patterns_found[0] if len(patterns_found) > 0 else "unknown",
                                   "#aaaaaa")
        node_colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(G, seed=42, k=1.5)
    edge_weights = [G[u][v]["weight"] * 0.3 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ffffff",
                           alpha=0.25, width=edge_weights, arrows=True,
                           arrowsize=10, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=80, alpha=0.9)

    # Legend
    for pattern, color in pattern_colors.items():
        ax.scatter([], [], color=color, s=60, label=pattern.replace("_", "-"))
    ax.legend(loc="lower right", fontsize=9, facecolor="#2a2a4a",
              labelcolor="white", edgecolor="#555555")

    ax.set_title("Flagged Transaction Network (suspicious accounts)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")

    plt.tight_layout()
    path = OUT_DIR / "06_transaction_graph.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved → {path.name}")
