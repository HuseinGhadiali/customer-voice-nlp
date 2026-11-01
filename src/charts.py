"""Charts for the NLP notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from reviews import ROOT
from viz_style import BLUES, MUTED, SERIES, TEXT, apply_style, finish

IMG = ROOT / "images"
apply_style()


def sentiment_by_rating(d: pd.DataFrame, path=IMG / "sentiment_by_rating.png"):
    t = pd.crosstab(d.rating, d.pred, normalize="index")[["negative", "neutral", "positive"]]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    left = np.zeros(len(t))
    colors = {"negative": SERIES[1], "neutral": MUTED, "positive": SERIES[0]}
    for c in t.columns:
        ax.barh([f"{r}★" for r in t.index], t[c], left=left, color=colors[c], height=0.6, label=c, edgecolor="white", lw=1.5)
        left += t[c].to_numpy()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of reviews by RoBERTa-predicted sentiment")
    finish(fig, ax, "The text disagrees with the stars more often than you'd think",
           "Pretrained RoBERTa sentiment (no training) vs the reviewer's star rating", path)


def aspect_chart(rates: pd.Series, path=IMG / "complaint_aspects.png"):
    r = rates.sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(r.index, r / 100, color=[SERIES[0] if v == r.max() else SERIES[0] for v in r], height=0.6)
    for y, v in enumerate(r):
        ax.text(v / 100 + 0.005, y, f"{v:.0f}%", va="center", fontsize=9, color=TEXT)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, r.max() / 100 * 1.15)
    ax.set_xlabel("Share of 1-2★ reviews mentioning the aspect (multi-label, zero-shot score ≥ 0.5)")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "What 1-2★ reviewers actually complain about", "Zero-shot DeBERTa-v3 aspect tagging, no labelled training data", path)


def aspect_heatmap(m: pd.DataFrame, path=IMG / "aspects_by_category.png"):
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(m.to_numpy(), cmap=LinearSegmentedColormap.from_list("b", BLUES), aspect="auto", vmin=0, vmax=max(60, m.to_numpy().max()))
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=25, ha="right")
    ax.set_yticks(range(m.shape[0]), [f"{i}" for i in m.index])
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 35 else TEXT)
    finish(fig, ax, "Complaint mix differs by category: fix the right thing per category",
           "% of negative reviews tagged with each aspect (categories with 15+ negative reviews in the sample)", path)


def topic_map(xy: np.ndarray, labels: np.ndarray, names: dict, path=IMG / "topic_map.png"):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color=MUTED, alpha=0.35, lw=0)
    order = pd.Series(labels).value_counts().index[:8]
    for i, k in enumerate(order):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=3, color=SERIES[i], alpha=0.55, lw=0)
        cx, cy = np.median(xy[m], axis=0)
        ax.text(cx, cy, names[k], fontsize=8.5, color=TEXT, ha="center", va="center", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": SERIES[i], "lw": 1, "alpha": 0.9})
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    finish(fig, ax, "Topic map of negative and neutral reviews",
           "MiniLM sentence embeddings → UMAP 2-D → k-means topics, labelled with c-TF-IDF keywords (8 largest shown)", path)


def intent_models(res: pd.DataFrame, path=IMG / "intent_models.png"):
    r = res.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    best = r.accuracy.idxmax()
    ax.barh(r.model, r.accuracy, color=[SERIES[0] if i == best else MUTED for i in r.index], height=0.55)
    for y, (a, f, s) in enumerate(zip(r.accuracy, r.macro_f1, r.train_seconds)):
        ax.text(a + 0.004, y, f"{a:.1%} acc · {f:.3f} macro-F1 · {s / 60:.1f} min", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0.8, 1.06)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "Frozen sentence embeddings beat a CPU-budget fine-tune",
           "Test accuracy on 3,080 real banking-app support queries, 77 intents (training time on a 4-core CPU)", path)


def routing_chart(curves: dict, path=IMG / "routing_tradeoff.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c.auto_routed_pct / 100, c.accuracy_on_auto_routed_pct / 100, marker="o", markersize=4, color=SERIES[i], label=name)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Tickets auto-routed (rest go to a human triage queue)")
    ax.set_ylabel("Accuracy on auto-routed tickets")
    ax.legend(loc="lower left")
    finish(fig, ax, "Confidence thresholds trade automation for accuracy",
           "Each point is a confidence threshold from 0.30 to 0.95", path)
