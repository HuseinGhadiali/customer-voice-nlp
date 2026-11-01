"""Render assets/banner.png: real review embeddings (MiniLM -> UMAP) as a glowing galaxy, coloured by topic."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from banner import glow_scatter, make_banner  # noqa: E402
from reviews import CACHE, embed  # noqa: E402

topic = pd.read_parquet(CACHE / "topic_docs.parquet")
E = embed(topic.text)
from sklearn.cluster import KMeans  # noqa: E402
import umap  # noqa: E402

labels = KMeans(12, n_init=10, random_state=0).fit_predict(E)
xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(E)


def art(ax, acc):
    palette = acc + ["#ffffff", "#a0f1ff", "#ffb86c", "#7cffcb", "#f78fb3", "#9b5de5", "#fee440", "#53d8fb"]
    for k in range(12):
        m = labels == k
        glow_scatter(ax, xy[m, 0], xy[m, 1], palette[k % len(palette)], s=2.2, alpha=0.6)
    ax.text(xy[:, 0].min(), xy[:, 1].min() - 1.2, "each dot = a real Flipkart review · MiniLM embeddings → UMAP · colour = discovered topic",
            color="white", alpha=0.55, fontsize=9)
    ax.set_ylim(xy[:, 1].min() - 1.6, xy[:, 1].max() + 0.5)


make_banner(Path(__file__).parent / "banner.png", "Customer Voice NLP",
            "Sentiment · zero-shot aspects · topics · fine-tuned DistilBERT",
            ["Hugging Face", "Transformers", "PyTorch", "Sentence-BERT"], art, theme="neon",
            emoji_free_kicker="NLP · 30K reviews + 13K support tickets")
