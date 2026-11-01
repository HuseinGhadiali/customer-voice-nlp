"""Flipkart review analytics with Hugging Face models: sentiment, rating-text mismatch, zero-shot aspects, topic discovery."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ASPECTS = {   # single-concept NLI hypotheses (compound "A or B" hypotheses score poorly) -> short name
    "The product broke or has a defect.": "Defective / broken",
    "The product stopped working after some time.": "Stopped working",
    "The delivery was late.": "Late delivery",
    "The item arrived damaged.": "Arrived damaged",
    "The product is overpriced.": "Overpriced",
    "The seller refused a return or refund.": "Return / refund refused",
    "The customer service or warranty support was bad.": "Poor service / warranty",
    "The product makes too much noise.": "Too noisy",
    "The product is different from what was described.": "Not as described",
}

CATEGORY_RULES = [  # (regex on product name, category); first match wins
    (r"dishwasher|place settings", "Dishwashers"), (r"chimney|hood", "Kitchen chimneys"),
    (r"food processor|food factory|masterchef|grinder|nutri", "Food processors & mixers"),
    (r"rice cooker|pressure cooker|kettle", "Kettles & cookers"), (r"cooler", "Air coolers"),
    (r"ceiling fan|exhaust fan|bldc", "Fans"),
    (r"bluetooth|headset|neckband|speaker|soundbar|home theatre|bullets|rockerz|tweeter|amplifier|stereo", "Audio"),
    (r"camera|dslr", "Cameras"), (r"hard disk|seagate|wd ", "Storage"), (r"nest|google", "Smart home"),
    (r"stabilizer|remote|set ?top|stand|dth|tata sky|airtel", "TV accessories"),
    (r"perfume|parfum|kajal|lakm|mask|spray", "Beauty & fragrance"),
    (r"skate|cricket|bat|stumps|ball|gym|fitness|net", "Sports & fitness"),
    (r"shoe|glove|safety|lab coat", "Safety wear"), (r"tds|meter|tester", "Water testers"),
    (r"diaper|commode|pill|medicine|covid|hot water|warm bag|pain|enema", "Health & personal care"),
    (r"pencil|doms|crayon|stationery|art|magic book", "Stationery"),
]


def product_category(name: str) -> str:
    n = name.lower()
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, n):
            return cat
    return "Other"


def load_reviews() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("ml-hub/flipkart-reviews")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
    df["product"] = df.product_name.str.replace(r"\?+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df["category"] = df["product"].map(product_category)
    df["text"] = df.text.fillna("").str.strip()
    df = df.dropna(subset=["Rate"]).copy()
    df["rating"] = df.Rate.astype(int)
    df["star_sentiment"] = np.select([df.rating >= 4, df.rating == 3], ["positive", "neutral"], "negative")
    df["n_words"] = df.text.str.split().str.len()
    return df.drop_duplicates(["product", "text", "rating"]).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, per_rating: int = 1200, min_words: int = 3, seed: int = 0) -> pd.DataFrame:
    d = df[df.n_words >= min_words]
    return d.groupby("rating", group_keys=False).apply(lambda g: g.sample(min(len(g), per_rating), random_state=seed)).reset_index(drop=True)


# ------------------------------------------------------------------ transformers (cached)
def _cached(name: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = fn()
    out.to_parquet(path)
    return out


def transformer_sentiment(texts: pd.Series, cache_name: str = "sentiment_roberta") -> pd.DataFrame:
    def run():
        from transformers import pipeline
        clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True, max_length=128, top_k=None)
        order = texts.str.len().sort_values().index          # length-sorted batches = far less padding on CPU
        res = clf(texts.loc[order].tolist(), batch_size=64)
        rows = [{d["label"]: d["score"] for d in r} for r in res]
        out = pd.DataFrame(rows, index=order).loc[texts.index][["negative", "neutral", "positive"]]
        out["pred"] = out.idxmax(axis=1)
        return out
    return _cached(cache_name, run)


def zero_shot_aspects(texts: pd.Series, cache_name: str = "aspects_zeroshot") -> pd.DataFrame:
    def run():
        from transformers import pipeline
        zs = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
        labels = list(ASPECTS)
        order = texts.str.len().sort_values().index
        res = zs(texts.loc[order].str.slice(0, 600).tolist(), candidate_labels=labels, multi_label=True, batch_size=32,
                 hypothesis_template="{}")
        rows = [dict(zip([ASPECTS[label] for label in r["labels"]], r["scores"])) for r in res]
        return pd.DataFrame(rows, index=order).loc[texts.index][list(ASPECTS.values())]
    return _cached(cache_name, run)


def embed(texts: pd.Series, cache_name: str = "embeddings_minilm") -> np.ndarray:
    path = CACHE / f"{cache_name}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    e = SentenceTransformer(EMBED_MODEL).encode(texts.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, e)
    return e


# ------------------------------------------------------------------ analysis helpers (pure, tested)
def class_tfidf_keywords(texts: pd.Series, labels: np.ndarray, top_n: int = 6) -> dict:
    """c-TF-IDF (as in BERTopic): treat each cluster as one document; score words by in-cluster frequency x rarity across clusters."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    domain = {"product", "good", "bad", "nice", "dont", "working", "quality", "flipkart", "worst", "poor", "use", "buy", "ok",
              "money", "waste", "really", "just", "like", "got", "item", "bought", "purchase", "worth", "better", "best"}
    stop = sorted(ENGLISH_STOP_WORDS | domain)
    docs = pd.Series(texts.to_numpy()).groupby(labels).apply(" ".join)
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=1)
    tf = cv.fit_transform(docs).toarray().astype(float)
    tf = tf / tf.sum(1, keepdims=True)
    idf = np.log(1 + tf.shape[0] / (tf > 0).sum(0))
    score = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {lab: vocab[np.argsort(-score[i])[:top_n]].tolist() for i, lab in enumerate(docs.index)}


def mismatches(df: pd.DataFrame, sent: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Reviews whose text sentiment contradicts the star rating (hidden complaints in 4-5 star reviews and vice versa)."""
    d = df.join(sent)
    hidden_complaints = d[(d.rating >= 4) & (d.negative >= threshold)].assign(kind="4-5★ but negative text")
    hidden_praise = d[(d.rating <= 2) & (d.positive >= threshold)].assign(kind="1-2★ but positive text")
    return pd.concat([hidden_complaints, hidden_praise])[["kind", "product", "rating", "text", "negative", "positive"]]
