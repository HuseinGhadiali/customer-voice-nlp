"""Banking77 support-query intent classification: TF-IDF baseline, sentence embeddings, fine-tuned DistilBERT,
and confidence-based auto-routing."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"


def load_banking77():
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names
    train, test = ds["train"].to_pandas(), ds["test"].to_pandas()
    return train, test, names


def evaluate(y_true, proba: np.ndarray, model: str, seconds: float) -> dict:
    pred = proba.argmax(1)
    return {"model": model, "accuracy": accuracy_score(y_true, pred), "macro_f1": f1_score(y_true, pred, average="macro"),
            "train_seconds": round(seconds, 1)}


def tfidf_baseline(train, test):
    t = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=10, max_iter=3000)
    clf.fit(vec.fit_transform(train.text), train.label)
    return clf.predict_proba(vec.transform(test.text)), time.time() - t


def embedding_model(train, test, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    t = time.time()
    enc = SentenceTransformer(model_name)
    xtr = enc.encode(train.text.tolist(), batch_size=128, normalize_embeddings=True)
    xte = enc.encode(test.text.tolist(), batch_size=128, normalize_embeddings=True)
    clf = LogisticRegression(C=20, max_iter=3000).fit(xtr, train.label)
    return clf.predict_proba(xte), time.time() - t


def finetune_distilbert(train, test, n_labels: int, epochs: int = 3, out_dir: Path = CACHE / "distilbert-banking77"):
    """Fine-tune distilbert-base-uncased with the HF Trainer (CPU-friendly settings)."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    torch.manual_seed(0)
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_labels)

    def enc(df):
        return Dataset.from_pandas(df[["text", "label"]]).map(lambda b: tok(b["text"], truncation=True, max_length=48), batched=True)

    dtr, dte = enc(train), enc(test)
    args = TrainingArguments(output_dir=str(out_dir), num_train_epochs=epochs, per_device_train_batch_size=32,
                             per_device_eval_batch_size=128, learning_rate=6e-5, warmup_steps=60, weight_decay=0.01,
                             lr_scheduler_type="linear", logging_steps=50, save_strategy="no", report_to=[], seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=dtr, data_collator=DataCollatorWithPadding(tok))
    t = time.time()
    trainer.train()
    seconds = time.time() - t
    logits = trainer.predict(dte).predictions
    proba = np.exp(logits - logits.max(1, keepdims=True))
    proba /= proba.sum(1, keepdims=True)
    history = [h for h in trainer.state.log_history if "loss" in h]
    return proba, seconds, history


THRESHOLDS = np.round(np.arange(0.3, 0.99, 0.05), 2)


def routing_curve(y_true, proba: np.ndarray, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Auto-route a ticket only when the model is confident; the rest go to a human queue."""
    conf, pred = proba.max(1), proba.argmax(1)
    rows = []
    for th in thresholds:
        m = conf >= th
        rows.append({"threshold": th, "auto_routed_pct": 100 * m.mean(),
                     "accuracy_on_auto_routed_pct": 100 * (pred[m] == np.asarray(y_true)[m]).mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def top_confusions(y_true, proba, names, k: int = 10) -> pd.DataFrame:
    pred = proba.argmax(1)
    df = pd.DataFrame({"true": np.asarray(names)[np.asarray(y_true)], "pred": np.asarray(names)[pred]})
    wrong = df[df.true != df.pred]
    return wrong.value_counts().head(k).rename("count").reset_index()


def run_all(epochs: int = 2) -> None:
    """Train every model once and cache predictions (the notebook reads the cache)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    train, test, names = load_banking77()
    results, probs = [], {}
    p, s = tfidf_baseline(train, test)
    results.append(evaluate(test.label, p, "TF-IDF + LogisticRegression", s))
    probs["tfidf"] = p
    p, s = embedding_model(train, test)
    results.append(evaluate(test.label, p, "MiniLM sentence embeddings + LogisticRegression", s))
    probs["minilm"] = p
    p, s, hist = finetune_distilbert(train, test, len(names), epochs)
    results.append(evaluate(test.label, p, f"DistilBERT fine-tuned ({epochs} epochs, CPU)", s))
    probs["distilbert"] = p
    np.savez_compressed(CACHE / "banking77_test_probs.npz", **probs, y=test.label.to_numpy())
    pd.DataFrame(results).to_csv(REPORTS / "intent_model_comparison.csv", index=False)
    (CACHE / "distilbert_history.json").write_text(json.dumps(hist))
    print(pd.DataFrame(results))


if __name__ == "__main__":
    run_all()
