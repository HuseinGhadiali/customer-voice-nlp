"""Run the slow transformer steps once and cache them under data/cache (the notebook reads the cache)."""
import time

from reviews import CACHE, embed, load_reviews, stratified_sample, transformer_sentiment, zero_shot_aspects

t = time.time()
df = load_reviews()
sample = stratified_sample(df)
sample.to_parquet(str(CACHE / "review_sample.parquet"))
transformer_sentiment(sample.text)
print("sentiment done", round(time.time() - t), "s", flush=True)
neg = df[(df.rating <= 2) & (df.n_words >= 6)].sample(300, random_state=0)
neg.to_parquet(str(CACHE / "negative_sample.parquet"))
zero_shot_aspects(neg.text)
print("aspects done", round(time.time() - t), "s", flush=True)
topic = df[(df.rating <= 3) & (df.n_words >= 5)]
topic.to_parquet(str(CACHE / "topic_docs.parquet"))
embed(topic.text)
print("embeddings done", round(time.time() - t), "s", flush=True)
