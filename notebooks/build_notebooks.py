import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}

reviews = [
    md("# Part 1: What are Flipkart customers really saying?\n"
       "**Data:** 40K real Flipkart product reviews across 205 products (Hugging Face Hub: `ml-hub/flipkart-reviews`), loaded at run time and not redistributed.\n\n"
       "**Models (all from 🤗 Hugging Face, CPU only):**\n"
       "| Task | Model | Training needed |\n|---|---|---|\n"
       "| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | none (pretrained) |\n"
       "| Complaint aspects | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (zero-shot NLI) | none |\n"
       "| Topic discovery | `sentence-transformers/all-MiniLM-L6-v2` embeddings + UMAP + k-means + c-TF-IDF | none |\n\n"
       "Slow transformer steps are cached by `src/precompute.py`; this notebook reads the cache."),
    code("import sys, warnings\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')\nimport numpy as np, pandas as pd\n"
         "from IPython.display import Image, display\nfrom reviews import *\nimport charts\n"
         "df = load_reviews()\nprint(f'{len(df):,} reviews · {df[\"product\"].nunique()} products · median {df.n_words.median():.0f} words')\n"
         "display(df.rating.value_counts(normalize=True).sort_index().round(3).to_frame('share'))\ndf.category.value_counts().head(10)"),
    md("## 1. Sentiment: a pretrained transformer vs a model trained on star labels\nStratified sample of 1,200 reviews per star rating (≥3 words). "
       "The supervised baseline (TF-IDF + logistic regression) is trained on all *other* (de-duplicated) reviews using stars as labels."),
    code("from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\n"
         "from sklearn.metrics import f1_score, classification_report\n"
         "sample = pd.read_parquet(CACHE / 'review_sample.parquet')\nsent = transformer_sentiment(sample.text)\n"
         "train = df[~df.text.isin(sample.text)]\nvec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)\n"
         "clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(vec.fit_transform(train.text), train.star_sentiment)\n"
         "base_pred = clf.predict(vec.transform(sample.text))\n"
         "comp = pd.DataFrame({'macro_F1_vs_star_label': [f1_score(sample.star_sentiment, base_pred, average='macro'),\n"
         "                                                f1_score(sample.star_sentiment, sent.pred, average='macro')]},\n"
         "                    index=[f'TF-IDF + LogReg (trained on {len(train):,} star-labelled reviews)', 'RoBERTa sentiment (zero training)'])\ncomp.round(3)"),
    code("d = sample.join(sent)\ncharts.sentiment_by_rating(d)\ndisplay(Image('../images/sentiment_by_rating.png'))"),
    md("### Hidden complaints: 4–5★ reviews whose *text* is clearly negative\nThese are invisible to any star-based dashboard."),
    code("mm = mismatches(sample, sent)\nprint(mm.kind.value_counts().to_string())\n"
         "pd.set_option('display.max_colwidth', 140)\nmm[mm.kind.str.startswith('4-5')].sort_values('negative', ascending=False).head(8)[['rating', 'text', 'negative']]"),
    md("## 2. Zero-shot aspect tagging of complaints\n300 one- and two-star reviews (≥6 words), 9 business aspects, multi-label. No labelled data needed: "
       "an NLI model scores each single-concept hypothesis (e.g. *\"The delivery was late.\"*) against the review."),
    code("neg = pd.read_parquet(CACHE / 'negative_sample.parquet')\nasp = zero_shot_aspects(neg.text)\nflags = asp >= 0.5\n"
         "rates = 100 * flags.mean()\ncharts.aspect_chart(rates)\nprint(f'{(flags.sum(1) == 0).mean():.0%} of complaints match none of the 9 aspects')\n"
         "display(Image('../images/complaint_aspects.png'))"),
    code("cat = neg.join(flags).groupby('category')\nm = (100 * cat[list(flags.columns)].mean()).loc[cat.size()[lambda s: s >= 15].index]\n"
         "charts.aspect_heatmap(m.round(0))\nm.round(0).to_csv('../reports/aspects_by_category.csv')\ndisplay(Image('../images/aspects_by_category.png'))"),
    code("# the most confident example for each aspect (sanity check of the zero-shot model)\n"
         "pd.DataFrame({a: [neg.text.loc[asp[a].idxmax()][:160], round(asp[a].max(), 3)] for a in asp.columns}, index=['example', 'score']).T"),
    md("## 3. Topic discovery with sentence embeddings\nAll 1–3★ reviews with ≥5 words → MiniLM embeddings → k-means (k = 12) → UMAP for the 2-D map → "
       "**c-TF-IDF** (the BERTopic idea) to name each topic."),
    code("from sklearn.cluster import KMeans\nimport umap\ntopic = pd.read_parquet(CACHE / 'topic_docs.parquet')\nE = embed(topic.text)\n"
         "labels = KMeans(12, n_init=10, random_state=0).fit_predict(E)\nkw = class_tfidf_keywords(topic.text, labels)\n"
         "names = {k: ' · '.join(v[:3]) for k, v in kw.items()}\n"
         "xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(E)\ncharts.topic_map(xy, labels, names)\n"
         "summary = (topic.assign(topic=labels).groupby('topic').agg(reviews=('text', 'size'), avg_rating=('rating', 'mean'))\n"
         "           .assign(keywords=lambda t: t.index.map(lambda k: ', '.join(kw[k])), example=lambda t: t.index.map(lambda k: topic.text[labels == k].iloc[0][:110]))\n"
         "           .sort_values('reviews', ascending=False))\nsummary.round(2).to_csv('../reports/topics.csv')\ndisplay(Image('../images/topic_map.png'))\nsummary.round(2)"),
    md("## Takeaways\nSee the README. In short: star ratings hide complaints, a zero-training transformer beats a supervised TF-IDF baseline on sentiment, "
       "and zero-shot aspects turn 300 complaints into a ranked fix-list per category without labelling a single review."),
]

intent = [
    md("# Part 2: Routing banking support tickets with fine-tuned transformers\n"
       "**Data:** Banking77 (PolyAI, CC-BY-4.0): 13,083 real customer queries to a banking app, labelled into **77 fine-grained intents** "
       "(e.g. *card_arrival*, *pending_top_up*, *exchange_rate*).\n\n"
       "**Business question:** can incoming tickets be routed to the right queue automatically, and how many can be routed safely without a human?\n\n"
       "| Model | Idea |\n|---|---|\n| TF-IDF + logistic regression | classic baseline |\n"
       "| MiniLM sentence embeddings + logistic regression | frozen 🤗 encoder, light classifier |\n"
       "| DistilBERT fine-tuned | full fine-tuning with the 🤗 `Trainer`, 2 epochs on CPU (a deliberately small compute budget) |\n\n"
       "Training runs once in `src/intent.py` (cached predictions); this notebook analyses the results."),
    code("import sys, json, warnings\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')\nimport numpy as np, pandas as pd\n"
         "from IPython.display import Image, display\nfrom intent import *\nimport charts\n"
         "train, test, names = load_banking77()\nprint(f'{len(train):,} train · {len(test):,} test · {len(names)} intents')\ntest.sample(5, random_state=1).assign(intent=lambda d: d.label.map(dict(enumerate(names))))"),
    code("res = pd.read_csv('../reports/intent_model_comparison.csv')\ncharts.intent_models(res)\ndisplay(res.round(4))\ndisplay(Image('../images/intent_models.png'))"),
    code("hist = pd.DataFrame(json.loads((CACHE / 'distilbert_history.json').read_text()))\nhist[['epoch', 'loss']].iloc[::3].round(3).T"),
    md("**Reading the result:** with only 2 epochs on CPU, DistilBERT is under-trained (loss still falling), and a *frozen* MiniLM encoder "
       "plus logistic regression wins at about a ninth of the compute. The Banking77 paper reports about 93% for fully fine-tuned BERT, so the fine-tune "
       "should overtake with more epochs on a GPU. On a CPU budget, embeddings + a linear model are the pragmatic choice."),
    md("## Safe automation: route only confident predictions (best model: MiniLM + LogReg)"),
    code("P = np.load(CACHE / 'banking77_test_probs.npz')\ny = P['y']\n"
         "curves = {'DistilBERT fine-tuned': routing_curve(y, P['distilbert']), 'MiniLM + LogReg': routing_curve(y, P['minilm']),\n"
         "          'TF-IDF + LogReg': routing_curve(y, P['tfidf'])}\ncharts.routing_chart(curves)\n"
         "c = curves['MiniLM + LogReg']\ndisplay(c.round({\"threshold\": 2, \"auto_routed_pct\": 1, \"accuracy_on_auto_routed_pct\": 1}))\ndisplay(Image('../images/routing_tradeoff.png'))"),
    code("op = c[c.accuracy_on_auto_routed_pct >= 98].iloc[0]\n"
         "print(f\"At confidence ≥ {op.threshold:.2f}, the MiniLM model auto-routes {op.auto_routed_pct:.0f}% of tickets at {op.accuracy_on_auto_routed_pct:.1f}% accuracy; \"\n"
         "      f\"the remaining {100 - op.auto_routed_pct:.0f}% go to human triage.\")"),
    md("## Where does it still get confused?"),
    code("top_confusions(y, P['minilm'], names, 10)"),
    md("Most confusions are between genuinely overlapping intents (e.g. *pending* vs *failed* transfers or top-ups). "
       "These are candidates for merging queues or adding a clarifying question in the chatbot."),
]
nbf.write(nbf.v4.new_notebook(cells=reviews, metadata=meta), "01_review_analytics.ipynb")
nbf.write(nbf.v4.new_notebook(cells=intent, metadata=meta), "02_support_intent_routing.ipynb")
