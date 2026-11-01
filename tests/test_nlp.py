"""Fast tests for the pure helpers (no model downloads)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from intent import routing_curve, top_confusions  # noqa: E402
from reviews import ASPECTS, class_tfidf_keywords, mismatches, product_category  # noqa: E402


def test_product_category_rules():
    assert product_category("Crompton 75 L Desert Air Cooler (White)") == "Air coolers"
    assert product_category("boAt Rockerz 510 Super Extra Bass Bluetooth Headset") == "Audio"
    assert product_category("IFB Neptune VX Free Standing 12 Place Settings") == "Dishwashers"
    assert product_category("Hindware Ripple 90 Auto Clean Wall Mounted Chimney") == "Kitchen chimneys"
    assert product_category("Something unknown") == "Other"


def test_class_tfidf_finds_distinctive_words():
    texts = pd.Series(["battery drains fast", "battery dies quickly battery", "late delivery courier", "delivery very late courier"])
    kw = class_tfidf_keywords(texts, np.array([0, 0, 1, 1]), top_n=2)
    assert "battery" in kw[0] and ("delivery" in kw[1] or "courier" in kw[1] or "late" in kw[1])


def test_mismatch_detection():
    df = pd.DataFrame({"product": ["a", "b", "c"], "rating": [5, 1, 5], "text": ["x", "y", "z"]})
    sent = pd.DataFrame({"negative": [0.95, 0.01, 0.1], "positive": [0.02, 0.97, 0.8]})
    m = mismatches(df, sent)
    assert sorted(m.kind) == ["1-2★ but positive text", "4-5★ but negative text"]


def test_routing_curve_monotonic_coverage():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5) * 0.3, 500)
    y = proba.argmax(1)
    y[:100] = (y[:100] + 1) % 5                      # some errors
    c = routing_curve(y, proba)
    assert (np.diff(c.auto_routed_pct) <= 0).all()   # higher threshold -> fewer auto-routed
    assert c.accuracy_on_auto_routed_pct.dropna().between(0, 100).all()


def test_top_confusions_counts_only_errors():
    proba = np.eye(3)[[0, 1, 1, 2]]
    out = top_confusions(np.array([0, 1, 0, 2]), proba, ["a", "b", "c"])
    assert out.to_dict("records") == [{"true": "a", "pred": "b", "count": 1}]


def test_aspect_labels_unique():
    assert len(set(ASPECTS.values())) == len(ASPECTS)
