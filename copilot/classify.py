"""Ticket category: char n-gram TF-IDF + logistic regression. Chinese has no spaces; character 1-3 grams need no
tokenizer and cope with typos. Cheap enough to run on every ticket; only low-confidence tickets go further."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).resolve().parent.parent / "data"


def load(split: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines()]


def train(rows: list[dict]):
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
                          LogisticRegression(C=4.0, max_iter=2000))
    model.fit([r["text"] for r in rows], [r["label"] for r in rows])
    return model


def predict(model, texts: list[str]) -> list[tuple[str, float]]:
    proba = model.predict_proba(texts)
    idx = proba.argmax(1)
    return [(model.classes_[i], float(p[i])) for i, p in zip(idx, proba)]


_MODEL = None


def default_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = train(load("train"))
    return _MODEL
