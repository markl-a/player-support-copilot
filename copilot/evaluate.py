"""python -m copilot.evaluate -> results/eval.json"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from .classify import load, predict, train
from .fields import extract, mask
from .pipeline import MIN_CONFIDENCE, handle

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    tr, te = load("train"), load("test")
    model = train(tr)
    texts = [mask(r["text"], keep=set(extract(r["text"])["order_ids"])) for r in te]
    preds = predict(model, texts)
    y, yhat, conf = [r["label"] for r in te], [p for p, _ in preds], [c for _, c in preds]
    auto = [i for i, c in enumerate(conf) if c >= MIN_CONFIDENCE]
    order_hits = sum(1 for r in te if r["order_id"] and r["order_id"] in extract(r["text"])["order_ids"])
    order_total = sum(1 for r in te if r["order_id"])
    handled = [handle(r["text"], model=model) for r in te]
    report = {
        "split": "test tickets use sentence templates never seen in training",
        "n_train": len(tr), "n_test": len(te),
        "classification": {"accuracy_all": round(sum(a == b for a, b in zip(y, yhat)) / len(y), 3),
                           "macro_f1_all": round(f1_score(y, yhat, average="macro"), 3),
                           "auto_share": round(len(auto) / len(y), 3),
                           "accuracy_on_auto": round(sum(y[i] == yhat[i] for i in auto) / max(1, len(auto)), 3),
                           "min_confidence": MIN_CONFIDENCE},
        "per_class": {k: {m: round(v, 3) for m, v in d.items() if m in ("precision", "recall", "f1-score")}
                      for k, d in classification_report(y, yhat, output_dict=True, zero_division=0).items() if k in set(y)},
        "coverage_curve": [{"min_confidence": th, "auto_share": round(sum(c >= th for c in conf) / len(conf), 3),
                            "accuracy_on_auto": round(sum(y[i] == yhat[i] for i, c in enumerate(conf) if c >= th) / max(1, sum(c >= th for c in conf)), 3)}
                           for th in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)],
        "order_id_extraction_recall": round(order_hits / max(1, order_total), 3),
        "pipeline_status": dict(Counter(h.status for h in handled)),
        "confusions": dict(Counter(f"{a} -> {b}" for a, b in zip(y, yhat) if a != b).most_common(8)),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
