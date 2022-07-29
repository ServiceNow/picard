"""Cosql response accuracy metric."""

import datasets
from typing import Dict, Any


def compute_sacrebleu_metric(predictions, references) -> Dict[str, Any]:
    sacrebleu = datasets.load_metric("sacrebleu")
    pres = [prediction for prediction in predictions]
    refs = [[reference["label"]] for reference in references]
    results = sacrebleu.compute(predictions=pres, references=refs)
    score = round(results["score"], 2)
    return {
        "sacrebleu": float(score),
    }