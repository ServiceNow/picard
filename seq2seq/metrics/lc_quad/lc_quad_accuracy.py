"""Cosql intent accuracy metric."""

import re
from typing import Dict, Any

# Replace multi space and only keep one
def spaceReplace(i):
    i = re.sub(' +', ' ', i).split(' ')
    return i


def compute_accuracy_metric(predictions, references) -> Dict[str, Any]:
    acc, total = 0, 0
    for prediction, reference in zip(predictions, references):
        p = prediction.lower()
        r = reference["query"].lower()
        consice_p = re.sub('[\W_]+', '', p)
        consice_r = re.sub('[\W_]+', '', r)
        if consice_p == consice_r:
            acc +=1
        total += 1
    return {
        "accuracy": float(acc/total),
    }