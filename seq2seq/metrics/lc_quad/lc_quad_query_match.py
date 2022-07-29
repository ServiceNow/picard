"""Cosql intent accuracy metric."""

import re
from typing import Dict, Any
from unittest import result


def parse_triples(sparql):
    p1 = re.compile(r'[{](.*?)[}]', re.S)  #最小匹配
    # p2 = re.compile(r'[{](.*)[}]', re.S)   #贪婪匹配
    result1 = re.findall(p1, sparql)
    # result2 = re.findall(p2, sparql)
    # print(sparql)
    if len(result1)>0:
        triple_l = result1[0].strip().split(". ")
        triple_l = [triple.strip().replace(" .", "").strip().replace(" ", "").lower() for triple in triple_l]
        triple_s = set(triple_l)
    else:
        triple_s = set()
    return triple_s


def query_match_one(prediction, reference):
    pred_s = parse_triples(prediction)
    ref_s = parse_triples(reference)
    
    if pred_s == ref_s:
        return 1
    else:
        return 0


def compute_query_match_metric(predictions, references) -> Dict[str, Any]:
    qm, total = 0, 0
    for prediction, reference in zip(predictions, references):
        qm += query_match_one(prediction, reference["query"])
        total += 1
    return {
        "query_match": float(qm/total),
    }