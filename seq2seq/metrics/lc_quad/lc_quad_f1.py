import requests
from typing import Dict, Any


def empty(r):
    if not r:
        return True
    if 'boolean' not in r:
        if 'results' in r:
            if 'bindings' in r['results']:
                if not r['results']['bindings']:
                    return True
                if {} in r['results']['bindings']:
                    return True
    return False


def hitkg(query, url="http://localhost:8892/sparql", typeq="target"):
    try:
        #print(query)
        proxies = { "http": None, "https": None}
        r = requests.get(url, params={'format': 'json', 'query': query}, proxies =proxies)
        json_format = r.json()
        #print(json_format)
        results = json_format
        if empty(results) and typeq == 'target':
            # print("Empty")
            return "Empty"
        else:
            # print("find result")
            #sys.exit(1)
            return results
    except Exception as err:
        #print(err)
        if typeq == 'target':
            print("no response in target query")
            #sys.exit(1)
        return ''


def parse_answer_from_result(result_dict):
    if 'boolean' in result_dict:
        return result_dict['boolean']
    elif 'results' in result_dict:
        raw_results = result_dict['results']['bindings']
        result = [item[result_dict['head']['vars'][0]]['value'] for item in raw_results]
        return result
    else:
        return None


def compute_prf1_one(prediction, reference):
    gold_result = hitkg(reference)
    gold_answer = parse_answer_from_result(gold_result)
    pred_result = hitkg(prediction)
    pred_answer = parse_answer_from_result(pred_result)
    if gold_answer is None:
        if pred_answer is None:
            recall, precision, f1 = 1, 1, 1
        else:
            recall, precision, f1 = 0, 0, 0
    elif isinstance(gold_answer, bool):
        if gold_answer == pred_answer:
            recall, precision, f1 = 1, 1, 1
        else:
            recall, precision, f1 = 0, 0, 0
    elif isinstance(gold_answer, list):
        if pred_answer is None:
            recall, precision, f1 = 0, 0, 0
        else:
            pred_answer = set(pred_answer)
            gold_answer = set(gold_answer)
            if len(pred_answer.intersection(gold_answer)) != 0:
                precision = len(pred_answer.intersection(gold_answer)) / len(pred_answer)
                recall = len(pred_answer.intersection(gold_answer)) / len(gold_answer)
                f1 = (2 * recall * precision / (recall + precision))
            else:
                recall, precision, f1 = 0, 0, 0
                
    result_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return result_dict


def compute_f1_metric(predictions, references) -> Dict[str, Any]:
    from tqdm import tqdm
    precision, recall, f1= 0., 0., 0.
    total = 0
    for pred, ref in tqdm(zip(predictions, references)):
        prediction = pred
        reference = ref['query']

        result_dict = compute_prf1_one(prediction, reference)
        p, r, f= result_dict["precision"], result_dict["recall"], result_dict["f1"]
        precision += p
        recall += r
        f1 += f

        total += 1

    return {
        "precision": float(precision/total),
        "recall": float(recall/total),
        "f1": float(f1/total)
    }
