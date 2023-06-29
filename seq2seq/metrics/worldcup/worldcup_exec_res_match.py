# this file is for compute_metrics in the ...
"""Worldcup exec match metric"""

from typing import Dict, Tuple, List, Any, Set
from itertools import product
from collections import defaultdict
import re
import random
import datetime
from utils.sql_database import SQLDatabase
from functools import lru_cache

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    res = False
    details = ''
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        if s1 == s2:
            res = True
        else:
            details = f'Results are not identical in ORDERED comparison\nGT:{s1}\nGT{s2}'
                 
    else:
        if set(s1) == set(s2):
            res = True
        else:
            details = f'Results are not identical in UNORDERED comparison\nGT:{s1}\nGT{s2}'
    
    return res, details


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> Tuple[bool, str]:
    details = '\n'
    if len(result1) == 0 and len(result2) == 0:
        return True, details

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False, details

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        details += f'results do not have the same number of columns, they are different, GT: {num_cols}, PT: {result2[0]}'
        return False, details

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False, details

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                details += f'ORDERED: FOUND matched permutation\nGT: {result1}\nPT: {result2}\nPT_perm: {result2_perm}'
                return True, details
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                details += f'UNORDERED: FOUND matched permutation\nGT: {set(result1)}\nPT: {set(result2)}\nPT_perm: {set(result2_perm)}'
                return True, details

    if order_matters:
        details += f'Results are not identical in ORDERED comparison\nGT:{tab1_sets_by_columns}\nGT{result2}'
    else:
        details += f'Results are not identical in UNORDERED comparison\nGT:{tab1_sets_by_columns}\nGT{result2}'
    return False, details


def replace_cur_year(query: str, cur_year=None) -> str:
    if cur_year is None:
        cur_year = datetime.datetime.now().year
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", cur_year, query, flags=re.IGNORECASE
    )

def compute_exec_res_match(predictions, references) -> Dict[str, Any]:
    n = len(predictions) * 1.0
    acc = 0
    for _prediction, reference in zip(predictions, references):
        gt_res = get_results(reference['query'], reference['db_uri'], reference['db_schema'])
        prediction = _prediction.split('|')[-1].strip()
        pred_res = get_results(prediction, reference['db_uri'], reference['db_schema'])
        res, _ = result_eq(gt_res, pred_res, order_matters=True)
        acc += int(res)
    return {
        "exec_res_match": acc/n
    }

# @TODO implement in async I/O
@lru_cache(maxsize=1000)
def get_results(query, db_uri, db_schema):
    db = SQLDatabase.from_uri(db_uri, schema=db_schema)
    res = db.run(command=query, fetch="many", fmt="list", limit_num=100)
    return res
        
