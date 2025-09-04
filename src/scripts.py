import json
import time
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
from langchain_ollama import OllamaLLM
import statistics
from stars_agent import StarsAgent
from utils import timeout, time_monitor


from typing import Callable, List, Tuple

Triple = Tuple[int, int, int]



def blotto_cmp(a, b):
    aw = sum(1 for x, y in zip(a, b) if x > y)
    bw = sum(1 for x, y in zip(a, b) if y > x)

    if aw > bw:
        return 1
    elif aw == bw:
        return 0
    else:
        return -1


def compare_all(a_list, b_list):
    all_result = []
    for a in a_list:
        total = 0
        no_loss = 0
        for b in b_list:
            if blotto_cmp(a, b) >= 0:
                no_loss += 1
            total += 1
        all_result.append([a, no_loss*1.0/total])
    return sorted(all_result, key=lambda x: x[1], reverse=True), np.average([i[1] for i in all_result])


def make_sum20_generator(rule: Callable[[int, int, int], bool]) -> List[Triple]:
    return [(i, j, k)
            for i in range(21)
            for j in range(21 - i)
            for k in [20 - i - j]
            if rule(i, j, k)]

def two_ten(i: int, j: int, k: int) -> bool:
    return (i==0 and j==k) or (j==0 and i==k) or (k==0 and i==j)

def two_five(i: int, j: int, k: int) -> bool:
    return (i==0 and j==k) or (j==0 and i==k) or (k==0 and i==j)

def at_least_one_gt13(i: int, j: int, k: int) -> bool:
    return i > 13 or j > 13 or k > 13

def at_least_one_gt11(i: int, j: int, k: int) -> bool:
    return i > 11 or j > 11 or k > 11

def at_least_one_lt2(i: int, j: int, k: int) -> bool:
    return i < 2 or j < 2 or k < 2

def at_least_one_lt2_and_even(i: int, j: int, k: int) -> bool:
    return ((i < 2 and statistics.stdev([j, k]) < 2. )
            or (j < 2 and statistics.stdev([i, k]) < 2.0)
            or (k < 2 and statistics.stdev([i, j]) < 2.0) )

def at_least_one_lt3_and_even(i: int, j: int, k: int) -> bool:
    return ((i < 3 and statistics.stdev([j, k]) < 2. )
            or (j < 3 and statistics.stdev([i, k]) < 2.0)
            or (k < 3 and statistics.stdev([i, j]) < 2.0) )

def at_least_one_lt4_and_even(i: int, j: int, k: int) -> bool:
    return ((i < 4 and statistics.stdev([j, k]) < 2. )
            or (j < 4 and statistics.stdev([i, k]) < 2.0)
            or (k < 4 and statistics.stdev([i, j]) < 2.0) )

def average(i: int, j: int, k: int) -> bool:
    return statistics.stdev([i, j, k]) < 2.0

def no_condition(i: int, j: int, k: int) -> bool:
    return True


if __name__ == "__main__":

    func_list = [at_least_one_lt2, at_least_one_lt2_and_even, at_least_one_lt3_and_even, average, no_condition, at_least_one_gt13, at_least_one_gt11, two_ten, two_five]
    for i in func_list:
        for j in func_list:
            result_list1, avg = compare_all(make_sum20_generator(i), make_sum20_generator(j))
            if avg > 0.5:
                print(avg, i.__name__, j.__name__)