"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It provides token-efficiency and
hedging penalties via delta functions.
"""
from typing import List, Union, Tuple
import re

from rewards_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from utils import extract_answer, grade_answer_sympy, grade_answer_mathd, count_hedging_markers
import random
import numpy as np
import math

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END   = "</think>"


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput, ignore_think_token = False) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        elif THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if not ignore_think_token:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            else:
                model_solution = model_response
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

     
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)



# --- Length delta functions ---

def get_delta_score(num_tokens: int, used_tokens: int):
    z = (used_tokens - num_tokens) / 500
    return max(0.1, math.exp(-z*z / 2))


def get_delta_score_linear(num_tokens: int, used_tokens: int, alpha: float = 1/3000):
    z = abs(used_tokens - num_tokens) * alpha
    delta = 1.0 - z
    return max(0.0, min(1.0, delta))


def get_delta_score_linear_both(num_tokens: int, used_tokens: int, alpha: float = 0.002):
    if num_tokens < 0:
        delta = used_tokens - abs(num_tokens)
        sc = (-alpha * delta) if delta >= 0 else (alpha * -delta)
        sc = max(-1.0, min(1.0, sc))
        return (sc + 1.0) / 2.0
    return get_delta_score_linear(num_tokens, used_tokens, alpha)


def get_delta_score_sigmoid(num_tokens: int, used_tokens: int, alpha: float = 0.01):
    d = (used_tokens - num_tokens) * alpha
    s = 1.0 / (1.0 + math.exp(-d))
    return max(0.0, min(1.0, s))


def get_delta_score_sigmoid_exact(num_tokens: int, used_tokens: int, alpha: float = 0.01):
    d = abs(num_tokens - used_tokens) * alpha
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-d))))


def get_binary_score(num_tokens: int, used_tokens: int):
    return 1.0 if used_tokens <= num_tokens else 0.0


# --- Hedging delta functions ---

def get_delta_score_hedge_linear(hedge_count: int, beta: float) -> float:
    return max(0.0, 1.0 - beta * hedge_count)


def get_delta_score_hedge_sigmoid(hedge_count: int, beta: float) -> float:
    d = hedge_count * beta
    return 1.0 / (1.0 + math.exp(d))


# --- GPQA (unchanged) ---

def gpqa_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm=False,
                   num_tokens: int = -1, valid_response_length: int = -1):
    cfg = RewardConfig()
    cfg.use_math_orm = enable_llm
    def get_choice(r):
        for i in range(len(r)-1, -1, -1):
            c = r[i]
            if c in ('A','B','C','D') and not r[i-1].isalpha():
                return c
        return ''
    return 1.0 if get_choice(solution_str) == ground_truth else 0.0


def math_reward_fn(
    solution_str: str,
    ground_truth: Union[str, List[str]],
    num_tokens: int = -1,
    valid_response_length: int = -1,
    ignore_think_token: bool = False,
    reward_config: RewardConfig | None = None,
    return_delta_score: bool = False,
) -> Union[float, Tuple[float, float]]:

    # Defaults
    if reward_config is None:
        reward_config = RewardConfig()
    alpha = float(getattr(reward_config, "alpha", 0.0))
    beta  = float(getattr(reward_config, "beta", 0.0))

    # 1) Correctness
    rm = RewardMathFn(reward_config)
    rr = rm(
        RewardInput(problem=solution_str,
                    problem_type=RewardType.MATH,
                    model_response=solution_str,
                    ground_truth={"answer": ground_truth}),
        ignore_think_token=ignore_think_token
    )
    correctness = float(rr.is_correct)

    # 2) Hedging delta
    hedge_delta = 1.0
    if beta > 0:
        hcnt = count_hedging_markers(solution_str)
        hedge_delta = (get_delta_score_hedge_sigmoid(hcnt, beta)
                       if reward_config.sigmoid_reward
                       else get_delta_score_hedge_linear(hcnt, beta))

    # If no shaping flags, just correctness * hedge
    shaping_on = any((reward_config.linear_reward,
                      reward_config.multiplier_reward,
                      reward_config.sigmoid_reward))
    if not shaping_on:
        final = correctness * hedge_delta
        return (final, hedge_delta) if return_delta_score else final

    # 3) Length delta
    length_delta = 1.0
    if num_tokens != -1 and alpha > 0:
        if num_tokens < 0:  # LCPO-Max
            length_delta = (get_delta_score_sigmoid(num_tokens, float(valid_response_length), alpha)
                            if reward_config.sigmoid_reward
                            else get_delta_score_linear_both(num_tokens, float(valid_response_length), alpha))
        else:               # LCPO-Exact
            length_delta = (get_delta_score_sigmoid_exact(num_tokens, float(valid_response_length), alpha)
                            if reward_config.sigmoid_reward
                            else get_delta_score_linear(num_tokens, float(valid_response_length), alpha))
        if getattr(reward_config, "debug", False):
            print(f"[DBG] lenΔ={length_delta:.3f} corr={correctness} num={num_tokens} used={valid_response_length}")

    # 4) Combine
    if reward_config.multiplier_reward:
        combined_delta = max(0.0, length_delta * hedge_delta)
        final = correctness * combined_delta
    else:
        combined_delta = (length_delta + hedge_delta) / 2.0
        final = (correctness + combined_delta) / 2.0

    final = max(0.0, min(1.0, final))

    return (final, combined_delta) if return_delta_score else final



# --- majority_at_k (unchanged) ---

def majority_at_k(
    generations: List[str],
    ground_truths: Union[str, List[str]],
    k: int = -1,
    problem: str = "",
    enable_llm: bool = False,
    ignore_think_token: bool = False,
    shuffle: bool = False
) -> str:
    if not isinstance(ground_truths, list) and not isinstance(ground_truths, np.ndarray):
        ground_truths = [ground_truths]
    processed_gt = []
    for truth in ground_truths:
        t = str(truth)
        if "\\boxed" in t:
            ext = extract_answer(t)
            if ext:
                processed_gt.append(ext)
        else:
            processed_gt.append(t)
    if k > 0 and k < len(generations):
        gens = random.sample(generations, k) if shuffle else generations[:k]
    else:
        gens = generations

    answers = []
    for g in gens:
        if ignore_think_token:
            g = re.sub(r'<think>.*?</think>', '', g, flags=re.DOTALL)
        if "\\boxed" in g:
            ext = extract_answer(g)
            if ext:
                answers.append(ext)
        else:
            answers.append(g)

    clusters, counts = [], []
    for ans in answers:
        found = False
        for i, rep in enumerate(clusters):
            if grade_answer_mathd(ans, rep) or grade_answer_sympy(ans, rep):
                counts[i] += 1
                found = True
                break
        if not found:
            clusters.append(ans)
            counts.append(1)
    if not clusters:
        return 0.0
    idx = counts.index(max(counts))
    final = clusters[idx]
    for gt in processed_gt:
        if grade_answer_mathd(final, gt) or grade_answer_sympy(final, gt):
            return 1.0
    return 0.0


if __name__ == "__main__":
    cfg = RewardConfig()
    reward = RewardMathFn(cfg)
    inp = RewardInput(
        problem="x+1=2",
        problem_type=RewardType.MATH,
        model_response="The answer is \boxed{1}",
        ground_truth={"answer": "1"}
    )
    print(reward(inp))
