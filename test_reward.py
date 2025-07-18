# test_reward.py
import sys; sys.path.insert(0, ".")
from math_reward import RewardMathFn
from rewards_types import RewardConfig, RewardInput, RewardType

cfg = RewardConfig()
cfg.alpha = 0.1
cfg.beta  = 0.2

reward_fn = RewardMathFn(cfg)

def run_case(resp, truth, desc):
    inp = RewardInput(
        problem="",
        problem_type=RewardType.MATH,
        model_response=resp,
        ground_truth={"answer": truth}
    )
    out = reward_fn(inp)
    # grab the raw numbers by re-computing inside test:
    # (you can also add debug prints inside RewardMathFn itself)
    # but let's just print what we got:
    print(
        f"{desc:25s} → reward={out.reward:.3f}  correct={out.is_correct}"
    )

cases = [
    ("The answer is \\boxed{42}", "42", "Exact correct"),
    ("Answer: 42",               "42", "No box but correct"),
    ("I think maybe \\boxed{42}", "42", "With hedging"),
    ("\\boxed{24}",              "42", "Wrong answer"),
    ("<think>…</think> 42",      "42", "With CoT tags"),
    ("42 extra words here",      "42", "Length deviation"),
]

for resp, truth, desc in cases:
    run_case(resp, truth, desc)
