# test_math_reward.py
from math_reward import RewardMathFn
from rewards_types import RewardInput, RewardType, RewardConfig

# 1) Build a RewardConfig with your α/β
cfg = RewardConfig()
cfg.alpha = 0.1   # length penalty weight
cfg.beta  = 0.2   # hedging penalty weight

# 2) Instantiate your reward fn
reward_fn = RewardMathFn(cfg)

# 3) Helper to run one test
def run_case(resp:str, truth, desc:str):
    inp = RewardInput(
        problem="",
        problem_type=RewardType.MATH,
        model_response=resp,
        ground_truth={"answer": truth}
    )
    out = reward_fn(inp)
    print(f"{desc:20s} → reward={out.reward:0.3f}, correct={out.is_correct}")

# 4) Test cases
run_case("The answer is \\boxed{42}",              "42",    "Exact correct")
run_case("Answer: 42  ",                            "42",    "No box but correct")
run_case("I think maybe the answer is \\boxed{42}", "42",    "Hedging word ‘maybe’")
run_case("The answer is \\boxed{24}",               "42",    "Wrong answer")
run_case("<think>…</think> 42",                     "42",    "With CoT tags")
run_case("42 but extra words here to bloat length", "42",    "Length deviation")

# 5) If you have the math_reward_fn wrapper, you can also do:
from math_reward import math_reward_fn
print("Wrapper call:", math_reward_fn("42", "42", num_tokens=1, valid_response_length=3, reward_config=cfg))
