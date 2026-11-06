"""The ARENA 2.4.2 multiplication task (copied from chapter2_rl/instructions/pages/06_[2.4.2]_RLVR.md)."""
import random
import re

DIFFICULTY_LADDER = [
    ("3x2", (100, 999), (10, 99)),
    ("4x2", (1000, 9999), (10, 99)),
    ("3x3", (100, 999), (100, 999)),
    ("4x3", (1000, 9999), (100, 999)),
    ("4x4 (b<3000)", (1000, 9999), (1000, 2999)),
    ("4x4", (1000, 9999), (1000, 9999)),
    ("4x5", (1000, 9999), (10000, 99999)),
    ("5x5", (10000, 99999), (10000, 99999)),
]

# 2.4.2's sampler: temperature 1.0, top-k 50, top-p 0.95, 768 new tokens
SAMPLING = dict(temperature=1.0, top_k=50, top_p=0.95, max_tokens=768)

# Seeds: practice problems (the editor may use these freely) vs the held-out test set (harness only)
PRACTICE_SEED, TEST_SEED = 1, 987654321


def mult_byhand(rng: random.Random, difficulty: int = 0) -> tuple[str, int]:
    _, (a_lo, a_hi), (b_lo, b_hi) = DIFFICULTY_LADDER[difficulty]
    a, b = rng.randint(a_lo, a_hi), rng.randint(b_lo, b_hi)
    prompt = f"Compute {a} * {b}. Reason step by step, then give the final answer as \\boxed{{N}}.\n"
    return prompt, a * b


def problems(seed: int, difficulty: int, n: int) -> list[tuple[str, int]]:
    rng = random.Random(seed * 1000 + difficulty)
    return [mult_byhand(rng, difficulty) for _ in range(n)]


def extract_boxed_int(text: str) -> int | None:
    matches = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", text)
    return int(matches[-1]) if matches else None


def reward(text: str, answer: int) -> float:
    boxed = extract_boxed_int(text)
    return 0.0 if boxed is None else 1.0 if boxed == answer else 0.1


TEST_N = 2000      # held-out problems per rung (the test set evaluates a prefix of these)


def practice(difficulty: int, n: int, offset: int = 0) -> list[tuple[str, int]]:
    """Practice problems: the practice stream with every held-out test problem removed."""
    test = {p for p, _ in problems(TEST_SEED, difficulty, TEST_N)}
    rng, out = random.Random(PRACTICE_SEED * 1000 + difficulty), []
    while len(out) < offset + n:
        p = mult_byhand(rng, difficulty)
        if p[0] not in test:
            out.append(p)
    return out[offset:]
