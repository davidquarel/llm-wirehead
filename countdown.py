"""Countdown: reach a target with + - * / using each given number exactly once. Procedural, exactly verifiable,
with disjoint practice and holdout streams (different seed spaces) of unlimited size.

  problems(split, n_nums, n, offset) -> [(prompt, problem)]      split: "practice" or "holdout"
  reward(text, problem) -> 1.0 (correct) / 0.0
"""
import ast, operator, random, re

PRACTICE_SEED, HOLDOUT_SEED = 1_000_003, 7_777_777          # disjoint streams
PROMPT = ("Using the numbers {nums}, make an expression that equals {target}. You may use +, -, *, / and parentheses, "
          "and you must use each number exactly once. Reason step by step, then give the final expression (numbers "
          "and operators only) as \\boxed{{...}}.")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _make(rng, n_nums):
    """A solvable problem: random numbers combined by a random expression into a positive integer target."""
    while True:
        nums = [rng.randint(1, 100) if rng.random() < 0.7 else rng.randint(1, 12) for _ in range(n_nums)]
        vals = [(float(x), str(x)) for x in nums]
        rng.shuffle(vals)
        while len(vals) > 1:
            (a, ea), (b, eb) = vals.pop(), vals.pop()
            op = rng.choice("+-*/")
            if op == "/" and (b == 0 or a % b):
                op = rng.choice("+-*")
            v = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
            vals.append((v, f"({ea} {op} {eb})"))
        t, expr = vals[0]
        if t == int(t) and 1 <= t <= 999:
            return {"nums": sorted(nums), "target": int(t), "witness": expr[1:-1]}


def problems(split, n_nums, n, offset=0):
    base = PRACTICE_SEED if split == "practice" else HOLDOUT_SEED
    out = []
    for i in range(offset, offset + n):
        rng = random.Random(base * 1000 + n_nums * 1_000_000_007 + i)
        p = _make(rng, n_nums)
        out.append((PROMPT.format(nums=p["nums"], target=p["target"]), p))
    return out


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value, [node.value]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        (a, na), (b, nb) = _eval(node.left), _eval(node.right)
        return OPS[type(node.op)](a, b), na + nb
    raise ValueError("unsupported expression")


def extract(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if not m:
        return None
    e = m[-1].replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("×", "*").replace("÷", "/")
    e = re.sub(r"=.*$", "", e).strip()
    return e


def reward(text, problem):
    e = extract(text)
    if e is None:
        return 0.0
    try:
        v, used = _eval(ast.parse(e, mode="eval"))
    except Exception:
        return 0.0
    if sorted(int(x) for x in used if float(x).is_integer()) != problem["nums"] or len(used) != len(problem["nums"]):
        return 0.0
    return 1.0 if abs(v - problem["target"]) < 1e-6 else 0.0


if __name__ == "__main__":
    for k in (3, 4, 5):
        ps = problems("practice", k, 3)
        for prompt, p in ps[:1]:
            print(prompt); print("witness:", p["witness"], "->", reward(f"\\boxed{{{p['witness']}}}", p))
    a = {p["target"] for _, p in problems("practice", 4, 500)}; b = {(tuple(p["nums"]), p["target"]) for _, p in problems("holdout", 4, 500)}
    print("practice/holdout overlap (4 numbers, 500 each):", len({(tuple(p["nums"]), p["target"]) for _, p in problems("practice", 4, 500)} & b))
