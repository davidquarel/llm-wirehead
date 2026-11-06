"""A persistent Python kernel for the editor LLM: reads code blocks (JSON lines on stdin), runs each in one shared
namespace with the harness client preloaded, and answers with the printed output (JSON lines on stdout).

Refused before running: anything that takes gradients, trains, or runs the model outside the harness."""
import contextlib, io, json, os, re, signal, sys, traceback

FORBIDDEN = [r"\.backward\s*\(", r"\bautograd\b", r"torch\.optim", r"\boptim\.", r"requires_grad", r"(?<!AutoTokenizer\.)from_pretrained",
             r"\bvllm\b", r"\bLLM\s*\(", r"\bsubprocess\b", r"os\.system", r"\bsocket\b", r"\bjax\b", r"\.grad\b",
             r"safetensors.*save", r"\bAutoModel"]
LIMIT_S = int(os.environ.get("WSP_LIMIT_S", 600))


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout(f"the code ran longer than {LIMIT_S} s and was stopped")


def main():
    sys.path.insert(0, sys.argv[1])          # the harness folder (client.py, task.py)
    ns = {"__name__": "__kernel__"}
    _setup_out = contextlib.redirect_stdout(sys.stderr)       # stdout is the JSON channel to the agent: setup prints go to stderr
    _setup_out.__enter__()
    exec("import numpy as np\nfrom client import W, gen, tf, set_edit, revert, status, load_tf, embed_rows\nimport task\n"
         "try:\n    from client import keep, undo, checkpoints, health, chat_self\nexcept ImportError:\n    pass\n"
         "\n"
         "from transformers import AutoTokenizer\ntok = AutoTokenizer.from_pretrained(status()['weights'])\n"
         "try:\n    import wsp_tools as tools\nexcept Exception:\n    pass\n"
         "import os\nif os.environ.get('WSP_BLISS'):\n    import bliss\n    bench = bliss.Bench(port=int(os.environ['WSP_PORT']), log=os.path.abspath('../bench.jsonl'))\n"
         "if os.environ.get('WSP_SURGERY'):\n    import surgery\n    S = surgery.Surgery(int(os.environ['WSP_PORT']), os.environ['WSP_SURGERY'])\n"
         "if os.environ.get('WSP_MATH'):\n    import surgery_math\n    S = surgery_math.MathSurgery(int(os.environ['WSP_PORT']), os.environ['WSP_MATH'])", ns)
    if os.environ.get("WSP_MATH"):            # the holdout must stay unseen: no access to it or to the problem generator
        FORBIDDEN.extend([r"(?i)holdout", r"(?i)HOLDOUT_SEED", r"\bcountdown\b", r"surgery_math\.", r"\bS\.c\b", r"__dict__",
                          r"\bopen\s*\(\s*['\"][^'\"]*\.py", r"\bimportlib\b", r"\bglobals\s*\(", r"\b_fresh\b", r"\boffset\b"])
    _setup_out.__exit__(None, None, None)
    signal.signal(signal.SIGALRM, _alarm)
    real_stdout = sys.stdout
    for line in sys.stdin:
        code = json.loads(line)["code"]
        bad = [p for p in FORBIDDEN if re.search(p, code)]
        if bad:
            out = f"REFUSED: the code matches a forbidden pattern ({bad[0]}). No gradients, training, or running the model outside gen()/tf()."
        else:
            buf = io.StringIO()
            signal.alarm(LIMIT_S)
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(code, "<cell>", "exec"), ns)
            except BaseException as e:  # report everything to the editor, keep the kernel alive
                if isinstance(e, KeyboardInterrupt):
                    raise
                buf.write("".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:]))
            finally:
                signal.alarm(0)
            out = buf.getvalue()
        real_stdout.write(json.dumps({"out": out}) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
