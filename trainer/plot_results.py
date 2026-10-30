"""Draw docs/results.svg: the base model, run 2 and run 3 on the held-out questions, and the training curves of run 2 and run 3.

The chart is plain SVG written with the standard library (no plotting package, GitHub shows it in the README). It reads the eval and
train dumps in trainer/runs/ (git-ignored), so only the finished SVG is committed. Every number in it is computed from the dumps.

Run (from the trainer folder, any Python 3):
    python3 plot_results.py            # writes ../docs/results.svg
Test:
    python3 test_plot_results.py

All three evaluated models use the old exact-word reward (`reward_answer`) on the 126 questions of data/validation_dedup.jsonl, with the
BGE-M3 retriever at top 2. The training curves use the same `reward_answer` of the training rollouts (sampled at temperature 0.9).
"""
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

TRAINER_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINER_DIR.parent
RUNS = TRAINER_DIR / "runs"
VALIDATION = REPO_ROOT / "NeMo_Gym" / "resources_servers" / "mission_search" / "data" / "validation_dedup.jsonl"
OUT = REPO_ROOT / "docs" / "results.svg"

EVALS = {   # the three models, all evaluated with BGE-M3 at top 2 (the base comes from the run-3 eval session)
    "base": RUNS / "eval_run3_semantic" / "eval_base.jsonl",
    "run2": RUNS / "eval_bge_m3_top2" / "eval_final.jsonl",
    "run3": RUNS / "eval_run3_semantic" / "eval_final.jsonl",
}
TRAIN_DUMPS = {
    "run2": RUNS / "20261021-141800_run2_retrieval" / "train_dump.jsonl",
    "run3": RUNS / "20261030-080054_run3_semantic" / "train_dump.jsonl",
}
FIELD = "reward_answer"        # the old rule: the yardstick of Learnings L17
WINDOW = 25                    # training steps in one point of a curve


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question_key(text: str) -> str:
    """The same rule as normalize_question in convert_questions.py: lowercase, spaces collapsed."""
    return " ".join(text.lower().split())


def validation_keys(rows: list) -> set:
    """Question keys of the rows of validation_dedup.jsonl (the question follows the last "Question:" of the user message)."""
    return {question_key(r["responses_create_params"]["input"][-1]["content"].rsplit("Question:", 1)[-1]) for r in rows}


def one_per_question(rows: list, keep: set) -> dict:
    """{question key: first row} for the questions in `keep`. An old dump repeats a question; validation_dedup does not."""
    out = {}
    for r in rows:
        key = question_key(r["question"])
        if key in keep and key not in out:
            out[key] = r
    return out


def mean_se(values: list) -> tuple:
    n = len(values)
    mean = sum(values) / n
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)


def paired_diff(a: dict, b: dict, field: str = FIELD) -> tuple:
    """Mean and standard error of b - a over the questions in both dicts."""
    keys = sorted(set(a) & set(b))
    return mean_se([b[k][field] - a[k][field] for k in keys])


def window_means(rows: list, field: str = FIELD, size: int = WINDOW) -> list:
    """[(middle step, mean)] for the training steps 1..size, size+1..2*size, and so on."""
    by_window = {}
    for r in rows:
        by_window.setdefault((r["step"] - 1) // size, []).append(r[field])
    return [(w * size + 1 + (size - 1) / 2, sum(v) / len(v)) for w, v in sorted(by_window.items())]


def collect() -> dict:
    keep = validation_keys(read_jsonl(VALIDATION))
    evals = {name: one_per_question(read_jsonl(path), keep) for name, path in EVALS.items()}
    short = {name: len(rows) for name, rows in evals.items() if len(rows) != len(keep)}
    if short:
        raise SystemExit(f"These evals miss validation questions (found, expected {len(keep)}): {short}")
    return {
        "n": len(keep),
        "bars": {name: mean_se([r[FIELD] for r in rows.values()]) for name, rows in evals.items()},
        "run3_minus_run2": paired_diff(evals["run2"], evals["run3"]),
        "run3_minus_base": paired_diff(evals["base"], evals["run3"]),
        "curves": {name: window_means(read_jsonl(path)) for name, path in TRAIN_DUMPS.items()},
    }


# ---- drawing -------------------------------------------------------------------------------------------------------------------------
# Colours: categorical slots 1 and 2 of the data-viz reference palette (blue, orange), a neutral grey for the base model, and its
# light and dark surface tokens. The SVG has its own opaque background, so it is readable on any page theme.
WIDTH, HEIGHT = 1000, 560
X0_L, X1_L = 80, 470           # left panel: bars
X0_R, X1_R = 550, 930          # right panel: curves
Y_TOP, Y_BASE, Y_MAX = 130, 380, 0.8

CSS = """
.bg{fill:var(--surface)} .grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--muted);stroke-width:1}
.t1{fill:var(--ink);font:700 20px system-ui,sans-serif} .t2{fill:var(--ink2);font:13px system-ui,sans-serif}
.pt{fill:var(--ink);font:600 14px system-ui,sans-serif} .lab{fill:var(--ink);font:600 14px system-ui,sans-serif}
.val{fill:var(--ink);font:700 15px system-ui,sans-serif} .sub{fill:var(--ink2);font:12px system-ui,sans-serif}
.tick{fill:var(--ink2);font:12px system-ui,sans-serif} .err{stroke:var(--ink2);stroke-width:1.5;fill:none}
.base{fill:var(--base)} .c1{fill:var(--c1)} .c2{fill:var(--c2)}
.l1{stroke:var(--c1);stroke-width:2;fill:none} .l2{stroke:var(--c2);stroke-width:2;fill:none}
.m1{fill:var(--c1);stroke:var(--surface);stroke-width:2} .m2{fill:var(--c2);stroke:var(--surface);stroke-width:2}
svg{--surface:#fcfcfb;--grid:#e7e6e2;--ink:#0b0b0b;--ink2:#52514e;--muted:#8f8e89;--base:#9a9993;--c1:#2a78d6;--c2:#eb6834}
@media (prefers-color-scheme:dark){svg{--surface:#1a1a19;--grid:#2e2e2c;--ink:#f0efec;--ink2:#c3c2b7;--muted:#6f6e69;--base:#86857e;--c1:#3987e5;--c2:#d95926}}
"""


def _y(value: float) -> float:
    return Y_BASE - value / Y_MAX * (Y_BASE - Y_TOP)


def _f(x: float) -> str:
    return f"{x:.1f}"


def _text(x: float, y: float, cls: str, text: str, anchor: str = "start") -> str:
    return f'<text x="{_f(x)}" y="{_f(y)}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>'


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def verdict(diff: tuple) -> str:
    """Words for a paired difference (mean, standard error): "equals" when it is within 2 standard errors of zero."""
    mean, se = diff
    return "equals" if abs(mean) < 2 * se else ("is above" if mean > 0 else "is below")


def _axes(x0: float, x1: float) -> list:
    parts = []
    for tick in (0, 0.2, 0.4, 0.6, 0.8):
        y = _y(tick)
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{_f(y)}" y2="{_f(y)}" class="{"axis" if tick == 0 else "grid"}"/>')
        parts.append(_text(x0 - 8, y + 4, "tick", f"{tick:.1f}", "end"))
    return parts


def build_svg(data: dict) -> str:
    bars, curves, n = data["bars"], data["curves"], data["n"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img">',
             f"<style>{CSS}</style>",
             f"<title>Answer reward of the base model, run 2 and run 3</title>",
             "<desc>" + escape("; ".join(f"{name} {m:.3f} +- {se:.3f}" for name, (m, se) in bars.items())
                               + f" on {n} held-out questions (old exact-word rule). Run 3 minus run 2: "
                               f"{data['run3_minus_run2'][0]:+.3f} +- {data['run3_minus_run2'][1]:.3f}.") + "</desc>",
             f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}"/>']
    parts.append(_text(40, 40, "t1", f"Reward rises from {bars['base'][0]:.2f} to {bars['run3'][0]:.2f} with training; run 3 {verdict(data['run3_minus_run2'])} run 2"))
    parts.append(_text(40, 64, "t2", f"Old exact-word reward, mean ± 1 standard error. {n} held-out questions, greedy decoding, "
                                    "BGE-M3 retriever at top 2."))
    parts.append(_text(X0_L, 108, "pt", "Held-out questions"))
    parts.append(_text(X0_R, 108, "pt", "During training"))

    # left panel: three bars with error bars
    parts += _axes(X0_L, X1_L)
    labels = [("base", "base", "Base model", "untrained"), ("run2", "c1", "Run 2", "retrieval reward"), ("run3", "c2", "Run 3", "BGE-M3 + credit")]
    slot = (X1_L - X0_L) / len(labels)
    for i, (key, cls, name, sub) in enumerate(labels):
        mean, se = bars[key]
        cx = X0_L + slot * (i + 0.5)
        top, hi, lo = _y(mean), _y(mean + se), _y(max(mean - se, 0))
        parts.append(f'<path class="{cls}" d="M{_f(cx - 40)},{Y_BASE} V{_f(top + 4)} Q{_f(cx - 40)},{_f(top)} {_f(cx - 36)},{_f(top)} '
                     f'H{_f(cx + 36)} Q{_f(cx + 40)},{_f(top)} {_f(cx + 40)},{_f(top + 4)} V{Y_BASE} Z"/>')
        parts.append(f'<path class="err" d="M{_f(cx)},{_f(hi)} V{_f(lo)} M{_f(cx - 8)},{_f(hi)} H{_f(cx + 8)} M{_f(cx - 8)},{_f(lo)} H{_f(cx + 8)}"/>')
        parts.append(_text(cx, hi - 9, "val", f"{mean:.2f} ± {se:.2f}", "middle"))
        parts.append(_text(cx, Y_BASE + 22, "lab", name, "middle"))
        parts.append(_text(cx, Y_BASE + 39, "sub", sub, "middle"))
    d2, d3 = data["run3_minus_run2"], data["run3_minus_base"]
    parts.append(_text((X0_L + X1_L) / 2, Y_BASE + 68, "sub", f"Run 3 minus run 2: {_signed(d2[0])} ± {d2[1]:.3f}" + (" (no difference)" if verdict(d2) == "equals" else ""), "middle"))
    parts.append(_text((X0_L + X1_L) / 2, Y_BASE + 85, "sub", f"Run 3 minus base: {_signed(d3[0])} ± {d3[1]:.3f}", "middle"))

    # right panel: two curves
    parts += _axes(X0_R, X1_R)
    def x(step: float) -> float:
        return X0_R + (X1_R - X0_R) * step / 200
    for step in (0, 50, 100, 150, 200):
        parts.append(_text(x(step), Y_BASE + 18, "tick", str(step), "middle"))
    parts.append(_text((X0_R + X1_R) / 2, Y_BASE + 38, "sub", "training step", "middle"))
    for key, line_cls, mark_cls, name, dy in (("run2", "l1", "m1", "Run 2", 11), ("run3", "l2", "m2", "Run 3", -2)):
        points = [(x(s), _y(v)) for s, v in curves[key]]
        parts.append(f'<polyline class="{line_cls}" points="' + " ".join(f"{_f(px)},{_f(py)}" for px, py in points) + '"/>')
        parts += [f'<circle class="{mark_cls}" cx="{_f(px)}" cy="{_f(py)}" r="4.5"/>' for px, py in points]
        parts.append(_text(points[-1][0] + 10, points[-1][1] + dy, "sub", f"{name}  {curves[key][-1][1]:.2f}"))
    for i, (cls, name) in enumerate((("c1", "Run 2"), ("c2", "Run 3"))):
        lx = X0_R + 24 + i * 90
        parts.append(f'<circle class="{cls}" cx="{lx}" cy="{Y_TOP + 18}" r="5"/>')
        parts.append(_text(lx + 11, Y_TOP + 22, "sub", name))

    notes = ["Left: the three models on the same questions, scored by the old rule only (the semantic credit is not counted here).",
             "Run 2: old retriever in training, plus a retrieval reward. Run 3: BGE-M3 retriever, semantic credit, de-duplicated training data.",
             f"Right: reward of the training rollouts (temperature 0.9, same old rule), mean of each {WINDOW} steps.",
             "The two runs saw the questions in a different order, so the curves show a trend, not a test."]
    for i, note in enumerate(notes):
        parts.append(_text(40, 498 + i * 17, "sub", note))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_svg(data), encoding="utf-8")
    for name, (m, se) in data["bars"].items():
        print(f"{name:5s} {m:.3f} +- {se:.3f}  (n={data['n']})")
    print("run3 - run2 %+.3f +- %.3f | run3 - base %+.3f +- %.3f" % (*data["run3_minus_run2"], *data["run3_minus_base"]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
