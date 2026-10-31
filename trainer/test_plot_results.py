"""Why this file exists: plot_results.py draws docs/results.svg, the chart that shows the result of the project. A wrong window, a repeated
question counted twice or a hard-coded sentence would put a false number or a false claim into the README without any error. The synthetic
tests check each calculation against numbers worked out by hand. The real-data tests check the chart numbers against the eval summaries
(they skip when trainer/results/ is missing, and then test_every_file_the_chart_reads_is_in_trainer_results fails, so a skip cannot hide it). No GPU, no server, standard library only.

Run (from the trainer folder, any Python 3):
    python3 test_plot_results.py
Each test prints one PASS or FAIL line. The exit code is 1 if one test fails. The tests do not write any file.
"""
import json
import math
import re
import sys
import traceback
import xml.etree.ElementTree as ET

import plot_results as pr


def row(question, reward, step=1):
    return {"question": question, "reward_answer": reward, "step": step}


def val_row(question):
    return {"responses_create_params": {"input": [{"role": "system", "content": "s"}, {"role": "user", "content": "Context.\n\nQuestion: " + question + "\n"}]}}


def fake_data(base=0.2, run2=0.5, run3=0.5, d2=(0.0, 0.02), d3=(0.3, 0.04)):
    return {"n": 126, "bars": {"base": (base, 0.03), "run2": (run2, 0.04), "run3": (run3, 0.04)},
            "run3_minus_run2": d2, "run3_minus_base": d3,
            "curves": {"run2": [(13 + 25 * i, 0.2 + 0.05 * i) for i in range(8)], "run3": [(13 + 25 * i, 0.25 + 0.05 * i) for i in range(8)]}}


def test_mean_se_of_one_two_three_is_two_and_the_square_root_of_one_third():
    mean, se = pr.mean_se([1, 2, 3])
    assert mean == 2 and math.isclose(se, 1 / math.sqrt(3))


def test_question_key_ignores_case_and_repeated_spaces():
    assert pr.question_key("  What  WAS it?\n") == pr.question_key("what was it?")


def test_validation_keys_reads_the_text_after_the_last_question_label():
    assert pr.validation_keys([val_row("What was it?"), val_row("Why  Now?")]) == {"what was it?", "why now?"}


def test_one_per_question_keeps_the_first_row_and_drops_questions_outside_the_set():
    rows = [row("Q1?", 0.1), row("q1?", 0.9), row("Q2?", 0.5), row("Other?", 1.0)]
    out = pr.one_per_question(rows, {"q1?", "q2?"})
    assert sorted(out) == ["q1?", "q2?"] and out["q1?"]["reward_answer"] == 0.1


def test_paired_diff_uses_only_the_questions_in_both_sets_and_b_minus_a():
    a = {"q1": row("q1", 0.0), "q2": row("q2", 0.5), "q3": row("q3", 1.0)}
    b = {"q1": row("q1", 1.0), "q2": row("q2", 1.0)}
    mean, se = pr.paired_diff(a, b)
    assert math.isclose(mean, 0.75) and math.isclose(se, 0.25)   # differences 1.0 and 0.5


def test_window_means_put_step_25_in_the_first_window_and_step_26_in_the_second():
    rows = [row("q", 0.0, 1), row("q", 1.0, 25), row("q", 0.5, 26), row("q", 0.5, 50)]
    assert pr.window_means(rows) == [(13.0, 0.5), (38.0, 0.5)]


def test_the_title_says_equals_only_when_the_difference_is_inside_two_standard_errors():
    assert pr.verdict((-0.011, 0.019)) == "equals"
    assert pr.verdict((0.10, 0.02)) == "is above" and pr.verdict((-0.10, 0.02)) == "is below"


def test_the_svg_title_and_note_follow_the_data():
    same = pr.build_svg(fake_data(d2=(-0.011, 0.019)))
    higher = pr.build_svg(fake_data(d2=(0.10, 0.02)))
    assert "run 3 equals run 2" in same and "(no difference)" in same
    assert "run 3 is above run 2" in higher and "(no difference)" not in higher and "equals" not in higher


def test_the_svg_shows_each_bar_value_with_its_standard_error():
    text = pr.build_svg(fake_data(base=0.161, run2=0.569, run3=0.558))
    assert "0.16 ± 0.03" in text and "0.57 ± 0.04" in text and "0.56 ± 0.04" in text


def _numbers_outside_the_box(svg_text):
    bad = []
    for el in ET.fromstring(svg_text).iter():
        for attr in ("x", "cx", "x1", "x2"):
            if attr in el.attrib and not 0 <= float(el.attrib[attr]) <= pr.WIDTH:
                bad.append((el.tag, attr, el.attrib[attr]))
        for attr in ("y", "cy", "y1", "y2"):
            if attr in el.attrib and not 0 <= float(el.attrib[attr]) <= pr.HEIGHT:
                bad.append((el.tag, attr, el.attrib[attr]))
        for pair in re.findall(r"(-?[\d.]+),(-?[\d.]+)", el.attrib.get("points", "") + " " + el.attrib.get("d", "")):
            if not (0 <= float(pair[0]) <= pr.WIDTH and 0 <= float(pair[1]) <= pr.HEIGHT):
                bad.append((el.tag, "point", pair))
    return bad


def test_the_svg_parses_and_keeps_every_coordinate_inside_the_box_for_extreme_values():
    data = fake_data(base=0.0, run2=0.8, run3=0.8)
    data["bars"]["base"] = (0.0, 0.0)
    assert _numbers_outside_the_box(pr.build_svg(data)) == []


def test_the_svg_root_has_the_box_as_view_box_and_its_points_are_x_y_pairs():
    text = pr.build_svg(fake_data())
    root = ET.fromstring(text)
    assert root.tag.endswith("svg") and root.attrib["viewBox"] == f"0 0 {pr.WIDTH} {pr.HEIGHT}"
    assert "," not in re.sub(r"[\d.]+,[\d.]+", "", " ".join(el.attrib.get("points", "") for el in root.iter()))   # points are x,y pairs only


# ---- real data (skipped when trainer/results/ is missing) --------------------------------------------------------------------------
def _real_or_skip():
    needed = list(pr.EVALS.values()) + list(pr.TRAIN_DUMPS.values()) + [pr.VALIDATION]
    if not all(p.exists() for p in needed):
        print("      (skipped: trainer/results/ or the validation file is missing)")
        return None
    return pr.collect()


def test_every_file_the_chart_reads_is_in_trainer_results():
    needed = list(pr.EVALS.values()) + list(pr.TRAIN_DUMPS.values()) + [pr.VALIDATION, pr.RESULTS / "eval_run3_semantic" / "eval_summary.json"]
    missing = [str(p.relative_to(pr.REPO_ROOT)) for p in needed if not p.exists()]
    assert not missing, f"the chart needs these files and they are not in the repository: {missing}"


def test_real_bars_match_the_eval_summaries_on_126_questions():
    data = _real_or_skip()
    if data is None:
        return
    summary = {m["model"]: m for m in json.loads((pr.RESULTS / "eval_run3_semantic" / "eval_summary.json").read_text())}
    assert data["n"] == 126
    assert math.isclose(data["bars"]["base"][0], summary["base"]["reward"], abs_tol=1e-6)
    assert math.isclose(data["bars"]["run3"][0], summary["final"]["reward"], abs_tol=1e-6)
    assert math.isclose(data["bars"]["run3"][1], summary["final"]["reward_se"], abs_tol=1e-6)
    assert math.isclose(data["bars"]["run2"][0], 0.569, abs_tol=1e-3)


def test_real_paired_differences_match_the_numbers_in_the_report():
    data = _real_or_skip()
    if data is None:
        return
    m, se = data["run3_minus_run2"]
    assert math.isclose(m, -0.011, abs_tol=1e-3) and math.isclose(se, 0.019, abs_tol=1e-3)
    m, se = data["run3_minus_base"]
    assert math.isclose(m, 0.3966, abs_tol=1e-4) and math.isclose(se, 0.0414, abs_tol=1e-3)   # delta_reward_vs_base of the summary


def test_real_curves_have_eight_points_and_end_at_the_known_values():
    data = _real_or_skip()
    if data is None:
        return
    for name, last in (("run2", 0.419), ("run3", 0.461)):
        curve = data["curves"][name]
        assert [s for s, _ in curve] == [13 + 25 * i for i in range(8)]
        assert math.isclose(curve[-1][1], last, abs_tol=1e-3)


def test_real_svg_parses_and_stays_inside_the_box():
    data = _real_or_skip()
    if data is None:
        return
    assert _numbers_outside_the_box(pr.build_svg(data)) == []


def main():
    tests = [(n, f) for n, f in globals().items() if callable(f) and n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}\n" + "".join("      " + line for line in traceback.format_exc().splitlines(True)))
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
