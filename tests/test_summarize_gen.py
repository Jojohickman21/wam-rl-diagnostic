"""Tests for scripts/summarize_gen.py: parsing evaluation files, the bar fixed in TASK_SET.md and the
scoring of Josiah's predictions, on synthetic files in the exact format scripts/eval_policy.py writes."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_gen.py"
spec = importlib.util.spec_from_file_location("summarize_gen", SCRIPT)
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

G1, G2, G3 = "WamFamilyDesc-v0", "WamFamilyContact-v0", "WamFamilyContactFK-v0"


def write(directory, env, task, successes, spill=0.1, timeout=0.05):
    for seed, s in enumerate(successes, start=1):
        line = (f"{env} seed {seed} on {task} (yaw ±20°, 200 episodes): success {s:.3f} | valid fraction of "
                f"successes 0.990 | spill {spill:.3f} | timeout {timeout:.3f} | final distance (normalised) median 0.05\n")
        (directory / f"{env}__{seed}__{task}__20.txt").write_text(line)


def evals(tmp_path, **by_key):
    """by_key: name like G2_T2=[...] -> files for that variant and task."""
    envs = {"G1": G1, "G2": G2, "G3": G3}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for key, values in by_key.items():
        variant, task = key.split("_", 1)
        write(tmp_path, envs[variant], task, values)
    return sg.load(tmp_path)


# ---------- parsing ----------

def test_load_parses_every_field_and_skips_unfinished_files(tmp_path):
    write(tmp_path, G2, "T2", [0.9, 0.7], spill=0.2, timeout=0.1)
    (tmp_path / f"{G2}__3__T2__20.txt").write_text("")  # an evaluation still running
    (tmp_path / f"{G2}__4__T2__20.txt").write_text("Traceback (most recent call last):\n")
    ev = sg.load(tmp_path)
    assert ev[(G2, "T2")] == {1: dict(success=0.9, spill=0.2, timeout=0.1), 2: dict(success=0.7, spill=0.2, timeout=0.1)}


def test_successes_are_in_seed_order(tmp_path):
    ev = evals(tmp_path, G1_T2=[0.1, 0.2, 0.3])
    assert sg.successes(ev, G1, "T2") == [0.1, 0.2, 0.3]
    assert sg.successes(ev, G1, "T1") == []


# ---------- rank-sum ----------

def test_ranksum_is_one_for_identical_groups_and_minimal_for_full_separation():
    assert sg.ranksum_p([0.5] * 4, [0.5] * 4) == pytest.approx(1.0)
    # 4 vs 4 completely separated: 2 of the C(8,4) = 70 labellings are as extreme (two-sided).
    assert sg.ranksum_p([0.9, 0.8, 0.85, 0.95], [0.1, 0.2, 0.15, 0.25]) == pytest.approx(2 / 70)


# ---------- the bar ----------

def verdicts(ev):
    return {name: verdict for name, verdict, _ in sg.bar(ev)}


def test_validity_needs_both_family_boxes_at_080(tmp_path):
    ev = evals(tmp_path, G2_T4x=[0.8, 0.8], G2_T4xneg=[0.9, 0.9], G3_T4x=[0.9, 0.9], G3_T4xneg=[0.79, 0.79])
    v = verdicts(ev)
    assert v["Validity, G2 contact target"] is True
    assert v["Validity, G3 contact target + FK"] is False


def test_main_claim_needs_mean_070_and_no_overlap_with_the_label_policy(tmp_path):
    # G2: mean 0.75 but one seed at 0.50 overlaps the label policy's 0.22-0.56, so not met.
    # G3: every seed above 0.56 and mean exactly 0.70, so met (the bar is >= 0.70).
    ev = evals(tmp_path, G2_T2=[0.5, 0.9, 0.85, 0.75], G3_T2=[0.6, 0.8, 0.7, 0.7])
    v = verdicts(ev)
    assert v["Main claim, G2 on T2"] is False
    assert v["Main claim, G3 on T2"] is True


def test_main_claim_fails_below_070_even_without_overlap(tmp_path):
    ev = evals(tmp_path, G2_T2=[0.69, 0.69, 0.69])
    assert verdicts(ev)["Main claim, G2 on T2"] is False


def test_bar_is_pending_when_results_are_missing(tmp_path):
    ev = evals(tmp_path, G2_T4x=[0.9])  # no T4xneg yet, nothing on T2
    v = verdicts(ev)
    assert v["Validity, G2 contact target"] is None
    assert v["Main claim, G2 on T2"] is None
    assert "PENDING" in sg.report(ev)


# ---------- predictions ----------

def test_p1_needs_g2_above_both_g1_and_section10s_best(tmp_path):
    assert sg.predictions(evals(tmp_path, G1_T2=[0.3, 0.3], G2_T2=[0.5, 0.5]))["P1"] is True


def test_p1_is_false_when_g2_beats_g1_but_not_section10(tmp_path):
    assert sg.predictions(evals(tmp_path, G1_T2=[0.2, 0.2], G2_T2=[0.4, 0.4]))["P1"] is False


def test_p1_prime_needs_strictly_above_080(tmp_path):
    assert sg.predictions(evals(tmp_path / "at", G2_T2=[0.8, 0.8]))["P1'"] is False
    assert sg.predictions(evals(tmp_path / "above", G2_T2=[0.81, 0.81]))["P1'"] is True


def test_p2_true_when_g3_and_g2_cannot_be_separated(tmp_path):
    p = sg.predictions(evals(tmp_path, G2_T2=[0.7, 0.8, 0.6, 0.9], G3_T2=[0.75, 0.65, 0.85, 0.7]))
    assert p["P2"] is True


def test_p2_false_when_g3_is_clearly_different(tmp_path):
    p = sg.predictions(evals(tmp_path, G2_T2=[0.1, 0.2, 0.15, 0.25], G3_T2=[0.9, 0.8, 0.85, 0.95]))
    assert p["P2"] is False


def test_p3_prime_needs_g1_strictly_below_060(tmp_path):
    assert sg.predictions(evals(tmp_path / "at", G1_T2=[0.6, 0.6]))["P3'"] is False
    assert sg.predictions(evals(tmp_path / "below", G1_T2=[0.59, 0.59]))["P3'"] is True


def test_predictions_are_pending_without_data(tmp_path):
    assert sg.predictions(sg.load(tmp_path)) == {"P1": None, "P1'": None, "P2": None, "P3'": None}


# ---------- report ----------

def test_report_lists_every_variant_and_task_present(tmp_path):
    ev = evals(tmp_path, G1_T2=[0.3], G2_T2=[0.8], G2_T1=[0.9])
    text = sg.report(ev)
    assert "| G1 joint descriptor | T2 | 1 | 0.300 |" in text
    assert "| G2 contact target | T1 | 1 | 0.900 |" in text
    assert "Seeds evaluated on T2: G1 1, G2 1, G3 0" in text


def test_p2_false_when_ranges_overlap_but_the_rank_sum_separates_them(tmp_path):
    # One G2 seed (0.60) reaches above one G3 seed (0.50), so the ranges overlap, but 15 of the 16
    # seeds are ordered G2 < G3: the rank-sum test separates them (p < 0.05), so P2 must be false.
    g2 = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.60]
    g3 = [0.50, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
    assert sg.overlap(g2, g3) and sg.ranksum_p(g2, g3) < 0.05
    assert sg.predictions(evals(tmp_path, G2_T2=g2, G3_T2=g3))["P2"] is False
