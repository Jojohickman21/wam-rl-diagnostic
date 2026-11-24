"""Tests for scripts/analyze_multitask.py, on small synthetic TensorBoard logs written the way
scripts/train_ppo.py writes them: one `task/success` and one `task/index` scalar per finished
episode, at the current global step, so several episodes can share a step."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_multitask.py"
spec = importlib.util.spec_from_file_location("analyze_multitask", SCRIPT)
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)

ENV = "WamMultiTaskFake-v0"


def write_run(root, seed, episodes, finished=True, drop_index=0):
    """episodes: list of (step, task index, success). Returns the run folder."""
    run = root / f"{ENV}__train_ppo__{seed}__1795000000"
    writer = SummaryWriter(str(run))
    for i, (step, task, success) in enumerate(episodes):
        writer.add_scalar("task/success", float(success), step)
        if i < len(episodes) - drop_index:
            writer.add_scalar("task/index", float(task), step)
    writer.close()
    if finished:
        (run / "train_ppo.cleanrl_model").write_bytes(b"")
    return run


def interleaved(n, success_for, tasks=(0, 1, 2), per_step=2):
    """n episodes cycling through `tasks`, `per_step` episodes sharing each 2,048-step update."""
    out, count = [], {t: 0 for t in tasks}
    for i in range(n):
        task = tasks[i % len(tasks)]
        out.append(((i // per_step) * 2048, task, success_for(task, count[task])))
        count[task] += 1
    return out


# ---------- steps_to ----------

def test_steps_to_needs_a_full_window():
    assert am.steps_to(np.ones(199), np.arange(199)) == "never"


def test_steps_to_reports_the_step_where_the_window_first_exceeds_the_level():
    success = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
    steps = np.arange(1, 9) * 1000
    # Windows of 4: 0, .25, .5, .75, 1.0; the first above 0.9 ends at episode 8, step 8000.
    assert am.steps_to(success, steps, window=4) == "8k"


def test_steps_to_needs_strictly_more_than_the_level():
    # Two successes in a window of 4 average to exactly 0.5 in floating point, so this sits on the
    # boundary: reaching the level is not enough, it has to be exceeded.
    success = np.array([1, 1, 0, 0], float)
    assert am.steps_to(success, np.arange(4) * 1000, level=0.5, window=4) == "never"
    assert am.steps_to(np.array([1, 1, 1, 0], float), np.arange(4) * 1000, level=0.5, window=4) == "3k"


def test_steps_to_never_when_success_stays_low():
    assert am.steps_to(np.zeros(500), np.arange(500)) == "never"


# ---------- summarize_run ----------

def test_summarize_run_splits_episodes_by_task_even_when_steps_repeat(tmp_path):
    # T0 always succeeds, T1 never, T4x alternates; T4xneg never appears. Two episodes per step.
    run = write_run(tmp_path, 1, interleaved(750, lambda task, k: {0: 1, 1: 0, 2: k % 2}[task]))
    row = am.summarize_run(run)
    assert row["seed"] == "1"
    assert row["finals"] == {"T0": 1.0, "T1": 0.0, "T4x": 0.5, "T4xneg": None}
    # T0's 200th episode is episode 597 overall, logged at update 298: step 298 * 2048 = 610,304.
    assert row["reached"]["T0"] == "610k"
    assert row["reached"]["T1"] == row["reached"]["T4x"] == "never"
    assert row["pooled"] == pytest.approx((250 + 0 + 125) / 750)


def test_summarize_run_finals_use_only_each_tasks_last_500_episodes(tmp_path):
    # 600 T0 episodes; only episodes 100-199 succeed. The last 500 hold 100 of them (0.2); the last
    # 400 would hold none and all 600 would give 1/6, so only a 500-episode window gives 0.2.
    episodes = [(i * 2048, 0, 100 <= i < 200) for i in range(600)]
    row = am.summarize_run(write_run(tmp_path, 1, episodes))
    assert row["finals"]["T0"] == pytest.approx(0.2)
    assert row["pooled"] == pytest.approx(100 / 600)


def test_summarize_run_pooled_final_uses_the_last_2000_episodes(tmp_path):
    # 2,500 episodes; only 500-999 succeed. The last 2,000 hold all 500 of them (0.25); the last
    # 1,000 would hold none and all 2,500 would give 0.2.
    episodes = [(i * 2048, i % 4, 500 <= i < 1000) for i in range(2500)]
    assert am.summarize_run(write_run(tmp_path, 1, episodes))["pooled"] == pytest.approx(0.25)


def test_summarize_run_refuses_unpaired_series(tmp_path):
    run = write_run(tmp_path, 1, interleaved(30, lambda task, k: 1), drop_index=1)
    with pytest.raises(ValueError, match="task/index"):
        am.summarize_run(run)


# ---------- report ----------

def test_report_skips_unfinished_runs_and_orders_seeds_numerically(tmp_path):
    for seed in (10, 2):
        write_run(tmp_path, seed, interleaved(40, lambda task, k: 1, tasks=(0, 1, 2, 3)))
    write_run(tmp_path, 3, interleaved(40, lambda task, k: 1, tasks=(0, 1, 2, 3)), finished=False)
    rows = [line for line in am.report(ENV, tmp_path).splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert [row.split(" | ")[0] for row in rows] == ["| 2", "| 10"]


def test_report_marks_a_task_without_episodes_and_leaves_it_out_of_the_mean(tmp_path):
    write_run(tmp_path, 1, interleaved(30, lambda task, k: 1))  # no T4xneg episodes
    lines = am.report(ENV, tmp_path).splitlines()
    seed_row, mean_row = lines[4], lines[5]  # blank, heading, header, separator, then rows
    assert seed_row.split(" | ")[4] == "no episodes"
    assert mean_row.split(" | ")[4] == "no episodes"
    assert "nan" not in "\n".join(lines)


def test_report_mean_row_gives_the_mean_and_range_over_seeds(tmp_path):
    write_run(tmp_path, 1, interleaved(40, lambda task, k: 1, tasks=(0, 1, 2, 3)))
    write_run(tmp_path, 2, interleaved(40, lambda task, k: k % 2, tasks=(0, 1, 2, 3)))
    mean_row = am.report(ENV, tmp_path).splitlines()[-1]
    assert mean_row.startswith("| mean | 0.750 (range 0.500–1.000) |")
    assert mean_row.count("0.750 (range 0.500–1.000)") == 4


def test_report_header_matches_the_table_in_results(tmp_path):
    write_run(tmp_path, 1, interleaved(8, lambda task, k: 1, tasks=(0, 1, 2, 3)))
    lines = am.report(ENV, tmp_path).splitlines()
    assert lines[:3] == [
        "",
        f"## {ENV}",
        "| Seed | T0 final / steps to 90% | T1 final / steps to 90% | T4x final / steps to 90% "
        "| T4xneg final / steps to 90% | Pooled final |",
    ]
