"""Results of the G1-G3 generalisation batch, scored against the bar and predictions fixed in
TASK_SET.md before the runs started.

Reads runs/diagnostics/gen_evals/ (written by scripts/run_gen_evals.sh).

    python scripts/summarize_gen.py
"""
import collections
import glob
import itertools
import re
import statistics as st
from pathlib import Path

PATTERN = (r"(\S+) seed (\d) on (\S+) \(yaw ±20°, 200 episodes\): success ([\d.]+) \| valid fraction of "
           r"successes (\S+) \| spill ([\d.]+) \| timeout ([\d.]+)")
VARIANTS = [("WamFamilyDesc-v0", "G1 joint descriptor"), ("WamFamilyContact-v0", "G2 contact target"),
            ("WamFamilyContactFK-v0", "G3 contact target + FK")]
TASKS = ["T2", "T1", "T4x", "T4xneg"]
LABEL_POLICY_T2 = (0.22, 0.56)  # §10.2: the label policy's seed range on T2, 8 seeds
SECTION10_BEST_T2 = 0.422  # §10.2: the label policy's mean on T2, the best zero-shot result before this batch
DEFAULT_DIR = Path(__file__).parents[1] / "runs" / "diagnostics" / "gen_evals"


def ranksum_p(a, b):
    """Exact two-sided Mann-Whitney p: share of all relabellings at least as extreme as observed
    (same test as scripts/summarize_followup.py)."""
    pooled, n = a + b, len(a)

    def u(x, y):
        return sum((xi > yi) + 0.5 * (xi == yi) for xi in x for yi in y)

    center, observed = n * len(b) / 2, abs(u(a, b) - n * len(b) / 2)
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        chosen = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        hits += abs(u(x, y) - center) >= observed - 1e-12
    return hits / total


def overlap(a, b):
    return not (max(a) < min(b) or max(b) < min(a))


def load(directory=DEFAULT_DIR):
    """{(env, task): {seed: {success, spill, timeout}}} from every evaluation file. Files that don't
    match (e.g. an evaluation still running, so empty) are skipped."""
    ev = collections.defaultdict(dict)
    for path in glob.glob(str(Path(directory) / "*__20.txt")):
        m = re.match(PATTERN, Path(path).read_text())
        if m:
            ev[(m[1], m[3])][int(m[2])] = dict(success=float(m[4]), spill=float(m[6]), timeout=float(m[7]))
    return ev


def successes(ev, env, task):
    d = ev.get((env, task), {})
    return [d[k]["success"] for k in sorted(d)]


def table(ev):
    lines = ["| Variant | Task | Seeds | Success per seed | Mean | Range | Mean spill | Mean timeout |",
             "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for env, name in VARIANTS:
            d = ev.get((env, task), {})
            if not d:
                continue
            s = successes(ev, env, task)
            lines.append(f"| {name} | {task} | {len(s)} | {' / '.join(f'{x:.3f}' for x in s)} | {st.mean(s):.3f} | "
                         f"{min(s):.3f}–{max(s):.3f} | {st.mean(v['spill'] for v in d.values()):.3f} | "
                         f"{st.mean(v['timeout'] for v in d.values()):.3f} |")
    return "\n".join(lines)


def bar(ev):
    """The bar fixed in TASK_SET.md. Each entry: (name, verdict or None if data missing, detail)."""
    out = []
    for env, name in VARIANTS[1:]:
        box = [successes(ev, env, t) for t in ("T4x", "T4xneg")]
        if not all(box):
            out.append((f"Validity, {name}", None, "no T4x/T4xneg results yet"))
            continue
        means = [st.mean(s) for s in box]
        out.append((f"Validity, {name}", min(means) >= 0.80,
                    f"T4x mean {means[0]:.3f}, T4xneg mean {means[1]:.3f} (needs both >= 0.80)"))
    for env, name in VARIANTS[1:]:
        s = successes(ev, env, "T2")
        if not s:
            out.append((f"Main claim, {name[:2]} on T2", None, "no T2 results yet"))
            continue
        met = st.mean(s) >= 0.70 and not overlap(s, LABEL_POLICY_T2)
        out.append((f"Main claim, {name[:2]} on T2", met,
                    f"mean {st.mean(s):.3f}, range {min(s):.3f}–{max(s):.3f} vs label policy "
                    f"{LABEL_POLICY_T2[0]}–{LABEL_POLICY_T2[1]} (needs mean >= 0.70 and no overlap)"))
    return out


def comparisons(ev):
    """Between-variant comparisons on T2: (label, means, ranges overlap, rank-sum p)."""
    g = {name[:2]: successes(ev, env, "T2") for env, name in VARIANTS}
    out = []
    for a, b in (("G2", "G1"), ("G3", "G2"), ("G3", "G1")):
        if g[a] and g[b]:
            out.append((f"{a} vs {b}", (st.mean(g[a]), st.mean(g[b])), overlap(g[a], g[b]), ranksum_p(g[a], g[b])))
    return out


def predictions(ev):
    """Josiah's predictions as logged in TASK_SET.md. Value None when the data needed is missing."""
    g1, g2, g3 = (successes(ev, env, "T2") for env, _ in VARIANTS)
    return {
        "P1": None if not (g1 and g2) else st.mean(g2) > st.mean(g1) and st.mean(g2) > SECTION10_BEST_T2,
        "P1'": None if not g2 else st.mean(g2) > 0.80,
        "P2": None if not (g2 and g3) else overlap(g2, g3) and ranksum_p(g2, g3) >= 0.05,
        "P3'": None if not g1 else st.mean(g1) < 0.60,
    }


def report(ev):
    word = {True: "MET", False: "NOT MET", None: "PENDING"}
    lines = [table(ev), "", "**Bar (TASK_SET.md, fixed before running)**"]
    lines += [f"- {name}: {word[verdict]}. {detail}" for name, verdict, detail in bar(ev)]
    lines += [f"- {label} on T2: means {m[0]:.3f} vs {m[1]:.3f}, ranges overlap: {ov}, rank-sum p = {p:.3f}"
              for label, m, ov, p in comparisons(ev)]
    truth = {True: "TRUE", False: "FALSE", None: "PENDING"}
    meaning = {"P1": "G2 beats G1 and §10's best (0.422) on T2", "P1'": "G2 mean on T2 above 0.80",
               "P2": "G3 not separable from G2 on T2", "P3'": "G1 mean on T2 below 0.60"}
    lines += ["", "**Josiah's predictions (TASK_SET.md, logged before running)**"]
    lines += [f"- {key} ({meaning[key]}): {truth[value]}" for key, value in predictions(ev).items()]
    seeds = ", ".join(f"{name[:2]} {len(successes(ev, env, 'T2'))}" for env, name in VARIANTS)
    lines += ["", f"Seeds evaluated on T2: {seeds}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(load()))
