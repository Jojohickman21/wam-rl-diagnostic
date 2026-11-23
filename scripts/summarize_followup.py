"""Tables behind RESULTS.md §10.2: held-out T2 and in-distribution success, every evaluated seed.

Reads runs/diagnostics/evals/ (written by scripts/run_followup_evals.sh). Comparisons report
whether seed ranges overlap (the bar) and an exact two-sided rank-sum (Mann-Whitney) p across
seeds, as a second look.

    python scripts/summarize_followup.py
"""
import collections
import glob
import itertools
import re
import statistics as st

PATTERN = (r"(\S+) seed (\d) on (\S+) \(yaw ±20°, 200 episodes\): success ([\d.]+) \| valid fraction of "
           r"successes (\S+) \| spill ([\d.]+) \| timeout ([\d.]+)")
ev = collections.defaultdict(dict)
for path in glob.glob("runs/diagnostics/evals/*__20.txt"):
    m = re.match(PATTERN, open(path).read())
    ev[(m[1], m[3])][int(m[2])] = dict(success=float(m[4]), spill=float(m[6]), timeout=float(m[7]))


def ranksum_p(a, b):
    """Exact two-sided Mann-Whitney p: share of all relabellings at least as extreme as observed."""
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


def seeds(env, task):
    d = ev[(env, task)]
    return [d[k]["success"] for k in sorted(d)], d


POLICIES = [("WamMultiTask-v0", "Descriptor, 4 training tasks"), ("WamMultiTaskLabel-v0", "Task label, same 4 tasks"),
            ("WamTaskT1-v0", "T1 only"), ("WamTaskT4x-v0", "T4x only"), ("WamTaskT2-v0", "T2 only (ceiling)")]
print("| Policy | Seeds | Success per seed | Mean | Median | Range | Mean spill | Mean timeout |")
print("|---|---|---|---|---|---|---|---|")
t2 = {}
for env, name in POLICIES:
    s, d = seeds(env, "T2")
    t2[env] = s
    print(f"| {name} | {','.join(str(k) for k in sorted(d))} | {' / '.join(f'{x:.3f}' for x in s)} | {st.mean(s):.3f} | "
          f"{st.median(s):.3f} | {min(s):.3f}–{max(s):.3f} | {st.mean(v['spill'] for v in d.values()):.3f} | "
          f"{st.mean(v['timeout'] for v in d.values()):.3f} |")
t2["controls"] = t2["WamTaskT1-v0"] + t2["WamTaskT4x-v0"]
print("\n| Comparison | Means | Ranges overlap | p |\n|---|---|---|---|")
for a, b, label in [("WamMultiTask-v0", "WamMultiTaskLabel-v0", "Descriptor vs label"),
                    ("WamMultiTask-v0", "WamTaskT1-v0", "Descriptor vs T1-only"),
                    ("WamMultiTask-v0", "WamTaskT4x-v0", "Descriptor vs T4x-only"),
                    ("WamMultiTask-v0", "controls", "Descriptor vs both controls pooled"),
                    ("WamMultiTaskLabel-v0", "WamTaskT1-v0", "Label vs T1-only"),
                    ("WamMultiTaskLabel-v0", "WamTaskT4x-v0", "Label vs T4x-only"),
                    ("WamMultiTaskLabel-v0", "controls", "Label vs both controls pooled")]:
    x, y = t2[a], t2[b]
    overlap = not (max(x) < min(y) or max(y) < min(x))
    print(f"| {label} | {st.mean(x):.3f} vs {st.mean(y):.3f} | {'yes' if overlap else 'no'} | {ranksum_p(x, y):.3f} |")
print("\n| Task | Descriptor | Label | Ranges overlap | p |\n|---|---|---|---|---|")
for task in ("T0", "T1", "T4x", "T4xneg"):
    a, _ = seeds("WamMultiTask-v0", task)
    b, _ = seeds("WamMultiTaskLabel-v0", task)
    overlap = not (max(a) < min(b) or max(b) < min(a))
    print(f"| {task} | {' / '.join(f'{x:.3f}' for x in a)} (mean {st.mean(a):.3f}) | "
          f"{' / '.join(f'{x:.3f}' for x in b)} (mean {st.mean(b):.3f}) | {'yes' if overlap else 'no'} | {ranksum_p(a, b):.3f} |")
