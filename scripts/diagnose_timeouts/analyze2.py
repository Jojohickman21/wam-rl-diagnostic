import sys, numpy as np
eps = np.concatenate([np.load(f, allow_pickle=True) for f in sys.argv[1:]])
J4LO, J4HI = -0.19198, 3.927
def cls(e): return "timeout" if e["timeout"] else ("success" if e["success"] > 0 else "spill")
for c in ("success", "timeout", "spill"):
    S = [e for e in eps if cls(e) == c]
    q4 = np.array([e["qend"][3] for e in S]); q4max = np.array([e["qtrail"][:, 3].max() for e in S]); q4min = np.array([e["qtrail"][:, 3].min() for e in S])
    print(f"{c:8s} n={len(S):5d} joint4 end: near upper(3.93) {np.mean(q4 > J4HI - .05):.2f} near lower(-0.19) {np.mean(q4 < J4LO + .05):.2f} | ever within 0.05 of upper {np.mean(q4max > J4HI - .05):.2f} | ever near lower {np.mean(q4min < J4LO + .05):.2f} | joint4 start mean {np.mean([e['q0'][3] for e in S]):.2f}")
T = [e for e in eps if e["timeout"]]
motion = np.array([np.abs(np.diff(e["qtrail"][-50:, :7], axis=0)).sum(0) for e in T])
print("timeouts: total joint travel over last 50 steps (rad), median per joint:", np.round(np.median(motion, 0), 2))
stuck = np.array([e["qend"][3] > J4HI - .05 for e in T])
print("timeouts with joint 4 at upper limit:", stuck.sum(), "of", len(T))
# when did joint 4 first reach the limit, relative to door state
first = [int(np.argmax(e["qtrail"][:, 3] > J4HI - .05)) for e, s in zip(T, stuck) if s]
door_at = [e["angles"][f] for e, f, in zip([e for e, s in zip(T, stuck) if s], first)]
print("step joint 4 first reaches upper limit, median:", np.median(first), "| door angle then, pct 25/50/75:", np.round(np.percentile(door_at, [25, 50, 75]), 2))
# timeout rate by yaw and hinge
yaw = np.degrees([e["yaw"] for e in eps]); to = np.array([e["timeout"] for e in eps]); hx = np.array([e["hinge"][0] for e in eps]); hy = np.array([e["hinge"][1] for e in eps])
for name, v, edges in (("yaw deg", yaw, [-20, -10, 0, 10, 20]), ("hinge x", hx, np.percentile(hx, [0, 25, 50, 75, 100])), ("hinge y", hy, np.percentile(hy, [0, 25, 50, 75, 100]))):
    b = np.clip(np.digitize(v, edges) - 1, 0, 3)
    print(f"timeout rate by {name} bins {np.round(edges, 2)}:", [f"{to[b == i].mean():.3f}" for i in range(4)])
q0 = np.array([e["q0"] for e in eps])
for j in range(4):
    b = np.clip(np.digitize(q0[:, j], np.percentile(q0[:, j], [0, 25, 50, 75, 100])) - 1, 0, 3)
    print(f"timeout rate by start joint {j+1} quartile:", [f"{to[b == i].mean():.3f}" for i in range(4)])
