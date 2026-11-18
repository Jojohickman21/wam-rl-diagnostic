import sys, numpy as np
eps = np.concatenate([np.load(f, allow_pickle=True) for f in sys.argv[1:]])
LO = np.array([-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319]); HI = -LO; HI[1], HI[3], HI[5] = 2.0944, 3.927, 3.14159
to = np.array([e["timeout"] for e in eps]); ok = np.array([e["success"] > 0 for e in eps])
T = [e for e in eps if e["timeout"]]
end = np.array([e["door_end"] for e in T])
print(f"{len(eps)} episodes, {to.sum()} timeouts")
print("final door angle pct 10/25/50/75/90:", np.round(np.percentile(end, [10, 25, 50, 75, 90]), 2))
fa_end = np.array([e["fa"][-1] for e in T]); wrong_end = np.array([e["wrong"][-1] > 0 for e in T])
print("last step forearm >1N:", np.mean(fa_end > 1).round(2), "median N when >1:", np.median(fa_end[fa_end > 1]).round(0) if (fa_end > 1).any() else "-", "| wrong contact:", wrong_end.mean().round(2))
print("tilt at end median deg:", np.degrees(np.median([e["tilt"][-1] for e in T])).round(1))
near = end < 0.5; far = end > 1.0
print(f"near-closed stalls (<0.5 rad): {near.sum()}, unmoved/far (>1.0 rad): {far.sum()}, between: {(~near & ~far).sum()}")
def describe(sel, name):
    S = [e for e, s in zip(T, sel) if s]
    if not S: return
    print(f"\n-- {name} ({len(S)})")
    last = [slice(-50, None)]
    contact = np.array([np.mean(e["fa"][-50:] > 1) for e in S])
    moved = np.array([e["angles"][-50] - e["angles"][-1] for e in S])
    print(" share of last 50 steps with forearm >1N, median:", np.median(contact).round(2), "| door closed over last 50 steps, median rad:", np.median(moved).round(3))
    print(" median forearm force over last 50 steps when >1N:", np.median(np.concatenate([e["fa"][-50:][e["fa"][-50:] > 1] for e in S] + [[np.nan]])).round(0))
    lev = np.concatenate([e["lever"][-50:] for e in S]); lev = lev[np.isfinite(lev)]
    print(" contact lever arm from hinge (m) pct 10/50/90:", np.round(np.percentile(lev, [10, 50, 90]), 3) if len(lev) else "-")
    q = np.array([e["qend"] for e in S]); margin = np.minimum(q - LO, HI - q)
    print(" joints within 0.05 rad of a limit at end (share of episodes, joints 1-7):", np.round(np.mean(margin < 0.05, 0), 2))
    lead = np.array([np.abs(e["qtrail"][-1, 7:] - e["qtrail"][-1, :7]) for e in S])
    print(" |target - q| at end, median per joint:", np.round(np.median(lead, 0), 3))
    print(" share with any joint lead >= 0.09 (blocked):", np.mean(lead.max(1) >= 0.09).round(2))
    print(" wrong contact on last 50 steps (any), share:", np.mean([np.any(e["wrong"][-50:] > 0) for e in S]).round(2))
    print(" min door angle reached in episode, median:", np.median([e["angles"].min() for e in S]).round(2))
describe(near, "near-closed stalls"); describe(far, "door far open at timeout"); describe(~near & ~far, "between")
# reference: successes' lever arm and joint state when door is at 0.1-0.3 rad
print("\n-- successes, steps with door 0.1-0.3 rad")
lev = np.concatenate([e["lever"][(e["angles"] > .1) & (e["angles"] < .3)] for e in eps if e["success"] > 0]); lev = lev[np.isfinite(lev)]
print(" contact lever arm pct 10/50/90:", np.round(np.percentile(lev, [10, 50, 90]), 3))
# cabinet pose
def pose(sel):
    return np.array([[*e["hinge"], np.degrees(e["yaw"]), np.degrees(e["slide"]), e["door0"]] for e, s in zip(eps, sel) if s])
for name, sel in (("success", ok), ("timeout", to)):
    P = pose(sel); print(f"\n{name}: hinge x, hinge y, yaw deg, slide deg, door0 — mean", np.round(P.mean(0), 3), "sd", np.round(P.std(0), 3))
