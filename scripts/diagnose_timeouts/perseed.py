import numpy as np
for s in (1, 2, 3):
    eps = np.load(f"runs/diagnostics/a20_s{s}.npy", allow_pickle=True)
    to = np.array([e["timeout"] for e in eps]); ok = np.array([e["success"] > 0 for e in eps])
    yaw = np.abs(np.degrees([e["yaw"] for e in eps])); q4 = np.array([e["q0"][3] for e in eps])
    hard = (yaw > 10) & (q4 > 2.75)
    lim = np.mean([(e["qtrail"][:, 3].max() > 3.877) or (e["qtrail"][:, 3].min() < -0.142) for e in eps if e["timeout"]])
    print(f"seed {s}: n={len(eps)} timeout {to.mean():.3f} | |yaw|>10: timeout {to[yaw > 10].mean():.3f} success {ok[yaw > 10].mean():.3f} "
          f"| hard (|yaw|>10 & joint4 start>2.75, n={hard.sum()}): timeout {to[hard].mean():.3f} success {ok[hard].mean():.3f} | timeouts hitting a joint-4 limit {lim:.2f}")
