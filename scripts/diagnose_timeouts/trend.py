import glob, sys, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for env in sys.argv[1:]:
    for run in sorted(glob.glob(f"runs/{env}__*")):
        e = EventAccumulator(run, size_guidance={"scalars": 0}); e.Reload()
        L = e.Scalars("charts/episodic_length"); steps = np.array([x.step for x in L]); to = np.array([x.value >= 250 for x in L])
        sp = np.array([x.value for x in e.Scalars("task/spill")])
        w = [(0, 5e5), (5e5, 1e6), (1e6, 1.5e6), (1.5e6, 2.1e6)]
        print(env.split("-")[0][17:], run.split("__")[2], "timeout by 0.5M window:", [f"{to[(steps >= a) & (steps < b)].mean():.3f}" for a, b in w],
              "| spill:", [f"{sp[(steps >= a) & (steps < b)].mean():.3f}" for a, b in w])
