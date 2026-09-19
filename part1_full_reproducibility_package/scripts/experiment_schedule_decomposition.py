"""Noiseless run under the FINITE-SHOT trial's own schedule and stopping rule."""
import sys, json; sys.path.insert(0, "src")
import numpy as np
from cases import get_case
from problem_cache import load_or_build
import finite_shot as fs

class ZeroRNG:
    def normal(self, loc=0.0, scale=1.0, size=None):
        a = np.asarray(scale, dtype=float)
        return np.zeros(a.shape if size is None else size)
    def standard_normal(self, size=None):
        return np.zeros(() if size is None else size)

c = sys.argv[1]
pr = load_or_build(get_case(c), cache_dir=".cache")
code = {"m2": "M2_BAIFCIG", "m3": "M3_BAIFCUG"}
out = {}
for m in ("m2", "m3"):
    bound = json.load(open(f"runs/{c}/{c}_{code[m]}_summary.json"))["total_context_shots"]
    real = fs.simulate(pr, m, planning_bound=bound, n_trials=1, seed=0)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: ZeroRNG()      # noqa: E731
    try:
        zero = fs.simulate(pr, m, planning_bound=bound, n_trials=1, seed=0)
    finally:
        np.random.default_rng = orig
    noisy = fs.simulate(pr, m, planning_bound=bound, n_trials=200, seed=0)
    out[m] = dict(bound=bound, sched=zero.shots_mean, noisy=noisy.shots_mean,
                  sem=noisy.shots_sem)
    print(f"{c:30s} {m}: bound={bound:14,.0f} schedule-only={zero.shots_mean:14,.0f}"
          f" ({zero.shots_mean/bound:5.2f}x)  noisy={noisy.shots_mean:14,.0f}"
          f" ({noisy.shots_mean/bound:5.2f}x)  noise-on-top={noisy.shots_mean/zero.shots_mean:5.2f}x",
          flush=True)
r = out["m2"]["sched"] / out["m3"]["sched"]
print(f"{'':30s} M2/M3: bound={out['m2']['bound']/out['m3']['bound']:5.2f}"
      f"  schedule-only={r:5.2f}  realised={out['m2']['noisy']/out['m3']['noisy']:5.2f}")
