"""How do the H4 2.0 HF correlated trials actually end?"""
import sys, json, collections; sys.path.insert(0,"src")
import numpy as np
from cases import get_case
from problem_cache import load_or_build
import finite_shot as fs
c="H4_square_stretch_side2p0_HF"
pr=load_or_build(get_case(c), cache_dir=".cache")
truth=pr.abs_gradients; leader=int(np.argmax(truth))
order=np.argsort(-truth)
print("top4:", [(pr.labels[i], round(float(truth[i]),6)) for i in order[:4]])
print("MAXIMUM_ROUNDS", fs.MAXIMUM_ROUNDS, "MIN_RADIUS_FRACTION", fs.MINIMUM_RADIUS_FRACTION)
orig=fs._sequential_trial
info=[]
def wrapped(problem, sigma_sums, sigmas, z, shrink, tolerance, rng, shared, cache=None,
            stop_at_tolerance=False, factors=None, covariances=None):
    o=orig(problem, sigma_sums, sigmas, z, shrink, tolerance, rng, shared, cache,
           stop_at_tolerance, factors, covariances)
    info.append(o); return o
fs._sequential_trial=wrapped
bound=json.load(open(f"runs/{c}/{c}_M3_BAIFCUG_summary.json"))["total_context_shots"]
res=fs.simulate(pr,"m3",planning_bound=bound,n_trials=250,seed=0,
                noise_model="correlated",rule="marginal")
fs._sequential_trial=orig
rounds=collections.Counter(o.rounds for o in info)
sel=collections.Counter(pr.labels[o.selected] for o in info)
hit_cap=sum(1 for o in info if o.rounds>=fs.MAXIMUM_ROUNDS)
print("correct rate", res.correct_rate, " rounds histogram", dict(sorted(rounds.items())))
print("hit MAXIMUM_ROUNDS:", hit_cap, "of", len(info))
print("selected:", sel.most_common(5))
bad=[o for o in info if not o.correct]
print("wrong trials: n=%d  median rounds=%s  median shots=%.3g"%(len(bad),
      np.median([o.rounds for o in bad]) if bad else None,
      np.median([o.shots for o in bad]) if bad else float('nan')))
good=[o for o in info if o.correct]
print("right trials: median rounds=%s  median shots=%.3g"%(
      np.median([o.rounds for o in good]), np.median([o.shots for o in good])))
