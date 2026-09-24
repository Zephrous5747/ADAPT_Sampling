import sys, numpy as np
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from experiment_lih_mechanism import traced_trial, ZeroRNG
from cases import get_case
from problem_cache import load_or_build
from finite_shot import _AllocationCache
from shot_models import z_from_delta, DEFAULT_DELTA
import m2_baifcig, m3_baifcug
from m3_baifcug import run as m3run
cases = sys.argv[1:]
print(f"{'case':32s}{'M3 breadth':>11}{'M2 nl/bnd':>10}{'M3 nl/bnd':>10}{'1st elim r/r0':>14}{'bound M2/M3':>12}{'noiseless M2/M3':>16}")
for c in cases:
    p = load_or_build(get_case(c), cache_dir='.cache')
    n=p.n_generators; z=z_from_delta(DEFAULT_DELTA,n)
    sig=p.parent_fragment_sigmas(); cache=_AllocationCache(sig)
    ss=np.array([s.sum() for s in p.individual_fragment_sigmas(p.individual_fc_groups())])
    b2=m2_baifcig.run(p).total_shots; r3=m3run(p); b3=r3.total_shots if hasattr(r3,'total_shots') else float(np.sum(r3.shots_per_context))
    # breadth share of M3 bound: allocation at first breakpoint
    from bai import elimination_thresholds
    from shot_models import allocate_context_shots, epsilon_from_radius
    t0 = elimination_thresholds(p.abs_gradients)[0]
    br = allocate_context_shots(sig[list(t0.active),:], epsilon_from_radius(t0.radius,z)).sum()/b3
    out={}
    for m in ('m2','m3'):
        cost,left,_ = traced_trial(p,m,z,0.95,ZeroRNG(),sig,cache,ss,True)
        out[m]=cost; 
        if m=='m3':
            from bai import deficits
            r0 = deficits(p.abs_gradients).max()/2
            first = np.nanmax(left)/r0
    print(f"{c:32s}{br:>11.1%}{out['m2']/b2:>10.3f}{out['m3']/b3:>10.3f}{first:>14.3f}{b2/b3:>12.2f}{out['m2']/out['m3']:>16.2f}", flush=True)
