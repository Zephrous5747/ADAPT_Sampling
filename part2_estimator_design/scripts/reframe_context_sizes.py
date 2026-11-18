import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np
import part1_bridge
from contexts import build_context_library

for case in ["H4_square_eq_side1p0_HF", "LiH_R1p6_HF", "BeH2_HF", "H2O_eq_HF", "H2O_dz8o_eq_HF"]:
    t0 = time.time()
    try:
        problem = part1_bridge.load_problem(case)
        lib = build_context_library(problem, "canonical")
        members = np.array([len(c.members) for c in lib.contexts])
        home = np.array([len(c.home_members) for c in lib.contexts])
        print(case, "contexts", len(members), "|M_alpha| (all products the circuit resolves) mean %.1f median %d max %d" % (members.mean(), np.median(members), members.max()),
              "| home members mean %.1f max %d" % (home.mean(), home.max()), "| %.0fs" % (time.time() - t0), flush=True)
    except Exception as e:
        print(case, "ERR", type(e).__name__, str(e)[:100], flush=True)
