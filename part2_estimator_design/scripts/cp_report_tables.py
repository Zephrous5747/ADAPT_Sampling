"""Tables and figures of the CP-1.2 results report (reports/prior/CP12_results_report.tex).

Metric everywhere: the shot ratio R = shots(data-only design, arm NEUTRAL, trust 1) / shots(arm); R > 1 means fewer shots than the data-only design.
  Test B: ratio of the mean shots consumed by the selection, paired bootstrap over replicate ids (10 000 resamples).
  Test A: ratio of the mean true worst-arm estimator variance at identical data, V_data-only / V_arm.  At a fixed design the variance
          scales as 1/shots, so this is the ratio of the shots needed for the same variance.
Writes reports/prior/cp12_tabs.tex (fragments included by the report) and the figures cp12_fig_*.pdf.
    python scripts/cp_report_tables.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "..", "reports", "prior")
RNG = np.random.default_rng(12345)
NBOOT = 10000
M = "$-$"


def load(d, kind, name):
    path = os.path.join(RUNS, d, kind, name + ".jsonl")
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def boot_pct(a, b):
    """ratio mean(b) / mean(a) (b: data-only baseline, a: arm), paired bootstrap; returns (value, lo, hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = RNG.integers(0, n, (NBOOT, n))
    r = b[idx].mean(1) / a[idx].mean(1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return b.mean() / a.mean(), lo, hi


def fmt(v, star=None, dec=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    d = (1 if abs(v) >= 10 else 2) if dec is None else dec
    return f"{v:.{d}f}" + ("$^*$" if star else "")


# ---------------------------------------------------------------- Test A
def testa_rows(d, state):
    rows = load(d, "testa", state)
    R = collections.defaultdict(dict)
    meta = None
    for r in rows:
        if r["arm"] == "meta":
            meta = meta or r
            continue
        R[(r["N"], r["mode"], r["arm"])][r["rep"]] = r["vmax"]
    return R, meta


def testa_pct(R, N, mode, arm):
    A, B = R.get((N, mode, arm)), R.get((N, mode, "NEUTRAL"))
    if not A or not B:
        return None
    reps = sorted(set(A) & set(B))
    v, lo, hi = boot_pct([A[r] for r in reps], [B[r] for r in reps])
    return v, (lo <= 1 <= hi)


SPECS = [  # label, dir, state, c128, static, prev, family
    ("H$_2$ 0.7414~\\AA, $m=0$", "cp_order", "H2_eq_m0", "CLASSICAL_CONST", None, None, "H2/H4"),
    ("H$_2$ 1.5~\\AA, $m=1$", "cp_order", "H2_str_m1", "CLASSICAL_CONST", None, "PREVIOUS_CONST", "H2/H4"),
    ("H$_4$ 1.0~\\AA, $m=1$", "cp_v3", "H4_a1_m1", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", "H2/H4"),
    ("H$_4$ 2.0~\\AA, $m=3$", "cp_v3", "H4_a2_m3", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", "H2/H4"),
    ("LiH 1.6~\\AA, $m=2$", "cp_v2", "LiH_r16_m2", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", "LiH exact"),
    ("LiH 3.0~\\AA, $m=3$", "cp_v2", "LiH_r30_m3", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", "LiH exact"),
    ("LiH 3.0~\\AA, $m=5$", "cp_v2", "LiH_r30_m5", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", "LiH exact"),
    ("LiH 3.0~\\AA, $m=6$", "cp_v2", "LiH_r30_m6", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", "LiH exact"),
    ("LiH 1.6~\\AA, $m=2$, 1 determinant", "cp_v2", "LiH_r16_m2_K1", "CLASSICAL_T128", "CLASSICAL_Tinf", None, "LiH wrong"),
    ("LiH 3.0~\\AA, $m=3$, 2 determinants", "cp_v2", "LiH_r30_m3_K2", "CLASSICAL_T128", "CLASSICAL_Tinf", None, "LiH wrong"),
    ("LiH 3.0~\\AA, $m=5$, 3 determinants", "cp_v2", "LiH_r30_m5_K3", "CLASSICAL_T128", "CLASSICAL_Tinf", None, "LiH wrong"),
] + [(f"H$_2$O {g} $m={m}$", "cp_v8", f"H2O_{k}_m{m}", "CLASSICAL_T128", "CLASSICAL_Tinf", "PREVIOUS_Tinf", f"water {g}")
     for g, k, ms in (("eq.", "eq", (3, 7, 11, 16)), ("str.", "str", (3, 8, 13, 17))) for m in ms]

TESTA = {}
for label, d, state, c128, static, prev, fam in SPECS:
    R, meta = testa_rows(d, state)
    err = 1000 * (meta["E_c"] - meta["E_q"]) if meta else None
    TESTA[state] = dict(label=label, fam=fam, err=err, R=R, c128=c128, static=static, prev=prev, dir=d)


def tab_testa():
    L = []
    L.append("\\begin{tabular}{lrrrrrrr}\\toprule")
    L.append(" & error & \\multicolumn{2}{c}{A1, trust 128} & \\multicolumn{2}{c}{A2, trust 128} & A1 static & A2 previous\\\\")
    L.append("State & mHa & $N=32$ & $N=128$ & $N=32$ & $N=128$ & $N=128$ & $N=128$\\\\\\midrule")
    last = None
    for s, t in TESTA.items():
        if last is not None and t["fam"].split()[0] != last.split()[0]:
            L.append("\\midrule")
        last = t["fam"]
        cells = []
        for mode, N in (("A1", 32), ("A1", 128), ("A2", 32), ("A2", 128)):
            p = testa_pct(t["R"], N, mode, t["c128"])
            cells.append(fmt(*p) if p else "---")
        p = testa_pct(t["R"], 128, "A1", t["static"]) if t["static"] else None
        cells.append(fmt(*p) if p else "---")
        p = testa_pct(t["R"], 128, "A2", t["prev"]) if t["prev"] else None
        cells.append(fmt(*p) if p else "---")
        e = "---" if t["err"] is None else (f"{abs(t['err']):.2f}" if abs(t["err"]) < 0.005 else f"{t['err']:+.2f}".replace("-", M))
        L.append(f"{t['label']} & {e} & " + " & ".join(cells) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")
    return "\n".join(L)


def tab_testa_small():
    """N = 8 and 32, A1 and A2, trust 128 and previous for every state (compact)"""
    L = ["\\begin{tabular}{lrrrrrrrr}\\toprule",
         " & \\multicolumn{4}{c}{classical, trust 128} & \\multicolumn{4}{c}{previous-state prior}\\\\",
         " & \\multicolumn{2}{c}{A1} & \\multicolumn{2}{c}{A2} & \\multicolumn{2}{c}{A1} & \\multicolumn{2}{c}{A2}\\\\",
         "State & $N=8$ & $N=32$ & $N=8$ & $N=32$ & $N=8$ & $N=32$ & $N=8$ & $N=32$\\\\\\midrule"]
    for s, t in TESTA.items():
        cells = []
        for arm in (t["c128"], t["prev"]):
            for mode in ("A1", "A2"):
                for N in (8, 32):
                    p = testa_pct(t["R"], N, mode, arm) if arm else None
                    cells.append(fmt(*p) if p else "---")
        L.append(f"{t['label']} & " + " & ".join(cells) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")
    return "\n".join(L)


def fig_testa():
    groups = [("H2/H4", "H$_2$, H$_4$", "0.5"), ("LiH exact", "LiH, accurate classical state", "tab:blue"),
              ("LiH wrong", "LiH, truncated classical state (16, 31, 51 mHa)", "tab:red"),
              ("water eq.", "water equilibrium", "tab:green"), ("water str.", "water stretched", "tab:orange")]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    styles = ["-", "--", "-.", ":"]
    for i, mode in enumerate(("A1", "A2")):
        for g, title, color in groups:
            k = 0
            for s, t_ in TESTA.items():
                if t_["fam"] != g:
                    continue
                ys = [testa_pct(t_["R"], N, mode, t_["c128"]) for N in (8, 32, 128)]
                if any(y is None for y in ys):
                    continue
                ax[i].plot([8, 32, 128], [y[0] for y in ys], styles[k % 4], color=color, marker="o", ms=3, lw=1.3,
                           label=title if k == 0 else None)
                k += 1
        ax[i].set_xscale("log", base=2)
        ax[i].set_xticks([8, 32, 128]); ax[i].set_xticklabels(["8", "32", "128"])
        ax[i].axhline(1, color="k", lw=0.6); ax[i].grid(alpha=0.25)
        ax[i].set_xlabel("outcomes per context $N$ in the data")
        ax[i].set_title("A1: fixed allocation" if mode == "A1" else "A2: joint splitting and allocation", fontsize=10)
    ax[0].set_ylabel("shot ratio for the same worst-arm\nvariance, data-only / prior")
    ax[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cp12_fig_testa.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Test B
TESTB = [  # label, [(dir, state)] per arm name -> (dir, file arm)
    # state label, neutral (dir,state,arm), arms: list of (arm label, dir, state, source)
]


def tb(d, state, src):
    r = load(d, "testb", f"{state}__{src}")
    return None if r is None else {x["rep"]: x for x in r}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 1)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def tb_summary(base, arm):
    reps = sorted(set(base) & set(arm))
    a = np.array([arm[r]["selection_shots"] for r in reps], float)
    b = np.array([base[r]["selection_shots"] for r in reps], float)
    v, lo, hi = boot_pct(a, b)
    return dict(n=len(reps), v=v, lo=lo, hi=hi)


def tb_stats(arm):
    reps = sorted(arm)
    sh = np.array([arm[r]["selection_shots"] for r in reps], float)
    cert = sum(arm[r]["status"] == "certified" for r in reps)
    cor = sum(arm[r]["status"] == "certified" and arm[r]["correct"] for r in reps)
    wrong = cert - cor
    unres = len(reps) - cert
    fb = sum(v for r in reps for k, v in arm[r]["solver"].items() if k == "alternating_after_conic_failure")
    nd = sum(v for r in reps for v in arm[r]["solver"].values())
    return dict(n=len(reps), cor=cor, wrong=wrong, unres=unres, mean=sh.mean(), med=float(np.median(sh)), p90=float(np.percentile(sh, 90)),
                fb=fb, nd=nd, elim=sum(bool(arm[r]["maximizer_eliminated"]) for r in reps),
                hf=sum(arm[r]["history_failures"] + arm[r]["block_failures"] for r in reps),
                wall=np.mean([arm[r]["wall_seconds"] for r in reps]) / 3600)


def fsci(x):
    return f"{x:.2e}".replace("e+0", "e").replace("e+", "e")


TB_LIH = [  # state label, neutral (dir,state), [(arm label, dir, state, src)]
    ("LiH 1.6~\\AA, $m=2$ (30)", ("cp_v5", "LiH_r16_m2"),
     [("trust 128", "cp_v5", "LiH_r16_m2", "CLASSICAL"), ("trust $10^3$", "cp_v5", "LiH_r16_m2", "CLASSICAL_T1000"),
      ("trust $10^4$", "cp_v5", "LiH_r16_m2", "CLASSICAL_T10000"), ("trust $10^5$", "cp_v5", "LiH_r16_m2", "CLASSICAL_T100000"),
      ("static", "cp_v5", "LiH_r16_m2", "CLASSICAL_INF"), ("previous, trust 128$^\\dagger$", "cp_v4", "LiH_r16_m2", "PREVIOUS", ("cp_v4", "LiH_r16_m2"))]),
    ("LiH 3.0~\\AA, $m=3$ (30)", ("cp_v5", "LiH_r30_m3"),
     [("trust 128", "cp_v5", "LiH_r30_m3", "CLASSICAL"), ("trust $10^3$", "cp_v5", "LiH_r30_m3", "CLASSICAL_T1000"),
      ("trust $10^4$", "cp_v5", "LiH_r30_m3", "CLASSICAL_T10000"), ("trust $10^5$", "cp_v5", "LiH_r30_m3", "CLASSICAL_T100000"),
      ("static", "cp_v5", "LiH_r30_m3", "CLASSICAL_INF"), ("previous, trust 128$^\\dagger$", "cp_v4", "LiH_r30_m3", "PREVIOUS", ("cp_v4", "LiH_r30_m3")),
      ("static, 2 determinants (31 mHa)", "cp_v5", "LiH_r30_m3_K2", "CLASSICAL_INF")]),
    ("LiH 3.0~\\AA, $m=5$ (26)", ("cp_v6", "LiH_r30_m5"),
     [("trust 128", "cp_v6", "LiH_r30_m5", "CLASSICAL"), ("trust $10^3$", "cp_v6", "LiH_r30_m5", "CLASSICAL_T1000"),
      ("trust $10^4$", "cp_v6", "LiH_r30_m5", "CLASSICAL_T10000"), ("trust $10^5$", "cp_v6", "LiH_r30_m5", "CLASSICAL_T100000"),
      ("static", "cp_v6", "LiH_r30_m5", "CLASSICAL_INF"), ("static, 3 determinants (51 mHa)", "cp_v6", "LiH_r30_m5_K3", "CLASSICAL_INF")]),
    ("LiH 3.0~\\AA, $m=6$ (26)", ("cp_v7", "LiH_r30_m6"),
     [("trust 128", "cp_v7", "LiH_r30_m6", "CLASSICAL"), ("trust $10^5$", "cp_v7", "LiH_r30_m6", "CLASSICAL_T100000"),
      ("static", "cp_v7", "LiH_r30_m6", "CLASSICAL_INF")]),
]
TB_OLD = [  # H2 / H4 (flaw does not apply: fewer than 128 contexts) and water (uniform allocation)
    ("H$_2$ 0.7414~\\AA, $m=0$ (30)", ("cp_order", "H2_eq_m0"), [("classical, energy rule", "cp_order", "H2_eq_m0", "CLASSICAL")]),
    ("H$_2$ 1.5~\\AA, $m=1$ (30)", ("cp_order", "H2_str_m1"), [("classical, energy rule", "cp_order", "H2_str_m1", "CLASSICAL"), ("previous, energy rule", "cp_order", "H2_str_m1", "PREVIOUS")]),
    ("H$_4$ 1.0~\\AA, $m=1$ (30)", ("cp_v3", "H4_a1_m1"), [("classical, trust 128", "cp_v3", "H4_a1_m1", "CLASSICAL"), ("previous, trust 128", "cp_v3", "H4_a1_m1", "PREVIOUS")]),
    ("H$_4$ 2.0~\\AA, $m=3$ (30)", ("cp_v3", "H4_a2_m3"), [("classical, trust 128", "cp_v3", "H4_a2_m3", "CLASSICAL"), ("previous, trust 128", "cp_v3", "H4_a2_m3", "PREVIOUS")]),
]


def _lih_rho(label, state, extra=()):
    arms = [("trust 128", "cp_v9", state, "CLASSICAL"), ("trust $10^5$", "cp_v9", state, "CLASSICAL_T100000"), ("static", "cp_v9", state, "CLASSICAL_INF")]
    return (label, ("cp_v9", state), arms + list(extra))


TB_RHO_LIH = [  # Test B at rho = 0.1 (rho-good stop), LiH, 30 replicates
    _lih_rho("LiH 1.6~\\AA, $m=2$ (30)", "LiH_r16_m2"),
    _lih_rho("LiH 3.0~\\AA, $m=3$ (30)", "LiH_r30_m3", [("trust 128, 2 determinants (31 mHa)", "cp_v9", "LiH_r30_m3_K2", "CLASSICAL"),
                                                       ("static, 2 determinants (31 mHa)", "cp_v9", "LiH_r30_m3_K2", "CLASSICAL_INF")]),
    _lih_rho("LiH 3.0~\\AA, $m=5$ (30)", "LiH_r30_m5", [("trust 128, 3 determinants (51 mHa)", "cp_v9", "LiH_r30_m5_K3", "CLASSICAL"),
                                                       ("static, 3 determinants (51 mHa)", "cp_v9", "LiH_r30_m5_K3", "CLASSICAL_INF")]),
    _lih_rho("LiH 3.0~\\AA, $m=6$ (30)", "LiH_r30_m6"),
]
TB_RHO_WATER = [(f"H$_2$O {g} $m={m}$ (20)", ("cp_v10", f"H2O_{k}_m{m}"),
                 [("trust 128", "cp_v10", f"H2O_{k}_m{m}", "CLASSICAL"), ("static", "cp_v10", f"H2O_{k}_m{m}", "CLASSICAL_INF")])
                for g, k, ms in (("eq.", "eq", (3, 7, 11, 16)), ("str.", "str", (3, 8, 13, 17))) for m in ms]


def tab_testb(spec, with_hist=False):
    L = ["\\begin{tabular}{@{}llrrrrrrr@{}}\\toprule",
         "State (replicates) & Arm & \\parbox{1.6cm}{\\raggedleft correct / wrong / unres.} & mean & median & 90th pct. & \\parbox{3cm}{\\raggedleft ratio $\\mathcal R_{\\rm sel}$ [95\\% CI]} & \\parbox{1.1cm}{\\raggedleft fallback designs} & \\parbox{1.1cm}{\\raggedleft con-flicts}\\\\\\midrule"]
    for lab, (bd, bs), arms in spec:
        base = tb(bd, bs, "NEUTRAL")
        if not base:
            continue
        s0 = tb_stats(base)
        L.append(f"{lab} & data-only & {s0['cor']} / {s0['wrong']} / {s0['unres']} & {fsci(s0['mean'])} & {fsci(s0['med'])} & {fsci(s0['p90'])} & --- & {100*s0['fb']/s0['nd']:.0f}\\% & {s0['elim']}\\\\")
        for arm_spec in arms:
            al, d, st, src = arm_spec[:4]
            a = tb(d, st, src)
            if not a:
                continue
            s = tb_stats(a)
            b2 = tb(*arm_spec[4], "NEUTRAL") if len(arm_spec) > 4 else base
            c = tb_summary(b2, a)
            star = "" if (c["lo"] > 1 or c["hi"] < 1) else "$^*$"
            if s["unres"] > 0.5 * s["n"] or s0["unres"] > 0.5 * s0["n"]:
                L.append(f" & {al} & {s['cor']} / {s['wrong']} / {s['unres']} & {fsci(s['mean'])} & {fsci(s['med'])} & {fsci(s['p90'])} & not defined (unresolved) & {100*s['fb']/s['nd']:.0f}\\% & {s['elim']}\\\\")
                continue
            L.append(f" & {al} & {s['cor']} / {s['wrong']} / {s['unres']} & {fsci(s['mean'])} & {fsci(s['med'])} & {fsci(s['p90'])} & {fmt(c['v'])}{star} [{fmt(c['lo'])}, {fmt(c['hi'])}] & {100*s['fb']/s['nd']:.0f}\\% & {s['elim']}\\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    return "\n".join(L)


def fig_success():
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    cols = {"data-only": "k", "trust 128": "tab:blue", "trust $10^5$": "tab:orange", "static": "tab:red"}
    for a, (lab, (bd, bs), arms) in zip(ax, TB_LIH):
        series = [("data-only", tb(bd, bs, "NEUTRAL"))] + [(al, tb(d, st, src)) for al, d, st, src, *_ in arms if al in ("trust 128", "trust $10^5$", "static")]
        for name, arm in series:
            if not arm:
                continue
            n = len(arm)
            sh = np.sort([r["selection_shots"] for r in arm.values() if r["status"] == "certified" and r["correct"]])
            xs = np.concatenate([[sh[0] * 0.5], sh]) if len(sh) else np.array([])
            ys = np.arange(0, len(sh) + 1) / n
            a.step(np.concatenate([[xs[0]], sh]), ys, where="post", color=cols[name], label=name, lw=1.6)
        a.set_xscale("log"); a.set_ylim(0, 1.02); a.grid(alpha=0.25)
        a.set_title(lab.split(" (")[0].replace("~\\AA", " A").replace("$", ""), fontsize=9)
        a.set_xlabel("cumulative selection shots")
    ax[0].set_ylabel("fraction certified and correct")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cp12_fig_success.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- error dependence
def fig_error():
    pts = []  # (error mHa, ratio, lo, hi, label, kind)
    def add(err, bd, bs, d, st, src, lab, kind):
        base, arm = tb(bd, bs, "NEUTRAL"), tb(d, st, src)
        if base and arm:
            c = tb_summary(base, arm)
            pts.append((err, c["v"], c["lo"], c["hi"], lab, kind))
    E = {k: 1000 * (TESTA[k]["err"]) for k in ("LiH_r16_m2", "LiH_r30_m3", "LiH_r30_m5", "LiH_r30_m6", "LiH_r30_m3_K2", "LiH_r30_m5_K3", "LiH_r16_m2_K1")}
    for kind, src in (("static", "CLASSICAL_INF"), ("trust 128", "CLASSICAL")):
        add(abs(E["LiH_r16_m2"]) / 1000, "cp_v5", "LiH_r16_m2", "cp_v5", "LiH_r16_m2", src, "m=2", kind)
        add(abs(E["LiH_r30_m3"]) / 1000, "cp_v5", "LiH_r30_m3", "cp_v5", "LiH_r30_m3", src, "m=3", kind)
        add(abs(E["LiH_r30_m5"]) / 1000, "cp_v6", "LiH_r30_m5", "cp_v6", "LiH_r30_m5", src, "m=5", kind)
        add(abs(E["LiH_r30_m6"]) / 1000, "cp_v7", "LiH_r30_m6", "cp_v7", "LiH_r30_m6", src, "m=6", kind)
        add(abs(E["LiH_r30_m3_K2"]) / 1000, "cp_v5", "LiH_r30_m3", "cp_v5", "LiH_r30_m3_K2", src, "m=3, 2 det.", kind)
        add(abs(E["LiH_r30_m5_K3"]) / 1000, "cp_v6", "LiH_r30_m5", "cp_v6", "LiH_r30_m5_K3", src, "m=5, 3 det.", kind)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind, color, off in (("static", "tab:red", 1.0), ("trust 128", "tab:blue", 1.12)):
        P = [p for p in pts if p[5] == kind]
        x = np.array([max(p[0], 0.05) * off for p in P]); y = np.array([p[1] for p in P])
        ax[0].errorbar(x, y, yerr=[y - np.array([p[2] for p in P]), np.array([p[3] for p in P]) - y], fmt="o", color=color, label=kind, capsize=2, ms=4)
        for xi, yi, p in zip(x, y, P):
            ax[0].annotate(p[4], (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[0].axhline(1, color="k", lw=0.6); ax[0].set_xscale("log")
    ax[0].set_xlabel("classical-state energy error $|E_c-E_q|$ (mHa; 0 plotted at 0.05)")
    ax[0].set_ylabel("Test B: shot ratio $R_{\\mathrm{sel}}$ (data-only / arm)")
    ax[0].set_title("LiH, shots to certify", fontsize=9); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25)
    for s, t in TESTA.items():
        if t["err"] is None:
            continue
        for mode, mk in (("A1", "o"),):
            p = testa_pct(t["R"], 128, mode, t["static"]) if t["static"] else None
            q = testa_pct(t["R"], 128, mode, t["c128"])
            if p:
                ax[1].plot(max(abs(t["err"]), 0.05), p[0], "s", color="tab:red", ms=4)
            if q:
                ax[1].plot(max(abs(t["err"]), 0.05) * 1.12, q[0], "o", color="tab:blue", ms=4)
    ax[1].plot([], [], "s", color="tab:red", label="static"); ax[1].plot([], [], "o", color="tab:blue", label="trust 128")
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_xscale("log")
    ax[1].set_xlabel("classical-state energy error $|E_c-E_q|$ (mHa)"); ax[1].set_ylabel("Test A (A1, $N=128$): shot ratio\nfor the same variance (data-only / prior)")
    ax[1].set_title("all states with a classical prior", fontsize=9); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cp12_fig_error.pdf")); plt.close(fig)
    return pts


# ---------------------------------------------------------------- calibration
def tau_rule(x, d0, s):
    return 1.0 / (d0 + 1.0 - math.exp(-(x / s) ** 2))


def pct_loss(l):
    """calibration loss = mean log(V_prior / V_data-only); returns the ratio V_data-only / V_prior."""
    return math.exp(-l)


def tab_calibration():
    c = json.load(open(os.path.join(RUNS, "cp_order", "calibration.json")))
    rec = c["records"]
    states_of = {"H2": ["H2_eq_m0", "H2_str_m1"], "H4": ["H4_a1_m1", "H4_a2_m3"], "LiH": ["LiH_r16_m1", "LiH_r30_m3"]}
    L = ["\\begin{tabular}{llrrrrrrr}\\toprule",
         "Held-out & Source & $d_0$ & $s$ & $x_+$ / $y_+$ (held-out) & assigned strength & rule, training (shots) & constant 128, training (shots) & rule vs constant 128, held-out$^\\S$\\\\\\midrule"]
    for fold, v in c["folds"].items():
        for src in ("CLASSICAL", "PREVIOUS"):
            if src not in v["fitted"]:
                continue
            e, k = v["fitted"][src]["energy"], v["fitted"][src]["const"]
            xs = []
            for s in states_of[fold]:
                for j in range(3):
                    r = rec.get(f"{s}|{j}")
                    if r:
                        val = r["x_plus"] if src == "CLASSICAL" else r["y_plus"]
                        if val is not None:
                            xs.append(val)
            if xs:
                xr = f"{min(xs):.2f}--{max(xs):.2f}"
                tr = [tau_rule(x, e["d0"], e["s"]) for x in xs]
                ta = f"{min(tr):.0f}--{max(tr):.0f}" if max(tr) >= 10 else f"{min(tr):.1f}--{max(tr):.1f}"
            else:
                xr, ta = "undefined", "---"
            # held-out Test A (cp_order): rule vs constant at N=32, A1
            rr, cc = [], []
            for s in states_of[fold]:
                R, _ = testa_rows("cp_order", s)
                armr = "CLASSICAL" if src == "CLASSICAL" else "PREVIOUS"
                armc = "CLASSICAL_CONST" if src == "CLASSICAL" else "PREVIOUS_CONST"
                p, q = testa_pct(R, 32, "A1", armr), testa_pct(R, 32, "A1", armc)
                if p and q:
                    rr.append(p[0]); cc.append(q[0])
            held = f"{np.mean(rr):.2f} vs {np.mean(cc):.2f}" if rr else "---"
            flag = "$^\\P$" if e.get("range_limited") else ""
            L.append(f"{fold} & {src.lower()} & {e['d0']:.4f} & {e['s']:.2f}{flag} & {xr} & {ta} & {fmt(pct_loss(e['loss']))} & {fmt(pct_loss(k['loss']))} & {held}\\\\")
    L.append("\\bottomrule\\end{tabular}")
    return "\n".join(L)


def tab_lih_cal():
    c = json.load(open(os.path.join(RUNS, "cp_lih_cal", "calibration.json")))
    names = {"exact": "exact priors (LiH 1.6 $m{=}1$, LiH 3.0 $m{=}3$)", "degraded": "truncated priors (1 and 2 determinants)",
             "mixed": "exact and truncated priors"}
    L = ["\\begin{tabular}{llrrrrr}\\toprule", "Fit set & Source & $d_0$ & $s$ & rule (shots) & best constant (shots) & constant\\\\\\midrule"]
    for g, v in c["folds"].items():
        for src, f in v["fitted"].items():
            e, k = f["energy"], f["const"]
            val = "$\\infty$" if math.isinf(k["value"]) else f"{k['value']:.0f}"
            L.append(f"{names.get(g, g)} & {src.lower()} & {e['d0']:.4f} & {e['s']:.1f} & {fmt(pct_loss(e['loss']))} & {fmt(pct_loss(k['loss']))} & {val}\\\\")
    L.append("\\bottomrule\\end{tabular}")
    return "\n".join(L)


def fig_trust():
    c = json.load(open(os.path.join(RUNS, "cp_order", "calibration.json")))
    rec = c["records"]
    states_of = {"H2": ["H2_eq_m0", "H2_str_m1"], "H4": ["H4_a1_m1", "H4_a2_m3"], "LiH": ["LiH_r16_m1", "LiH_r30_m3"]}
    col = {"H2": "tab:green", "H4": "tab:purple", "LiH": "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for i, src in enumerate(("CLASSICAL", "PREVIOUS")):
        xx = np.linspace(0, 2.0, 300)
        for fold, v in c["folds"].items():
            e = v["fitted"][src]["energy"]
            ax[i].plot(xx, [tau_rule(x, e["d0"], e["s"]) for x in xx], color=col[fold], lw=1.2, label=f"rule fitted without {fold}")
            for s in states_of[fold]:
                for j in range(3):
                    r = rec.get(f"{s}|{j}")
                    key = "x_plus" if src == "CLASSICAL" else "y_plus"
                    if r and r[key] is not None:
                        ax[i].plot(r[key], tau_rule(r[key], e["d0"], e["s"]), "o", color=col[fold], ms=4)
        ax[i].axhline(128, color="k", ls="--", lw=1, label="constant 128")
        ax[i].set_yscale("log"); ax[i].set_xlabel("$x_+$" if src == "CLASSICAL" else "$y_+$")
        ax[i].set_ylabel("assigned trust $\\tau$"); ax[i].set_title(src.lower() + " source", fontsize=9); ax[i].grid(alpha=0.25)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cp12_fig_trust.pdf")); plt.close(fig)


# ---------------------------------------------------------------- exact-energy variant
def tab_xe():
    L = ["\\begin{tabular}{lrrrrrrrr}\\toprule", " & \\multicolumn{4}{c}{$N=32$} & \\multicolumn{4}{c}{$N=128$}\\\\",
         " & \\multicolumn{2}{c}{A1} & \\multicolumn{2}{c}{A2} & \\multicolumn{2}{c}{A1} & \\multicolumn{2}{c}{A2}\\\\",
         "State & 128 & rule & 128 & rule & 128 & rule & 128 & rule\\\\\\midrule"]
    for st, lab in (("LiH_r16_m2", "LiH 1.6~\\AA, $m=2$"), ("LiH_r30_m3", "LiH 3.0~\\AA, $m=3$"), ("LiH_r30_m5", "LiH 3.0~\\AA, $m=5$"),
                    ("LiH_r30_m6", "LiH 3.0~\\AA, $m=6$"), ("LiH_r30_m3_K2", "$m=3$, 2 determinants (31 mHa)"), ("LiH_r30_m5_K3", "$m=5$, 3 determinants (51 mHa)")):
        R, _ = testa_rows("cp_v3x", st)
        cells = []
        for N in (32, 128):
            for mode in ("A1", "A2"):
                for arm in ("CLASSICAL_T128", "CLASSICAL_XE"):
                    p = testa_pct(R, N, mode, arm)
                    cells.append(fmt(*p) if p else "---")
        L.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")
    return "\n".join(L)


def main():
    parts = {
        "testa": tab_testa(), "testa_small": tab_testa_small(),
        "testb_lih": tab_testb(TB_LIH), "testb_old": tab_testb(TB_OLD),
        "testb_rholih": tab_testb(TB_RHO_LIH), "testb_rhowater": tab_testb(TB_RHO_WATER),
        "calib": tab_calibration(), "calib_lih": tab_lih_cal(), "xe": tab_xe(),
    }
    with open(os.path.join(OUT, "cp12_tabs.tex"), "w", encoding="utf-8") as f:
        for k, v in parts.items():
            f.write(f"% ---- {k}\n\\newcommand{{\\tab{k.replace('_', '')}}}{{%\n{v}\n}}\n\n")
    fig_testa(); fig_success(); pts = fig_error(); fig_trust()
    for p in pts:
        print("ERRPT", p)


if __name__ == "__main__":
    main()
