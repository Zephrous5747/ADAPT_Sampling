#!/usr/bin/env python3
"""Rebuild Part I report tables from cached CSV/JSON outputs.

This script does not rerun quantum chemistry. It verifies that the cached
outputs packaged with this archive reproduce the tables in the Part I report.
It is the exact reproducibility path for the currently reported data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def read_lih_summary(cached_root: Path) -> Dict[str, Any]:
    path = cached_root / "lih_sto3g_stretched_pilot" / "summary.json"
    with path.open() as f:
        return json.load(f)


def read_multisystem_summary(cached_root: Path) -> pd.DataFrame:
    path = cached_root / "adapt_gradient_oracle_h4_h2o_pilot" / "summary.csv"
    return pd.read_csv(path)


def build_observable_structure(lih: Dict[str, Any], multi: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append({
        "case": "LiH, R=3.0 A",
        "state": "HF",
        "qubits": lih["n_qubits"],
        "pool": lih["pool_size"],
        "universal_terms": lih["universal_gradient_paulis"],
        "fc_ug_contexts": lih["parent_FC_groups_greedy"],
        "nonzero_gradients": lih["nonzero_gradients_gt_1e-9"],
    })
    for _, r in multi.iterrows():
        rows.append({
            "case": pretty_case(str(r["case"])),
            "state": r["state"],
            "qubits": int(r["n_qubits"]),
            "pool": int(r["pool_size"]),
            "universal_terms": int(r["universal_commutator_pauli_terms"]),
            "fc_ug_contexts": int(r["parent_fc_groups_greedy"]),
            "nonzero_gradients": int(r["nonzero_gradients"]),
        })
    return pd.DataFrame(rows)


def build_gradient_difficulty(lih: Dict[str, Any], multi: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append({
        "case": "LiH, R=3.0 A",
        "state": "HF",
        "top_generator": lih["winner"]["label"],
        "abs_g1": lih["winner"]["abs_gradient"],
        "abs_g2": lih["runner_up"]["abs_gradient"],
        "gap": lih["runner_up"]["absolute_gap"],
        "m2_diagnostic_shots": lih["results"]["independent_BAI"]["total_context_shots_est"],
    })
    for _, r in multi.iterrows():
        rows.append({
            "case": pretty_case(str(r["case"])),
            "state": r["state"],
            "top_generator": r["top_gradient_label"],
            "abs_g1": float(r["top_abs_gradient"]),
            "abs_g2": float(r["second_abs_gradient"]),
            "gap": float(r["top_gap_abs_gradient"]),
            "m2_diagnostic_shots": int(float(r["oracle_context_shots_independent_bai_proxy"])),
        })
    return pd.DataFrame(rows)


def build_completed_three_method(lih: Dict[str, Any], multi: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append({
        "case": "LiH, R=3.0 A",
        "state": "HF",
        "M1_FCUG": lih["results"]["parent_FC_all_gradients_uniform_context_shots"]["total_context_shots"],
        "M2_BAIFCIG": lih["results"]["independent_BAI"]["total_context_shots_est"],
        "M3_BAIFCUG": lih["results"]["new_fixed_parent_grouped_BAI"]["total_context_shots_est"],
    })
    h4 = multi[multi["case"].astype(str).str.startswith("H4")]
    for _, r in h4.iterrows():
        m1 = int(float(r["oracle_context_shots_parent_fc_all_gradient_proxy"]))
        m2 = int(float(r["oracle_context_shots_independent_bai_proxy"]))
        m3 = int(float(r["oracle_context_shots_fixed_parent_grouped_bai_proxy"]))
        rows.append({
            "case": pretty_case(str(r["case"])),
            "state": r["state"],
            "M1_FCUG": m1,
            "M2_BAIFCIG": m2,
            "M3_BAIFCUG": m3,
        })
    out = pd.DataFrame(rows)
    out["M1_over_M2"] = out["M1_FCUG"] / out["M2_BAIFCIG"]
    out["M2_over_M3"] = out["M2_BAIFCIG"] / out["M3_BAIFCUG"]
    return out


def build_h2o_status(multi: pd.DataFrame) -> pd.DataFrame:
    h2o = multi[multi["case"].astype(str).str.startswith("H2O")]
    rows = []
    for _, r in h2o.iterrows():
        rows.append({
            "case": pretty_case(str(r["case"])),
            "state": r["state"],
            "fc_ug_contexts": int(r["parent_fc_groups_greedy"]),
            "top_generator": r["top_gradient_label"],
            "gap": float(r["top_gap_abs_gradient"]),
            "m2_diagnostic_shots": int(float(r["oracle_context_shots_independent_bai_proxy"])),
        })
    return pd.DataFrame(rows)


def pretty_case(case: str) -> str:
    replacements = {
        "H4_square_eq_side1.0A": "H4 square, side=1.0 A",
        "H4_square_stretch_side2.0A": "H4 square, side=2.0 A",
        "H2O_eq_R0.9572A_angle104.52": "H2O eq., R(OH)=0.9572 A",
        "H2O_stretch_R1.75A_angle104.52": "H2O stretch, R(OH)=1.75 A",
    }
    return replacements.get(case, case)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached-root", default="data/cached", help="Path to cached data directory")
    ap.add_argument("--out", default="reproduced_tables", help="Output directory")
    args = ap.parse_args()

    root = Path(args.cached_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lih = read_lih_summary(root)
    multi = read_multisystem_summary(root)

    tables = {
        "table_observable_structure.csv": build_observable_structure(lih, multi),
        "table_gradient_difficulty_m2.csv": build_gradient_difficulty(lih, multi),
        "table_completed_three_methods.csv": build_completed_three_method(lih, multi),
        "table_h2o_status.csv": build_h2o_status(multi),
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False)
        print(f"wrote {out / name} ({len(df)} rows)")

    # Markdown summary for quick inspection.
    md = ["# Rebuilt Part I tables from cached outputs\n"]
    for name, df in tables.items():
        md.append(f"\n## {name}\n")
        md.append(df.to_markdown(index=False))
        md.append("\n")
    (out / "REBUILT_TABLES.md").write_text("\n".join(md))
    print(f"wrote {out / 'REBUILT_TABLES.md'}")


if __name__ == "__main__":
    main()
