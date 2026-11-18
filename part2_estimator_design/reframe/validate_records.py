#!/usr/bin/env python3
"""Validate summary records; does not certify scientific correctness."""
from pathlib import Path
import argparse, hashlib, json, math
root=Path(__file__).resolve().parent
protocol=json.loads((root/"protocol.json").read_text())
digest=hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()
allowed={"certified","unresolved","fallback","interval_conflict","optimizer_failure",
         "target_reached","trajectory_cap","step_cap","stationary"}
required={"record_id","protocol_version","protocol_sha256","code_commit","environment_file",
 "system_id","state_id","state_sha256","hamiltonian_sha256","pool_sha256","contexts_sha256",
 "method","pool_id","rho","delta","seed","status","selection_shots","block_shots",
 "selected_index","used_exact_moments_for_design","evidence_directory","wall_seconds","peak_memory_bytes"}
def validate(r):
    errors=[]
    missing=required-set(r)
    if missing: return ["missing fields: "+", ".join(sorted(missing))]
    if r["protocol_version"]!=protocol["protocol_version"]: errors.append("protocol version mismatch")
    if r["protocol_sha256"]!=digest: errors.append("protocol hash mismatch")
    if r["method"] not in protocol["primary_methods"]+[protocol["supplementary_method"]]:
        errors.append("unknown method")
    if r["status"] not in allowed: errors.append("unknown status")
    if not 0<=r["rho"]<1 or not 0<r["delta"]<1: errors.append("invalid rho/delta")
    if r["used_exact_moments_for_design"] is not False:
        errors.append("exact moments cannot enter primary empirical design")
    def count(x): return isinstance(x,int) and not isinstance(x,bool) and x>=0
    if not count(r["selection_shots"]): errors.append("invalid shot count")
    if not isinstance(r["block_shots"],list) or not all(count(x) for x in r["block_shots"]):
        errors.append("invalid block counts")
    elif sum(r["block_shots"])!=r["selection_shots"]:
        errors.append("block shots do not sum to total")
    if r["status"] in {"certified","fallback"} and not count(r["selected_index"]):
        errors.append("selected index required")
    for key in ("wall_seconds","peak_memory_bytes"):
        if not isinstance(r[key],(int,float)) or not math.isfinite(r[key]) or r[key]<0:
            errors.append("invalid "+key)
    for key in ("code_commit","environment_file","evidence_directory","state_id"):
        if not isinstance(r[key],str) or not r[key].strip(): errors.append("empty "+key)
    for key in ("state_sha256","hamiltonian_sha256","pool_sha256","contexts_sha256"):
        value=r[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            errors.append("invalid "+key)
    # Paths are deliberately checked by the archive audit, not inferred from this summary.
    return errors
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("records",type=Path)
    args=ap.parse_args()
    seen=set(); errors=[]; n=0
    for line_no,line in enumerate(args.records.read_text().splitlines(),1):
        if not line.strip(): continue
        n+=1
        try:
            r=json.loads(line)
            e=validate(r)
            if r.get("record_id") in seen: e.append("duplicate record_id")
            seen.add(r.get("record_id"))
            errors.extend(f"line {line_no}: {x}" for x in e)
        except (ValueError,TypeError,KeyError) as exc:
            errors.append(f"line {line_no}: malformed record ({exc})")
    if n==0: errors.append("no records: no numerical evidence")
    if errors: raise SystemExit("\n".join(errors))
    print(f"{n} summary records structurally valid; source/block audit still required.")

