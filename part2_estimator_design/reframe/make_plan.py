#!/usr/bin/env python3
"""Inspect or export a task plan; never starts a backend."""
from pathlib import Path
import argparse, hashlib, json
from collections import Counter
root=Path(__file__).resolve().parent
p=json.loads((root/"protocol.json").read_text())
digest=hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()
tasks=[]
def add(stage,system,state,method,setting,trials,pool="UCCSD"):
    for trial in range(trials):
        key="|".join(map(str,[p["protocol_version"],stage,state,pool,method,setting["id"],trial]))
        tasks.append(dict(task_id=hashlib.sha256(key.encode()).hexdigest()[:20],
          protocol_version=p["protocol_version"],protocol_sha256=digest,stage=stage,
          system_id=system,state_id=state,pool_id=pool,method=method,
          setting_id=setting["id"],rho=setting["rho"],delta=setting["delta"],trial=trial,
          seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big"),
          execution_status="blocked_pending_backend_inputs_validation_and_cost_gate"))
strict={"id":"strict","rho":0.0,"delta":0.05}
for state in p["fixed_states"]:
    group=state["group"]
    n=p["fixed"]["trials_"+("active_space" if group=="active_space" else group)]
    methods=p["primary_methods"]+([p["supplementary_method"]] if group=="core" else [])
    for method in methods:
        add("fixed_"+group,state["system"],state["id"],method,strict,n)
for system in p["supported_pool"]["systems"]:
    for step in p["supported_pool"]["steps"]:
        for method in p["primary_methods"]+[p["supplementary_method"]]:
            add("supported_pool",system,f"{system}_QEB_ADAPT{step}",method,strict,
                p["supported_pool"]["trials"],"QEB")
for system in p["trajectories"]["systems"]:
    for method in p["primary_methods"]:
        for setting in p["trajectories"]["settings"]:
            add("trajectory",system,system+"_HF",method,setting,p["trajectories"]["trials"])
for system in ("H2O_cas_eq","H2O_cas_str"):
    for setting in p["trajectories"]["settings"]:
        if setting["id"] in p["active_space_horizon"]["settings"]:
            add("active_space_horizon",system,system+"_HF","FC_SE_CS",setting,
                p["active_space_horizon"]["trials"])
parser=argparse.ArgumentParser()
parser.add_argument("--tasks",type=Path,help="Optional JSONL task-plan output")
args=parser.parse_args()
print(json.dumps({"protocol_sha256":digest,"backend_implemented":False,
 "production_trial_tasks":dict(Counter(t["stage"] for t in tasks)),
 "total_trial_tasks":len(tasks),
 "note":"Pilots, algebra checks and R4 diagnostics are prerequisites specified in workflow/SI; no jobs launched."},indent=2))
if args.tasks:
    args.tasks.write_text("".join(json.dumps(t,sort_keys=True)+"\n" for t in tasks))

