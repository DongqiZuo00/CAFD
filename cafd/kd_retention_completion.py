"""Seal an already-saved run under the user's explicit all-success-skip clarification."""
import hashlib,json,pathlib,subprocess,time
from collections import Counter
from cafd.kd_retention_finalize import _publish_once,_pretty

ROOT=pathlib.Path("/blue/du.j/jinjiaguo/CAFD")
RUN="mistral_cafd_kd_retention_replay_s2027"
REP=ROOT/"reports/kd_retention_replay_s2027"
OUT=ROOT/"runs/cafd/experiments"/RUN
ART=ROOT/"artifacts/cafd/experiments"/RUN
PROTOCOL={"authorization":"user_approved_200_rounds_with_199_actual_updates",
 "preserved_round_budget":200,"actual_optimizer_updates":199,
 "skipped_all_success_rounds":[194],"additional_training_authorized":False}
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]

def main():
 assert pathlib.Path.cwd().resolve()==ROOT
 auth=read(REP/"protocol_clarification_20260908/user_authorization.json")
 assert auth["protocol_clarification"]==PROTOCOL
 assert not (ART/"finalization_identity.json").exists(), "test must not have started"
 cfg=read(OUT/"config.json");costs=read(ART/"costs.json")
 controller=read(OUT/"controller_state.json");manifest=read(OUT/"data_manifest.json")
 assert cfg["profile"]=="formal" and cfg["total_rounds"]==200 and cfg["seed"]==2027
 assert controller["completed_rounds"]==200 and controller["completed_full_blocks"]==20 and controller["pending"] is None
 assert costs["rollout_rounds"]==200 and costs["actual_optimizer_updates"]==199
 train=rows(ART/"training_curve.jsonl")
 assert [r["round"] for r in train]==list(range(1,201))
 assert sum(r["optimizer_step"] for r in train)==199
 skipped=[r for r in train if not r["optimizer_step"]]
 assert [r["round"] for r in skipped]==[194]
 for r in skipped:
  assert r["kd_loss"]==r["rl_loss"]==r["gradient_norm"]==0
  assert len(r["group_routing"])==4
  assert all(len(g["full_pass"])==8 and all(g["full_pass"]) and g["rewards"]==[1.0]*8 and not g["use_kd"] and not g["use_rl"] for g in r["group_routing"])
 raw=rows(OUT/"raw_rollouts.jsonl")
 assert len(raw)==6400 and len({(r["round"],r["row_id"]) for r in raw})==800
 assert all(r["full_pass"] is True and r["reward"]==1 for r in raw if r["round"]==194)
 dev=rows(ART/"development_curve.jsonl");devout=rows(OUT/"development_outputs.jsonl")
 assert [r["round"] for r in dev]==[0,10,40,80,120,160,200]
 assert Counter(r["round"] for r in devout)==Counter({r:64 for r in [0,10,40,80,120,160,200]})
 for d in dev:
  rs=[r for r in devout if r["round"]==d["round"]]
  assert sum(r["full_pass"] for r in rs)==d["correct"]
 best=read(OUT/"selection_in_progress.json")
 assert best["round"]==max(dev,key=lambda r:(r["correct"],-r["round"]))["round"]==120
 assert best["correct"]==31 and pathlib.Path(best["checkpoint"]).is_dir()
 resume=OUT/"resume.pt"
 assert resume.stat().st_size>48_000_000_000
 assert resume.stat().st_mtime_ns <= (ART/"costs.json").stat().st_mtime_ns
 allocation=read(OUT/"allocation.json")
 sacct=subprocess.run(["sacct","-X","-n","-P","-j","41361573","-o","JobIDRaw,State,ElapsedRaw,ExitCode"],capture_output=True,text=True,check=True).stdout.strip()
 assert "41361573|FAILED|" in sacct and sacct.endswith("|1:0")
 assert "round budget completed without required optimizer updates" in (ROOT/"logs/cafd/kd-retention-formal-41361573.err").read_text()
 result=dict(status="STUDENT_TRAINING_COMPLETE",formal_result=True,rounds=200,actual_updates=199,
  predictive_decisions=controller["predictive_decisions"],multiaction_mpc_decisions=controller["mpc_multiaction_decisions"],
  final_u=controller["u"],selected=best,frozen_test_evaluated=False,costs=costs,job_id="41361573",
  node=subprocess.run(["sacct","-X","-n","-P","-j","41361573","-o","NodeList"],capture_output=True,text=True,check=True).stdout.strip(),gpu=allocation["gpu_name"],data_limitations=manifest.get("limitations",[]),
  protocol_clarification=PROTOCOL,completion_recovery={"reason":"User-authorized199 actual updates after all-success skip at round194",
   "original_job_exit":"FAILED1:0 from strict update-count completion guard, after fullround200 resume saved",
   "training_or_model_loading_performed":False})
 _publish_once(OUT/"selected.json",_pretty(dict(status="frozen",selection_rule="development_full_pass_earliest_tie",**best)))
 _publish_once(OUT/"complete.json",_pretty(result))
 _publish_once(ART/"result.json",_pretty(result))
 state=ROOT/"state/cafd/experiments"/RUN/"state.json"
 temp=state.with_suffix(".completion.tmp");temp.write_text(_pretty(result));temp.replace(state)
 (REP/"protocol_clarification_20260908/completion_recovery.json").write_text(_pretty({"status":"PASS","actual_updates":199,"rounds":200,"selected_round":120,"test_not_started":True,"no_training_or_model_loading":True,"sacct":sacct,"resume_bytes":resume.stat().st_size,"protocol_clarification":PROTOCOL}))
 print(json.dumps({"status":"STUDENT_TRAINING_COMPLETE","rounds":200,"actual_updates":199,"selected_round":120,"no_training_or_model_loading":True}))
if __name__=="__main__":main()
