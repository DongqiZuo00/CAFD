#!/usr/bin/env python3
"""Independently clone the published commit and restore/verify all CAFD records."""
import argparse,datetime,hashlib,json,os,re,subprocess,sys
from pathlib import Path
BASE=Path("/blue/du.j/jinjiaguo/CAFD/backup/github_20260908")
URL="https://github.com/DongqiZuo00/CAFD.git"
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--commit",required=True)
 args=ap.parse_args()
 if not re.fullmatch(r"[0-9a-f]{40}",args.commit):raise ValueError("Exact 40-character commit required")
 if BASE.resolve()!=BASE:raise ValueError("Unexpected backup path")
 clone=BASE/"verify_clone";destination=BASE/"restore_verification";auditpath=BASE/"github_roundtrip_audit.json"
 if clone.exists() or destination.exists() or auditpath.exists():raise ValueError("Verification paths must be new")
 local_manifest_sha=digest(BASE/"source-files.jsonl")
 local_restore_sha=digest(BASE/"restore_backup.py")
 local_parts_sha=digest(BASE/"parts.json")
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 env={**os.environ,"GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":"/dev/null","GIT_TERMINAL_PROMPT":"0",
      "GIT_NO_REPLACE_OBJECTS":"1","GIT_OPTIONAL_LOCKS":"0","PYTHONDONTWRITEBYTECODE":"1","TMPDIR":str(BASE)}
 def command(args,cwd=BASE):
  p=subprocess.run(args,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  if p.returncode:raise RuntimeError("Command failed: "+args[0]+" (exit "+str(p.returncode)+")")
  return p.stdout
 def git(args,cwd=BASE):
  return command(["git","--no-replace-objects","-c","credential.helper=","-c","core.hooksPath=/dev/null",*args],cwd)
 print("clone_public_https",flush=True)
 git(["clone","--no-checkout","--no-tags",URL,str(clone)])
 git(["checkout","--detach",args.commit],clone)
 actual=git(["rev-parse","HEAD"],clone).strip()
 if actual!=args.commit:raise RuntimeError("Checked-out commit differs")
 remote_manifest=clone/"backup/2026-09-08/source-files.jsonl"
 remote_parts=clone/"backup/2026-09-08/parts.json"
 remote_restore=clone/"restore_backup.py"
 if digest(remote_manifest)!=local_manifest_sha:raise RuntimeError("Published source manifest differs")
 if digest(remote_parts)!=local_parts_sha:raise RuntimeError("Published part manifest differs")
 if digest(remote_restore)!=local_restore_sha:raise RuntimeError("Published restore script differs from reviewed script")
 parts=json.loads(remote_parts.read_text())
 if len(parts["parts"])!=62 or parts["chunk_size_bytes"]!=6*1024*1024:raise RuntimeError("Unexpected chunk layout")
 print("verify_downloaded_chunks",flush=True)
 full=hashlib.sha256();offset=0
 for row in parts["parts"]:
  p=remote_parts.parent/row["name"]
  if not p.resolve().is_relative_to(remote_parts.parent) or p.is_symlink():raise RuntimeError("Unsafe chunk")
  data=p.read_bytes()
  if len(data)!=row["size_bytes"] or offset!=row["offset"]:raise RuntimeError("Chunk size/offset mismatch")
  if hashlib.sha256(data).hexdigest()!=row["sha256"]:raise RuntimeError("Chunk SHA256 mismatch")
  if hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()!=row["git_blob_sha"]:raise RuntimeError("Chunk Git blob mismatch")
  full.update(data);offset+=len(data)
 if offset!=parts["archive_size_bytes"] or full.hexdigest()!=parts["archive_sha256"]:raise RuntimeError("Downloaded archive hash differs")
 print("restore_repository_script",flush=True)
 restore_output=command([sys.executable,str(remote_restore),"--destination",str(destination)],clone)
 with (BASE/"github_roundtrip_restore.stdout.log").open("x") as f:f.write(restore_output)
 print("verify_restored_files",flush=True)
 expected={}
 for line in remote_manifest.read_text().splitlines():
  row=json.loads(line)
  if row["path"] in expected:raise RuntimeError("Duplicate source manifest path")
  expected[row["path"]]=row
 total=0
 for i,(name,row) in enumerate(expected.items()):
  q=destination/name
  if not q.resolve().is_relative_to(destination) or q.is_symlink() or not q.is_file():raise RuntimeError("Missing/unsafe restored path: "+name)
  if q.stat().st_size!=row["size"] or digest(q)!=row["sha256"]:raise RuntimeError("Restored file checksum mismatch: "+name)
  total+=row["size"]
  if i%1000==0:print("verified",i+1,"of",len(expected),flush=True)
 actual_files={q.relative_to(destination).as_posix() for q in destination.rglob("*") if q.is_file() or q.is_symlink()}
 if actual_files!=set(expected):raise RuntimeError("Restored file set differs from manifest")
 if len(expected)!=parts["source_file_count"]:raise RuntimeError("Restored file count differs")
 audit={"status":"passed","started_at":started,"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "public_repository":URL,"commit":actual,"clone_path":str(clone),"destination":str(destination),
        "chunk_count":len(parts["parts"]),"chunk_sha256_and_git_blob_verified":True,
        "archive_size_bytes":offset,"archive_sha256":full.hexdigest(),
        "source_manifest_sha256":local_manifest_sha,"parts_manifest_sha256":local_parts_sha,
        "restore_script_sha256":local_restore_sha,"repository_restore_script_exit_code":0,
        "restored_file_count":len(expected),"restored_bytes":total,
        "all_restored_sha256_match":True,"exact_file_set_matches":True,
        "network_source":"Independent public HTTPS clone from GitHub; no local archive used for restoration.",
        "original_sources_modified":False}
 with auditpath.open("x") as f:json.dump(audit,f,indent=2);f.write("\n")
 print(json.dumps(audit),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as e:
  failure={"status":"failed","error_type":type(e).__name__,"reason":str(e) if isinstance(e,(RuntimeError,ValueError)) else "operation_failed_see_type","at":datetime.datetime.now(datetime.timezone.utc).isoformat()}
  p=BASE/"github_roundtrip_failure.json"
  if not p.exists():
   with p.open("x") as f:json.dump(failure,f,indent=2);f.write("\n")
  print(json.dumps(failure),flush=True);sys.exit(1)
