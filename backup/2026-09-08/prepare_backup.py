#!/usr/bin/env python3
"""Original-path CAFD working-record backup. Only writes its own backup directory."""
import collections, datetime as dt, gzip, hashlib, io, json, os, re, stat
import subprocess, sys, tarfile, tempfile, time, zipfile
from pathlib import Path, PurePosixPath
ROOT=Path("/blue/du.j/jinjiaguo/CAFD")
OUT=ROOT/"backup/github_20260908"
ROOT_FILES=["README.md","pyproject.toml",".gitignore",".gitmodules"]
ROOT_TREES=["cafd","configs","scripts","tests","experiments","docs","reports","artifacts","logs","data/cafd","runs/cafd","state/cafd","tmp"]
EXCLUDED_DIRS={".venv",".cache","__pycache__",".pytest_cache",".git"}
BINARY_SUFFIXES={".pt",".pth",".bin",".safetensors",".npy",".npz"}
SECRET_NAMES={".gitconfig",".git-credentials",".netrc","_netrc","id_rsa","id_dsa","id_ecdsa","id_ed25519"}
CHUNK=1024*1024
OVERLAP=8192
PATTERNS=[
 ("private_key",b"PRIVATE KEY",rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
 ("pgp_private_key",b"PGP PRIVATE",rb"-----BEGIN PGP PRIVATE KEY BLOCK-----"),
 ("putty_private_key",b"PuTTY-User",rb"PuTTY-User-Key-File-[23]:"),
 ("github_classic_token",b"gh",rb"\bgh[pousr]_[A-Za-z0-9]{30,255}\b"),
 ("github_fine_grained_token",b"github_pat_",rb"\bgithub_pat_[A-Za-z0-9_]{40,255}\b"),
 ("huggingface_token",b"hf_",rb"\bhf_[A-Za-z0-9]{30,100}\b"),
 ("openai_token",b"sk-",rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,255}\b"),
 ("anthropic_token",b"sk-ant-",rb"\bsk-ant-[A-Za-z0-9_-]{30,255}\b"),
 ("aws_access_id",b"A",rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
 ("google_api_key",b"AIza",rb"\bAIza[A-Za-z0-9_-]{35}\b"),
 ("google_oauth_token",b"ya29.",rb"\bya29\.[A-Za-z0-9_-]{30,512}\b"),
 ("slack_token",b"xox",rb"\bxox[baprs]-[A-Za-z0-9-]{20,255}\b"),
 ("slack_webhook",b"hooks.slack.com",rb"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{20,255}"),
 ("gitlab_token",b"glpat-",rb"\bglpat-[A-Za-z0-9_-]{20,255}\b"),
 ("npm_token",b"npm_",rb"\bnpm_[A-Za-z0-9]{30,255}\b"),
 ("stripe_live_key",b"k_live_",rb"\b[rs]k_live_[A-Za-z0-9]{16,255}\b"),
 ("azure_storage_key",None,rb"(?i)\bAccountKey\s*=\s*[A-Za-z0-9+/]{80,90}={0,2}"),
 ("credential_url",b"://",rb"(?i)\b(?:https?|ssh|ftp|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s/:@\"']{1,100}:[^\s/@\"']{3,255}@"),
 ("authorization_bearer",None,rb"(?i)\bAuthorization[\"']?\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9_.~+/-]{16,512}"),
 ("authorization_basic",None,rb"(?i)\bAuthorization[\"']?\s*[:=]\s*[\"']?Basic\s+[A-Za-z0-9+/]{16,512}={0,2}"),
 ("jwt_token",b"eyJ",rb"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}\b")]
REGEXES=[(name,hint,re.compile(pattern)) for name,hint,pattern in PATTERNS]
SECRET_KEY=rb"(?:[A-Za-z0-9_]*(?:api_key|access_token|auth_token|client_secret|secret_access_key|password|passwd|private_key)|token|secret)"
ASSIGNMENT=re.compile(rb"(?i)(?<![A-Za-z0-9_])"+SECRET_KEY+rb"[\"']?\s*[:=]\s*[\"']([^\"'\r\n]{4,512})[\"']")
ENV_ASSIGNMENT=re.compile(rb"(?im)^\s*(?:export\s+)?"+SECRET_KEY+rb"\s*=\s*([A-Za-z0-9_+/.=-]{8,512})\s*$")
QUERY_SECRET=re.compile(rb"(?i)[?&](?:api_key|access_token|auth_token|client_secret|password|secret|token)=([^&#\s\"']{8,512})")
PLACEHOLDER=re.compile(rb"(?i)^(?:none|null|false|true|undefined|your[_ -].*|example.*|dummy.*|fake.*|test(?:[_ -].*)?|changeme|change_me|redacted|masked|placeholder|insert[_ -].*|[xX*]{4,}|\$\{.*\}|<.*>)$")
scan_state={"findings":[],"coverage":collections.Counter(),"archives":[],"rule_ids":[x[0] for x in PATTERNS]+["quoted_secret_assignment","env_secret_assignment","query_secret","credential_container_path"]}
found_keys=set()
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()
def write_json(name,obj):
 with (OUT/name).open("x",encoding="utf8") as f:json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True);f.write("\n")
def write_jsonl(name,rows):
 with (OUT/name).open("x",encoding="utf8") as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
def snapshot(p):
 s=p.lstat()
 return {"size":s.st_size,"mtime_ns":s.st_mtime_ns,"ctime_ns":s.st_ctime_ns,"mode":stat.S_IMODE(s.st_mode),"device":s.st_dev,"inode":s.st_ino}
def unchanged(p,r):
 now=snapshot(p)
 if now!={k:r[k] for k in now}:raise RuntimeError("source_changed:"+str(p.relative_to(ROOT)))
def finding(path,rule):
 key=(path,rule)
 if key not in found_keys:found_keys.add(key);scan_state["findings"].append({"path":path,"rule":rule})
def fake(value):
 value=value.strip()
 return bool(PLACEHOLDER.fullmatch(value) or value.isdigit() or any(x in value for x in [b"os.environ",b"os.getenv",b"$"+b"{",b"process.env"]))
def scan_bytes(path,b):
 # ASCII-compatible text plus UTF-16 ASCII strings; binary files are never deserialized.
 variants=[b]
 if b"\x00" in b[:4096]:variants.append(b.replace(b"\x00",b""))
 for raw in variants:
  lower=raw.lower()
  for name,hint,rx in REGEXES:
   if hint is not None and hint not in raw:continue
   if name=="azure_storage_key" and b"accountkey" not in lower:continue
   if name.startswith("authorization_") and b"authorization" not in lower:continue
   if rx.search(raw):finding(path,name)
  if any(k in lower for k in [b"token",b"secret",b"password",b"passwd",b"api_key",b"private_key"]):
   for rx,rule in [(ASSIGNMENT,"quoted_secret_assignment"),(ENV_ASSIGNMENT,"env_secret_assignment"),(QUERY_SECRET,"query_secret")]:
    for m in rx.finditer(raw):
     if not fake(m.group(1)):finding(path,rule)
def digest_stream(f,scan_path=None):
 h=hashlib.sha256();prev=b"";n=0
 while True:
  b=f.read(CHUNK)
  if not b:break
  h.update(b);n+=len(b)
  if scan_path is not None:scan_bytes(scan_path,prev+b);prev=(prev+b)[-OVERLAP:]
 return h.hexdigest(),n
def hash_file(p):
 with p.open("rb") as f:return digest_stream(f)[0]
def secret_container(path):
 p=PurePosixPath(path.replace("\\","/"))
 return p.name in SECRET_NAMES or any(a in {".git",".ssh"} for a in p.parts) or "/.aws/credentials" in "/"+str(p)
def kind(name):
 low=name.lower()
 if low.endswith((".tar.gz",".tgz",".tar",".tar.bz2",".tbz2",".tar.xz",".txz")):return "tar"
 if low.endswith((".zip",".whl",".docx",".xlsx",".pptx")):return "zip"
 if low.endswith(".gz"):return "gzip"
 return None
def scan_archive(fileobj,label,archive_kind,depth=0):
 if depth>12:raise RuntimeError("archive_recursion_limit:"+label)
 info={"path":label,"kind":archive_kind,"regular_files":0,"bytes":0,"links":0,"embedded_binary_files":0}
 scan_state["archives"].append(info)
 def consume(f,name,size):
  member_label=label+"!"+name
  info["regular_files"]+=1;info["bytes"]+=size;scan_state["coverage"]["archive_member_files"]+=1
  if secret_container(name):finding(member_label,"credential_container_path")
  if Path(name).suffix.lower() in BINARY_SUFFIXES:info["embedded_binary_files"]+=1
  nested=kind(name)
  if nested:
   with tempfile.SpooledTemporaryFile(max_size=8*CHUNK,dir=OUT) as tmp:
    while True:
     b=f.read(CHUNK)
     if not b:break
     tmp.write(b)
    tmp.seek(0);scan_archive(tmp,member_label,nested,depth+1)
  else:
   digest_stream(f,member_label)
 if archive_kind=="tar":
  with tarfile.open(fileobj=fileobj,mode="r|*") as tar:
   for m in tar:
    if m.issym() or m.islnk():info["links"]+=1;continue
    if m.isfile():
     f=tar.extractfile(m)
     if f is None:raise RuntimeError("archive_member_unreadable:"+label+"!"+m.name)
     with f:consume(f,m.name,m.size)
 elif archive_kind=="zip":
  with zipfile.ZipFile(fileobj) as z:
   for m in z.infolist():
    if m.is_dir():continue
    if m.flag_bits&1:raise RuntimeError("encrypted_archive_member:"+label+"!"+m.filename)
    if stat.S_ISLNK(m.external_attr>>16):info["links"]+=1;continue
    with z.open(m) as f:consume(f,m.filename,m.file_size)
 else:
  with gzip.GzipFile(fileobj=fileobj,mode="rb") as f:consume(f,label.rsplit("/",1)[-1][:-3],0)
def inventory():
 included=[];excluded=[];missing=[];directories=[]
 def visit(p,inherited=None):
  rel=p.relative_to(ROOT).as_posix();s=p.lstat();rec={"path":rel,**snapshot(p)}
  if stat.S_ISLNK(s.st_mode):
   excluded.append({**rec,"reason":"symlink_record_only_not_followed","target":os.readlink(p)});return
  reason=inherited
  if p.name in EXCLUDED_DIRS:reason="excluded_directory:"+p.name
  if stat.S_ISDIR(s.st_mode):
   if reason:excluded.append({**rec,"reason":reason,"type":"directory"})
   else:directories.append(rec)
   with os.scandir(p) as entries:
    for e in sorted(entries,key=lambda e:e.name):visit(Path(e.path),reason)
   return
  if not stat.S_ISREG(s.st_mode):excluded.append({**rec,"reason":"non_regular_file"});return
  if reason:excluded.append({**rec,"reason":reason});return
  if p.suffix.lower() in BINARY_SUFFIXES:excluded.append({**rec,"reason":"model_optimizer_probe_binary_extension"});return
  if p.suffix.lower()==".pyc":excluded.append({**rec,"reason":"python_bytecode"});return
  if secret_container(rel):finding(rel,"credential_container_path")
  included.append(rec)
 roots=ROOT_FILES+ROOT_TREES
 if (ROOT/"archive").exists():
  for p in sorted((ROOT/"archive").iterdir()):
   if "cafd" in p.name.lower():roots.append(p.relative_to(ROOT).as_posix())
 for name in roots:
  p=ROOT/name
  if not p.exists() and not p.is_symlink():missing.append(name)
  else:visit(p)
 return sorted(included,key=lambda x:x["path"]),sorted(excluded,key=lambda x:x["path"]),missing,roots,sorted(directories,key=lambda x:x["path"])
def git(args,**kwargs):
 env={**os.environ,"GIT_OPTIONAL_LOCKS":"0","GIT_TERMINAL_PROMPT":"0","GIT_NO_REPLACE_OBJECTS":"1"}
 p=subprocess.run(["git","--no-replace-objects","-c","core.hooksPath=/dev/null",*args],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
 if p.returncode:raise RuntimeError("git_command_failed:"+args[0]+":"+str(p.returncode))
 return p.stdout
def scan_git():
 refs=git(["show-ref","--head"]).decode("utf8").splitlines()
 rows=git(["rev-list","--objects","--all"]).decode("utf8").splitlines()
 objects=[(line.partition(" ")[0],line.partition(" ")[2]) for line in rows]
 type_rows=git(["cat-file","--batch-check=%(objectname) %(objecttype) %(objectsize)"],input=("\n".join(x[0] for x in objects)+"\n").encode()).decode().splitlines()
 paths=dict(objects);counter=collections.Counter();total=0
 for line in type_rows:
  oid,typ,size=line.split();size=int(size);counter[typ]+=1;total+=size
  if typ not in {"blob","commit","tag"}:continue
  label="git:"+oid+":"+paths.get(oid,"")
  if paths.get(oid) and secret_container(paths[oid]):finding(label,"credential_container_path")
  data=git(["cat-file",typ,oid])
  if len(data)!=size:raise RuntimeError("git_object_size_changed:"+oid)
  scan_bytes(label,data);scan_state["coverage"]["git_objects_scanned"]+=1
  nested=kind(paths.get(oid,""))
  if nested:scan_archive(io.BytesIO(data),label,nested)
 return {"refs_before":refs,"object_counts":dict(counter),"object_bytes":total,"git_local_configuration_included":False,"reachable_history_scanned":True,"replace_objects_disabled":True}
def main():
 started=stamp();t0=time.monotonic()
 if OUT.resolve()!=OUT or ROOT.resolve()!=ROOT:raise RuntimeError("unexpected_root_symlink")
 for name in ["working-records.tar.gz","original-history.bundle","credential-scan.json","archive-audit.json","source-files.jsonl"]:
  if (OUT/name).exists():raise RuntimeError("refusing_overwrite:"+name)
 print("inventory",flush=True)
 records,excluded,missing,scope,directories=inventory()
 write_jsonl("excluded-files.jsonl",excluded)
 print("scan_source_files",len(records),"bytes",sum(r["size"] for r in records),flush=True)
 for i,r in enumerate(records):
  p=ROOT/r["path"];unchanged(p,r)
  with p.open("rb") as f:r["sha256"],n=digest_stream(f,r["path"])
  if n!=r["size"]:raise RuntimeError("read_size_mismatch:"+r["path"])
  archive_kind=kind(r["path"])
  if archive_kind:
   with p.open("rb") as f:scan_archive(f,r["path"],archive_kind)
  unchanged(p,r);scan_state["coverage"]["source_files_scanned"]+=1;scan_state["coverage"]["source_bytes_scanned"]+=n
  if i%500==0:print("scanned",i+1,"of",len(records),flush=True)
 print("scan_git_reachable_history",flush=True)
 git_audit=scan_git()
 write_jsonl("source-files.jsonl",records)
 with (OUT/"source-SHA256SUMS").open("x",encoding="utf8") as f:
  for r in records:f.write(r["sha256"]+"  "+r["path"]+"\n")
 by_scope={}
 for name in scope:
  rr=[r for r in records if r["path"]==name or r["path"].startswith(name+"/")]
  by_scope[name]={"files":len(rr),"bytes":sum(r["size"] for r in rr)}
 summary={"started_at":started,"root":str(ROOT),"scope":scope,"missing_optional_roots":missing,"source_files":len(records),"source_bytes":sum(r["size"] for r in records),"excluded_entries":len(excluded),"excluded_regular_bytes":sum(r["size"] for r in excluded if r.get("type")!="directory" and r.get("reason")!="symlink_record_only_not_followed"),"symlinks_recorded":sum(r["reason"]=="symlink_record_only_not_followed" for r in excluded),"by_scope":by_scope,"preservation":"Original paths/bytes; existing archives unchanged; temporary test records retained; symlinks indexed without following."}
 write_json("source-summary.json",summary)
 scan_state["status"]="blocked" if scan_state["findings"] else "passed"
 scan_state["finished_at"]=stamp()
 scan_state["limitations"]="Pattern-based credential inspection; no matching values/context written. Candidate raw bytes, recursive archive members, reachable Git blobs/commit/tag messages scanned. Model binary source contents not read."
 write_json("credential-scan.json",scan_state)
 if scan_state["findings"]:
  write_json("archive-audit.json",{"status":"blocked","reason":"possible_credentials","findings":scan_state["findings"],"archive_created":False,"bundle_created":False})
  print(json.dumps({"status":"blocked","findings":scan_state["findings"]}),flush=True);return 2
 print("credential_scan_passed; building_git_bundle",flush=True)
 git(["bundle","create",str(OUT/"original-history.bundle"),"--all"])
 git(["bundle","verify",str(OUT/"original-history.bundle")])
 git_audit["refs_after"]=git(["show-ref","--head"]).decode("utf8").splitlines()
 if git_audit["refs_before"]!=git_audit["refs_after"]:raise RuntimeError("git_refs_changed")
 git_audit.update({"status":"passed","bundle_sha256":hash_file(OUT/"original-history.bundle"),"bundle_bytes":(OUT/"original-history.bundle").stat().st_size,"bundle_heads":git(["bundle","list-heads",str(OUT/"original-history.bundle")]).decode("utf8").splitlines()})
 write_json("git-bundle-audit.json",git_audit)
 print("building_working_records",flush=True)
 packed=OUT/"working-records.tar.gz"
 class CheckedReader:
  def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
  def read(self,n=-1):
   b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
 with packed.open("xb") as raw:
  with gzip.GzipFile(filename="",mode="wb",fileobj=raw,compresslevel=1,mtime=0) as gz:
   with tarfile.open(fileobj=gz,mode="w|",format=tarfile.PAX_FORMAT) as tar:
    for d in directories:
     t=tarfile.TarInfo(d["path"]);t.type=tarfile.DIRTYPE;t.mode=d["mode"];t.mtime=d["mtime_ns"]/1e9;t.pax_headers={"mtime":f'{d["mtime_ns"]//1000000000}.{d["mtime_ns"]%1000000000:09d}'};tar.addfile(t)
    for i,r in enumerate(records):
     p=ROOT/r["path"];unchanged(p,r)
     t=tarfile.TarInfo(r["path"]);t.size=r["size"];t.mode=r["mode"];t.mtime=r["mtime_ns"]/1e9;t.pax_headers={"mtime":f'{r["mtime_ns"]//1000000000}.{r["mtime_ns"]%1000000000:09d}'}
     with p.open("rb") as f:reader=CheckedReader(f);tar.addfile(t,reader)
     if reader.n!=r["size"] or reader.h.hexdigest()!=r["sha256"]:raise RuntimeError("archive_input_hash_mismatch:"+r["path"])
     unchanged(p,r)
     if i%1000==0:print("packed",i+1,"of",len(records),flush=True)
 print("verify_archive_bytes",flush=True)
 expected={r["path"]:r for r in records};seen=set();verified_bytes=0
 with tarfile.open(packed,"r|gz") as tar:
  for m in tar:
   if m.isdir():continue
   if not m.isfile() or m.name not in expected or m.name in seen:raise RuntimeError("unexpected_archive_member:"+m.name)
   f=tar.extractfile(m)
   with f:h,n=digest_stream(f)
   r=expected[m.name]
   if n!=r["size"] or h!=r["sha256"]:raise RuntimeError("archive_checksum_mismatch:"+m.name)
   seen.add(m.name);verified_bytes+=n
 if seen!=set(expected):raise RuntimeError("archive_member_set_mismatch")
 print("verify_source_stability",flush=True)
 for r in records:unchanged(ROOT/r["path"],r)
 again,excluded_again,missing_again,scope_again,dirs_again=inventory()
 if [(x["path"],x["size"],x["mtime_ns"],x["ctime_ns"]) for x in again]!=[(x["path"],x["size"],x["mtime_ns"],x["ctime_ns"]) for x in records]:raise RuntimeError("source_inventory_changed")
 if excluded_again!=excluded:raise RuntimeError("excluded_inventory_changed")
 if git(["show-ref","--head"]).decode().splitlines()!=git_audit["refs_before"]:raise RuntimeError("git_refs_changed_after_archive")
 audit={"status":"passed","started_at":started,"finished_at":stamp(),"elapsed_seconds":round(time.monotonic()-t0,3),"source_files":len(records),"source_bytes":summary["source_bytes"],"verified_archive_files":len(seen),"verified_archive_bytes":verified_bytes,"archive_bytes":packed.stat().st_size,"archive_sha256":hash_file(packed),"source_stat_before_after":"all unchanged (size,mtime_ns,ctime_ns,mode,device,inode)","source_hash_at_scan_equals_hash_read_into_archive":True,"archive_member_sha256_matches_manifest":True,"archive_member_set_matches_manifest":True,"excluded_inventory_unchanged":True,"git_refs_unchanged":True,"credential_scan_status":"passed","bundle_sha256":git_audit["bundle_sha256"],"source_modified":False,"uploaded":False}
 write_json("archive-audit.json",audit)
 machine={"archive_path":str(packed),"archive_size":audit["archive_bytes"],"archive_sha256":audit["archive_sha256"],"source_file_count":len(records),"source_bytes":summary["source_bytes"],"excluded_entry_count":len(excluded),"credential_scan_status":"passed","archive_audit_status":"passed","bundle_path":str(OUT/"original-history.bundle"),"bundle_size":git_audit["bundle_bytes"],"bundle_sha256":git_audit["bundle_sha256"]}
 write_json("manifest-summary.json",machine)
 outputs=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {"BACKUP-SHA256SUMS","prepare.stdout.log"})
 with (OUT/"BACKUP-SHA256SUMS").open("x",encoding="utf8") as f:
  for p in outputs:f.write(hash_file(p)+"  "+p.name+"\n")
 print(json.dumps(machine),flush=True)
 return 0
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:
  failure={"status":"failed","error_type":type(e).__name__,"reason":str(e) if isinstance(e,RuntimeError) else "operation_failed_see_type","at":stamp(),"archive_may_be_incomplete":True}
  if not (OUT/"failure-audit.json").exists():write_json("failure-audit.json",failure)
  print(json.dumps(failure),flush=True);sys.exit(1)
