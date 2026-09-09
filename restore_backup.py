"""Restore the byte-for-byte CAFD working-record backup into a new directory."""
import argparse, hashlib, json, os, shutil, tarfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--destination",required=True,type=Path)
    ap.add_argument("--manifest",type=Path,default=Path(__file__).resolve().parent/"backup/2026-09-08/parts.json")
    args=ap.parse_args()
    manifest=args.manifest.resolve()
    data=json.loads(manifest.read_text(encoding="utf-8"))
    dst=args.destination.resolve()
    if dst.exists() and any(dst.iterdir()): raise SystemExit("Destination must be new or empty.")
    dst.mkdir(parents=True,exist_ok=True)
    archive=dst/".cafd-working-records.tar.gz"
    try:
        total=0; h=hashlib.sha256()
        with archive.open("xb") as out:
            for part in data["parts"]:
                p=(manifest.parent/part["name"]).resolve()
                if not p.is_relative_to(manifest.parent): raise ValueError("Unsafe part path")
                if p.stat().st_size!=part["size_bytes"] or digest(p)!=part["sha256"]:
                    raise ValueError("Part checksum mismatch: "+part["name"])
                with p.open("rb") as f:
                    for b in iter(lambda:f.read(1024*1024),b""):
                        out.write(b);h.update(b);total+=len(b)
        if total!=data["archive_size_bytes"] or h.hexdigest()!=data["archive_sha256"]:
            raise ValueError("Archive checksum mismatch")
        with tarfile.open(archive,"r:gz") as tar:
            members=tar.getmembers()
            for member in members:
                target=(dst/member.name).resolve()
                if not target.is_relative_to(dst) or not (member.isfile() or member.isdir()):
                    raise ValueError("Unsafe archive member: "+member.name)
            for member in members:
                target=dst/member.name
                if member.isdir(): target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open("xb") as out: shutil.copyfileobj(src,out)
                os.chmod(target,member.mode & 0o777)
                os.utime(target,(member.mtime,member.mtime))
        archive.unlink()
        print("Restored and verified:",dst)
    except Exception:
        print("Restore failed; partial files remain only in the chosen destination.")
        raise
if __name__=="__main__": main()
