"""Point every builder's newest finished render at its current cache key (after changing how
caches are keyed, or bumping a BUILD number on purpose to keep old renders). Unfinished caches are
removed. Run from the repo: python3 tools/migrate_caches.py"""
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/puppet")
import city3d  # noqa: E402
import closeup3d  # noqa: E402
import fire3d  # noqa: E402
import guitar3d  # noqa: E402
import kit3d  # noqa: E402
import scenery3d  # noqa: E402


def done(d):
    return os.path.exists(f"{d}/meta.json")


def move_newest(target):
    if done(target):
        return
    base = target.rsplit("-", 1)[0]
    cands = [c for c in glob.glob(base + "-*") if done(c) and c != target]
    if cands:
        src = max(cands, key=os.path.getmtime)
        shutil.rmtree(target, ignore_errors=True)
        os.rename(src, target)
        print("->", os.path.relpath(target, ROOT))


for style in (None, "ben"):
    if style:
        os.environ["BENANIM_STYLE"] = style
    else:
        os.environ.pop("BENANIM_STYLE", None)
    for mod, kind in ((kit3d, "kits"), (guitar3d, "instruments"), (fire3d, "fires"), (scenery3d, "scenery"), (city3d, "scenery")):
        for f in glob.glob(f"{ROOT}/public/{kind}/*.json"):
            name = os.path.basename(f)[:-5]
            if name.startswith("_"):
                continue
            try:
                move_newest(mod.cache_dir(name))
            except Exception:  # noqa: BLE001  (a spec of another builder in the same folder)
                pass
for d in glob.glob(f"{ROOT}/out/closeupcache/*"):
    if not done(d):
        shutil.rmtree(d, ignore_errors=True)
        continue
    job = json.load(open(f"{d}/job.json"))
    new = closeup3d.job_dir(job)
    if new != d and not os.path.exists(new):
        os.rename(d, new)
print("ok")
