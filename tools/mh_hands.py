"""Cut the hands (with a bit of forearm) out of MakeHuman's base mesh (CC0, makehumancommunity.org),
with their bone joints and skin weights, into public/hands/mh_<side>.npz for puppet/closeup3d.py.

  python3 tools/mh_hands.py /path/to/makehuman/data   (base.obj, default.mhskel, default_weights.mhw)

Saved in the hand's own frame (meters): x = wrist -> middle knuckle, y = toward the index side,
z = the back of the hand; scaled so wrist -> middle knuckle = 9.4 cm. Bones (columns of the
weights): palm, index 1-3, middle 1-3, ring 1-3, pinky 1-3, thumb 1-3.
"""
import json
import os
import sys

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/mh"
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "hands")

V, VT, F, FT, grp = [], [], [], [], None
for line in open(f"{src}/base.obj"):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("vt "):
        VT.append([float(x) for x in line.split()[1:3]])
    elif line.startswith("g "):
        grp = line.split()[1]
    elif line.startswith("f ") and grp == "body":
        idx = [p.split("/") for p in line.split()[1:]]
        F.append([int(p[0]) - 1 for p in idx])
        FT.append([int(p[1]) - 1 for p in idx])
V, VT = np.array(V), np.array(VT)
skel = json.load(open(f"{src}/default.mhskel"))
W = json.load(open(f"{src}/default_weights.mhw"))["weights"]


def joint(name):
    return V[skel["joints"][name]].mean(0)


def head(b):
    return joint(skel["bones"][b]["head"])


def tail(b):
    return joint(skel["bones"][b]["tail"])


for side, S in (("right", "R"), ("left", "L")):
    fingers = {"index": 2, "middle": 3, "ring": 4, "pinky": 5, "thumb": 1}
    bones = [["wrist", "metacarpal1", "metacarpal2", "metacarpal3", "metacarpal4", "lowerarm02", "lowerarm01"]]
    for nm in ("index", "middle", "ring", "pinky", "thumb"):
        for j in (1, 2, 3):
            bones.append([f"finger{fingers[nm]}-{j}"])
    w = np.zeros((len(V), len(bones)))
    for bi, names in enumerate(bones):
        for b in names:
            for vi, wt in W.get(f"{b}.{S}", []):
                w[vi, bi] += wt
    hand_w = w[:, 1:].sum(1) + w[:, 0] - sum(np.array([wt for vi, wt in W.get(f"lowerarm01.{S}", [])]).sum() * 0 for _ in [0])
    keep_v = w.sum(1) > 0.35
    # cut the forearm about 9 cm (base-mesh units are dm) behind the wrist
    wr = head(f"wrist.{S}")
    mid = head(f"finger3-1.{S}")
    f = (mid - wr) / np.linalg.norm(mid - wr)
    keep_v &= ((V - wr) @ f) > -0.9
    faces = [i for i, fc in enumerate(F) if all(keep_v[v] for v in fc)]
    used = sorted({v for i in faces for v in F[i]})
    remap = {v: k for k, v in enumerate(used)}
    # the hand's frame
    idx_k, pin_k = head(f"finger2-1.{S}"), head(f"finger5-1.{S}")
    lat = idx_k - pin_k
    lat -= (lat @ f) * f
    lat /= np.linalg.norm(lat)
    dz = np.cross(f, lat) if side == "right" else -np.cross(f, lat)
    # back of the hand: the side the knuckles' tops are on (check against the mesh normal sense)
    R = np.stack([f, lat, dz])  # rows
    scale = 0.094 / np.linalg.norm(mid - wr)
    loc = lambda p: (R @ (np.asarray(p) - wr).T).T * scale  # noqa: E731
    verts = loc(V[used])
    if (verts[:, 2] > 0).mean() < 0.3:  # sanity: most of the back of the hand should be at z > 0 or near 0
        print(side, "warning: dorsal axis looks flipped")
    fc = [[remap[v] for v in F[i]] for i in faces]
    uv_idx = [FT[i] for i in faces]
    chains = {}
    for nm, k in fingers.items():
        pts = [head(f"finger{k}-1.{S}"), head(f"finger{k}-2.{S}"), head(f"finger{k}-3.{S}"), tail(f"finger{k}-3.{S}")]
        chains[nm] = loc(pts)
    ww = w[used]
    ww /= np.maximum(ww.sum(1, keepdims=True), 1e-9)
    tri = [q for q in fc]
    np.savez_compressed(f"{out_dir}/mh_{side}.npz", verts=verts.astype(np.float32), weights=ww.astype(np.float32),
                        faces=np.array([q + ([q[-1]] if len(q) == 3 else []) for q in fc], np.int32),
                        uv=np.array([[VT[t] for t in (u + ([u[-1]] if len(u) == 3 else []))] for u in uv_idx], np.float32),
                        **{f"chain_{k}": v.astype(np.float32) for k, v in chains.items()})
    print(side, len(used), "verts", len(fc), "faces", {k: np.round(v, 3).tolist() for k, v in chains.items() if k in ("index", "thumb")})
open(f"{out_dir}/README.md", "w").write("Hands cut from the MakeHuman base mesh and default rig (CC0 1.0, makehumancommunity.org) by tools/mh_hands.py.\n")
