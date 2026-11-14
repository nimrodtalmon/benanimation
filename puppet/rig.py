"""Loading a rigged character from public/parts/<name>/parts.json, the same way for every cast
class: the puppet config (head landmarks over the body drawing), the two arms, the hands' style.

  j = rig.load(name)
  cfg = rig.puppet_cfg(name, j, scale, crop_h_src)     # for warp.Puppet
  arm_l, arm_r = rig.arms(name, j)                      # viewer's left / right Limbs
  style = rig.hand_style(j)                             # Ben's flat marker look for code hands

Head landmarks come from parts.json "head" (characters built by build_bust.py) or, for older
characters, from warp.CHARS[name]; either way they are shifted by the drawing's side padding.
"""
import json
import os

import numpy as np

import hand as handrig
import warp
from limb import Limb, join_parts
from warp import load_rgba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parts_dir(name):
    return f"{ROOT}/public/parts/{name}"


def load(name):
    return json.load(open(f"{parts_dir(name)}/parts.json"))


def puppet_cfg(name, j, scale, crop_h_src, body_file=None, src_h=None):
    """warp.Puppet config: the body drawing with its head landmarks (shifted by the padding)."""
    pad = j.get("pad", 0)
    hd = j.get("head") or warp.CHARS[name]
    sx = lambda v: v + pad  # noqa: E731
    body_file = body_file or ("body.png" if os.path.exists(f"{parts_dir(name)}/body.png") else "body_nolegs.png")
    src_h = src_h or (hd.get("src_h") if "src_h" in hd else 1000)
    cfg = dict(hd)
    cfg.update(full=f"public/parts/{name}/{body_file}", src_h=src_h, height=src_h * scale, crop_h_src=crop_h_src,
               head_ells=[(sx(x), y, rx, ry) for x, y, rx, ry in hd["head_ells"]],
               neck=(sx(hd["neck"][0]), sx(hd["neck"][1]), hd["neck"][2], hd["neck"][3]),
               face=(sx(hd["face"][0]),) + tuple(hd["face"][1:]), pivot=(sx(hd["pivot"][0]), hd["pivot"][1]),
               mouth=(sx(hd["mouth"][0]),) + tuple(hd["mouth"][1:]),
               eyes=[(sx(x), y, rx, ry) for x, y, rx, ry in hd["eyes"]],
               irises=[(sx(x), y, r) for x, y, r in hd.get("irises", [])],
               eye_hide=[(sx(x), y, rx, ry) for x, y, rx, ry in hd.get("eye_hide", [])])
    if "lid_color" in hd:
        cfg["lid_color"] = (sx(hd["lid_color"][0]), hd["lid_color"][1])
    cfg.setdefault("lid_from", 150)  # defaults, for rigs that leave them out
    cfg.setdefault("lash", (20, 18, 19))
    cfg.setdefault("lash_w", 12)
    return cfg


def arms(name, j, **kw):
    """The character's two arms (Limbs): its upper arm + forearm drawings, mirrored for the other side."""
    parts = parts_dir(name)
    up = load_rgba(f"{parts}/{j['upper_arm']['file']}")
    fo = load_rgba(f"{parts}/{j['forearm']['file']}")
    u, f, jt = join_parts(up, fo, j["upper_arm"], j["forearm"])
    arm_l = Limb(u, f, jt, j["arm_width"], inner=-1, **kw)  # viewer's left arm, bends toward the body
    uf, ff = u[:, ::-1].copy(), f[:, ::-1].copy()  # mirrored copy for the other arm
    wj = u.shape[1] - 1
    jm = {k: [wj - v[0], v[1]] for k, v in jt.items()}
    return arm_l, Limb(uf, ff, jm, j["arm_width"], inner=1, **kw)


def hand_style(j):
    hs = j["hand"]
    return handrig.Style(skin=np.array(hs["skin_bgr"]) / 255, ink=(0.075, 0.07, 0.08), ink_w=0.03, shade=0.0, flat=True)
