"""Distor on drums: cast/kit_player.py with Distor's rig (kept for older clips)."""
from cast.kit_player import KitPlayer


def DistorDrummer(song, *a, **kw):
    return KitPlayer(song, "distor", *a, **kw)
