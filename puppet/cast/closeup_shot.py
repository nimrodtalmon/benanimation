"""A close-up shot as a cast member: the venue's picture behind, thrown far out of focus, and the
3D close-up (puppet/closeup3d.py) of a player's hands on their instrument on top."""
import cv2
import numpy as np

import rig
from closeup3d import CloseUp, closeup_job
from production import H, W


class CloseUpShot:
    def __init__(self, song, rig_name, kind, t0, t1, light="stage", res=(1024, 576), samples=10, model=None):
        self.j = rig.load(rig_name)
        job = closeup_job(song, rig_name, kind, t0, t1, light=light, res=res, samples=samples, model=model)
        self.cu = CloseUp(job) if job else None

    def draw(self, frame, t):
        small = cv2.resize(frame, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        frame[:] = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H), interpolation=cv2.INTER_LINEAR) * 0.75
        if self.cu is None:
            return
        im = self.cu.at(t)
        if im.dtype != np.float32 or im.max() > 1.5:
            im = im.astype(np.float32) / (65535 if im.max() > 255 else 255)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
        a = np.clip(im[..., 3:4], 0, 1)
        frame *= 1 - a
        frame += im[..., :3] * a
