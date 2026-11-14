"""A band recording -> a score (song.json) for the puppets: drum hits, bass notes, guitar strums,
beat grid and sections. Needs the stems from Demucs (drums, bass, other):

  python3 -m demucs -n htdemucs -d cpu -o <stems> song.wav
  python3 puppet/transcribe.py <song.wav> <stems/htdemucs/song> <out.json> [--credit "..."]

Drums: onsets per frequency band of the drum stem (kick: lows, snare: body + noise, hat: highs,
crash: highs that ring on). Bass: onsets + pitch (pYIN). Guitar: onsets of the 'other' stem;
chords from chroma (major/minor/power templates). Sections: loudness changes on the bar grid.
"""
import json
import sys

import librosa
import numpy as np
import scipy.signal as sig

SR = 22050
HOP = 256
GM = dict(KICK=36, SNARE=38, HAT=42, OPEN_HAT=46, CRASH=49)
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load(path):
    return librosa.load(path, sr=SR, mono=True)[0]


def band(y, lo, hi):
    if lo and hi:
        sos = sig.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    elif lo:
        sos = sig.butter(4, lo, btype="high", fs=SR, output="sos")
    else:
        sos = sig.butter(4, hi, btype="low", fs=SR, output="sos")
    return sig.sosfiltfilt(sos, y)


def env(y):
    return librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]


def onsets(y, delta=0.07, wait=0.07):
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    f = librosa.onset.onset_detect(onset_envelope=o, sr=SR, hop_length=HOP, delta=delta, wait=int(wait * SR / HOP), units="frames")
    return f, o


def vel(e, f, ref):
    return int(np.clip(50 + 77 * e[f] / ref, 40, 127))


def peaks(e, height, dist=0.09, prom=0.15):
    """Peaks of a level curve normalized to its 95th percentile -> (frames, normalized levels)."""
    n = e / (np.percentile(e, 95) + 1e-9)
    f, _ = sig.find_peaks(n, height=height, distance=max(1, int(dist * SR / HOP)), prominence=prom)
    return f, n


def drums(y):
    """-> [{t, piece, vel}] from the drum stem: peaks of the level in three bands."""
    e_lo, e_mid, e_hi = env(band(y, None, 110)), env(band(y, 160, 900)), env(band(y, 6000, 10500))
    out = []
    v = lambda x: int(np.clip(45 + 75 * x, 40, 127))  # noqa: E731
    f, n = peaks(e_lo, 0.35, 0.09, 0.2)
    out += [dict(t=k * HOP / SR, piece=GM["KICK"], vel=v(n[k])) for k in f]
    f, n = peaks(e_mid, 0.7, 0.12, 0.3)  # the loud mid peaks are the snare; small ones are bleed and ghosts
    out += [dict(t=k * HOP / SR, piece=GM["SNARE"], vel=v(n[k])) for k in f]
    f, n = peaks(e_hi, 0.3, 0.09, 0.15)
    ring = int(0.3 * SR / HOP)
    for k in f:
        later = n[min(k + ring, len(n) - 1)]
        crash = n[k] > 0.95 and later > 0.55 * n[k]
        out.append(dict(t=k * HOP / SR, piece=GM["CRASH"] if crash else GM["HAT"], vel=v(n[k])))
    out.sort(key=lambda d: d["t"])
    # one crash per ring: drop hats inside a ringing crash (they are the crash)
    clean, last_crash = [], -9
    for d in out:
        if d["piece"] == GM["CRASH"]:
            if d["t"] - last_crash < 0.5:
                continue
            last_crash = d["t"]
        elif d["piece"] == GM["HAT"] and d["t"] - last_crash < 0.25:
            continue
        clean.append(d)
    # bleed gate: no hits where the drum stem as a whole is near silent (other instruments leak
    # a little into the stem's high band)
    e_all = env(y)
    e_all = np.convolve(e_all, np.ones(int(0.5 * SR / HOP)) / int(0.5 * SR / HOP), mode="same")
    ref = np.percentile(e_all, 95)
    clean = [d for d in clean if e_all[min(len(e_all) - 1, int(d["t"] * SR / HOP))] > 0.15 * ref]
    # the envelope peak trails the attack by ~2 frames: move hits to the attack
    for d in clean:
        d["t"] = max(0.0, d["t"] - 2 * HOP / SR)
    return clean


def notes(y, fmin, fmax, delta=0.07):
    """Onsets + median pitch until the next onset -> [{t, dur, pitch, vel}]."""
    f, o = onsets(y, delta, 0.08)
    e = env(y)
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, hop_length=HOP * 2)
    ref = np.percentile(e[f], 90) if len(f) else 1
    out = []
    for i, k in enumerate(f):
        k1 = f[i + 1] if i + 1 < len(f) else len(e) - 1
        kk = min(k + 3, len(e) - 1)
        if e[kk] < 0.12 * ref:
            continue
        seg = f0[k // 2: max(k // 2 + 1, k1 // 2)]
        seg = seg[np.isfinite(seg)]
        p = int(round(float(np.median(librosa.hz_to_midi(seg))))) if len(seg) else None
        # duration: until the level falls to 30% or the next onset
        dur_f = k1 - k
        below = np.nonzero(e[kk:k1] < 0.3 * e[kk])[0]
        if len(below):
            dur_f = int(below[0]) + 3
        out.append(dict(t=k * HOP / SR, dur=round(dur_f * HOP / SR, 3), pitch=p, vel=vel(e, kk, ref)))
    return out


def chords(y, times):
    """Chord name per strum from chroma (root + quality: '', 'm' or '5')."""
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    tmpl = {}
    for r in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7)), ("5", (0, 7))):
            v = np.zeros(12)
            v[[(r + i) % 12 for i in iv]] = 1
            tmpl[NOTES[r] + q] = v / np.linalg.norm(v)
    out = []
    for i, t in enumerate(times):
        a = int(t * SR / HOP)
        b = int(times[i + 1] * SR / HOP) if i + 1 < len(times) else a + 40
        c = C[:, a:max(a + 2, b)].mean(1)
        c = c / (np.linalg.norm(c) + 1e-9)
        name = max(tmpl, key=lambda n: tmpl[n] @ c - (0.04 if n.endswith("5") else 0))
        out.append(name)
    return out


def sections(y, bar, duration, t0=0.0):
    """Bars grouped by loudness: quiet / full / stop (near silence between hits)."""
    e = env(y)
    per = []
    t = t0
    while t < duration - 1e-6:
        a, b = int(t * SR / HOP), int(min(duration, t + bar) * SR / HOP)
        per.append((t, min(duration, t + bar), float(np.median(e[a:b])) if b > a else 0.0))
        t += bar
    full = np.percentile([p[2] for p in per], 75)
    lab = ["full" if p[2] > 0.6 * full else ("stop" if p[2] < 0.15 * full else "quiet") for p in per]
    out = []
    for (a, b, _), l in zip(per, lab):
        if out and out[-1]["name"] == l:
            out[-1]["t1"] = b
        else:
            out.append(dict(name=l, t0=a, t1=b))
    return out


def transcribe(mix, stems, credit=""):
    y = load(mix)
    duration = len(y) / SR
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    yd, yb, yo = load(f"{stems}/drums.wav"), load(f"{stems}/bass.wav"), load(f"{stems}/other.wav")
    dr = drums(yd)
    bass = notes(yb, 30, 330)
    gt = notes(yo, 70, 1000, 0.09)
    names = chords(yo + 0.5 * yb, [g["t"] for g in gt])
    # strum direction: down on the beat, up between beats
    spb = 60 / bpm
    guitar = []
    for g, n in zip(gt, names):
        ph = ((g["t"] - (bt[0] if len(bt) else 0)) / spb) % 1
        down = ph < 0.25 or ph > 0.75
        root = librosa.note_to_midi(n.rstrip("m5") + "2")
        guitar.append(dict(t=round(g["t"], 3), dur=g["dur"], root=int(root), chord=n, dir="down" if down else "up", accent=g["vel"] > 100))
    return dict(bpm=round(bpm, 2), duration=round(duration, 3), beats=[round(float(b), 3) for b in bt],
                sections=sections(y, 4 * spb, duration, float(bt[0]) if len(bt) else 0.0), credit=credit,
                drums=[dict(beat=None, piece=d["piece"], vel=d["vel"], t=round(d["t"], 3)) for d in dr],
                bass=[dict(t=round(b["t"], 3), dur=b["dur"], pitch=b["pitch"], vel=b["vel"]) for b in bass],
                vocal=[], guitar=guitar)


if __name__ == "__main__":
    a = sys.argv[1:]
    credit = ""
    if "--credit" in a:
        i = a.index("--credit")
        credit = a[i + 1]
        a = a[:i] + a[i + 2:]
    s = transcribe(a[0], a[1], credit)
    json.dump(s, open(a[2], "w"), indent=0)
    from collections import Counter
    print("bpm", s["bpm"], "dur", s["duration"], "drums", Counter(d["piece"] for d in s["drums"]), "bass", len(s["bass"]), "guitar", len(s["guitar"]))
    print("sections", [(x["name"], round(x["t0"], 1), round(x["t1"], 1)) for x in s["sections"]])
