"""Style reel: the same 10 s of the band (song 18-28 s: end of the verse, the chorus, a pause, the
next line) in several styles, one after the other, labeled. For choosing a direction.

  python3 clips/stylereel/clip.py one 3      -> out/reel-3.mp4 (one style)
  python3 clips/stylereel/clip.py all [2]    -> every style of round 1 (or 2), 2 at a time, then the reel
  python3 clips/stylereel/clip.py reel       -> out/stylereel.mp4 from the rendered styles
  python3 clips/stylereel/clip.py stills 3 20 24
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/puppet")
from cast.coriander import Guitarist  # noqa: E402
from cast.distor import Singer  # noqa: E402
from cast.pepper import Drummer  # noqa: E402
from production import W, HandDrawn, Looks, Production, TVStatic  # noqa: E402
from song import Song  # noqa: E402

song = Song.default()
T0, T1 = 18.0, 28.0

# shots like clip 1 (a cut per bar) ...
BAR_CUTS = [(18, 20, "singer_full"), (20, 22, "WIDE"), (22, 24, "guitarist"), (24, 26, "drummer"), (26, 28, "singer")]
# ... or Clint Eastwood cutting: the same moment from 2-3 angles, cuts on beats (1 beat = 0.5 s)
BEAT_CUTS = [(18, 19, "singer"), (19, 19.5, "singer_close"), (19.5, 20, "drummer"), (20, 21, "WIDE"), (21, 21.5, "singer_close"),
             (21.5, 22, "singer"), (22, 23, "guitarist"), (23, 23.5, "drummer_close"), (23.5, 24, "drummer"), (24, 25, "WIDE"),
             (25, 25.5, "guitarist"), (25.5, 26, "singer_full"), (26, 27, "singer_close"), (27, 28, "singer")]

STYLES = {
    1: dict(label="1  clip 1 as it was", fluid=False, cuts=BAR_CUTS),
    2: dict(label="2  fluid motion + subtle hand-drawn finish", fluid=True, look="hd", cuts=BAR_CUTS),
    3: dict(label="3  2 + Clint Eastwood camera (drift, depth, calm heads)", fluid=True, look="hd", ce=True, cuts=BAR_CUTS),
    4: dict(label="4  3 + Clint Eastwood cutting (beats, several angles)", fluid=True, look="hd", ce=True, cuts=BEAT_CUTS),
    5: dict(label="5  4 in the white Clint Eastwood opening look", fluid=True, look="hd", ce=True, cuts=BEAT_CUTS, stage="white"),
    6: dict(label="6  4 at night, light from the ground, TV static", fluid=True, look="static", ce=True, cuts=BEAT_CUTS, stage="night"),
    7: dict(label="7  4 on twos with blended in-betweens", fluid=True, look="hd", ce=True, cuts=BEAT_CUTS, twos=True, ghost=0.5),
    # round 2 (all with faces for the drummer and guitarist)
    11: dict(label="A  6 + faces (drummer, guitarist)", fluid=True, look="static", ce=True, cuts=BEAT_CUTS, stage="night", face=True),
    12: dict(label="B  A + bolder camera (punch-ins on crashes, sway)", fluid=True, look="static", ce=True, cuts=BEAT_CUTS, stage="night",
             face=True, punch=0.07, roll=1.2, drift=0.08),
    13: dict(label="C  A with the colored stage lights", fluid=True, look="static", ce=True, cuts=BEAT_CUTS, stage="stage", face=True),
    14: dict(label="D  A + worn-tape TV (color bleed, scan lines)", fluid=True, look="vhs", ce=True, cuts=BEAT_CUTS, stage="night", face=True),
    15: dict(label="E  A with calmer cutting (one cut per bar)", fluid=True, look="static", ce=True, cuts=BAR_CUTS, stage="night", face=True),
}
ROUND = {1: [1, 2, 3, 4, 5, 6, 7], 2: [11, 12, 13, 14, 15]}


def production(k):
    st = STYLES[k]
    fl, ce = st["fluid"], st.get("ce", False)
    hd = 0.6 if ce else 1.0
    fc = st.get("face", False)
    sing = dict(fluid=fl, head=hd, mouth_twos=ce) if fl else {}
    cast = {
        "singer": lambda: Singer(song, **sing),
        "singer_close": lambda: Singer(song, scale=0.62, top=-10, **sing),
        "singer_full": lambda: Singer(song, scale=0.245, top=40, **sing),
        "drummer": lambda: Drummer(song, head=hd, face=fc),
        "drummer_close": lambda: Drummer(song, scale=0.85, top=-60, head=hd, face=fc),
        "guitarist": lambda: Guitarist(song, head=hd, face=fc),
        "w_drummer": lambda: Drummer(song, scale=0.40, top=150, head=hd, face=fc),
        "w_singer": lambda: Singer(song, scale=0.28, cx=W * 0.24, top=470, **sing),
        "w_guitarist": lambda: Guitarist(song, scale=0.42, cx=W * 0.76, top=int(330 - 120 * 0.42), head=hd, face=fc),
    }
    wide = ("w_drummer", "w_singer", "w_guitarist")
    shots = [(0, T0, "singer")] + [(a, b, wide if w == "WIDE" else w) for a, b, w in st["cuts"]] + [(T1, song.duration + 1, "singer")]
    look = {None: None, "hd": HandDrawn(), "static": Looks(HandDrawn(), TVStatic(0.12)),
            "vhs": Looks(HandDrawn(), TVStatic(0.14, vhs=0.8))}[st.get("look")]
    if st.get("stage") == "white":
        look = HandDrawn(paper=0.0)  # paper grain reads as dirt on white
    kw = dict(drift=st.get("drift", 0.05), parallax=0.55, ghost=st.get("ghost", 0.35)) if ce else {}
    if st.get("punch"):
        import numpy as np
        kw["punch"] = (np.array([d["t"] for d in song.data["drums"] if d["piece"] == 49]), st["punch"])
        kw["roll"] = st.get("roll", 0.0)
    return Production(song, cast, shots, name=f"reel-{k}", script=os.path.abspath(__file__), look=look, twos=st.get("twos", False),
                      stage=st.get("stage", "stage"), crowd_wide=st.get("stage", "stage") == "stage", **kw)


def one(k):
    return production(k).render_range(T0, T1, f"{ROOT}/out/reel-{k}.mp4")


def reel(rnd=1):
    ks = ROUND[rnd]
    ins = sum([["-i", f"{ROOT}/out/reel-{k}.mp4"] for k in ks], [])
    lab = lambda k: STYLES[k]["label"].replace(":", "\\\\:")  # noqa: E731
    flt = "".join(f"[{i}:v]drawtext=text='{lab(k)}':x=30:y=30:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v{i}];" for i, k in enumerate(ks))
    flt += "".join(f"[v{i}][{i}:a]" for i in range(len(ks))) + f"concat=n={len(ks)}:v=1:a=1[v][a]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + ins + ["-filter_complex", flt, "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                    "-crf", "25", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", f"{ROOT}/out/stylereel{'' if rnd == 1 else rnd}.mp4"], check=True)
    return f"{ROOT}/out/stylereel{'' if rnd == 1 else rnd}.mp4"


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "one":
        print(one(int(sys.argv[2])))
    elif what == "stills":
        production(int(sys.argv[2])).stills([float(a) for a in sys.argv[3:]], f"{ROOT}/out/reel-{sys.argv[2]}-still")
    elif what == "reel":
        print(reel(int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    elif what == "all":  # all [round]
        rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        ks = ROUND[rnd]
        for i in range(0, len(ks), 2):
            ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "one", str(k)]) for k in ks[i:i + 2]]
            for p in ps:
                p.wait()
        print(reel(rnd))
