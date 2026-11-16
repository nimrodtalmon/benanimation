"""A short pop-rock song, written as a score.

The score is the single source of truth: it is rendered to audio (MIDI ->
FluidSynth) and exported as JSON for the animation (every hit, strum and sung
note, in seconds).

  python music/score.py            # -> music/song.mid, public/song/song.wav, src/song.json
"""
import json
import subprocess

import mido

BPM = 120
SPB = 60 / BPM  # seconds per beat
TPB = 480  # MIDI ticks per beat

# Sections in bars (4 beats each)
SECTIONS = [("intro", 2), ("verse", 8), ("chorus", 8), ("outro", 2)]

# Chords per bar: name -> (guitar power chord root, bass root)
CHORDS = {"Em": (40, 28), "C": (48, 36), "G": (43, 31), "D": (50, 38)}
VERSE = ["Em", "C", "G", "D"] * 2
CHORUS = ["C", "G", "D", "Em"] * 2

KICK, SNARE, HAT, OPEN_HAT, CRASH, TOM_HI, TOM_MID, TOM_LO = 36, 38, 42, 46, 49, 50, 47, 45

# Vocal melody: (beat within section, length in beats, MIDI pitch)
E4, G4, A4, B4, D4, C5, D5, E5 = 64, 67, 69, 71, 62, 72, 74, 76
PHRASE_A = [(1, .5, E4), (1.5, .5, G4), (4, 1, G4), (5, 1, A4), (6, 1, G4), (7, 1, E4), (8, 1, E4), (9, .5, D4), (9.5, 1.5, E4)]
PHRASE_B = [(12, .5, D4), (12.5, .5, E4), (13, 1, G4), (14, 1.5, B4), (15.5, .5, A4), (16, 1, G4), (17, 1, E4), (18, 2, D4)]
PHRASE_B2 = [(12, .5, D4), (12.5, .5, E4), (13, 1, G4), (14, 1.5, B4), (15.5, .5, A4), (16, 1, G4), (17, 1, A4), (18, 2.5, E4)]
VERSE_MELODY = PHRASE_A + PHRASE_B + [(b + 16, l, p) for b, l, p in PHRASE_A] + [(b + 16, l, p) for b, l, p in PHRASE_B2]
CHORUS_HALF = [(0, .5, G4), (.5, .5, A4), (1, 1, B4), (2, 1, B4), (3, .5, A4), (3.5, .5, G4), (4, 1, B4), (5, 2, D5),
               (8, .5, D5), (8.5, .5, C5), (9, 1, B4), (10, 1, A4), (11, .5, G4), (11.5, .5, A4), (12, 3, B4)]
CHORUS_MELODY = CHORUS_HALF + [(b + 16, l, p) for b, l, p in CHORUS_HALF[:-1]] + [(28, 1, D5), (29, 1, E5), (30, 2, E5)]
VOWELS = ["oh", "ah", "oo", "ah", "ee", "oh", "ah"]


def build():
    ev = {"drums": [], "guitar": [], "bass": [], "vocal": [], "sections": [], "bpm": BPM}
    beat = 0.0
    for name, bars in SECTIONS:
        ev["sections"].append({"name": name, "t0": beat * SPB, "t1": (beat + bars * 4) * SPB})
        for bar in range(bars):
            b0 = beat + bar * 4
            last = bar == bars - 1
            chord = {"intro": "Em", "verse": VERSE[bar % 8], "chorus": CHORUS[bar % 8], "outro": "Em"}[name]
            g, bs = CHORDS[chord]

            # drums
            def d(at, piece, vel=100):
                ev["drums"].append({"beat": b0 + at, "piece": piece, "vel": vel})

            if name == "outro" and bar == 1:
                d(0, CRASH, 120); d(0, KICK, 120)
            elif name == "outro":
                for i in range(8):
                    d(i * .5, [TOM_HI, TOM_HI, TOM_MID, TOM_MID, TOM_LO, TOM_LO, SNARE, SNARE][i], 95 + i * 3)
            else:
                if bar == 0 and name != "intro":
                    d(0, CRASH, 115)
                if name == "chorus" and bar % 2 == 0 and bar:
                    d(0, CRASH, 105)
                kicks = [0, 1.5, 2] if name != "chorus" else [0, .5, 2, 2.5]
                for k in kicks:
                    d(k, KICK, 110)
                fill = last and name in ("verse", "intro")
                for s in (1, 3):
                    if not (fill and s == 3):
                        d(s, SNARE, 112)
                for i in range(8 if not fill else 6):
                    d(i * .5, OPEN_HAT if name == "chorus" and i % 2 == 0 else HAT, 70 + (12 if i % 2 == 0 else 0))
                if fill:
                    for i, p in enumerate([SNARE, SNARE, TOM_HI, TOM_MID]):
                        d(3 + i * .25, p, 100 + i * 5)

            # guitar: 8th-note power chords (palm-muted in the verse), bass: 8ths on the root
            if name == "outro":
                if bar == 1:
                    ev["guitar"].append({"beat": b0, "len": 4, "root": g, "chord": chord, "dir": "down", "accent": True})
                    ev["bass"].append({"beat": b0, "len": 4, "note": bs})
            else:
                for i in range(8):
                    muted = name in ("verse", "intro")
                    ev["guitar"].append({"beat": b0 + i * .5, "len": .22 if muted else .48, "root": g, "chord": chord,
                                         "dir": "down" if i % 2 == 0 else "up", "accent": i == 0})
                    ev["bass"].append({"beat": b0 + i * .5, "len": .45, "note": bs + (12 if i == 7 and name == "chorus" else 0)})
        melody = {"verse": VERSE_MELODY, "chorus": CHORUS_MELODY}.get(name, [])
        for i, (b, l, p) in enumerate(melody):
            ev["vocal"].append({"beat": beat + b, "len": l, "pitch": p, "vowel": VOWELS[i % len(VOWELS)]})
        beat += bars * 4
    ev["beats_total"] = beat
    return ev


def to_midi(ev, path):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    msgs = []  # (tick, order, msg)

    def note(ch, beat, length, pitch, vel):
        t0 = int(beat * TPB)
        t1 = int((beat + length) * TPB)
        msgs.append((t0, 1, mido.Message("note_on", channel=ch, note=pitch, velocity=vel)))
        msgs.append((t1, 0, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))

    for e in ev["drums"]:
        note(9, e["beat"], .25, e["piece"], e["vel"])
    for e in ev["guitar"]:
        r = e["root"]
        vel = 112 if e["accent"] else (96 if e["dir"] == "down" else 84)
        for p in (r, r + 7, r + 12):
            note(1, e["beat"] + (0.01 if p != r else 0), e["len"], p, vel)
    for e in ev["bass"]:
        note(2, e["beat"], e["len"], e["note"], 105)
    for e in ev["vocal"]:
        note(3, e["beat"], e["len"] * .95, e["pitch"], 118)
        note(4, e["beat"], e["len"] * .95, e["pitch"], 70)

    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM)))
    for ch, prog in [(1, 30), (2, 33), (3, 52), (4, 53)]:  # distortion gtr, finger bass, voice oohs, synth voice
        track.append(mido.Message("program_change", channel=ch, program=prog))
    track.append(mido.Message("control_change", channel=3, control=91, value=70))  # reverb on vocals
    track.append(mido.Message("control_change", channel=4, control=91, value=70))
    track.append(mido.Message("control_change", channel=1, control=10, value=44))  # guitar a bit left
    for ch, vol in [(1, 78), (2, 96), (3, 127), (4, 110), (9, 118)]:  # mix: keep the voice on top
        track.append(mido.Message("control_change", channel=ch, control=7, value=vol))
    now = 0
    for t, _, m in sorted(msgs, key=lambda x: (x[0], x[1])):
        track.append(m.copy(time=t - now))
        now = t
    mid.save(path)


def main():
    ev = build()
    for k in ("drums", "guitar", "bass", "vocal"):
        for e in ev[k]:
            e["t"] = round(e["beat"] * SPB, 4)
            if "len" in e:
                e["dur"] = round(e["len"] * SPB, 4)
    ev["duration"] = ev["beats_total"] * SPB + 3  # let the last chord ring
    to_midi(ev, "music/song.mid")
    subprocess.run(["fluidsynth", "-ni", "-g", "0.9", "-r", "44100", "-F", "/tmp/song_raw.wav",
                    "/usr/share/sounds/sf2/FluidR3_GM.sf2", "music/song.mid"], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "/tmp/song_raw.wav", "-t", str(ev["duration"]),
                    "-af", "loudnorm=I=-14:TP=-1.5", "-ar", "44100", "public/song/song.wav"], check=True)
    json.dump(ev, open("src/song.json", "w"))
    print(f"{ev['duration']:.1f}s, {len(ev['drums'])} drum hits, {len(ev['guitar'])} strums, {len(ev['vocal'])} sung notes")


if __name__ == "__main__":
    main()
