# Placeholder voices for the opening (first-person cut): shapes the raw SAPI lines into three voices, places them on the timeline and mixes them
# with the sound sketch. Run with the system Python (numpy).   python mix_voices.py
import glob
import os
import subprocess
import wave

import numpy as np

FF = glob.glob(r"C:\Users\mike\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe")[0]
RAW = r"C:\ai\unedited\audio\voices\raw"
OUT = r"C:\ai\unedited\audio\voices"
SR = 48000
DUR = 17.6
T_ON = 8.6            # vision comes on: before this he hears the world through the rock


def run(args):
    subprocess.run([FF, "-y", "-v", "error"] + args, check=True)


def shape(src, dst, chain):
    """raw SAPI line -> trimmed, processed, 48 kHz stereo wav"""
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=-1:stop_duration=0.35:stop_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse"
    run(["-i", src, "-af", trim + "," + chain, "-ar", str(SR), "-ac", "2", dst])


def load(path):
    with wave.open(path, "rb") as w:
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return a.reshape(-1, ch)


def save(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


# --- the voices
MENDER = ("rubberband=pitch=1.17,{tempo}highpass=f=140,equalizer=f=1700:t=q:w=1:g=3,lowpass=f=3200,"
          "acompressor=threshold=0.1:ratio=3:attack=5:release=60,aecho=0.7:0.6:45:0.18,volume=3.6")
RELIC = ("rubberband=pitch=0.78:formant=preserved,{tempo}highpass=f=70,equalizer=f=150:t=q:w=1:g=5,equalizer=f=2500:t=q:w=1.5:g=-3,"
         "acrusher=bits=11:mode=log:aa=1:mix=0.35,asoftclip=type=tanh,lowpass=f=3600,acompressor=threshold=0.1:ratio=3,volume=2.2")
TALLY = ("rubberband=pitch=1.03,{tempo}highpass=f=170,equalizer=f=3200:t=q:w=1:g=2.5,"
         "chorus=0.7:0.9:48:0.4:0.22:1.5,aecho=0.8:0.88:5:0.28,acompressor=threshold=0.12:ratio=2.5,volume=2.0")

lines = [  # id, voice chain, tempo, start seconds, cut at (seconds, None = whole line)
    ("M1", MENDER, 1.00, 2.00, None),
    ("M3", MENDER, 1.15, 3.95, None),
    ("M5", MENDER, 1.30, 6.55, 8.75),
    ("R1", RELIC, 1.00, 9.05, None),
    ("T1", TALLY, 1.15, 11.10, None),
]
buses = {"M": np.zeros((int(DUR * SR), 2)), "R": np.zeros((int(DUR * SR), 2)), "T": np.zeros((int(DUR * SR), 2))}
for lid, chain, tempo, t0, cut in lines:
    tp = "atempo=%.3f," % tempo if abs(tempo - 1.0) > 1e-3 else ""
    dst = os.path.join(OUT, lid + "_s.wav")
    shape(os.path.join(RAW, lid + ".wav"), dst, chain.format(tempo=tp))
    x = load(dst)
    if cut is not None:
        n = int((cut - t0) * SR)
        x = x[:n].copy()
        fade = int(0.04 * SR)
        x[-fade:] *= np.linspace(1, 0, fade)[:, None]
    i0 = int(t0 * SR)
    buses[lid[0]][i0:i0 + len(x)] += x[: len(buses[lid[0]]) - i0]
    print(lid, "start %.2f  length %.2f s" % (t0, len(x) / SR))
for k, name in (("M", "mender"), ("R", "relic"), ("T", "tally")):
    save(os.path.join(OUT, "stem_%s.wav" % name), buses[k])

# --- the sound sketch (re-made for 17.6 s)
ns = {}
exec(open(r"C:\ai\unedited\ep1_opening_audio5.py", encoding="utf-8").read().replace("DUR = 15.6", "DUR = %.1f" % DUR), ns)
print(ns["make_audio"](os.path.join(OUT, "sfx_v04.wav"))["seconds"], "s of sfx")

# --- mix: everything outside is muffled until his vision comes on; his own voice and Tally's are not
fc = ("[0:a]lowpass=f=450:enable='lt(t,%(t)s)',volume=0.8[s];"
      "[1:a]lowpass=f=520:enable='lt(t,%(t)s)'[m];[m][2:a][3:a]amix=inputs=3:normalize=0[vo];[vo]asplit=2[vo1][vo2];"
      "[s][vo1]sidechaincompress=threshold=0.015:ratio=6:attack=15:release=400:makeup=1[sd];"
      "[sd][vo2]amix=inputs=2:normalize=0,alimiter=limit=0.93,afade=t=out:st=%(f)s:d=0.8[o]") % {"t": T_ON, "f": DUR - 0.8}
run(["-i", os.path.join(OUT, "sfx_v04.wav"), "-i", os.path.join(OUT, "stem_mender.wav"), "-i", os.path.join(OUT, "stem_relic.wav"),
     "-i", os.path.join(OUT, "stem_tally.wav"), "-filter_complex", fc, "-map", "[o]", "-ar", str(SR), r"C:\ai\unedited\audio\ep1_opening_mix_v04.wav"])
print("mix written")
