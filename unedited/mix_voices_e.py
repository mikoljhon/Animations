# Opening mix with the ElevenLabs takes (user's own voice IDs), placed on the 17.6 s first-person cut. python mix_voices_e.py [take letters M1 M3 M5 R1 T1 e.g. a b a a a]
# Long dead pauses inside a take are shortened (the acting keeps its rhythm); everything outside Relic is muffled until his vision comes on.
import glob
import os
import subprocess
import sys
import wave

import numpy as np

FF = glob.glob(r"C:\Users\mike\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe")[0]
RAW = r"C:\ai\unedited\audio\voices\raw_e"
OUT = r"C:\ai\unedited\audio\voices"
SR = 48000
DUR = 17.6
T_ON = 8.6
TAKES = dict(zip(["M1", "M3", "M5", "R1", "T1"], sys.argv[1:6])) if len(sys.argv) >= 6 else {"M1": "a", "M3": "b", "M5": "a", "R1": "a", "T1": "a"}

COMPRESS = ("silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.04:stop_periods=-1:stop_duration=0.30:stop_threshold=-42dB:stop_silence=0.24,"
            "areverse,silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.04,areverse")
CHAINS = {
    "M": "highpass=f=110,lowpass=f=3600,aecho=0.8:0.55:38:0.12,loudnorm=I=-20:TP=-2:LRA=8",        # a covered face
    "R": "highpass=f=60,equalizer=f=140:t=q:w=1:g=2.5,loudnorm=I=-18:TP=-2:LRA=8",                    # heard from inside his own head
    "T": "highpass=f=120,loudnorm=I=-20:TP=-2:LRA=8",
}
# id, start second, cut at second (None = whole)
PLACE = [("M1", 0.10, None), ("M3", 1.95, None), ("M5", 7.20, 8.85), ("R1", 8.95, None), ("T1", 11.25, None)]


def run(args):
    subprocess.run([FF, "-y", "-v", "error"] + args, check=True)


def load(path):
    with wave.open(path, "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    return (np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0).reshape(-1, ch)


def save(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


buses = {k: np.zeros((int(DUR * SR), 2)) for k in "MRT"}
report = {}
for lid, t0, cut in PLACE:
    src = os.path.join(RAW, lid + TAKES[lid] + ".mp3")
    dst = os.path.join(OUT, lid + "_e.wav")
    run(["-i", src, "-af", COMPRESS + "," + CHAINS[lid[0]], "-ar", str(SR), "-ac", "2", dst])
    x = load(dst)
    if cut is not None:
        x = x[: int((cut - t0) * SR)].copy()
        fade = int(0.05 * SR)
        x[-fade:] *= np.linspace(1, 0, fade)[:, None]
    i0 = int(t0 * SR)
    buses[lid[0]][i0:i0 + len(x)] += x[: len(buses[lid[0]]) - i0]
    report[lid] = (t0, round(len(x) / SR, 2), round(t0 + len(x) / SR, 2))
    print(lid, "take", TAKES[lid], "start %.2f, length %.2f s, ends %.2f" % report[lid])
for k, name in (("M", "mender"), ("R", "relic"), ("T", "tally")):
    save(os.path.join(OUT, "stem_e_%s.wav" % name), buses[k])
fc = ("[0:a]lowpass=f=450:enable='lt(t,%(t)s)',volume=0.7[s];"
      "[1:a]lowpass=f=520:enable='lt(t,%(t)s)'[m];[m][2:a][3:a]amix=inputs=3:normalize=0[vo];[vo]asplit=2[vo1][vo2];"
      "[s][vo1]sidechaincompress=threshold=0.015:ratio=6:attack=15:release=400:makeup=1[sd];"
      "[sd][vo2]amix=inputs=2:normalize=0,alimiter=limit=0.93,afade=t=out:st=%(f)s:d=0.5[o]") % {"t": T_ON, "f": DUR - 0.5}
run(["-i", os.path.join(OUT, "sfx_v04.wav"), "-i", os.path.join(OUT, "stem_e_mender.wav"), "-i", os.path.join(OUT, "stem_e_relic.wav"),
     "-i", os.path.join(OUT, "stem_e_tally.wav"), "-filter_complex", fc, "-map", "[o]", "-ar", str(SR), r"C:\ai\unedited\audio\ep1_opening_mix_v07.wav"])
print("mix written; Tally ends at %.2f s, so 'Log... empty' is at about %.2f s (frame %d)" % (report["T1"][2], report["T1"][2] - 0.7, int((report["T1"][2] - 0.7) * 24) + 1))
