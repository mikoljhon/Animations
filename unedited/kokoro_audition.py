# Audition reel: several Kokoro voices per character say one of their lines. python kokoro_audition.py
import os
import sys

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

K = Kokoro(r"C:\ai\kokoro\kokoro-v1.0.onnx", r"C:\ai\kokoro\voices-v1.0.bin")
voices = sorted(K.get_voices())
print("VOICES", voices)


def f0(x, sr):
    """rough median pitch (Hz) by autocorrelation on voiced frames"""
    x = x - x.mean()
    out = []
    n = int(0.04 * sr)
    for i in range(0, len(x) - 2 * n, n // 2):
        w = x[i:i + 2 * n]
        if np.sqrt((w ** 2).mean()) < 0.02:
            continue
        ac = np.correlate(w, w, "full")[len(w) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.35 * ac[0]:
            out.append(sr / k)
    return float(np.median(out)) if out else 0.0


cast = {
    "RELIC": ("Tally? Tally! What happened?", ["am_onyx", "am_fenrir", "am_michael", "bm_george", "bm_lewis", "am_adam"], 0.92),
    "TALLY": ("I'm here. I'm here. Systems online. Vitals stable. Log... empty.", ["af_sarah", "af_nicole", "bf_emma", "af_heart", "bf_isabella", "af_nova"], 1.0),
    "MENDER": ("Come on. What have you got for me? Dead as the rock. You beautiful thing.", ["am_puck", "bm_fable", "af_jessica", "am_liam", "bm_daniel", "af_river"], 1.08),
}
sr_out = 24000
pieces = []
gap = np.zeros(int(0.5 * sr_out))
report = {}
for who, (line, cands, speed) in cast.items():
    for i, v in enumerate(cands, 1):
        if v not in voices:
            print("missing", v)
            continue
        lab, sr = K.create("%s, option %d." % (who.capitalize(), i), voice=v, speed=1.0, lang="en-us")
        smp, sr = K.create(line, voice=v, speed=speed, lang="en-us")
        report["%s %d %s" % (who, i, v)] = round(f0(smp, sr), 0)
        pieces += [lab, np.zeros(int(0.25 * sr)), smp, gap]
        print(who, i, v, "pitch ~%d Hz, %.1f s" % (report["%s %d %s" % (who, i, v)], len(smp) / sr))
audio = np.concatenate(pieces)
sf.write(r"C:\ai\unedited\audio\voices\kokoro_audition.wav", audio, sr_out)
print("written", round(len(audio) / sr_out, 1), "s")
