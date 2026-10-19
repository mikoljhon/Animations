# Opening lines with Kokoro voices (placeholders until real voices), fitted to their windows in the 17.6 s first-person cut.
#   python kokoro_voices.py [relic_voice] [tally_voice] [mender_voice]
import os
import subprocess
import sys
import wave

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

RELIC_V = sys.argv[1] if len(sys.argv) > 1 else "am_onyx"
TALLY_V = sys.argv[2] if len(sys.argv) > 2 else "af_sarah"
MENDER_V = sys.argv[3] if len(sys.argv) > 3 else "af_river"
K = Kokoro(r"C:\ai\kokoro\kokoro-v1.0.onnx", r"C:\ai\kokoro\voices-v1.0.bin")
RAW = r"C:\ai\unedited\audio\voices\raw_k"
os.makedirs(RAW, exist_ok=True)
# id, voice, text, window in seconds (start, end available), slowest and fastest speed
LINES = [
    ("M1", MENDER_V, "Come on. What have you got for me?", 1.8, (0.95, 1.35)),
    ("M3", MENDER_V, "Dead as the rock. You beautiful thing. You are the jackpot.", 2.45, (0.95, 1.40)),
    ("M5", MENDER_V, "No. No, no, no. The scanner said...", 2.2, (0.95, 1.35)),
    ("R1", RELIC_V, "Tally? Tally! What happened?", 1.9, (0.85, 1.25)),
    ("T1", TALLY_V, "I'm here. I'm here. Systems online. Vitals stable. Log... empty.", 5.7, (0.95, 1.25)),
]
for lid, voice, text, window, (lo, hi) in LINES:
    base, sr = K.create(text, voice=voice, speed=1.0, lang="en-us")
    n0 = int(np.abs(base).max() > 0) and len(base) / sr
    # trim silence at the ends to measure the speech itself
    idx = np.where(np.abs(base) > 0.01)[0]
    speech = (idx[-1] - idx[0]) / sr if len(idx) else n0
    speed = float(np.clip(speech / window, lo, hi))
    smp, sr = K.create(text, voice=voice, speed=speed, lang="en-us")
    sf.write(os.path.join(RAW, lid + ".wav"), smp, sr)
    print(lid, voice, "speed %.2f" % speed, "speech %.2f s (window %.2f)" % (speech / speed, window))
