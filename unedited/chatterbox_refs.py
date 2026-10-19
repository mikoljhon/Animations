# Reference clips for Chatterbox (it clones the timbre of a short clip). Made with Kokoro until we have real recordings. system Python.
import os

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

K = Kokoro(r"C:\ai\kokoro\kokoro-v1.0.onnx", r"C:\ai\kokoro\voices-v1.0.bin")
OUT = r"C:\ai\chatterbox\refs"
os.makedirs(OUT, exist_ok=True)
TEXT = ("The rest of the night was quiet. I listened to the pump tick in the dark, and I kept still, because "
        "moving would have meant remembering how, and that is a harder thing than it sounds.")
for name, voice, speed in (("relic", "am_onyx", 0.9), ("tally", "af_sarah", 1.0), ("mender", "af_river", 1.05)):
    smp, sr = K.create(TEXT, voice=voice, speed=speed, lang="en-us")
    sf.write(os.path.join(OUT, name + ".wav"), smp, sr)
    print(name, voice, round(len(smp) / sr, 1), "s")
