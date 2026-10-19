# Opening lines with Chatterbox (clones the timbre of a reference clip, with an emotion control). Run with the Chatterbox venv:
#   set HF_HOME=C:\ai\chatterbox\hf ; C:\ai\chatterbox\venv\Scripts\python.exe chat_voices.py [only_ids...]
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"C:\ai\chatterbox\hf")
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

REFS = r"C:\ai\chatterbox\refs"
OUT = r"C:\ai\unedited\audio\voices\raw_c"
os.makedirs(OUT, exist_ok=True)
# id, reference voice, text, emotion exaggeration, cfg_weight (lower = slower, more deliberate)
LINES = [
    ("M1", "mender", "Come on. What have you got for me?", 0.55, 0.5),
    ("M3", "mender", "Dead as the rock. You beautiful thing. You are the jackpot.", 0.8, 0.5),
    ("M5", "mender", "No. No, no, no. The scanner said...", 0.95, 0.4),
    ("R1", "relic", "Tally? Tally! What happened?", 0.9, 0.35),
    ("T1", "tally", "I'm here. I'm here. Systems online. Vitals stable. Log... empty.", 0.45, 0.5),
]
only = set(sys.argv[1:])
t0 = time.time()
model = ChatterboxTTS.from_pretrained(device="cpu")
print("model loaded in %.0f s" % (time.time() - t0), flush=True)
for lid, ref, text, exag, cfg in LINES:
    if only and lid not in only:
        continue
    t1 = time.time()
    torch.manual_seed(7)
    wav = model.generate(text, audio_prompt_path=os.path.join(REFS, ref + ".wav"), exaggeration=exag, cfg_weight=cfg)
    ta.save(os.path.join(OUT, lid + ".wav"), wav, model.sr)
    print(lid, "%.1f s of speech in %.0f s" % (wav.shape[-1] / model.sr, time.time() - t1), flush=True)
