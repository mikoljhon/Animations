# reel_encode.py : put the two reel views side by side with captions and encode the MP4.
#   python reel_encode.py <tag> <out.mp4> [kd]  (frames from reel_vid.py, clip spans from anim_cache5\spans_<tag>.json; a take made by
#   kd_beats.py also has anim_cache5\beats_<tag>.json, whose beat names become the titles)
import sys, json, glob, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
tag, out = sys.argv[1], sys.argv[2]
NAME = "the KillDroid" if len(sys.argv) > 3 and sys.argv[3] == "kd" else "Relic"
R = r"C:\ai\unedited"
spans = json.load(open(R + r"\anim_cache5\spans_%s.json" % tag))
desc = {}
for n in spans:
    try:
        desc[n] = str(np.load(R + r"\mocap_ref\segs\%s.npz" % n, allow_pickle=True)["desc"])
    except Exception:
        desc[n] = ""
beats = {}
holds = []          # beats without a capture clip (kd_beats.py's final hold): (start, end, label)
if os.path.exists(R + r"\anim_cache5\beats_%s.json" % tag):
    for b in json.load(open(R + r"\anim_cache5\beats_%s.json" % tag))["beats"]:
        beats[b["clip"]] = b["label"]
        if b["clip"] not in spans:
            holds.append((b["start"], b["end"], b["label"]))
qa = json.load(open(R + r"\anim_cache5\qa_%s.json" % tag))
A = sorted(glob.glob(R + r"\renders\reel_%s_A\f_*.jpg" % tag))
ff = glob.glob(r"C:\Users\mike\AppData\Local\Microsoft\WinGet\Packages\**\ffmpeg.exe", recursive=True)[0]
f_big = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
f_sm = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 17)
Wd, Hd, top, bot = 1440, 720, 44, 34
p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (Wd, Hd + top + bot), "-r", "24", "-i", "-",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", out], stdin=subprocess.PIPE)
verdict = "QA %s" % ("PASS" if qa.get("pass") else "FAIL")
for pa in A:
    f = int(os.path.basename(pa)[2:6])
    pb = pa.replace("_A\\", "_B\\")
    im = Image.new("RGB", (Wd, Hd + top + bot), (14, 14, 16))
    im.paste(Image.open(pa).convert("RGB"), (0, top))
    im.paste(Image.open(pb).convert("RGB"), (720, top))
    dr = ImageDraw.Draw(im)
    clip = next((n for n, (s, ln, m) in spans.items() if s <= f <= s + ln), None)
    hold = next((lb for s, e, lb in holds if s <= f <= e), None)
    if clip in beats:
        title = "%s   (%s: %s)" % (beats[clip], clip, desc.get(clip, ""))
    elif hold:
        title = hold
    else:
        title = ("%s  -  %s" % (clip, desc.get(clip, ""))) if clip else "transition (blended joint by joint)"
    dr.text((12, 10), title, fill=(255, 255, 255), font=f_big)
    dr.text((Wd - 330, 12), "frame %d    %s" % (f, verdict), fill=(120, 230, 120) if qa.get("pass") else (255, 90, 90), font=f_sm)
    dr.text((12, top + Hd + 8), "LEFT: %s beside the real performer (white figure = raw motion capture)" % NAME, fill=(200, 200, 200), font=f_sm)
    dr.text((732, top + Hd + 8), "RIGHT: %s, close three-quarter view" % NAME, fill=(200, 200, 200), font=f_sm)
    p.stdin.write(im.tobytes())
p.stdin.close()
p.wait()
print("ENCODED", out, len(A), "frames")
