# pose_sheet.py : compose pose_test.py renders into one labelled sheet.   python pose_sheet.py <tag> [<tag2>]  (two tags: side by side)
import sys
from PIL import Image, ImageDraw, ImageFont
tags = [t for t in sys.argv[1:] if t != "close"]
VIEWS = ("cfront", "cleft", "cback", "cright") if "close" in sys.argv else ("front", "left", "back", "right")
NAMES = ["arms out to the sides", "arms straight overhead", "guard: elbows fully bent", "punch reach + other arm back", "high front kick",
         "side kick", "deep crouch", "twist + lean + head turn"]
Wv, Hv = 230, 294
font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 14)
S = Image.new("RGB", (Wv * 4 * len(tags), Hv * len(NAMES)), (20, 20, 20))
dr = ImageDraw.Draw(S)
for t, tag in enumerate(tags):
    for i, nm in enumerate(NAMES):
        for j, v in enumerate(VIEWS):
            try:
                im = Image.open(r"C:\ai\unedited\renders\posetest_%s\%d_%s.png" % (tag, i, v)).convert("RGB").resize((Wv, Hv))
            except FileNotFoundError:
                continue
            S.paste(im, ((t * 4 + j) * Wv, i * Hv))
        dr.text((t * 4 * Wv + 6, i * Hv + 4), "%s  |  %s" % (tag, nm), fill=(255, 230, 0), font=font)
out = r"C:\ai\unedited\renders\posetest_%s%s.png" % ("_vs_".join(tags), "_close" if "close" in sys.argv else "")
S.save(out)
print(out, S.size)
