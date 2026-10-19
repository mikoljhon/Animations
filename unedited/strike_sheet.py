# strike_sheet.py : compose the strike_views.py renders into one sheet per take:
#   one row per picked frame: Relic (or the KillDroid) front | real front | Relic side | real side, labelled.
#   python strike_sheet.py <tag>   -> renders\strikes_<tag>.png
import sys, json
from PIL import Image, ImageDraw, ImageFont
tag = sys.argv[1]
d = r"C:\ai\unedited\renders\sv_%s" % tag
picked = json.load(open(d + r"\frames.json"))
W_, H_ = 260, 310
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 15)
except OSError:
    font = ImageFont.load_default()
rows = [(clip, lab, f, info.get("who", "relic")) for clip, info in picked.items() for lab, f in info["frames"]]
S = Image.new("RGB", (W_ * 4, H_ * len(rows)), (20, 20, 20))
dr = ImageDraw.Draw(S)
NAME = {"relic": "RELIC ", "kd": "KILLDROID "}
for r, (clip, lab, f, rig_who) in enumerate(rows):
    for c, (who, view) in enumerate(((rig_who, "front"), ("real", "front"), (rig_who, "side"), ("real", "side"))):
        im = Image.open(d + r"\%s_%04d_%s_%s.png" % (clip, f, who, view)).convert("RGB").resize((W_, H_))
        S.paste(im, (c * W_, r * H_))
        dr.text((c * W_ + 6, r * H_ + 4), NAME.get(who, "REAL ") + view, fill=(255, 220, 0) if who != "real" else (120, 200, 255), font=font)
    dr.text((6, r * H_ + H_ - 22), "%s  %s  f%d" % (clip, lab, f), fill=(255, 255, 255), font=font)
S.save(r"C:\ai\unedited\renders\strikes_%s.png" % tag)
print("sheet", S.size, len(rows), "rows")
