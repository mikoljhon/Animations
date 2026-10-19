# mender_v03_boards.py : image boards for the Mender v03 restyle (system Python + Pillow, no Blender):
#   python mender_v03_boards.py turn        turnaround + close-ups + notes          -> renders\mender_v03_turnaround.png
#   python mender_v03_boards.py beforeafter v02 (old look) vs v03, same light, same cameras -> renders\mender_v03_before_after.png
#   python mender_v03_boards.py lineup      the cast at true scale, same light       -> renders\cast_lineup_v03.png
#   python mender_v03_boards.py cave        the cave two-shot, v02 render vs v03     -> renders\mender_v03_cave_before_after.png
import os, sys
from PIL import Image, ImageDraw, ImageFont

R = r"C:\ai\unedited\renders"
BG = (18, 19, 22)
INK = (232, 230, 227)
DIM = (150, 150, 155)
MARK = (64, 196, 255)
F = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(os.path.join(F, name), size)


def fit(im, w, h):
    return im.resize((w, h), Image.LANCZOS)


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def load(name):
    return Image.open(os.path.join(R, name)).convert("RGB")


VIEWS = [("front", "Front"), ("q3", "Three-quarter (mechanical arm)"), ("side", "Right side"), ("back", "Back three-quarter"),
         ("q3left", "Three-quarter (glove hand, cargo pocket)")]
CLOSE = [("head", "Dented steel hard hat, tape repair, ribbed headlamp on a hinged bracket, brass goggles"),
         ("arm", "Bare-steel arm: bolted plates, rams, clamp bands, Relic's brass caps on wrist, knuckles and finger pins"),
         ("legs", "Folds bunching above the cuffs, top-stitched seams, stitched knee panels, mud building up toward the boots"),
         ("patch", "Hand-sewn patches with frayed raw edges and loose threads, cargo pocket with flap and snaps"),
         ("pack", "Lid flap, ladder-lock buckles and loose strap tails, three-strand rope, rolled foam mat, ribbed jerrycan, caged lantern")]
NOTES = [("Same look as Relic and the KillDroid", "Every Mender material now comes from style_mats.worn(), the house look: faded, grimy safety-orange "
          "coveralls (sun-bleached on top, grime in the folds, mud building up toward the boots), worn leather and webbing, chipped yellow "
          "paint over steel, bare steel with Relic's brass caps, dust settled on everything that faces up."),
         ("Geometry", "Coverall resampled 3-4x denser with real folds; seams with double top-stitching; nine frayed patches; zipper placket; "
          "cargo pocket; stitched knee and elbow panels. Dented hat with a crease and a tape repair, curved peak. Buckles, D-ring, strap "
          "tails, flapped pouches with snaps, frame belt buckle, combination wrench. Arm: plates, bolts, rams, brass caps."),
         ("Rig and files", "Bone names, rig and body proportions unchanged (Mender_Rig, 49 bones). mender_v03.blend (studio) and "
          "mender_v03_cave.blend (the cave copy MenderC_Rig shares every mesh, so it is restyled too). Old files untouched."),
         ("Kept on purpose", "The silver Registry port with its blue LED stays clean: the one clean thing on Mender.")]


def turn():
    pad, top = 30, 128
    vw, vh = 456, 600
    W = pad + 5 * (vw + pad)
    im = Image.new("RGB", (W, 2400), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 24), "Mender v03  ·  restyled to the house look", font=font("bahnschrift.ttf", 48), fill=INK)
    d.text((pad, 84), "Same rig and proportions as v02 (1.6 m, 1.82 m to the antenna tip).  Cast light (the KillDroid v02 studio).",
           font=font("segoeui.ttf", 23), fill=DIM)
    fl = font("segoeuib.ttf", 21)
    for i, (nm, lab) in enumerate(VIEWS):
        x = pad + i * (vw + pad)
        im.paste(fit(load("mender_v03_%s.png" % nm), vw, vh), (x, top))
        for j, line in enumerate(wrap(d, lab, fl, vw - 10)):
            d.text((x + 6, top + vh + 6 + j * 25), line, font=fl, fill=DIM)
    y = top + vh + 70
    cs = vw
    fs = font("segoeui.ttf", 19)
    for i, (nm, lab) in enumerate(CLOSE):
        x = pad + i * (cs + pad)
        im.paste(fit(load("mender_v03_cu_%s.png" % nm), cs, cs), (x, y))
        for j, line in enumerate(wrap(d, lab, fs, cs - 10)):
            d.text((x + 6, y + cs + 6 + j * 23), line, font=fs, fill=DIM)
    y += cs + 110
    ft, fb = font("segoeuib.ttf", 24), font("segoeui.ttf", 21)
    colw = (W - 3 * pad) // 2
    rowh = 150
    for k, (t, body) in enumerate(NOTES):
        x = pad + (k % 2) * (colw + pad)
        yy = y + (k // 2) * rowh
        d.text((x, yy), t, font=ft, fill=MARK)
        for j, line in enumerate(wrap(d, body, fb, colw)):
            d.text((x, yy + 34 + j * 26), line, font=fb, fill=INK)
    y += 2 * rowh + 10
    out = os.path.join(R, "mender_v03_turnaround.png")
    im.crop((0, 0, W, y)).save(out)
    print("SAVED", out, (W, y))


def beforeafter():
    pad, top, lab = 30, 120, 150
    vw, vh = 380, 500
    cols = ["front", "q3", "back"]
    cus = ["head", "arm", "legs", "patch"]
    cw = 380
    W = lab + pad + len(cols) * (vw + pad) + len(cus) * (cw + pad)
    H = top + 2 * (vh + pad) + 70
    names = ["Front", "Three-quarter", "Back", "Hat, goggles, respirator", "Mechanical arm", "Lower legs and boots", "Patch and cargo pocket"]
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 24), "Mender: before and after", font=font("bahnschrift.ttf", 46), fill=INK)
    d.text((pad, 80), "Same rig, same proportions, same cast light and cameras.  Top: v02 (relic_v07_mender.blend).  Bottom: v03 (mender_v03.blend).",
           font=font("segoeui.ttf", 22), fill=DIM)
    fl = font("segoeuib.ttf", 26)
    for r, tag, name in ((0, "v02", "Before\nv02"), (1, "v03", "After\nv03")):
        y = top + r * (vh + pad)
        for j, line in enumerate(name.split("\n")):
            d.text((pad, y + vh // 2 - 30 + j * 34), line, font=fl, fill=MARK if r else DIM)
        x = lab + pad
        for c in cols:
            im.paste(fit(load("mender_%s_%s.png" % (tag, c)), vw, vh), (x, y))
            x += vw + pad
        for c in cus:
            src = load("mender_%s_cu_%s.png" % (tag, c))
            im.paste(fit(src, cw, cw), (x, y + (vh - cw) // 2))
            x += cw + pad
    x, y = lab + pad, top + 2 * (vh + pad) - pad + 8
    fs = font("segoeuib.ttf", 21)
    for k, nm in enumerate(names):
        w_ = vw if k < len(cols) else cw
        d.text((x + 6, y), nm, font=fs, fill=DIM)
        x += w_ + pad
    out = os.path.join(R, "mender_v03_before_after.png")
    im.save(out)
    print("SAVED", out, im.size)


def lineup():
    a = load("cast_lineup_v03_front.png")
    b = load("cast_lineup_v03_q3.png")
    c = load("cast_lineup_v03_pair.png")
    pad, top = 30, 136
    W = a.width + 2 * pad
    im = Image.new("RGB", (W, top + a.height + b.height + c.height + 3 * pad + 60), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 22), "The cast at true scale, same light", font=font("bahnschrift.ttf", 44), fill=INK)
    fs = font("segoeui.ttf", 21)
    d.text((pad, 76), "Mender v03 1.6 m  ·  Relic 2.1 m  ·  KillDroid v02 2.9 m.", font=fs, fill=DIM)
    d.text((pad, 102), "All three now share the house look of style_mats.py: worn paint and bare steel, Relic's brass, grime, "
                       "dust on top, mud low down.", font=fs, fill=DIM)
    im.paste(a, (pad, top))
    im.paste(b, (pad, top + a.height + pad))
    im.paste(c, (pad + (a.width - c.width) // 2, top + a.height + b.height + 2 * pad))
    out = os.path.join(R, "cast_lineup_v03.png")
    im.save(out)
    print("SAVED", out, im.size)


def cave():
    a = load("mender_cave_03_two_shot_dug.png")
    b = load("mender_v03_cave_two_shot_dug.png")
    pad, top = 30, 110
    w, h = 1200, 675
    im = Image.new("RGB", (2 * w + 3 * pad, top + h + 70), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 22), "Mender in the cave: before and after", font=font("bahnschrift.ttf", 44), fill=INK)
    d.text((pad, 76), "The dug-out two-shot from the episode 1 opening, same camera.  Left: v02 (relic_v08_mender_cave.blend).  "
                      "Right: v03 (mender_v03_cave.blend).", font=font("segoeui.ttf", 21), fill=DIM)
    im.paste(fit(a, w, h), (pad, top))
    im.paste(fit(b, w, h), (2 * pad + w, top))
    fl = font("segoeuib.ttf", 22)
    d.text((pad + 8, top + h + 8), "v02", font=fl, fill=DIM)
    d.text((2 * pad + w + 8, top + h + 8), "v03", font=fl, fill=MARK)
    out = os.path.join(R, "mender_v03_cave_before_after.png")
    im.save(out)
    print("SAVED", out, im.size)


if __name__ == "__main__":
    {"turn": turn, "beforeafter": beforeafter, "lineup": lineup, "cave": cave}[sys.argv[1] if len(sys.argv) > 1 else "turn"]()
