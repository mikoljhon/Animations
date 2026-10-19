# kd2_boards.py : image boards for the KillDroid v02 redesign (system Python + Pillow, no Blender):
#   python kd2_boards.py v01ref     reference vs v01 with the closely matching elements numbered -> renders\killdroid_v01_vs_ref.png
#   python kd2_boards.py concepts   the three redesign directions on one board                    -> renders\kd2_concepts_board.png
import os, sys
from PIL import Image, ImageDraw, ImageFont

R = r"C:\ai\unedited\renders"
REF = r"C:\ai\unedited\refs\killdroid_ref.jpg"
BG = (18, 19, 22)
INK = (232, 230, 227)
DIM = (150, 150, 155)
MARK = (64, 196, 255)
F = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(os.path.join(F, name), size)


def fit_h(im, h):
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


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


def marker(draw, xy, n, r=17):
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(10, 12, 16), outline=MARK, width=3)
    f = font("segoeuib.ttf", 20)
    draw.text((x, y - 1), str(n), font=f, fill=MARK, anchor="mm")


# elements of v01 that follow the reference closely: (number, title, text, marks on ref (x, y in the original 395x498), marks on v01_0 (640x900))
MATCHES = [
    (1, "Silhouette", "Twice a man's width: a small helmeted head sunk between huge shoulders, V-shaped torso, long arms hanging to mid-thigh, armoured column legs, big boots.",
     [(330, 40)], [(560, 120)]),
    (2, "Red muscle + black plates", "Bare red-brown muscle (pecs, eight-pack abs) framed by black metal plates worn to rust red.",
     [(196, 205)], [(330, 385)]),
    (3, "Visor bar, two red eyes", "A black helmet with one horizontal visor bar holding two glowing red eyes.",
     [(192, 60)], [(330, 160)]),
    (4, "Three-clawed steel hand", "The right forearm is an armoured sleeve that ends in three curved steel talons.",
     [(70, 335)], [(150, 730)]),
    (5, "Banded legs", "Greaves made of stacked rings and bands, knee caps and ankle cuffs on an armoured lower body.",
     [(125, 395), (262, 400)], [(205, 655), (470, 690)]),
    (6, "Round pauldrons", "Big domed shoulder armour, layered and larger on the claw side, smaller on the fist side.",
     [(100, 112), (292, 112)], [(120, 265), (505, 270)]),
    (7, "Same arm layout", "Mechanical claw arm on his right, a bare organic left arm with a bracer and a big fist.",
     [(318, 285)], [(520, 575)]),
    (8, "Side camera + chest strap", "A small camera barrel on the side of the helmet, and a diagonal strap / plate across the chest from the right shoulder.",
     [(158, 68), (150, 140)], [(400, 175), (290, 300)]),
]


def v01ref():
    H = 900
    ref = Image.open(REF).convert("RGB")
    sx = H / ref.height
    ref = fit_h(ref, H)
    v1 = Image.open(os.path.join(R, "killdroid_v01_0.png")).convert("RGB")
    s1 = H / v1.height
    v1 = fit_h(v1, H)
    pad, top, legend_w = 28, 120, 640
    W = pad + ref.width + pad + v1.width + pad + legend_w + pad
    Hh = top + H + pad + 40
    im = Image.new("RGB", (W, Hh), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 26), "KillDroid v01 vs the reference image", font=font("bahnschrift.ttf", 44), fill=INK)
    d.text((pad, 80), "Numbered: elements of v01 that closely follow the reference (credited on the image to Tsvetomir Georgiev, The Aaron Sims Company)",
           font=font("segoeui.ttf", 22), fill=DIM)
    x0, x1 = pad, pad + ref.width + pad
    im.paste(ref, (x0, top))
    im.paste(v1, (x1, top))
    fl = font("segoeuib.ttf", 22)
    d.text((x0 + 10, top + H + 6), "Reference (refs\\killdroid_ref.jpg)", font=fl, fill=DIM)
    d.text((x1 + 10, top + H + 6), "v01 (killdroid_v01.blend)", font=fl, fill=DIM)
    for n, title, text, mr, mv in MATCHES:
        for (x, y) in mr:
            marker(d, (x0 + x * sx, top + y * sx), n)
        for (x, y) in mv:
            marker(d, (x1 + x * s1, top + y * s1), n)
    lx, ly = x1 + v1.width + pad, top
    ft, fb = font("segoeuib.ttf", 25), font("segoeui.ttf", 21)
    for n, title, text, mr, mv in MATCHES:
        marker(d, (lx + 18, ly + 16), n)
        d.text((lx + 46, ly), title, font=ft, fill=INK)
        ly += 36
        for line in wrap(d, text, fb, legend_w - 50):
            d.text((lx + 46, ly), line, font=fb, fill=DIM)
            ly += 27
        ly += 16
    out = os.path.join(R, "killdroid_v01_vs_ref.png")
    im.save(out)
    print("SAVED", out, im.size)


CONCEPTS = [
    ("A", "RECLAIMER", "StarCapital's repossession unit: a closed corporate riot hull.",
     ["Head: tall helmet, ONE vertical ice-blue slit in a graphite face plate",
      "Weapon: pile-driver ram (piston + octagonal hammer head), no hand",
      "Colours: scuffed ivory shell, cobalt trim, graphite, hazard band",
      "Shoulders: solid angular blocks; legs: smooth shells, no bands",
      "Grabs with the armoured left fist"]),
    ("B", "BREAKER", "A demolition machine StarCapital refitted to hunt.",
     ["Head: no neck, a squat turret under an engine hump, SIX white work lamps",
      "Weapon: hydraulic crusher jaws (two jaws with teeth, hazard stripes)",
      "Colours: worn olive drab, dark steel, chrome rams, yellow bands",
      "Shoulders: octagonal actuator drums; legs: exposed hydraulic rams",
      "Exhaust stacks; can grab and lift with the jaws"]),
    ("C", "GROWN", "A vat-grown body held together by a surgical steel frame.",
     ["Head: bald pale skull, ONE amber lens bolted into the face, halo ring",
      "Weapon: hydraulic cleaver blade bolted to the right forearm",
      "Colours: pale grey flesh with veins, brushed steel, teal feed tubes",
      "Shoulders: bolted joint implants; steel ribcage + external spine",
      "Legs: orthopaedic braces and steel sabots; flesh left fist"]),
]


def concepts():
    hero_h, side_h, head_s, sil_h = 1000, 520, 300, 420
    col_w, pad, top = 720, 30, 130
    W = pad + 3 * (col_w + pad)
    sil_files = [("v01 (follows the reference)", "kd2_concept_v01_sil.png")] + [("%s  %s" % (k, t), "kd2_concept_%s_sil.png" % k) for k, t, _, _ in CONCEPTS]
    Hh = top + 70 + hero_h + 16 + side_h + 24 + 6 * 30 + 40 + 60 + sil_h + 60
    im = Image.new("RGB", (W, Hh), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 24), "KillDroid v02: three redesign directions", font=font("bahnschrift.ttf", 48), fill=INK)
    d.text((pad, 84), "Same role (2.85 m enforcer, brute strength, weapon on his right arm) and the same skeleton and bone names for all three. Pick one.",
           font=font("segoeui.ttf", 24), fill=DIM)
    fh, fp, fb = font("bahnschrift.ttf", 40), font("segoeuii.ttf", 22), font("segoeui.ttf", 21)
    for c, (k, title, pitch, bullets) in enumerate(CONCEPTS):
        x = pad + c * (col_w + pad)
        y = top
        d.text((x, y), "%s  %s" % (k, title), font=fh, fill=MARK)
        d.text((x, y + 46), pitch, font=fp, fill=INK)
        y += 84
        hero = Image.open(os.path.join(R, "kd2_concept_%s_hero.png" % k)).convert("RGB")
        im.paste(fit_h(hero, hero_h).crop((0, 0, col_w, hero_h)), (x, y))
        y += hero_h + 16
        side = fit_h(Image.open(os.path.join(R, "kd2_concept_%s_side.png" % k)).convert("RGB"), side_h)
        side = side.crop(((side.width - 374) // 2, 0, (side.width - 374) // 2 + 374, side_h))
        im.paste(side, (x, y))
        head = Image.open(os.path.join(R, "kd2_concept_%s_head.png" % k)).convert("RGB").resize((col_w - 374 - 16, col_w - 374 - 16), Image.LANCZOS)
        im.paste(head, (x + 374 + 16, y))
        ty = y + head.height + 12
        for b in bullets[:2]:
            for line in wrap(d, b, fb, col_w - 374 - 16):
                d.text((x + 374 + 16, ty), line, font=fb, fill=DIM)
                ty += 26
            ty += 6
        y += side_h + 24
        for b in bullets[2:]:
            d.text((x, y), "\u2022 " + b, font=fb, fill=DIM)
            y += 30
    y = top + 84 + hero_h + 16 + side_h + 24 + 3 * 30 + 50
    d.text((pad, y), "Front silhouettes, same scale", font=font("bahnschrift.ttf", 32), fill=INK)
    y += 50
    sw = (W - pad * 5) // 4
    for i, (label, fn) in enumerate(sil_files):
        sil = Image.open(os.path.join(R, fn)).convert("RGB")
        sil = fit_h(sil, sil_h)
        sx0 = pad + i * (sw + pad)
        im.paste(sil.crop(((sil.width - sw) // 2, 0, (sil.width - sw) // 2 + sw, sil_h)) if sil.width > sw else sil, (sx0, y))
        d.text((sx0, y + sil_h + 8), label, font=font("segoeuib.ttf", 22), fill=INK)
    out = os.path.join(R, "kd2_concepts_board.png")
    im = im.crop((0, 0, W, y + sil_h + 50))
    im.save(out)
    print("SAVED", out, im.size)


SWATCHES = [("olive paint", (0.060, 0.065, 0.040)), ("dark olive", (0.045, 0.049, 0.031)), ("bare steel", (0.36, 0.36, 0.37)),
            ("dark steel", (0.085, 0.085, 0.09)), ("brass caps", (0.42, 0.28, 0.09)), ("hazard yellow", (0.62, 0.40, 0.012)),
            ("rubber", (0.016, 0.016, 0.016)), ("work lamp", (1.0, 0.92, 0.78))]


def srgb(c):
    return tuple(int(round(255 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055))) for v in c)


def sheet():
    pad, top = 30, 120
    vw, vh = 456, 600
    W = pad + 4 * (vw + pad)
    im = Image.new("RGB", (W, 2000), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 24), "KillDroid v02  ·  KD-B7 “Breaker”", font=font("bahnschrift.ttf", 48), fill=INK)
    d.text((pad, 82), "StarCapital asset-recovery unit: a demolition machine refitted to hunt.  2.86 m hull, 2.91 m to the exhaust caps.  "
                      "Built in Relic's house style (style_mats.py).", font=font("segoeui.ttf", 23), fill=DIM)
    fl = font("segoeuib.ttf", 22)
    for i, (nm, lab) in enumerate((("front", "Front"), ("q3", "Three-quarter (weapon side)"), ("side", "Right side"), ("back", "Back three-quarter"))):
        x = pad + i * (vw + pad)
        im.paste(Image.open(os.path.join(R, "killdroid_v02_%s.png" % nm)).convert("RGB").resize((vw, vh), Image.LANCZOS), (x, top))
        d.text((x + 8, top + vh + 6), lab, font=fl, fill=DIM)
    y = top + vh + 50
    cs = 560
    for i, (nm, lab) in enumerate((("head", "Head: turret under the engine hump, six work lamps"), ("crusher", "Weapon: hydraulic crusher, jaws on their own bones"))):
        x = pad + i * (cs + pad)
        im.paste(Image.open(os.path.join(R, "killdroid_v02_%s.png" % nm)).convert("RGB").resize((cs, cs), Image.LANCZOS), (x, y))
        d.text((x + 8, y + cs + 6), lab, font=fl, fill=DIM)
    tx = pad + 2 * (cs + pad)
    tw = W - tx - pad
    ft, fb = font("segoeuib.ttf", 24), font("segoeui.ttf", 21)
    notes = [("Rig", "v01's 26 bones unchanged (0.000 m), plus claw1a.R (outer jaw) and claw2a.R (inner jaw): rotate them about local X to open "
                     "and close. 28 bones. Motion Assist and kd_ctrl.py checked on it."),
             ("Hydraulics", "12 rams (elbows, hips, knees, ankles, waist, jaws) re-aim themselves at their pivots as the joints move, "
                            "so they stay connected in any pose."),
             ("Lamps", "Six real spot lights plus glowing lenses, both driven by the rig property lamp_power (0 to 1): "
                       "set it to 0 for the beat where Relic kills the camera rig."),
             ("Surfaces", "Worn olive paint chipped to steel on the edges, grime, oil and rust streaks, dust on top, mud low down. "
                          "Relic's brass joint caps. Stencils: STARCAPITAL ASSET RECOVERY, KD-B7, DANGER, CRUSH POINT.")]
    ty = y
    for t, body in notes:
        d.text((tx, ty), t, font=ft, fill=MARK)
        ty += 32
        for line in wrap(d, body, fb, tw):
            d.text((tx, ty), line, font=fb, fill=INK)
            ty += 26
        ty += 12
    sw = (tw - 7 * 10) // 8
    for k, (nm, c) in enumerate(SWATCHES):
        sx = tx + k * (sw + 10)
        d.rectangle((sx, ty + 4, sx + sw, ty + 4 + 44), fill=srgb(c), outline=(70, 70, 75))
        for j, line in enumerate(wrap(d, nm, font("segoeui.ttf", 16), sw + 8)):
            d.text((sx, ty + 52 + j * 19), line, font=font("segoeui.ttf", 16), fill=DIM)
    y += cs + 50
    d.text((pad, y), "Pose test (kd_ctrl.py): guard, overhead wind-up, crouch, strike with the jaws open", font=font("bahnschrift.ttf", 28), fill=INK)
    y += 44
    ph = 420
    pw = int(ph * 640 / 800)
    for k in range(4):
        x = pad + k * (pw + pad)
        im.paste(Image.open(os.path.join(R, "killdroid_v02_pose%d.png" % k)).convert("RGB").resize((pw, ph), Image.LANCZOS), (x, y))
    y += ph + 30
    out = os.path.join(R, "killdroid_v02_sheet.png")
    im.crop((0, 0, W, y)).save(out)
    print("SAVED", out, (W, y))


CHANGES = [(1, "Silhouette", "a hunched V-torso with a small helmeted head", "no head or neck: a squat lamp turret under a box engine hump, short legs, huge forearms"),
           (2, "Colours", "bare red muscle and black plates", "worn olive paint over steel, Relic's brass caps, yellow and black hazard jaws, no flesh"),
           (3, "Head", "a visor bar with two red eyes and a side camera", "six white work lamps behind a wire guard, a small sensor pod"),
           (4, "Weapon", "a three-clawed steel hand", "a two-jaw hydraulic crusher with bolted teeth and its own rams"),
           (5, "Legs", "banded, ringed greaves", "box beams with exposed hydraulic rams, steel tread pads"),
           (6, "Shoulders", "round layered pauldrons", "octagonal actuator drums with brass caps"),
           (7, "Arms", "organic left arm with a bracer and a fist", "both arms mechanical; the left ends in a heavy mechanical fist")]


def v02ref():
    H = 900
    ref = fit_h(Image.open(REF).convert("RGB"), H)
    v2 = fit_h(Image.open(os.path.join(R, "killdroid_v02_vsref_render.png")).convert("RGB"), H)
    pad, top, legend_w = 28, 120, 700
    W = pad + ref.width + pad + v2.width + pad + legend_w + pad
    im = Image.new("RGB", (W, top + H + 70), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 26), "KillDroid v02 vs the reference image", font=font("bahnschrift.ttf", 44), fill=INK)
    d.text((pad, 80), "Same role (huge enforcer, brute strength, weapon on the right arm); every element v01 copied is replaced.",
           font=font("segoeui.ttf", 22), fill=DIM)
    x0, x1 = pad, pad + ref.width + pad
    im.paste(ref, (x0, top))
    im.paste(v2, (x1, top))
    fl = font("segoeuib.ttf", 22)
    d.text((x0 + 10, top + H + 8), "Reference (credited to Tsvetomir Georgiev)", font=fl, fill=DIM)
    d.text((x1 + 10, top + H + 8), "v02 (killdroid_v02.blend)", font=fl, fill=DIM)
    lx, ly = x1 + v2.width + pad, top
    ft, fb = font("segoeuib.ttf", 24), font("segoeui.ttf", 20)
    for n, title, was, now in CHANGES:
        marker(d, (lx + 18, ly + 16), n)
        d.text((lx + 46, ly), title, font=ft, fill=INK)
        ly += 33
        for line in wrap(d, "was: " + was, fb, legend_w - 50):
            d.text((lx + 46, ly), line, font=fb, fill=DIM)
            ly += 25
        for line in wrap(d, "now: " + now, fb, legend_w - 50):
            d.text((lx + 46, ly), line, font=fb, fill=INK)
            ly += 25
        ly += 12
    out = os.path.join(R, "killdroid_v02_vs_ref.png")
    im.save(out)
    print("SAVED", out, im.size)


def lineup():
    a = Image.open(os.path.join(R, "cast_lineup_v02_front.png")).convert("RGB")
    b = Image.open(os.path.join(R, "cast_lineup_v02_q3.png")).convert("RGB")
    pad, top = 30, 110
    W = a.width + 2 * pad
    im = Image.new("RGB", (W, top + a.height + b.height + pad + 60), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 22), "The cast at true scale, same light", font=font("bahnschrift.ttf", 44), fill=INK)
    d.text((pad, 76), "Mender 1.6 m  ·  Relic 2.1 m  ·  KillDroid v02 2.9 m.  The KillDroid now uses Relic's surfaces (worn steel, brass caps, grime); "
                      "Mender is still the clean, stylized one.", font=font("segoeui.ttf", 21), fill=DIM)
    im.paste(a, (pad, top))
    im.paste(b, (pad, top + a.height + pad))
    out = os.path.join(R, "cast_lineup_v02.png")
    im.save(out)
    print("SAVED", out, im.size)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "v01ref"
    {"v01ref": v01ref, "concepts": concepts, "sheet": sheet, "v02ref": v02ref, "lineup": lineup}[what]()
