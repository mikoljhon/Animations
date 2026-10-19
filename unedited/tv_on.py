# "Camera off" until the rock breaks, then an old TV / CRT switch-on. Works on the rendered frames, no re-render needed.
#   python tv_on.py            reads renders\pov_seq\f_####.png, writes renders\pov_tv\f_####.png
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

SRC = r"C:\ai\unedited\renders\pov_seq"
DST = r"C:\ai\unedited\renders\pov_tv"
N = 422
F_ON = 146                      # the big strike: the rock breaks and the camera switches on
W, H = 1280, 720
os.makedirs(DST, exist_ok=True)
rng = np.random.default_rng(12)


def ease_out(u, p=2.5):
    u = min(max(u, 0.0), 1.0)
    return 1.0 - (1.0 - u) ** p


def glow(canvas, radius, amount):
    g = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return np.clip(canvas + np.asarray(g, dtype=np.float32) * amount, 0, 255)


def scanlines(img, strength):
    out = img.copy()
    out[1::2] *= (1.0 - strength)
    return out


def tv_frame(k, src):
    """k = frames since the switch-on (0 = the first frame of the line)"""
    canvas = np.zeros((H, W, 3), np.float32)
    if k == 0:                                           # a dot-ish short line
        w = int(W * 0.12)
        canvas[H // 2 - 1:H // 2 + 2, W // 2 - w // 2:W // 2 + w // 2] = 255
        return glow(canvas, 6, 2.0)
    if k == 1:
        w = int(W * 0.55)
        canvas[H // 2 - 2:H // 2 + 3, W // 2 - w // 2:W // 2 + w // 2] = 255
        return glow(canvas, 9, 2.5)
    if k == 2:                                           # the full-width line
        canvas[H // 2 - 3:H // 2 + 4, :] = 255
        return glow(canvas, 14, 2.5)
    if k <= 9:                                           # opens vertically, squashed picture, over-bright
        u = (k - 2) / 8.0
        h = int(7 + (H - 7) * ease_out(u))
        img = np.asarray(Image.fromarray(src).resize((W, h), Image.LANCZOS), dtype=np.float32)
        gain = 1.0 + 1.3 * (1.0 - u)
        y0 = (H - h) // 2
        canvas[y0:y0 + h] = np.clip(img * gain + 60 * (1.0 - u), 0, 255)
        canvas[max(y0 - 1, 0):y0 + 2] = 255              # bright edges of the opening window
        canvas[min(y0 + h - 2, H - 2):y0 + h + 1] = 255
        canvas += rng.normal(0, 14 * (1.0 - u * 0.5), canvas.shape).astype(np.float32)
        return glow(scanlines(canvas, 0.18), 10, 0.9)
    # settled: flicker, wobble and a little picture tearing, scanlines fading out
    j = k - 10
    flick = [1.40, 0.82, 1.18, 0.92, 1.08, 0.97, 1.04, 1.0]
    gain = flick[j] if j < len(flick) else 1.0
    img = src.astype(np.float32) * gain
    if j < 6:
        shift = int(rng.integers(-9, 10)) if j % 2 == 0 else 0
        img = np.roll(img, shift, axis=1)
        if j in (1, 3):                                  # a torn band
            y = int(rng.integers(80, H - 160))
            img[y:y + 40] = np.roll(img[y:y + 40], int(rng.integers(-40, 41)), axis=1)
    img += rng.normal(0, max(10.0 - j * 1.4, 0.0), img.shape).astype(np.float32)
    s = max(0.16 - j * 0.02, 0.05)
    return np.clip(scanlines(img, s), 0, 255)


black = Image.new("RGB", (W, H), (0, 0, 0))
for f in range(1, N + 1):
    out = os.path.join(DST, "f_%04d.png" % f)
    if f < F_ON:
        black.save(out)
        continue
    k = f - F_ON
    src = np.asarray(Image.open(os.path.join(SRC, "f_%04d.png" % f)).convert("RGB"))
    if k > 16:
        Image.fromarray(src).save(out)
        continue
    Image.fromarray(np.clip(tv_frame(k, src), 0, 255).astype(np.uint8)).save(out)
print("done", N, "frames; camera switches on at frame", F_ON, "= %.2f s" % ((F_ON - 1) / 24.0))
