# ep1_opening_audio.py : procedural sound sketch for the 9 s opening animatic (cut 2): pump heartbeat, rock groan, pick strikes, gravel, ring buzz and
# hum, eye pings, Mender's gasp, the dropped pick, boot scuffs, Relic's servo. Placeholder sound only, no voices.
# Run with numpy (inside Blender's Python): exec(open(path).read()); make_audio(out_path). Times follow the frame numbers of build_opening2().
import numpy as np, wave, os

SR = 48000
DUR = 15.6
DROP_ = 211
SLOW, C1, C2, C3 = 1.35, 42, 91, 146


def G(f):
    return C3 + (f - 58.0) * SLOW if f >= 58 else f


def fg(f):
    return fr(G(f))



def fr(f):
    return (f - 1) / 24.0


def make_audio(out_path, seed=11):
    rng = np.random.default_rng(seed)
    N = int(SR * DUR)
    out = np.zeros((N, 2))

    def fft_filter(x, lo=None, hi=None, order=2):
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1.0 / SR)
        g = np.ones_like(f)
        if lo:
            g = g / (1.0 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
        if hi:
            g = g / (1.0 + (f / hi) ** (2 * order))
        return np.fft.irfft(X * g, len(x))

    def env_ad(n, a, tau):
        i = np.arange(n)
        att = np.minimum(i / max(a * SR, 1.0), 1.0)
        dec = np.exp(-np.maximum(i - a * SR, 0.0) / (tau * SR))
        return att * dec

    def place(y, t0, gain=1.0, pan=0.0):
        i0 = int(round(t0 * SR))
        i1 = min(i0 + len(y), N)
        if i0 >= N or i1 <= i0 or i0 < 0:
            return
        gl = np.cos((pan + 1) * np.pi / 4)
        gr = np.sin((pan + 1) * np.pi / 4)
        seg = y[:i1 - i0] * gain
        out[i0:i1, 0] += seg * gl
        out[i0:i1, 1] += seg * gr

    def thump(f0=62.0, tau=0.06, dur=0.45):
        n = int(dur * SR)
        t = np.arange(n) / SR
        return np.sin(2 * np.pi * (f0 + 28 * np.exp(-t / 0.02)) * t) * np.exp(-t / tau)

    def click(dur=0.012, tau=0.002):
        n = int(dur * SR)
        return rng.standard_normal(n) * np.exp(-np.arange(n) / (tau * SR))

    def noise_burst(dur, lo, hi, a=0.004, tau=0.05):
        n = int(dur * SR)
        y = fft_filter(rng.standard_normal(n), lo, hi)
        y = y / (np.abs(y).max() + 1e-9)
        return y * env_ad(n, a, tau)

    def trickle(dur, density, lo=700, hi=6000, tau=0.6):
        n = int(dur * SR)
        imp = (rng.random(n) < density / SR).astype(float) * rng.uniform(0.2, 1.0, n)
        k = rng.standard_normal(int(0.006 * SR)) * np.exp(-np.arange(int(0.006 * SR)) / (0.0015 * SR))
        y = np.fft.irfft(np.fft.rfft(imp) * np.fft.rfft(k, n), n)
        y = fft_filter(y, lo, hi)
        y = y / (np.abs(y).max() + 1e-9)
        return y * np.exp(-np.arange(n) / (tau * SR))

    def ping(f, dur=0.3, tau=0.07):
        n = int(dur * SR)
        t = np.arange(n) / SR
        return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.03)) * np.minimum(t / 0.004, 1.0) * np.exp(-t / tau)

    def knock(f, tau=0.035, dur=0.25):
        n = int(dur * SR)
        t = np.arange(n) / SR
        return np.sin(2 * np.pi * f * t) * np.exp(-t / tau)

    t_all = np.arange(N) / SR

    # --- ambience: cave rumble, a breath of air, and the long groan in the rock (0.3 s to 2.2 s)
    rum = fft_filter(np.cumsum(rng.standard_normal(N)), 25, 140)
    rum = rum / (np.abs(rum).max() + 1e-9)
    out[:, 0] += rum * 0.065
    out[:, 1] += np.roll(rum, 977) * 0.065
    air = fft_filter(rng.standard_normal(N), 1800, 6500)
    air = air / (np.abs(air).max() + 1e-9) * (0.5 + 0.5 * np.sin(2 * np.pi * t_all / 5.3 + 1.0)) ** 2
    out[:, 0] += air * 0.012
    out[:, 1] += np.roll(air, 1501) * 0.012
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    groan = (np.sin(2 * np.pi * np.cumsum(70 - 24 * t / 2.0) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(141 - 48 * t / 2.0) / SR)) * np.sin(np.pi * t / 2.0) ** 2
    place(groan, 0.3, 0.22, -0.2)
    place(trickle(1.4, 90, tau=0.5), 1.0, 0.05, 0.3)

    # --- the pump: slow and steady, then it speeds up when the ring lights (f74) and settles into a heartbeat
    s_ = 0.15
    while s_ < fg(74) - 0.45:
        place(thump(), s_, 0.55)
        place(click(), s_, 0.05)
        s_ += 1.25
    t = fg(74)
    for gap in (0.62, 0.50, 0.42, 0.36):
        place(thump(), t, 0.62)
        t += gap
    while t < DUR - 0.5:
        place(thump(58, 0.07), t, 0.60)
        place(thump(74, 0.05), t + 0.17, 0.34)
        t += 0.56

    # --- pick strikes (f8, f31, f58) and the slab giving way
    for f, g, pan in ((C1, 0.34, -0.15), (C2, 0.36, -0.1), (C3, 0.6, -0.1)):
        s = fr(f)
        # the swing: a breath in as the pick goes up, effort at the top, the whoosh of the downswing, a scrape as he wrenches it free
        place(noise_burst(0.45, 300, 2200, 0.12, 0.14), fr(f - 30), 0.07, 0.25)
        place(noise_burst(0.30, 350, 2400, 0.03, 0.10), fr(f - 12), 0.11, 0.25)
        n_ = int(0.30 * SR)
        wh = fft_filter(rng.standard_normal(n_), 250, 3500) * (np.arange(n_) / n_) ** 2.2
        place(wh / (np.abs(wh).max() + 1e-9), s - 0.30, g * 0.55, -0.1)
        place(noise_burst(0.35, 900, 4200, 0.02, 0.09), s + 0.15, 0.07, -0.1)
        place(noise_burst(0.12, 1500, 5200, 0.001, 0.025), s, g, pan)
        place(thump(90, 0.07), s, g * 1.3, pan)
        place(ping(2300, 0.2, 0.03), s, g * 0.25, pan)
    s = fr(C3) + 0.05
    place(noise_burst(1.7, 60, 420, 0.15, 0.55), s, 0.55, -0.1)         # the slab grinding off
    place(trickle(2.0, 400, tau=0.8), s + 0.05, 0.55, 0.1)              # gravel pouring down the armour
    place(thump(45, 0.25, 0.8), s, 0.7)

    # --- the ring: buzz in time with its flicker (f74 to 87), then a hum that swells
    ring = [(74, 0.9), (76, 0.1), (78, 0.7), (79, 0.0), (82, 1.0), (85, 0.55), (87, 1.0)]
    ft = np.array([fg(f) for f, _ in ring])
    fv = np.array([v for _, v in ring])
    i0, i1 = int(fg(74) * SR), int(fg(88) * SR)
    tt = t_all[i0:i1]
    idx = np.clip(np.searchsorted(ft, tt, side='right') - 1, 0, len(ft) - 1)
    amp = fv[idx]
    buzz = (np.sin(2 * np.pi * 100 * tt) + 0.5 * np.sin(2 * np.pi * 300 * tt) + 0.25 * np.sin(2 * np.pi * 500 * tt)) * 0.7
    buzz = buzz + fft_filter(rng.standard_normal(len(tt)), 800, 7000) * 0.35
    place(buzz * amp, fg(74), 0.11, 0.0)
    place(click(0.02, 0.004), fg(74), 0.25)
    place(click(0.02, 0.004), fg(82), 0.2)
    i0 = int(fg(84) * SR)
    th = t_all[i0:]
    sw = np.clip((th - fg(84)) / 2.0, 0, 1)
    sw = sw * sw * (3 - 2 * sw)
    hum_l = np.sin(2 * np.pi * 110 * th) + 0.4 * np.sin(2 * np.pi * 220 * th + 0.3) + 0.2 * np.sin(2 * np.pi * 330 * th)
    hum_r = np.sin(2 * np.pi * 110.8 * th) + 0.4 * np.sin(2 * np.pi * 221 * th + 0.9) + 0.2 * np.sin(2 * np.pi * 331 * th)
    pulse = 0.85 + 0.15 * np.sin(2 * np.pi * th / 1.8)
    out[i0:, 0] += hum_l * sw * pulse * 0.050
    out[i0:, 1] += hum_r * sw * pulse * 0.050

    # --- Mender: a breath in at the freeze (f78), a gasp when the eyes light (f104), the pick falling (f106), boots scrabbling (f109 to 125)
    place(noise_burst(0.35, 400, 2600, 0.08, 0.12), fg(78), 0.10, 0.25)
    place(noise_burst(0.45, 500, 2800, 0.05, 0.16), fg(104), 0.17, 0.25)
    for k, (g, extra) in enumerate(((0.55, 0.0), (0.25, 0.0), (0.12, 0.0))):
        s = fg(113) + (0.0, 0.10, 0.17)[k]
        place(knock(520, 0.03), s, g * 0.8, -0.1)
        place(noise_burst(0.1, 200, 1800, 0.001, 0.03), s, g, -0.1)
        place(thump(150, 0.05, 0.2), s, g * 0.8, -0.1)
    place(ping(2600, 0.5, 0.09), fg(113), 0.08, -0.1)                   # the steel head ringing on stone
    for s, g in ((4.52, 0.18), (4.66, 0.14), (4.79, 0.20), (4.95, 0.15), (5.05, 0.17), (5.17, 0.12)):
        place(noise_burst(0.09, 300, 3200, 0.004, 0.035), fg(1 + s * 24), g, 0.3)
    place(noise_burst(1.4, 120, 900, 0.2, 0.4), fg(1 + 4.4 * 24), 0.06, 0.3)         # cloth rustle
    place(trickle(1.6, 150, tau=0.6), fr(DROP_), 0.12, 0.0)               # stones sliding off Relic

    # --- the eyes (f99 to 106) and Relic's head lifting (f104 to 130): soft pings and a low servo
    for f, g, hz in ((99, 0.08, 880), (103, 0.09, 990), (104, 0.04, 740), (106, 0.10, 1180)):
        place(ping(hz, 0.35, 0.09), fg(f), g, 0.0)
    n = int((fg(131) - fg(104)) * SR)
    t = np.arange(n) / SR
    fq = 150 + 90 * np.sin(np.pi * t / (n / SR))
    ph = 2 * np.pi * np.cumsum(fq) / SR
    servo = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)) * np.sin(np.pi * t / (n / SR)) ** 2
    servo = fft_filter(servo, 60, 1100)
    place(servo / (np.abs(servo).max() + 1e-9), fg(104), 0.07, -0.2)
    place(click(0.02, 0.004), fg(192), 0.05, 0.0)                       # the blink

    # --- cave reverb: a stereo tail of decaying noise, 1.7 s
    n = int(1.7 * SR)
    ir = [fft_filter(rng.standard_normal(n), 120, 3200) * np.exp(-np.arange(n) / (0.5 * SR)) for _ in range(2)]
    ir = [x / np.sqrt((x ** 2).sum()) for x in ir]
    wet = np.zeros_like(out)
    for c in range(2):
        L = N + n
        W = np.fft.irfft(np.fft.rfft(out[:, c], L) * np.fft.rfft(ir[c], L), L)[:N]
        wet[:, c] = W
    mix = out + wet * 0.55
    # --- master: gentle limiter, fades, normalise
    mix = np.tanh(mix * 1.1)
    fade_in = np.minimum(t_all / 0.05, 1.0)
    fade_out = np.minimum((DUR - t_all) / 0.4, 1.0)
    mix *= (fade_in * fade_out)[:, None]
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89
    pcm = (mix * 32767).astype('<i2')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    rms = [round(float(np.sqrt((mix[int(a * SR):int((a + 0.5) * SR)] ** 2).mean())), 3) for a in np.arange(0, DUR - 0.5, 0.5)]
    return {"path": out_path, "seconds": round(N / SR, 2), "peak": round(float(np.abs(mix).max()), 3), "rms_per_half_second": rms}
