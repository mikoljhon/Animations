# mender_v03.py : Mender v03, restyled into the UNEDITED house look (style_mats.py) so the scavenger matches Relic and KillDroid v02.
# Every Mender material now comes from style_mats.worn(): faded, grimy coveralls with sun-bleached shoulders and mud building up toward
# the boots, worn leather and webbing, chipped yellow paint over a dented steel hard hat, a bare-steel mechanical arm with Relic's brass
# caps. The parts that read as toys got real geometry: folds and top-stitched seams on the coveralls, frayed hand-sewn patches, a cargo
# pocket and stitched reinforcement panels, a zipper placket, a dented hat with a tape repair, a ribbed headlamp on a hinged bracket,
# ladder-lock buckles and loose strap tails, flapped pouches with snaps, a frame belt buckle, a combination wrench, hydraulic rams and
# bolted plates on the arm, a three-strand rope, a rolled foam mat, a ribbed jerrycan, a caged lantern, a sprung antenna with a torn flag.
#
# It RESTYLES an existing Mender in place: meshes are rebuilt on the same datablocks and materials keep their names, so the cave copy
# (MenderC_*, which shares the studio copy's meshes) follows automatically. Rig, bone names and body proportions are not touched.
# Run headless, one Blender at a time (renders go through run_gated.ps1):
#   blender -b relic_v07_mender.blend      --python mender_v03.py -- build mender_v03.blend         studio file -> new file
#   blender -b relic_v08_mender_cave.blend --python mender_v03.py -- build mender_v03_cave.blend    cave file, both copies -> new file
#   blender -b mender_v03.blend            --python mender_v03.py -- turn v03     turnaround + close-ups under the cast light
#   blender -b relic_v07_mender.blend      --python mender_v03.py -- turn v02     the same shots of the old look (before / after)
#   blender -b killdroid_v02.blend         --python mender_v03.py -- lineup       Mender v03, Relic, KillDroid v02 at true scale
#   blender -b mender_v03_cave.blend       --python mender_v03.py -- cave         the dug-out two-shot in the cave with the new look
import bpy, bmesh, math, os, sys, types
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

_g = globals()
for _f in ("mender_build.py", "mender_scene.py", "mender_pose.py", "style_mats.py"):
    exec(compile(open(os.path.join(r"C:\ai\unedited", _f), encoding="utf-8").read(), _f, "exec"), _g)

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = ARGS[0] if ARGS else "dry"
REND = os.path.join(OUT_DIR, "renders")
CPRE = "MenderC_"
SCV = np.asarray(SC, float)
PROTECTED = {"relic_v07_mender.blend", "relic_v08_mender_cave.blend", "killdroid_v02.blend", "fight_v07.blend"}
SUIT_SLOTS = ['coverall', 'patch', 'pad', 'patch2', 'patch3', 'thread', 'thread2', 'smetal']
SI = {k: i for i, k in enumerate(SUIT_SLOTS)}


# ================================================================== small geometry helpers (design frame, metres, before SC)
def gauss(x, c, w):
    return np.exp(-((np.asarray(x, float) - c) / w) ** 2)


def unit_rows(A):
    A = np.asarray(A, float)
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)


def catmull(P, n_out):
    """uniform Catmull-Rom curve through the rows of P (any number of columns), n_out samples"""
    P = np.asarray(P, float)
    m = len(P)
    Q = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    t = np.linspace(0.0, m - 1.0, n_out)
    i = np.minimum(t.astype(int), m - 2)
    u = (t - i)[:, None]
    p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
    return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3)


def cumlen(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def at_len(P, s, extra=None):
    L = cumlen(P)
    s = np.clip(np.asarray(s, float), 0.0, L[-1])
    i = np.clip(np.searchsorted(L, s, side='right') - 1, 0, len(P) - 2)
    t = ((s - L[i]) / np.maximum(L[i + 1] - L[i], 1e-12))[:, None]
    out = P[i] + (P[i + 1] - P[i]) * t
    if extra is None:
        return out
    return out, extra[i] + (extra[i + 1] - extra[i]) * t


def resample(P, step, extra=None):
    L = cumlen(P)
    n = max(2, int(round(L[-1] / step)) + 1)
    return at_len(P, np.linspace(0.0, L[-1], n), extra)


def smooth_rows(A, k=2, closed=False):
    out = np.asarray(A, float).copy()
    if len(out) < 3:
        return out
    for _ in range(k):
        if closed:
            out = (np.roll(out, 1, 0) + 2 * out + np.roll(out, -1, 0)) / 4.0
        else:
            nxt = out.copy()
            nxt[1:-1] = (out[:-2] + 2 * out[1:-1] + out[2:]) / 4.0
            out = nxt
    return out


def rot_axis(axis, ang):
    x, y, z = unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ring_tube(P, rx, ry, sides, up):
    """the rings of mender_build.tube() (same parallel-transported frame), as an (n, sides, 3) array"""
    P = np.asarray(P, float)
    n = len(P)
    rx = np.broadcast_to(np.asarray(rx, float), (n,))
    ry = np.broadcast_to(np.asarray(ry, float), (n,))
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1)[:, None], 1e-9)
    up = np.asarray(up, float)
    N = up - T[0] * (up @ T[0])
    if np.linalg.norm(N) < 1e-6:
        N = np.cross(T[0], [0.0, 1.0, 0.0])
    N /= np.linalg.norm(N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    R = np.zeros((n, sides, 3))
    for i in range(n):
        if i > 0:
            N = N - T[i] * (N @ T[i])
            N /= np.linalg.norm(N)
        B = np.cross(T[i], N)
        R[i] = P[i] + np.outer(rx[i] * np.cos(ang), N) + np.outer(ry[i] * np.sin(ang), B)
    return R


def ring_polys(R, P, caps=(True, True)):
    n, sides = R.shape[:2]
    V = R.reshape(-1, 3)
    polys = [(i * sides + k, i * sides + (k + 1) % sides, (i + 1) * sides + (k + 1) % sides, (i + 1) * sides + k)
             for i in range(n - 1) for k in range(sides)]
    extra = []
    if caps[0]:
        ci = len(V) + len(extra)
        extra.append(P[0])
        polys += [(ci, (k + 1) % sides, k) for k in range(sides)]
    if caps[1]:
        ci = len(V) + len(extra)
        extra.append(P[-1])
        o = (n - 1) * sides
        polys += [(ci, o + k, o + (k + 1) % sides) for k in range(sides)]
    if extra:
        V = np.concatenate([V, np.array(extra)])
    return V, polys


def band(P, thick_dir, half_w, half_t, sides=12, ex=4.0, caps=(True, True)):
    """a flat strap or belt along polyline P: rounded-rectangle section, half_w wide, half_t thick along thick_dir (one vector or one per point)"""
    P = np.asarray(P, float)
    n = len(P)
    T = unit_rows(np.gradient(P, axis=0))
    D = np.asarray(thick_dir, float)
    D = np.broadcast_to(D, (n, 3)) if D.ndim == 1 else D
    Tn = unit_rows(D - T * np.einsum('ij,ij->i', D, T)[:, None])
    W = np.cross(Tn, T)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    a = np.sign(c) * np.abs(c) ** (2.0 / ex)
    b = np.sign(s) * np.abs(s) ** (2.0 / ex)
    hw = np.broadcast_to(np.asarray(half_w, float), (n,))
    ht = np.broadcast_to(np.asarray(half_t, float), (n,))
    R = P[:, None, :] + (hw[:, None, None] * a[None, :, None]) * W[:, None, :] + (ht[:, None, None] * b[None, :, None]) * Tn[:, None, :]
    return ring_polys(R, P, caps)


SGN8 = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
BOX_P = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def boxes(C, X, Y, Z, size):
    """many small boxes at once: centres C (k,3), right-handed unit axes X, Y, Z (k,3), full size (3,) or (k,3)"""
    C = np.asarray(C, float).reshape(-1, 3)
    k = len(C)
    if k == 0:
        return np.zeros((0, 3)), []
    S = np.broadcast_to(np.asarray(size, float), (k, 3)) / 2.0
    V = (C[:, None, :] + SGN8[None, :, 0:1] * (S[:, None, 0:1] * X[:, None, :]) + SGN8[None, :, 1:2] * (S[:, None, 1:2] * Y[:, None, :])
         + SGN8[None, :, 2:3] * (S[:, None, 2:3] * Z[:, None, :]))
    polys = [tuple(int(i) + 8 * j for i in q) for j in range(k) for q in BOX_P]
    return V.reshape(-1, 3), polys


def dashes(P, N, pitch, dash, wdt=0.0011, hgt=0.0008, lift=0.0002, closed=False, jit=0.0, rng=None, phase=0.5):
    """thread dashes (a running stitch) along polyline P lying on a surface with normals N"""
    P, N = np.asarray(P, float), np.asarray(N, float)
    if closed:
        P, N = np.concatenate([P, P[:1]]), np.concatenate([N, N[:1]])
    L = cumlen(P)
    if L[-1] < pitch:
        return np.zeros((0, 3)), []
    s = np.arange(pitch * phase, L[-1] - 0.3 * dash, pitch)
    C, Nn = at_len(P, s, N)
    T = unit_rows(at_len(P, np.minimum(s + 0.002, L[-1])) - at_len(P, np.maximum(s - 0.002, 0.0)))
    Nn = unit_rows(Nn - T * np.einsum('ij,ij->i', Nn, T)[:, None])
    if jit > 0 and rng is not None:
        for i in range(len(T)):
            T[i] = rot_axis(Nn[i], rng.uniform(-jit, jit)) @ T[i]
        C = C + np.cross(Nn, T) * rng.uniform(-0.0005, 0.0005, (len(C), 1))
    X = np.cross(T, Nn)
    return boxes(C + Nn * (lift + hgt / 2), X, T, Nn, (wdt, dash, hgt))


def ribbon(P, Nr, half_w=0.0030, hgt=0.0011, sink=0.0009):
    """a raised seam welt along P on a surface with normals Nr: its edges dip into the cloth"""
    P, Nr = np.asarray(P, float), unit_rows(Nr)
    T = unit_rows(np.gradient(P, axis=0))
    Bv = unit_rows(np.cross(Nr, T))
    a = np.array([-1.0, -0.55, 0.0, 0.55, 1.0]) * half_w
    b = np.array([-sink, 0.72 * hgt, hgt, 0.72 * hgt, -sink])
    G = P[:, None, :] + a[None, :, None] * Bv[:, None, :] + b[None, :, None] * Nr[:, None, :]
    n, m = G.shape[:2]
    polys = [(i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1) for i in range(n - 1) for j in range(m - 1)]
    return G.reshape(-1, 3), polys


def lathe_on(profile, center, axis, sides=24, y_hint=(0.0, 0.0, 1.0)):
    V, polys = lathe(profile, sides)
    a = unit(axis)
    yh = np.asarray(y_hint, float)
    if abs(yh @ a) > 0.95:
        yh = np.array([1.0, 0.0, 0.0])
    return place(V, center, basis(a, yh)), polys


def dome_cap(center, axis, R, h, sides=28, rim=0.18):
    """Relic's brass joint cap: a shallow dome on a thin rim with a small centre boss"""
    prof = [(0.0, h * 1.06), (R * 0.12, h * 1.06), (R * 0.16, h * 0.98), (R * 0.40, h * 0.93), (R * 0.62, h * 0.80), (R * 0.80, h * 0.58),
            (R * 0.93, h * 0.34), (R, h * rim), (R, 0.0), (R * 0.6, -0.0005), (0.0, -0.0005)]
    return lathe_on(prof[::-1], center, axis, sides)


def hexbolt(center, axis, r=0.0042, h=0.0028):
    a = unit(axis)
    return cyl(np.asarray(center, float) - a * 0.0008, np.asarray(center, float) + a * h, r, r * 0.96, sides=6)


def frame_on(n, t):
    """right-handed (X, Y=t, Z=n) for something lying on a surface with normal n, running along t"""
    n = unit(n)
    t = unit(np.asarray(t, float) - n * (np.asarray(t, float) @ n))
    return np.cross(t, n), t, n


def ladder_lock(center, n, t, along=0.020, across=0.034, thick=0.0038, bar=0.0034):
    """a ladder-lock strap buckle lying on a surface: rectangular frame with two cross bars"""
    X, Y, Z = frame_on(n, t)
    R = np.stack([X, Y, Z], 1)
    c = np.asarray(center, float) + Z * thick * 0.5
    items = []
    for sx in (-1, 1):
        items.append(rbox(c + X * sx * (across / 2 - bar / 2), (bar, along, thick), 0.0012, R, n=1))
    for sy in (-1, 0, 1):
        items.append(rbox(c + Y * sy * (along / 2 - bar / 2) * (1.0 if sy else 0.0), (across, bar * (0.8 if sy == 0 else 1.0), thick * 0.9), 0.0012, R, n=1))
    return merge(items)


def merge_items(items):
    """[(V, polys, key)] -> one (V, polys) per key, keeping order"""
    out = {}
    for V, P, k in items:
        out.setdefault(k, []).append((V, P))
    return [(*merge(v), k) for k, v in out.items()]


# ================================================================== the coverall: denser tubes with real folds
def _radial(R, P):
    return unit_rows(R - P[:, None, :])


def fold_torso(R, P):
    rad = _radial(R, P)
    x, z = R[..., 0], R[..., 2]
    th = np.arctan2(rad[..., 1], rad[..., 0])
    front = np.clip(-rad[..., 1], 0.0, 1.0)
    d = np.zeros(R.shape[:2])
    # gathers where the belt cinches the cloth, above and below it, and blousing over the belt
    g = gauss(z, 1.088, 0.030) + 0.8 * gauss(z, 0.988, 0.026)
    d += 0.0034 * g * np.sin(23.0 * th + 0.9 * np.sin(3.0 * th + 0.7))
    d += 0.0050 * gauss(z, 1.095, 0.032)
    # drag lines from the armpits
    for sx in (1.0, -1.0):
        wa = gauss(z, 1.30, 0.055) * gauss(x * sx, 0.165, 0.045)
        d += 0.0032 * wa * np.sin(2.0 * np.pi * (z + 0.55 * (np.abs(x) - 0.12)) / 0.034 + 1.1 * sx)
    # soft horizontal folds over the stomach
    d += 0.0018 * gauss(z, 1.19, 0.05) * front * np.sin(2.0 * np.pi * z / 0.045 + 0.4 * np.sin(2 * th))
    return R + d[..., None] * rad


def fold_leg(R, P, sx, seed):
    ph = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 8)
    rad = _radial(R, P)
    z = R[..., 2]
    th = np.arctan2(rad[..., 1], rad[..., 0] * sx)
    outer = rad[..., 0] * sx
    front = -rad[..., 1]
    d = np.zeros(R.shape[:2])
    # 1. bunching stacked above the elastic cuff (the hem itself is held by the cuff)
    w = np.clip((0.38 - z) / 0.16, 0.0, 1.0) ** 1.1 * np.clip((z - 0.188) / 0.022, 0.0, 1.0)
    wave = z + 0.010 * np.sin(2.0 * th + ph[0]) + 0.005 * np.sin(5.0 * th + ph[1])
    f = np.sin(2.0 * np.pi * wave / 0.038)
    d += 0.0062 * w * (np.maximum(f, 0.0) ** 0.8 - 0.30)
    # 2. creases behind the knee, a bagged knee in front
    d += 0.0040 * gauss(z, 0.50, 0.055) * np.clip(-front, 0.0, 1.0) ** 1.3 * np.sin(2.0 * np.pi * (z + 0.006 * np.sin(3.0 * th + ph[2])) / 0.027)
    d += 0.0042 * gauss(z, 0.505, 0.06) * np.clip(front, 0.0, 1.0) ** 2
    # 3. drag lines from the crotch across the inner front of the thigh
    xo = (R[..., 0] - P[:, None, 0]) * sx
    wc = gauss(z, 0.80, 0.07) * np.clip(0.7 * front - 0.5 * outer, 0.0, 1.0) ** 1.2
    d += 0.0036 * wc * np.sin(2.0 * np.pi * (z - 0.9 * xo) / 0.043 + ph[3])
    # 4. long soft drape folds down the shin and the thigh
    d += 0.0020 * np.sin(4.0 * th + ph[4]) * gauss(z, 0.31, 0.10)
    d += 0.0015 * np.sin(3.0 * th + ph[5]) * gauss(z, 0.72, 0.12)
    return R + d[..., None] * rad


def fold_arm(R, P, sx, el_u, seed, cuff):
    ph = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 6)
    rad = _radial(R, P)
    L = cumlen(P)
    u = (L / L[-1])[:, None] * np.ones(R.shape[1])[None, :]
    sl = u * L[-1]
    th = np.linspace(0, 2 * np.pi, R.shape[1], endpoint=False)[None, :] * np.ones(R.shape[0])[:, None]
    d = np.zeros(R.shape[:2])
    inner = unit((-0.30 * sx, -1.0, 0.0))
    d += 0.0040 * gauss(u, el_u, 0.065) * np.clip(rad @ inner, 0.0, 1.0) ** 1.2 * np.sin(2.0 * np.pi * (sl + 0.006 * np.sin(2.0 * th + ph[0])) / 0.022)
    d += 0.0030 * gauss(u, el_u, 0.06) * np.clip(rad @ np.array([0.0, 1.0, 0.0]), 0.0, 1.0) ** 2
    if cuff:
        wend = np.clip((u - 0.80) / 0.17, 0.0, 1.0) * np.clip((0.985 - u) / 0.025, 0.0, 1.0)
        f = np.sin(2.0 * np.pi * (sl + 0.005 * np.sin(3.0 * th + ph[1])) / 0.026)
        d += 0.0042 * wend * (np.maximum(f, 0.0) ** 0.8 - 0.30)
    d += 0.0026 * gauss(u, 0.16, 0.07) * np.clip(rad @ np.array([-sx, 0.0, 0.0]), 0.0, 1.0) * np.sin(2.0 * np.pi * (0.8 * sl + 0.3 * R[..., 2]) / 0.03 + ph[2])
    d += 0.0015 * np.sin(3.0 * th + ph[3]) * gauss(u, 0.35, 0.12)
    return R + d[..., None] * rad


class Ctx:
    """the coverall under construction: Suit accumulator, base tubes, BVH of the tubes, polygon -> body region"""
    def __init__(self):
        self.S = Suit()
        self.base = []           # (region, V, polys)
        self.ax = {}             # region -> dense centre path
        self.rings = {}          # region -> ring array
        self.tree, self.regs = None, None
        self.rng = np.random.default_rng(31)

    def add_base(self, region, V, polys):
        self.base.append((region, V, polys))
        j, b, bl = CH[region]
        self.S.host(V, polys, SI['coverall'], j, b, bl)

    def build_tree(self, extra=()):
        Vs, Ps, regs, off = [], [], [], 0
        for rg, V, P in list(self.base) + list(extra):
            Vs.append(V)
            Ps += [tuple(int(i) + off for i in p) for p in P]
            regs += [rg] * len(P)
            off += len(V)
        V = np.concatenate(Vs)
        return BVHTree.FromPolygons([tuple(v) for v in V], Ps), regs


def suit_chains():
    chains = {'torso': ([J['hip_c'], J['pelvis_t'], J['spine_t'], J['chest_t'], J['neck_t']], ['pelvis', 'spine', 'chest', 'neck'], 0.07)}
    for s in ('L', 'R'):
        chains['leg_' + s] = ([J['hip_' + s], J['kn_' + s], J['an_' + s]], ['thigh.' + s, 'shin.' + s], 0.07)
        chains['arm_' + s] = ([J['clav_' + s], J['sh_' + s], J['el_' + s], J['wr_' + s]], ['clavicle.' + s, 'upperarm.' + s, 'forearm.' + s], 0.06)
    CH.update(chains)


def weights_by_region(V, regs):
    V = np.asarray(V, float)
    regs = np.asarray(regs)
    out = {}
    for rg in np.unique(regs):
        m = regs == rg
        j, b, bl = CH[str(rg)]
        for bn, arr in chain_weights(V[m], j, b, bl).items():
            out.setdefault(bn, np.zeros(len(V)))[m] = arr
    return out


def suit_base(ctx):
    """build_suit()'s tubes (same control points and radii) resampled 3-4x denser along and around, then folded"""
    suit_chains()
    z = np.array([0.80, 0.87, 0.94, 1.04, 1.16, 1.29, 1.38, 1.44, 1.48, 1.51])
    rx = np.array([0.150, 0.188, 0.194, 0.178, 0.186, 0.208, 0.224, 0.178, 0.098, 0.090])
    ry = np.array([0.100, 0.116, 0.120, 0.110, 0.113, 0.127, 0.127, 0.103, 0.082, 0.078])
    cy = np.array([0.0, 0.0, 0.0, -0.002, -0.008, -0.013, -0.008, 0.0, 0.006, 0.008])
    C = catmull(np.stack([np.zeros_like(z), cy, z, rx, ry], 1), 46)
    R = fold_torso(ring_tube(C[:, :3], C[:, 3], C[:, 4], 48, (1.0, 0.0, 0.0)), C[:, :3])
    ctx.ax['torso'], ctx.rings['torso'] = C, R
    ctx.add_base('torso', *ring_polys(R, C[:, :3], (True, True)))
    for s, sx in (('L', 1.0), ('R', -1.0)):
        P = np.array([[0.095, 0.0, 0.97], [0.098, -0.003, 0.80], [0.103, -0.010, 0.63], [0.105, -0.015, 0.50],
                      [0.107, -0.005, 0.38], [0.109, 0.008, 0.28], [0.110, 0.014, 0.215], [0.110, 0.015, 0.172]]) * np.array([sx, 1, 1])
        t = np.linspace(0, 1, len(P))
        r = np.array([0.101, 0.099, 0.092, 0.085, 0.077, 0.072, 0.076, 0.082])
        r = r * (1 + 0.025 * np.sin(t * 17 + (0.0 if s == 'L' else 1.3)))
        C = catmull(np.column_stack([P, r]), 34)
        R = fold_leg(ring_tube(C[:, :3], C[:, 3], C[:, 3] * 1.02, 40, (1.0, 0.0, 0.0)), C[:, :3], sx, 11 if s == 'L' else 12)
        ctx.ax['leg_' + s], ctx.rings['leg_' + s] = C, R
        ctx.add_base('leg_' + s, *ring_polys(R, C[:, :3], (False, False)))
    for s, sx in (('L', 1.0), ('R', -1.0)):
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        pts = [sh + np.array([-0.02 * sx, 0, 0.005])] + [lerp(sh, el, u) for u in (0.18, 0.45, 0.75)] + [el]
        rad = [0.046, 0.063, 0.061, 0.056, 0.057]
        fa, far = (0.25, 0.55, 0.80), (0.054, 0.051, 0.0545)
        if s == 'R' and CYBORG_ARM:
            fa, far = (0.22, 0.45), (0.054, 0.060)
        pts += [lerp(el, wr, u) for u in fa]
        rad += list(far)
        C = catmull(np.column_stack([np.array(pts), np.array(rad)]), 32)
        L = cumlen(C[:, :3])
        el_u = L[int(round(4 / (len(pts) - 1) * 31))] / L[-1]
        R = fold_arm(ring_tube(C[:, :3], C[:, 3], C[:, 3], 32, (0.0, 1.0, 0.0)), C[:, :3], sx, el_u, 21 if s == 'L' else 22, cuff=(s == 'R'))
        ctx.ax['arm_' + s], ctx.rings['arm_' + s] = C, R
        ctx.add_base('arm_' + s, *ring_polys(R, C[:, :3], (True, True)))
    ctx.tree, ctx.regs = ctx.build_tree()


# ------------------------------------------------------------------ conformed pieces: patches, pads, pockets, placket
def slab_outline(w, h, M, ns, jag, rng, skew=0.0):
    phi = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    ex = 2.0 / ns
    O0 = np.stack([0.5 * w * np.sign(c) * np.abs(c) ** ex, 0.5 * h * np.sign(s) * np.abs(s) ** ex], 1)
    O0[:, 0] += skew * O0[:, 1]
    O = O0.copy()
    if jag > 0.0:
        a = rng.uniform(0.0, 2.0 * np.pi, 3)
        low = 0.5 + 0.25 * np.sin(3.0 * phi + a[0]) + 0.25 * np.sin(7.0 * phi + a[1])
        dlt = jag * (0.45 * low + 0.55 * rng.uniform(0.0, 1.0, M) ** 1.5)
        O = O0 * (1.0 - dlt / np.maximum(np.linalg.norm(O0, axis=1), 1e-6))[:, None]
    return phi, O0, O


def cast_down(tree, pts, n, up=0.035, maxd=0.09):
    """ray-cast points onto the surface along -n: (hit points, normals facing n, face indices or -1)"""
    S_, N_, F_ = [], [], []
    for p in pts:
        hit = tree.ray_cast(Vector(p + n * up), Vector(-n), maxd)
        if hit[0] is None:
            S_.append(p)
            N_.append(n)
            F_.append(-1)
        else:
            hn = np.array(hit[1])
            S_.append(np.array(hit[0]))
            N_.append(hn if hn @ n > 0 else -hn)
            F_.append(hit[2])
    return np.array(S_), unit_rows(np.array(N_)), np.array(F_)


def conform_slab(tree, c, n, R, w, h, thick, M=56, rings=(1.0, 0.86, 0.68, 0.48, 0.26), ns=8.0, jag=0.0, rng=None,
                 edge_thin=0.5, crown=0.0, lift=0.0006, sink=0.0020, peel=None, skew=0.0):
    """a flat piece of cloth or leather (w x h, rounded corners) laid onto a surface: every vertex is ray-cast along -n, so it follows
    folds and curvature; the rim is a short wall dipping into the cloth (or, under a peeled corner, just the piece's own thickness)"""
    rng = rng or np.random.default_rng(0)
    n = unit(n)
    phi, O0, O = slab_outline(w, h, M, ns, jag, rng, skew)
    pts2d = np.concatenate([O * f for f in rings] + [np.zeros((1, 2))])
    base = c[None, :] + pts2d[:, 0:1] * R[:, 0][None, :] + pts2d[:, 1:2] * R[:, 1][None, :]
    S_, N_, F_ = cast_down(tree, base, n)
    off = unit_rows(N_ + n[None, :])
    nr = len(rings)
    prof = np.concatenate([np.full(M, edge_thin if i == 0 else 1.0 + crown * (1.0 - rings[i] ** 2)) for i in range(nr)] + [[1.0 + crown]])
    top_h = lift + thick * prof
    ext = np.zeros(len(top_h))
    if peel is not None:
        pa, amt = peel
        dphi = np.angle(np.exp(1j * (phi - pa)))
        wpe = np.clip(1.0 - np.abs(dphi) / 0.55, 0.0, 1.0) ** 2
        ext = np.concatenate([amt * wpe * rings[i] ** 3 for i in range(nr)] + [[0.0]])
        top_h = top_h + ext
    T = S_ + off * top_h[:, None]
    Bt = S_[:M] - N_[:M] * sink
    lifted = ext[:M] > 0.0004
    Bt[lifted] = T[:M][lifted] - off[:M][lifted] * (thick * edge_thin + 0.0006)
    V = np.concatenate([T, Bt])
    polys = []
    for i in range(nr - 1):
        a0, b0 = i * M, (i + 1) * M
        for k in range(M):
            k2 = (k + 1) % M
            polys.append((a0 + k, a0 + k2, b0 + k2, b0 + k))
    o, ci = (nr - 1) * M, nr * M
    for k in range(M):
        polys.append((o + k, o + (k + 1) % M, ci))
    ntop = len(polys)
    bo = nr * M + 1
    for k in range(M):
        k2 = (k + 1) % M
        polys.append((k, bo + k, bo + k2, k2))
    F = np.concatenate([F_, F_[:M]])
    return dict(V=V, polys=polys, ntop=ntop, T=T, M=M, phi=phi, O0=O0, O=O, c=np.asarray(c, float), n=n, R=R, F=F,
                off=off, thick=thick, edge_thin=edge_thin, S=S_, SN=N_)


def regions_of(ctx, F, fallback):
    return [ctx.regs[f] if f >= 0 else fallback for f in F]


def slab_top_points(sl, O2):
    """points on the slab's own top surface above local 2D points O2"""
    tt = BVHTree.FromPolygons([tuple(v) for v in sl['T']], [tuple(p) for p in sl['polys'][:sl['ntop']]])
    c, n, R = sl['c'], sl['n'], sl['R']
    base = c[None, :] + O2[:, 0:1] * R[:, 0][None, :] + O2[:, 1:2] * R[:, 1][None, :]
    S_, N_, F_ = cast_down(tt, base, n, up=0.05, maxd=0.12)
    return S_, N_


def inset_outline(sl, inset, smooth=2):
    O0 = sl['O0']
    tg = np.roll(O0, -1, 0) - np.roll(O0, 1, 0)
    n2 = unit_rows(np.stack([tg[:, 1], -tg[:, 0]], 1))
    return smooth_rows(O0 - n2 * inset, smooth, closed=True)


def majority_region(ctx, F, fallback, regs=None):
    regs = regs or ctx.regs
    rv = [regs[f] for f in F if f >= 0]
    if not rv:
        return fallback
    vals, cnt = np.unique(rv, return_counts=True)
    return str(vals[np.argmax(cnt)])


def add_piece(ctx, sl, mat, fallback, regs=None):
    """a sewn-on piece follows ONE body region (the one under most of it): per-vertex regions tore pieces that straddle the junction
    of two coverall tubes as soon as the rig bends (the cave crouch pulled a seat patch 16 cm apart)"""
    rg = majority_region(ctx, sl['F'], fallback, regs)
    ctx.S.add(sl['V'], sl['polys'], mat, weights_by_region(sl['V'], [rg] * len(sl['V'])))
    return rg


def add_stitches(ctx, sl, inset, mat, region, pitch=0.0068, dash=0.0040, jit=0.0, wdt=0.0011):
    P, N = slab_top_points(sl, inset_outline(sl, inset))
    V, polys = dashes(P, N, pitch, dash, wdt=wdt, closed=True, jit=jit, rng=ctx.rng)
    if len(V):
        ctx.S.add(V, polys, mat, weights_by_region(V, [region] * len(V)))


def add_fringe(ctx, sl, count, mat, region, lens=(0.0020, 0.0065), long_=2):
    """loose threads where a patch's raw edge is fraying: short fibres lying on the cloth, a couple of longer strays"""
    rng = ctx.rng
    M, R, n = sl['M'], sl['R'], sl['n']
    O = sl['O']
    tg = np.roll(O, -1, 0) - np.roll(O, 1, 0)
    n2 = unit_rows(np.stack([tg[:, 1], -tg[:, 0]], 1))
    items = []
    for j, k in enumerate(rng.choice(M, count, replace=False)):
        start = sl['T'][k] - sl['off'][k] * sl['thick'] * sl['edge_thin'] * 0.5
        d3 = R[:, 0] * n2[k, 0] + R[:, 1] * n2[k, 1]
        d3 = rot_axis(n, rng.uniform(-0.75, 0.75)) @ d3
        L = rng.uniform(*lens) * (2.2 if j < long_ else 1.0)
        side = np.cross(n, d3) * rng.uniform(-0.25, 0.25) * L
        r = rng.uniform(0.00032, 0.00055)
        pts = [start]
        for q in (0.34, 0.67, 1.0):
            p = start + d3 * L * q + side * q * q
            S_, N_, F_ = cast_down(ctx.tree, p[None, :], n)
            pts.append(S_[0] + N_[0] * (r + 0.00015))
        items.append(tube(np.array(pts), np.array([r, r, r * 0.9, r * 0.7]), sides=4, caps=(True, True), up=tuple(n)))
    if items:
        V, P = merge(items)
        ctx.S.add(V, P, mat, weights_by_region(V, [region] * len(V)))


def hit_on(ctx, x, z, front=True, tree=None):
    tree = tree or ctx.tree
    hit = tree.ray_cast(Vector((x, -1.0 if front else 1.0, z)), Vector((0, 1.0 if front else -1.0, 0)))
    if hit[0] is None:
        return None, None
    p, n = np.array(hit[0]), unit(np.array(hit[1]))
    if (n[1] > 0) == front:
        n = -n
    return p, n


def frame_rot(n, y_hint=(0, 0, 1), rot=0.0):
    R = basis(n, y_hint)
    return R @ rotz(rot) if rot else R


def suit_details(ctx):
    rng = ctx.rng
    S = ctx.S
    # ---- the patches: frayed raw edges, hand stitching (some in light thread), loose threads, two corners starting to peel
    PATCHES = [(0.115, 0.66, 'leg_L', (0.088, 0.108), 1, 0.15, True, None),
               (-0.100, 0.32, 'leg_R', (0.078, 0.098), 3, -0.20, True, (2.4, 0.0022)),
               (0.262, 1.29, 'arm_L', (0.078, 0.074), 1, 0.25, True, None),
               (-0.289, 1.20, 'arm_R', (0.068, 0.092), 4, 0.10, True, None),
               (0.352, 0.99, 'arm_L', (0.058, 0.078), 3, -0.15, True, None),
               (-0.140, 1.10, 'torso', (0.092, 0.072), 4, 0.05, True, (5.6, 0.0020)),
               (0.105, 0.775, 'leg_L', (0.090, 0.092), 1, -0.12, False, None),
               (-0.105, 0.72, 'leg_R', (0.088, 0.102), 3, -0.20, False, None),
               (0.060, 1.16, 'torso', (0.078, 0.068), 3, -0.3, True, None)]
    for k, (x, z, reg, (w, h), mat, rot, front, peel) in enumerate(PATCHES):
        p, n = hit_on(ctx, x, z, front)
        if p is None:
            continue
        sl = conform_slab(ctx.tree, p, n, frame_rot(n, (0, 0, 1), rot), w, h, rng.uniform(0.0018, 0.0024), ns=rng.uniform(5.5, 9.0),
                          jag=rng.uniform(0.0020, 0.0030), rng=rng, edge_thin=0.45, peel=peel, skew=rng.uniform(-0.08, 0.08))
        rg = add_piece(ctx, sl, mat, reg)
        add_stitches(ctx, sl, 0.0052, SI['thread2'] if k % 3 != 1 else SI['thread'], rg, pitch=0.0078, dash=0.0044, jit=0.16)
        add_fringe(ctx, sl, int(rng.integers(16, 26)), mat, rg)
    # ---- a cargo pocket on the outer left thigh: gusseted body, flap with two snaps, double stitching
    p, n = hit_on(ctx, 0.190, 0.60)
    if p is not None:
        R = frame_rot(n)
        pk = conform_slab(ctx.tree, p, n, R, 0.086, 0.118, 0.0085, ns=10.0, jag=0.0003, rng=rng, crown=0.65, edge_thin=0.55)
        rg = add_piece(ctx, pk, SI['coverall'], 'leg_L')
        add_stitches(ctx, pk, 0.0042, SI['thread'], rg)
        add_stitches(ctx, pk, 0.0075, SI['thread'], rg)
        tree2, regs2 = ctx.build_tree(extra=[('leg_L', pk['V'], pk['polys'])])
        p2, n2 = hit_on(ctx, 0.190, 0.665, tree=tree2)
        if p2 is not None:
            fl = conform_slab(tree2, p2, n2, frame_rot(n2), 0.094, 0.042, 0.0042, ns=9.0, rng=rng, edge_thin=0.7, lift=0.0004)
            rgf = add_piece(ctx, fl, SI['coverall'], 'leg_L', regs=regs2)
            add_stitches(ctx, fl, 0.0040, SI['thread'], rgf)
            for sxl in (-0.026, 0.026):
                c = fl['c'] + fl['R'][:, 0] * sxl + fl['R'][:, 1] * (-0.008)
                Sx, Nx = slab_top_points(fl, np.array([[sxl, -0.008]]))
                V, P = merge([cyl(Sx[0] - Nx[0] * 0.0008, Sx[0] + Nx[0] * 0.0018, 0.0058, 0.0056, sides=16),
                              ellipsoid(Sx[0] + Nx[0] * 0.0018, (0.0042, 0.0042, 0.0014), 14, 6, basis(Nx[0]))])
                S.add(V, P, SI['smetal'], weights_by_region(V, [rgf] * len(V)))
    # ---- stitched reinforcement panels over the knees and the left elbow (heavy dark canvas)
    for s, sx in (('L', 1.0), ('R', -1.0)):
        kn = J['kn_' + s]
        p, n = hit_on(ctx, kn[0], kn[2] + 0.015)
        if p is None:
            continue
        kp = conform_slab(ctx.tree, p, n, frame_rot(n), 0.120, 0.145, 0.0050, ns=7.0, jag=0.0004, rng=rng, crown=0.25, edge_thin=0.6)
        rg = add_piece(ctx, kp, SI['pad'], 'leg_' + s)
        add_stitches(ctx, kp, 0.0045, SI['thread'], rg)
        add_stitches(ctx, kp, 0.0085, SI['thread'], rg)
        for yy in (-0.024, 0.024):
            xs = np.linspace(-0.050, 0.050, 26)
            P, N = slab_top_points(kp, np.stack([xs, np.full_like(xs, yy)], 1))
            V, Pp = dashes(P, N, 0.0068, 0.0040)
            S.add(V, Pp, SI['thread'], weights_by_region(V, [rg] * len(V)))
    p, n = hit_on(ctx, J['el_L'][0] + 0.03, J['el_L'][2], front=False)
    if p is not None:
        ep = conform_slab(ctx.tree, p, n, frame_rot(n), 0.085, 0.100, 0.0045, ns=7.0, rng=rng, crown=0.2, edge_thin=0.6)
        rg = add_piece(ctx, ep, SI['pad'], 'arm_L')
        add_stitches(ctx, ep, 0.0045, SI['thread'], rg)
    # ---- the zipper placket down the front, with the slider and pull at the top
    p, n = hit_on(ctx, 0.008, 1.192)
    if p is not None:
        R = frame_rot(n)
        pl = conform_slab(ctx.tree, p, n, R, 0.030, 0.385, 0.0028, M=72, rings=(1.0, 0.75, 0.45), ns=14.0, rng=rng, edge_thin=0.7)
        rg = add_piece(ctx, pl, SI['coverall'], 'torso')
        add_stitches(ctx, pl, 0.0045, SI['thread'], rg)
        St, Nt = slab_top_points(pl, np.array([[0.0, 0.180]]))
        Xa, Ya, Za = frame_on(Nt[0], R[:, 1])
        Rz = np.stack([Xa, Ya, Za], 1)
        V1, P1 = rbox(St[0] + Za * 0.0030, (0.011, 0.013, 0.0048), 0.0015, Rz, n=1)
        tab = rot_axis(Xa, -0.25) @ Rz
        V2, P2 = rbox(St[0] + Za * 0.0055 - Ya * 0.013, (0.0075, 0.022, 0.0016), 0.0008, tab, n=1)
        V, P = merge([(V1, P1), (V2, P2)])
        S.add(V, P, SI['smetal'], weights_by_region(V, [rg] * len(V)))
    # ---- top-stitched seams
    seams(ctx)
    # ---- the right sleeve is cut off and hemmed round the mechanical arm
    C = ctx.ax['arm_R']
    P = C[:, :3]
    Tn = unit(P[-1] - P[-2])
    rr = float(np.linalg.norm(ctx.rings['arm_R'][-1] - P[-1], axis=1).mean())
    V, Pp = torus(P[-1] - Tn * 0.004, rr + 0.0006, 0.0046, axis=Tn, nu=40, nv=8)
    S.add(V, Pp, SI['coverall'], weights_by_region(V, ['arm_R'] * len(V)))
    ring_i = len(P) - 3
    Pr = ctx.rings['arm_R'][ring_i]
    Nr = unit_rows(Pr - P[ring_i])
    V, Pp = dashes(Pr + Nr * 0.0003, Nr, 0.0068, 0.0040, closed=True)
    S.add(V, Pp, SI['thread'], weights_by_region(V, ['arm_R'] * len(V)))


def surface_path(ctx, origins, dirs, maxd=0.9, smooth=2, offset=0.0):
    P, N, F = [], [], []
    for o, d in zip(origins, dirs):
        d = unit(d)
        hit = ctx.tree.ray_cast(Vector(o), Vector(d), maxd)
        if hit[0] is None:
            continue
        nn = np.array(hit[1])
        if nn @ d > 0:
            nn = -nn
        P.append(np.array(hit[0]))
        N.append(nn)
        F.append(hit[2])
    if len(P) < 3:
        return None
    P, N = np.array(P), unit_rows(smooth_rows(np.array(N), 3))
    P = smooth_rows(P, smooth) + N * offset
    return P, N, np.array(F)


def add_seam(ctx, path, fallback, half_w=0.0031, hgt=0.0011, rows=(-0.62, 0.62), closed=False, force=None):
    """a welt with double top-stitching along a surface path. Every run of the path that lies on one coverall tube becomes its own
    welt that follows only that tube (where the path crosses onto another tube the welt stops, as the tubes part when the rig bends)"""
    if path is None:
        return
    P, N, F = path
    if closed:
        P, N, F = np.concatenate([P, P[:1]]), np.concatenate([N, N[:1]]), np.concatenate([F, F[:1]])
    reg = [force] * len(F) if force else regions_of(ctx, F, fallback)
    P, N = resample(P, 0.004, N)
    N = unit_rows(N)
    Lr = np.linspace(0.0, 1.0, len(P))
    idx = np.clip(np.round(Lr * (len(reg) - 1)).astype(int), 0, len(reg) - 1)
    regp = [reg[i] for i in idx]
    runs, start = [], 0
    for i in range(1, len(regp) + 1):
        if i == len(regp) or regp[i] != regp[start]:
            if i - start >= 6:
                runs.append((start, i, regp[start]))
            start = i
    for a, b, rg in runs:
        Pr, Nr = P[a:b], N[a:b]
        V, Pp = ribbon(Pr, Nr, half_w, hgt)
        ctx.S.add(V, Pp, SI['coverall'], weights_by_region(V, [rg] * len(V)))
        T = unit_rows(np.gradient(Pr, axis=0))
        Bv = unit_rows(np.cross(Nr, T))
        for r_ in rows:
            Q = Pr + Bv * (r_ * half_w) + Nr * (0.72 * hgt * 0.9)
            V, Pp = dashes(Q, Nr, 0.0064, 0.0038, wdt=0.0010)
            if len(V):
                ctx.S.add(V, Pp, SI['thread'], weights_by_region(V, [rg] * len(V)))


def seams(ctx):
    for s, sx in (('L', 1.0), ('R', -1.0)):
        C = ctx.ax['leg_' + s]
        zc, xc, yc = C[::-1, 2], C[::-1, 0], C[::-1, 1]
        # outer side seam, up over the hip to just under the belt (rays start in the gap between hip and hanging arm, not beyond the arm)
        zs = np.arange(0.993, 0.206, -0.004)
        org = [(np.interp(z, zc, xc) + sx * 0.16, np.interp(z, zc, yc), z) for z in zs]
        add_seam(ctx, surface_path(ctx, org, [(-sx, 0.0, 0.0)] * len(org), maxd=0.16), 'leg_' + s)
        # inseam
        zs = np.arange(0.720, 0.208, -0.004)
        org = [(0.0, np.interp(z, zc, yc), z) for z in zs]
        add_seam(ctx, surface_path(ctx, org, [(sx, 0.0, 0.0)] * len(org)), 'leg_' + s)
        # double hem stitching above the elastic cuff
        for zz in (0.203, 0.216):
            a = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            c0 = np.array([np.interp(zz, zc, xc), np.interp(zz, zc, yc), zz])
            org = [c0 + 0.105 * np.array([math.cos(t), math.sin(t), 0.0]) for t in a]       # start in the gap, not inside the other leg
            pth = surface_path(ctx, org, [(-math.cos(t), -math.sin(t), 0.0) for t in a], maxd=0.06, smooth=1)
            if pth is not None:
                V, Pp = dashes(pth[0] + pth[1] * 0.0002, pth[1], 0.0064, 0.0038, closed=True)
                ctx.S.add(V, Pp, SI['thread'], weights_by_region(V, ['leg_' + s] * len(V)))
        # sleeve seam down the back of the arm
        C = ctx.ax['arm_' + s]
        Pa = C[:, :3]
        L = cumlen(Pa)
        us = np.linspace(0.22, 0.97, 110) * L[-1]
        A = at_len(Pa, us)
        add_seam(ctx, surface_path(ctx, [a + np.array([0.0, 0.14, 0.0]) for a in A], [(0.0, -1.0, 0.0)] * len(A)), 'arm_' + s)
        # armhole: where the sleeve leaves the body
        add_seam(ctx, armhole(ctx, s, sx), 'arm_' + s, closed=True, force='arm_' + s)
        # shoulder seam from the collar to the armhole
        xs = np.arange(0.075, 0.196, 0.004) * sx
        add_seam(ctx, surface_path(ctx, [(x, 0.004, 1.9) for x in xs], [(0.0, 0.0, -1.0)] * len(xs)), 'torso')
    # front yoke across the chest, centre-front and seat seams below the placket and the pack
    xs = np.arange(-0.178, 0.1781, 0.004)
    add_seam(ctx, surface_path(ctx, [(x, -0.5, 1.372) for x in xs], [(0.0, 1.0, 0.0)] * len(xs)), 'torso')
    zs = np.arange(0.812, 0.996, 0.004)
    add_seam(ctx, surface_path(ctx, [(0.0, -0.5, z) for z in zs], [(0.0, 1.0, 0.0)] * len(zs)), 'torso')
    add_seam(ctx, surface_path(ctx, [(0.0, 0.5, z) for z in zs], [(0.0, -1.0, 0.0)] * len(zs)), 'torso')


def armhole(ctx, s, sx):
    """closed loop on the sleeve where it comes out of the torso (analytic torso ellipse, folds ignored)"""
    C = ctx.ax['torso']
    zt, rxt, ryt, cyt = C[:, 2], C[:, 3], C[:, 4], C[:, 1]
    R = ctx.rings['arm_' + s]
    Pa = ctx.ax['arm_' + s][:, :3]
    f = (R[..., 0] / np.interp(R[..., 2], zt, rxt)) ** 2 + ((R[..., 1] - np.interp(R[..., 2], zt, cyt)) / np.interp(R[..., 2], zt, ryt)) ** 2 - 1.0
    pts, nrm = [], []
    for k in range(R.shape[1]):
        col = f[:, k]
        out = np.where(col > 0.0)[0]
        if len(out) == 0:
            continue
        i = int(out[0])
        if i == 0:
            p = R[0, k]
        else:
            t = col[i - 1] / (col[i - 1] - col[i])
            p = R[i - 1, k] + (R[i, k] - R[i - 1, k]) * t
        ii = max(i, 1)
        ax_ = unit(Pa[min(ii + 1, len(Pa) - 1)] - Pa[ii - 1])
        rn = unit(p - Pa[ii])
        pts.append(p + ax_ * 0.0035)
        nrm.append(rn)
    if len(pts) < 8:
        return None
    P, N = smooth_rows(np.array(pts), 2, closed=True), unit_rows(smooth_rows(np.array(nrm), 2, closed=True))
    # snap onto the actual (folded) sleeve surface
    Q, NN, FF = [], [], []
    for p, nn in zip(P, N):
        hit = ctx.tree.ray_cast(Vector(p + nn * 0.02), Vector(-nn), 0.05)
        if hit[0] is None:
            Q.append(p)
            NN.append(nn)
            FF.append(-1)
        else:
            Q.append(np.array(hit[0]))
            NN.append(nn)
            FF.append(hit[2])
    return np.array(Q), np.array(NN), np.array(FF)


# ================================================================== rigid gear (design frame; each object = list of (V, polys, key))
HAT_C = np.array([0.0, -0.003, 1.652])
HAT_PROF = np.array([(0.134, 0.0), (0.136, 0.010), (0.132, 0.040), (0.122, 0.072), (0.102, 0.097), (0.070, 0.116), (0.034, 0.128), (0.0, 0.132)])
HAT_Q = HAT_C + np.array([0.0, 0.0, 0.030])


def hat_profile(n=24):
    C = catmull(HAT_PROF, n)
    C[:, 0] = np.maximum(C[:, 0], 0.0)
    C[-1] = HAT_PROF[-1]
    return C


def dome_point(az_deg, elev, prof=None):
    prof = hat_profile() if prof is None else prof
    t = elev * (len(prof) - 1)
    i = min(int(t), len(prof) - 2)
    u = t - i
    r, z = prof[i] * (1 - u) + prof[i + 1] * u
    a = math.radians(az_deg)
    return HAT_C + np.array([r * math.cos(a), 1.10 * r * math.sin(a), z])


def dent(V, center, rad, depth, elong=1.0, ang=0.0, crease=0.0):
    """press a smooth dent (or, with crease = length, a linear crease) into the dome around the surface point `center`"""
    D = V - HAT_Q
    Dn = unit_rows(D)
    uc = unit(center - HAT_Q)
    t1 = np.cross((0.0, 0.0, 1.0), uc)
    t1 = unit(t1) if np.linalg.norm(t1) > 1e-6 else np.array([1.0, 0.0, 0.0])
    t2 = np.cross(uc, t1)
    t1, t2 = math.cos(ang) * t1 + math.sin(ang) * t2, -math.sin(ang) * t1 + math.cos(ang) * t2
    a = (Dn @ t1) * 0.13
    b = (Dn @ t2) * 0.13
    front = (Dn @ uc) > 0.6
    if crease > 0:
        pr = depth * np.clip(1.0 - np.abs(b) / rad, 0.0, 1.0) ** 1.6 * np.clip(1.0 - (a / (crease / 2)) ** 2, 0.0, 1.0) ** 1.5
    else:
        # sheet steel dents with a crisp edge and a slight raised lip, not a soft bowl
        x = np.sqrt((a / elong) ** 2 + b ** 2) / rad
        pr = depth * np.clip(1.0 - x, 0.0, 1.0) ** 1.5 - 0.10 * depth * np.exp(-((x - 1.0) / 0.22) ** 2)
    pr = pr * front
    return V - Dn * pr[:, None]


def hat_items(rng):
    prof = hat_profile(40)
    V, P = lathe(prof, 144)
    V = place(V, HAT_C, scale=(1.0, 1.10, 1.0))
    V = dent(V, dome_point(-45, 0.55, prof), 0.032, 0.0105)
    V = dent(V, dome_point(140, 0.40, prof), 0.040, 0.0085, elong=1.6, ang=0.5)
    V = dent(V, dome_point(205, 0.22, prof), 0.015, 0.0050)
    V = dent(V, dome_point(-12, 0.30, prof), 0.022, 0.0065, elong=1.3, ang=1.2)
    V = dent(V, dome_point(160, 0.72, prof), 0.0045, 0.0028, ang=0.9, crease=0.060)
    for _ in range(7):
        while True:
            az, el = rng.uniform(-180, 180), rng.uniform(0.15, 0.85)
            if abs(((az + 90 + 180) % 360) - 180) > 28 and abs(az - 90) > 14 and not (5 < az < 50):
                break
        V = dent(V, dome_point(az, el, prof), rng.uniform(0.006, 0.010), rng.uniform(0.0008, 0.0015))
    out = {'Hat': [(V, P, 'hat')]}
    hat_tree = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(p) for p in P])
    # rim: the rolled edge, nicked at the front-left where the big dent is
    Vr, Pr = torus((0.0, -0.003, 1.655), 0.134, 0.0085, nu=72, nv=10)
    Vr[:, 1] = (Vr[:, 1] + 0.003) * 1.10 - 0.003
    az = np.degrees(np.arctan2((Vr[:, 1] + 0.003) / 1.1, Vr[:, 0]))
    nick = gauss(np.angle(np.exp(1j * np.radians(az + 45.0))), 0.0, 0.12)
    Vr[:, :2] -= unit_rows(np.column_stack([Vr[:, 0], Vr[:, 1] + 0.003])) * (0.0025 * nick)[:, None]
    Vr[:, 2] -= 0.0012 * nick
    out['HatRim'] = [(Vr, Pr, 'hat')]
    # crown ridge, rebuilt on the new dome profile
    ys = np.linspace(-0.130, 0.130, 40)
    zs = HAT_C[2] + np.interp(np.abs(ys) / 1.10, prof[::-1, 0], prof[::-1, 1]) + 0.004
    out['HatRidge'] = [(*tube(np.stack([np.zeros_like(ys), ys - 0.003, zs], 1), 0.017, 0.0075, sides=12, caps=(True, True), up=(1.0, 0.0, 0.0)), 'hat')]
    # front peak: a curved visor following the dome base, sloping down, with a rolled lip
    out['HatBrim'] = [peak(-90.0, 128.0, 0.124, 0.196, 0.0045, slope=14.0, z0=0.008)]
    out['HatNapeFlap'] = [peak(90.0, 70.0, 0.126, 0.150, 0.0040, slope=10.0, z0=0.006)]
    # suspension rivets on the sides, and a strip of cloth tape over an old crack
    rv = []
    for az_ in (0.0, 180.0, 35.0, 145.0):
        pnt = dome_point(az_, 0.17, prof)
        nn = unit(pnt - HAT_Q)
        rv.append(ellipsoid(pnt + nn * 0.0004, (0.0058, 0.0058, 0.0026), 14, 6, basis(nn)))
    out['Hat'].append((*merge(rv), 'steel'))
    c = dome_point(27.0, 0.42, prof)
    n = unit(c - HAT_Q)
    up_ = unit(dome_point(27.0, 0.50, prof) - dome_point(27.0, 0.34, prof))
    sl = conform_slab(hat_tree, c, n, basis(n, up_), 0.034, 0.128, 0.0007, M=48, rings=(1.0, 0.7, 0.4), ns=16.0, jag=0.0016,
                      rng=rng, edge_thin=0.7, lift=0.0004, sink=0.0012)
    out['Hat'].append((sl['V'], sl['polys'], 'tape'))
    return out


def peak(az_c, span, r0, r1, thick, slope=14.0, z0=0.008, nu=40, nv=8):
    """a curved hard-hat peak around the dome base (lathe frame, y stretched like the dome), as (V, polys, 'hat')"""
    az = np.radians(np.linspace(az_c - span / 2, az_c + span / 2, nu))
    tt = np.linspace(0.0, 1.0, nv)
    taper = 1.0 - 0.55 * (np.abs(np.linspace(-1, 1, nu)) ** 3)
    top, bot = np.zeros((nu, nv, 3)), np.zeros((nu, nv, 3))
    sl = math.tan(math.radians(slope))
    for i, a in enumerate(az):
        for j, t in enumerate(tt):
            r = r0 + (r1 - r0) * t * taper[i]
            z = z0 - (r - r0) * sl + 0.0035 * max(0.0, (t - 0.85) / 0.15) ** 2 * taper[i]
            p = HAT_C + np.array([r * math.cos(a), 1.10 * r * math.sin(a), z])
            top[i, j] = p + np.array([0, 0, thick / 2])
            bot[i, j] = p - np.array([0, 0, thick / 2])
    V = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3)])
    o = nu * nv
    polys = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a0, b0, c0, d0 = i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1
            polys.append((a0, d0, c0, b0))
            polys.append((o + a0, o + b0, o + c0, o + d0))
    for i in range(nu - 1):
        j = nv - 1
        a0, b0 = i * nv + j, (i + 1) * nv + j
        polys.append((a0, o + a0, o + b0, b0))
        j = 0
        a0, b0 = i * nv + j, (i + 1) * nv + j
        polys.append((b0, o + b0, o + a0, a0))
    for i in (0, nu - 1):
        for j in range(nv - 1):
            a0, d0 = i * nv + j, i * nv + j + 1
            q = (d0, a0, o + a0, o + d0) if i == 0 else (a0, d0, o + d0, o + a0)
            polys.append(q)
    V, polys = orient_outward(V, polys)
    return V, polys, 'hat'


def lamp_items():
    ax = np.array([0.0, -1.0, 0.0])
    c0 = np.array([0.0, -0.140, 1.728])
    out = {}
    housing = [(0.0, 0.0), (0.028, 0.0), (0.0335, 0.003), (0.034, 0.008), (0.034, 0.042), (0.0368, 0.043)]
    bezel = [(0.0368, 0.043), (0.0372, 0.046), (0.0372, 0.0512), (0.0355, 0.0540), (0.0290, 0.0540), (0.0283, 0.0525), (0.0280, 0.0470)]
    items = [(*lathe_on(housing, c0, ax, 36), 'steel'), (*lathe_on(bezel, c0, ax, 36), 'chrome')]
    for d in (0.010, 0.017, 0.024, 0.031):
        items.append((*lathe_on([(0.0338, d - 0.0013), (0.0395, d - 0.0008), (0.0395, d + 0.0008), (0.0338, d + 0.0013)], c0, ax, 36), 'steelDark'))
    items.append((*cyl((0.0, -0.163, 1.7615), (0.0, -0.163, 1.7695), 0.0056, 0.0050, sides=14), 'brass'))
    items.append((*rbox((0.0, -0.163, 1.7715), (0.003, 0.010, 0.004), 0.001), 'steelDark'))
    out['LampBody'] = items
    # hinged bracket: base plate on the dome, two ears, pivot pin, brass thumb screw
    m = [(*rbox((0.0, -0.1335, 1.716), (0.052, 0.011, 0.040), 0.003, n=2), 'steel')]
    for sx in (-1, 1):
        m.append((*rbox((0.0445 * sx, -0.1475, 1.7245), (0.0050, 0.030, 0.034), 0.0018, n=1), 'steel'))
    m.append((*cyl((-0.049, -0.152, 1.728), (0.049, -0.152, 1.728), 0.0034, 0.0034, sides=12), 'steelDark'))
    m.append((*cyl((0.0465, -0.152, 1.728), (0.0545, -0.152, 1.728), 0.0080, 0.0080, sides=16), 'brass'))
    m.append((*dome_cap((0.0545, -0.152, 1.728), (1.0, 0.0, 0.0), 0.0080, 0.0030, sides=16), 'brass'))
    for sx in (-1, 1):
        m.append((*hexbolt((0.016 * sx, -0.139, 1.704), (0.0, -1.0, 0.0), 0.0030, 0.0022), 'steelDark'))
    out['LampMount'] = m
    return out


def shoulder_straps_v03(ctx):
    out = {}
    tree = ctx.tree
    for side, sx in (('L', 1.0), ('R', -1.0)):
        pts, nrm = [], []
        zlast, nlast = 1.45, np.array([0.0, 0.0, 1.0])
        for yv in np.linspace(0.150, -0.060, 16):
            h = tree.ray_cast(Vector((0.105 * sx, yv, 2.2)), Vector((0, 0, -1.0)))
            if h[0] is not None:
                zlast = h[0].z
                nlast = np.array(h[1]) if h[1].z > 0 else -np.array(h[1])
            pts.append((0.105 * sx, yv, zlast))
            nrm.append(nlast)
        for zv in np.linspace(1.395, 1.06, 24):
            xv = (0.105 + (0.085 - 0.105) * (1.395 - zv) / 0.335) * sx
            h = tree.ray_cast(Vector((xv, -1.0, zv)), Vector((0, 1.0, 0)))
            if h[0] is not None:
                nn = np.array(h[1])
                pts.append((xv, h[0].y, zv))
                nrm.append(nn if nn[1] < 0 else -nn)
        P = smooth_rows(np.array(pts), 1)
        N = unit_rows(smooth_rows(np.array(nrm), 4))
        P, N = resample(P, 0.006, N)
        N = unit_rows(N)
        ks = int(np.argmax((P[:, 2] < 1.150) & (P[:, 1] < 0.0)))
        items = []
        pad = P[:ks + 1] + N[:ks + 1] * 0.0072
        items.append((*band(pad, N[:ks + 1], 0.030, 0.0060, sides=16, ex=3.0), 'strap'))
        # binding along both edges of the padded strap
        T = unit_rows(np.gradient(pad, axis=0))
        W = unit_rows(np.cross(N[:ks + 1], T))
        for e in (-1, 1):
            items.append((*tube(pad + W * e * 0.0285, 0.0026, sides=6, caps=(True, True), up=(0.0, 0.0, 1.0)), 'strap'))
        web = P[ks:] + N[ks:] * 0.0026
        items.append((*band(web, N[ks:], 0.0125, 0.0016, sides=10), 'strap'))
        Tk = unit(P[min(ks + 1, len(P) - 1)] - P[ks - 1])
        items.append((*ladder_lock(P[ks] + N[ks] * 0.0040, N[ks], Tk, along=0.022, across=0.034), 'buckle'))
        # the loose tail hanging from the buckle
        down = unit(np.array([0.10 * sx, -0.25, -1.0]))
        tail = np.array([P[ks] + N[ks] * 0.0075 + down * t + N[ks] * 0.004 * t / 0.07 for t in np.linspace(0.0, 0.072, 10)])
        items.append((*band(tail, N[ks], 0.0118, 0.0014, sides=10), 'strap'))
        if side == 'L':
            kd = int(np.argmin(np.abs(P[:, 2] - 1.33) + (P[:, 1] > 0) * 9))
            X_, Y_, Z_ = frame_on(N[kd], unit(P[kd + 1] - P[kd - 1]))
            c = P[kd] + N[kd] * 0.0140
            items.append((*torus(c, 0.0120, 0.0022, axis=Z_, nu=24, nv=6), 'steel'))
            items.append((*rbox(c - Y_ * 0.010, (0.030, 0.012, 0.0030), 0.0010, np.stack([X_, Y_, Z_], 1), n=1), 'strap'))
        out['ShoulderStrap' + side] = ('chest', items)
    # sternum strap with a side-release buckle, in front of the shoulder straps
    xs = np.linspace(-0.100, 0.100, 26)
    pth = surface_path(ctx, [(x, -0.6, 1.30) for x in xs], [(0.0, 1.0, 0.0)] * len(xs), smooth=1)
    items = []
    if pth is not None:
        P, N, F = pth
        P = P + N * 0.0150
        items.append((*band(P, N, 0.0105, 0.0017, sides=10), 'strap'))
        k = len(P) // 2
        X_, Y_, Z_ = frame_on(N[k], unit(P[k + 1] - P[k - 1]))
        Rb = np.stack([X_, Y_, Z_], 1)
        items.append((*rbox(P[k] + Z_ * 0.0032 - Y_ * 0.009, (0.026, 0.020, 0.0085), 0.0028, Rb, n=1), 'buckle'))
        items.append((*rbox(P[k] + Z_ * 0.0032 + Y_ * 0.011, (0.024, 0.020, 0.0070), 0.0026, Rb, n=1), 'buckle'))
    out['ChestStrap'] = ('chest', items)
    return out


# ---- the pack
CYP = 0.205
FLAP_LOW, FLAP_T = 1.275, 0.0055


def pack_bulge(x, z):
    return 0.006 * np.clip(1.0 - (np.asarray(x) / 0.175) ** 2, 0.0, 1.0) * np.clip(1.0 - ((np.asarray(z) - 1.215) / 0.235) ** 2, 0.0, 1.0)


def pack_back(x, z, flap=True):
    """outer surface (y) of the pack's back at (x, z), with the canvas bulge and, where it covers, the lid flap"""
    y = CYP + 0.095 + pack_bulge(x, z)
    if flap:
        on = np.clip((np.asarray(z) - FLAP_LOW) / 0.004, 0.0, 1.0) * np.clip((0.150 - np.abs(np.asarray(x))) / 0.004, 0.0, 1.0)
        y = y + FLAP_T * on
    return y


def pack_items(rng):
    out = {}
    cy = CYP
    V, P = rbox((0.0, cy, 1.215), (0.350, 0.190, 0.470), 0.020, n=7)
    back = V[:, 1] > cy + 0.060
    V[back, 1] += pack_bulge(V[back, 0], V[back, 2])
    side = np.abs(V[:, 0]) > 0.150
    V[side, 0] += np.sign(V[side, 0]) * 0.004 * np.clip(1.0 - ((V[side, 2] - 1.215) / 0.235) ** 2, 0, 1) * np.clip(1.0 - (np.abs(V[side, 1] - cy) / 0.095) ** 2, 0, 1)
    items = [(V, P, 'canvas')]
    # piping round the back panel
    u = np.linspace(0.0, 2 * np.pi, 160, endpoint=False)
    hx, hz, rc = 0.175 - 0.006, 0.235 - 0.006, 0.016
    ox = np.sign(np.cos(u)) * np.minimum(np.abs(np.cos(u)) * 4.0, 1.0)
    pts = []
    for t in u:
        c_, s_ = math.cos(t), math.sin(t)
        x = np.sign(c_) * min(abs(c_) ** 0.25, 1.0) * hx
        z = np.sign(s_) * min(abs(s_) ** 0.25, 1.0) * hz
        pts.append((x, cy + 0.095 - 0.0045, 1.215 + z))
    Pp = np.array(pts)
    Pp[:, 1] = CYP + 0.095 - 0.0045 + pack_bulge(Pp[:, 0], Pp[:, 2]) * 0.5
    Pp = np.concatenate([Pp, Pp[:1]])
    items.append((*tube(Pp, 0.0042, sides=8, caps=(False, False), up=(0.0, 1.0, 0.0)), 'canvas2'))
    out['PackBody'] = ('chest', items)
    out['PackFlap'] = ('chest', [lid_flap()])
    # bedroll: a grey wool blanket cinched by two straps with buckles
    xs = np.linspace(-0.17, 0.17, 60)
    rr = 0.072 * (1.0 - 0.09 * (gauss(xs, 0.10, 0.014) + gauss(xs, -0.10, 0.014)))
    rr *= 1.0 + 0.012 * np.sin(xs * 37.0)
    prof = [(0.0, xs[0] - 0.002)] + [(r, x) for r, x in zip(rr, xs)] + [(0.0, xs[-1] + 0.002)]
    Vb, Pb = lathe_on(prof, (0.0, cy, 1.505), (1.0, 0.0, 0.0), 40, y_hint=(0.0, 0.0, 1.0))
    items = [(Vb, Pb, 'blanket')]
    for sx in (-1.0, 1.0):
        rs = 0.072 * 0.91 + 0.0022
        a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.stack([np.full_like(a, 0.10 * sx), cy + rs * np.cos(a), 1.505 + rs * np.sin(a)], 1)
        ring = np.concatenate([ring, ring[:1]])
        nr = unit_rows(ring - np.array([0.10 * sx, cy, 1.505]))
        items.append((*band(ring, nr, 0.0115, 0.0016, sides=10, caps=(False, False)), 'strap'))
        top = np.array([0.10 * sx, cy, 1.505 + rs + 0.0016])
        items.append((*ladder_lock(top, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), along=0.020, across=0.030), 'buckle'))
        tail = np.array([top + np.array([0.0, 0.004 + 0.06 * t, 0.006 - 0.035 * t * t]) for t in np.linspace(0, 1, 8)])
        items.append((*band(tail, (0.0, 0.0, 1.0), 0.0110, 0.0013, sides=10), 'strap'))
    out['PackRoll'] = ('chest', items)
    Vb, Pb = rbox((0.0, cy, 0.94), (0.300, 0.135, 0.120), 0.014, n=4)
    out['PackBottom'] = ('chest', [(Vb, Pb, 'packbase')])
    # side pockets; the right one with a flap and a snap (the left one is under the jerrycan)
    for sx, nm in ((1.0, 'PackPocketL'), (-1.0, 'PackPocketR')):
        wx = 0.085 if sx > 0 else 0.075
        cx = 0.205 if sx > 0 else -0.200
        Vp, Pq = rbox((cx, cy - 0.005, 1.10), (wx, 0.120, 0.230), 0.012, n=4)
        outer = np.abs(Vp[:, 0] - cx) > wx / 2 - 0.004
        Vp[outer, 0] += sx * 0.004 * np.clip(1.0 - ((Vp[outer, 2] - 1.10) / 0.115) ** 2, 0, 1) * np.clip(1.0 - ((Vp[outer, 1] - cy + 0.005) / 0.06) ** 2, 0, 1)
        it = [(Vp, Pq, 'canvas2')]
        if sx < 0:
            fx = cx - wx / 2
            it.append((*rbox((cx - 0.001, cy - 0.005, 1.2185), (wx + 0.006, 0.126, 0.010), 0.004, n=2), 'canvas2'))
            it.append((*rbox((fx - 0.006, cy - 0.005, 1.192), (0.0055, 0.120, 0.050), 0.0024, n=2), 'canvas2'))
            it.append((*cyl((fx - 0.0085, cy - 0.005, 1.180), (fx - 0.0115, cy - 0.005, 1.180), 0.0058, 0.0055, sides=14), 'steel'))
        out[nm] = ('chest', it)
    # lid straps down the back, over the flap and the upper compression strap, into ladder-locks on the back panel
    for sx, nm in ((1.0, 'PackStrapL'), (-1.0, 'PackStrapR')):
        x = 0.10 * sx
        zs = np.linspace(1.452, 1.172, 46)
        ys = pack_back(x, zs) + np.where(zs > FLAP_LOW + 0.002, 0.0060, 0.0018)
        P = np.stack([np.full_like(zs, x), smooth_rows(ys, 3), zs], 1)
        top = np.array([[x, CYP + 0.06, 1.452 + 0.0035], [x, CYP + 0.085, 1.4505]])
        P = np.concatenate([top, P])
        it = [(*band(P, (0.0, 1.0, 0.0), 0.0130, 0.0016, sides=10), 'strap')]
        bk = np.array([x, pack_back(x, 1.166, False) + 0.0018, 1.166])
        it.append((*ladder_lock(bk, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), along=0.022, across=0.032), 'buckle'))
        anc = np.array([[x, pack_back(x, z, False) + 0.0016, z] for z in np.linspace(1.162, 1.105, 8)])
        it.append((*band(anc, (0.0, 1.0, 0.0), 0.0125, 0.0015, sides=10), 'strap'))
        tail = np.array([[x + 0.004 * sx * t, pack_back(x, 1.166 - 0.075 * t, False) + 0.0055 + 0.004 * t, 1.166 - 0.075 * t] for t in np.linspace(0, 1, 10)])
        it.append((*band(tail, (0.0, 1.0, 0.0), 0.0122, 0.0014, sides=10), 'strap'))
        out[nm] = ('chest', it)
    # horizontal compression straps: the upper one on the flap, the lower one on the panel
    for k, zz in enumerate((1.04, 1.36)):
        xs = np.linspace(-0.168, 0.168, 50)
        ys = pack_back(xs, zz) + 0.0018
        P = np.stack([xs, smooth_rows(ys, 2), np.full_like(xs, zz)], 1)
        out['PackBackStrap%d' % k] = ('chest', [(*band(P, (0.0, 1.0, 0.0), 0.0135, 0.0015, sides=10), 'strap')])
        bx = 0.090
        bk = np.array([bx, pack_back(bx, zz) + 0.0034, zz])
        it = [(*ladder_lock(bk, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), along=0.022, across=0.034), 'buckle')]
        tail = np.array([[bx + 0.004 + 0.055 * t, pack_back(bx + 0.055 * t, zz) + 0.0062, zz - 0.004 - 0.030 * t * t] for t in np.linspace(0, 1, 9)])
        it.append((*band(tail, (0.0, 1.0, 0.0), 0.0128, 0.0013, sides=10), 'strap'))
        out['PackBuckle%d' % k] = ('chest', it)
    # frame bars with rubber end caps
    for sx, nm in ((1.0, 'PackBarL'), (-1.0, 'PackBarR')):
        out[nm] = ('chest', [(*cyl((0.125 * sx, cy + 0.085, 1.10), (0.125 * sx, cy + 0.085, 1.60), 0.009, 0.009, sides=12), 'steel'),
                             (*cyl((0.125 * sx, cy + 0.085, 1.592), (0.125 * sx, cy + 0.085, 1.618), 0.0108, 0.0100, sides=12), 'rubber')])
    # a coil of three-strand rope (six loops, two per old rope object), hung clear of the right pocket
    for k in range(3):
        loops = [rope_ring((-0.268 - 0.0030 * j, cy + 0.015 + 0.004 * math.sin(1.7 * j), 1.02 + 0.011 * j), 0.080 - 0.0016 * j, 0.0105, j)
                 for j in (2 * k, 2 * k + 1)]
        out['Rope%d' % k] = ('chest', loops)
    out['RolledMat'] = ('chest', [rolled_mat((0.0, cy + 0.125, 0.835), 0.052, 0.36)])
    for sx in (-1.0, 1.0):
        rs = 0.0545
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.stack([np.full_like(a, 0.12 * sx), cy + 0.125 + rs * np.cos(a), 0.835 + rs * np.sin(a)], 1)
        ring = np.concatenate([ring, ring[:1]])
        out['RolledMat'][1].append((*tube(ring, 0.0030, sides=8, caps=(False, False), up=(1.0, 0.0, 0.0)), 'rubber'))
    out['Lantern'] = ('chest', lantern_items((0.205, cy + 0.02)))
    out.update(jerrycan_items((0.255, cy + 0.015, 1.215)))
    out.update(antenna_items(rng))
    return out


def lid_flap(nl=28, nx=22):
    """the pack lid: a canvas sheet over the top of the pack and down the back to FLAP_LOW, lower corners rounded"""
    path = []
    for t in np.linspace(0.0, 1.0, 10):
        path.append((CYP - 0.035 + t * 0.110, 1.450))
    for a in np.linspace(0.0, np.pi / 2, 8)[1:]:
        path.append((CYP + 0.075 + 0.020 * math.sin(a), 1.430 + 0.020 * math.cos(a)))
    for z in np.linspace(1.430, FLAP_LOW, 16)[1:]:
        path.append((CYP + 0.095, z))
    path = np.array(path)
    L = cumlen(path)
    ls = np.linspace(0.0, L[-1], nl)
    pa = at_len(path, ls)
    tg = unit_rows(np.gradient(pa, axis=0))          # path is (y, z)
    nrm = unit_rows(np.stack([np.zeros(nl), -tg[:, 1], tg[:, 0]], 1))
    nrm = np.where((nrm[:, 1:2] + nrm[:, 2:3]) < 0, -nrm, nrm)
    rc = 0.035
    outer, inner = np.zeros((nl, nx, 3)), np.zeros((nl, nx, 3))
    for i, l in enumerate(ls):
        hw = 0.150
        if l > L[-1] - rc:
            q = (l - (L[-1] - rc)) / rc
            hw = 0.150 - rc + rc * math.sqrt(max(1.0 - q * q, 0.0))
        for j, u in enumerate(np.linspace(-1.0, 1.0, nx)):
            x = u * hw
            y, z = pa[i, 0], pa[i, 1]
            bul = pack_bulge(x, z) if z < 1.432 else 0.0
            base = np.array([x, y + bul * nrm[i, 1], z])
            sag = 0.0012 * math.sin(3.0 * u + 1.0) * math.sin(math.pi * l / L[-1])
            outer[i, j] = base + nrm[i] * (FLAP_T + sag)
            inner[i, j] = base + nrm[i] * 0.0010
    V = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)])
    o = nl * nx
    polys = []
    for i in range(nl - 1):
        for j in range(nx - 1):
            a0, b0, c0, d0 = i * nx + j, i * nx + j + 1, (i + 1) * nx + j + 1, (i + 1) * nx + j
            polys.append((a0, b0, c0, d0))
            polys.append((o + d0, o + c0, o + b0, o + a0))
    for i in range(nl - 1):
        for j, sgn in ((0, 1), (nx - 1, -1)):
            a0, d0 = i * nx + j, (i + 1) * nx + j
            polys.append((a0, d0, o + d0, o + a0) if sgn > 0 else (d0, a0, o + a0, o + d0))
    for j in range(nx - 1):
        for i, sgn in ((0, 1), (nl - 1, -1)):
            a0, b0 = i * nx + j, i * nx + j + 1
            polys.append((b0, a0, o + a0, o + b0) if sgn > 0 else (a0, b0, o + b0, o + a0))
    V, polys = orient_outward(V, polys)
    return V, polys, 'canvas'


def rope_ring(c, Rm, r, seed):
    """a coil of three-strand rope: the section has three lobes that twist along the ring"""
    nu, nv = 140, 12
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    lobe = np.abs(np.cos(1.5 * (Vv - 4.0 * U)))
    rr = r * (0.78 + 0.22 * lobe ** 0.5)
    wob = 1.0 + 0.02 * np.sin(3.0 * U + seed)
    Rr = Rm * wob + rr * np.cos(Vv)
    X = np.zeros_like(U)
    Y, Z = Rr * np.cos(U), Rr * np.sin(U)
    X = rr * np.sin(Vv)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3) + np.asarray(c, float)
    polys = [(i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv) for i in range(nu) for j in range(nv)]
    V, polys = orient_outward(V, polys)
    return V, polys, 'rope'


def rolled_mat(c, R0, width, turns=3.4, t=0.0062, n=150):
    """a closed-cell foam mat rolled up: a thick spiral sheet, so the spiral shows on both ends"""
    th = np.linspace(0.0, 2 * np.pi * turns, n)
    rad = R0 - t * th / (2 * np.pi)
    ox, oz = rad * np.cos(th), rad * np.sin(th)
    ix, iz = (rad - t * 0.92) * np.cos(th), (rad - t * 0.92) * np.sin(th)
    xs = np.array([-width / 2, width / 2])
    V, polys = [], []
    for x in xs:
        for (a, b) in ((ox, oz), (ix, iz)):
            V += [(x, c[1] + a[k], c[2] + b[k]) for k in range(n)]
    V = np.array(V)
    n2 = n
    oL, iL, oR, iR = 0, n2, 2 * n2, 3 * n2
    for k in range(n2 - 1):
        polys.append((oL + k, oR + k, oR + k + 1, oL + k + 1))
        polys.append((iL + k + 1, iR + k + 1, iR + k, iL + k))
        polys.append((oL + k + 1, iL + k + 1, iL + k, oL + k))
        polys.append((oR + k, iR + k, iR + k + 1, oR + k + 1))
    for e in (0, n2 - 1):
        polys.append((oL + e, iL + e, iR + e, oR + e) if e == 0 else (oR + e, iR + e, iL + e, oL + e))
    V, polys = orient_outward(V, polys)
    return V, polys, 'mat'


def lantern_items(cxy):
    cx, cyy = cxy
    it = []
    it.append((*lathe_on([(0.0, 0.800), (0.038, 0.800), (0.041, 0.804), (0.041, 0.826), (0.036, 0.831)], (cx, cyy, 0.0), (0, 0, 1), 32), 'steel'))
    it.append((*lathe_on([(0.036, 0.889), (0.042, 0.892), (0.043, 0.898), (0.030, 0.915), (0.018, 0.922), (0.0, 0.923)], (cx, cyy, 0.0), (0, 0, 1), 32), 'steel'))
    it.append((*dome_cap((cx, cyy, 0.9225), (0, 0, 1), 0.016, 0.006, sides=20), 'brass'))
    for k in range(5):
        a = 2 * np.pi * k / 5 + 0.3
        p = np.array([cx + 0.0385 * math.cos(a), cyy + 0.0385 * math.sin(a), 0.0])
        it.append((*cyl(p + (0, 0, 0.829), p + (0, 0, 0.892), 0.0022, 0.0022, sides=6), 'steelDark'))
    for zz in (0.846, 0.874):
        it.append((*torus((cx, cyy, zz), 0.0385, 0.0021, nu=32, nv=6), 'steelDark'))
    a = np.linspace(0.0, np.pi, 24)
    bail = np.stack([np.full_like(a, cx) + 0.034 * np.cos(a), np.full_like(a, cyy), 0.915 + 0.030 * np.sin(a)], 1)
    it.append((*tube(bail, 0.0018, sides=6, caps=(True, True), up=(0, 1, 0)), 'steel'))
    return it


def jerrycan_items(c):
    c = np.asarray(c, float)
    out = {}
    V, P = rbox(c, (0.070, 0.150, 0.190), 0.007, n=5)
    it = [(V, P, 'jerry')]
    # pressed X on the outer face
    ang = math.atan2(0.150, 0.170)
    for sg in (-1, 1):
        R = rot_axis((1.0, 0.0, 0.0), sg * ang)
        it.append((*rbox(c + np.array([0.0352, 0.0, 0.0]), (0.004, 0.012, 0.182), 0.0016, R, n=1), 'jerry'))
    # the welded seam bead round the middle
    u = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)
    pts = [(c[0], c[1] + np.sign(math.cos(t)) * min(abs(math.cos(t)) ** 0.2, 1.0) * 0.0758, c[2] + np.sign(math.sin(t)) * min(abs(math.sin(t)) ** 0.2, 1.0) * 0.0958) for t in u]
    pts = np.concatenate([np.array(pts), np.array(pts[:1])])
    it.append((*tube(pts, 0.0028, sides=8, caps=(False, False), up=(1.0, 0.0, 0.0)), 'jerry'))
    out['Jerrycan'] = ('chest', it)
    # three handles on top, spout with a lever cap at the front
    hd = []
    for dy in (0.005, 0.035, 0.065):
        y = c[1] + dy
        loop = np.array([[c[0] - 0.024, y, c[2] + 0.093], [c[0] - 0.024, y, c[2] + 0.108], [c[0] - 0.012, y, c[2] + 0.114],
                         [c[0] + 0.012, y, c[2] + 0.114], [c[0] + 0.024, y, c[2] + 0.108], [c[0] + 0.024, y, c[2] + 0.093]])
        hd.append((*tube(catmull(loop, 16), 0.0040, sides=8, caps=(True, True), up=(0.0, 1.0, 0.0)), 'jerry'))
    out['JerryHandle'] = ('chest', hd)
    sp = c + np.array([0.0, -0.048, 0.093])
    out['JerryCap'] = ('chest', [(*cyl(sp, sp + (0, 0, 0.016), 0.0135, 0.0135, sides=18), 'steel'),
                                 (*cyl(sp + (0, 0, 0.016), sp + (0, 0, 0.026), 0.0160, 0.0155, sides=18), 'steel'),
                                 (*rbox(sp + np.array([0.0, 0.020, 0.024]), (0.010, 0.030, 0.005), 0.0018, rot_axis((1, 0, 0), 0.25), n=1), 'steelDark')])
    for k, zz in enumerate((1.155, 1.275)):
        it = [(*rbox((c[0], c[1], zz), (0.082, 0.160, 0.016), 0.005, n=2), 'strap'),
              (*ladder_lock(np.array([c[0] + 0.041, c[1], zz]), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), along=0.018, across=0.024, thick=0.0034), 'buckle')]
        out['JerryStrap%d' % k] = ('chest', it)
    return out


def antenna_items(rng):
    cy = CYP
    p0 = np.array([-0.145, cy + 0.04, 1.55])
    p1 = np.array([-0.150, cy + 0.07, 2.02])
    d = unit(p1 - p0)
    it = []
    ps = p0 + d * 0.022
    pe = p0 + d * 0.080
    it.append((*cyl(p0 - d * 0.006, ps, 0.0090, 0.0085, sides=14), 'steelDark'))
    th = np.linspace(0.0, 2 * np.pi * 7, 260)
    X, Y, _ = frame_on(d, (1.0, 0.0, 0.0))
    hel = np.array([ps + d * (0.058 * t / th[-1]) + (X * math.cos(t) + Y * math.sin(t)) * 0.0062 for t in th])
    it.append((*tube(hel, 0.0016, sides=6, caps=(True, True), up=tuple(X)), 'steel'))
    it.append((*cyl(ps, pe, 0.0035, 0.0035, sides=10), 'steel'))
    it.append((*cyl(pe, p1, 0.0036, 0.0021, sides=8), 'steel'))
    it.append((*ellipsoid(p1, (0.0042, 0.0042, 0.0042), 10, 6), 'steelDark'))
    clamp_a = np.array([-0.125, cy + 0.085, 1.585])
    clamp_b = p0 + d * ((1.585 - p0[2]) / d[2])
    it.append((*rbox((clamp_a + clamp_b) / 2, (0.012, np.linalg.norm(clamp_b - clamp_a) + 0.012, 0.012), 0.002,
                     basis((0, 0, 1), unit(clamp_b - clamp_a)), n=1), 'steelDark'))
    out = {'Antenna': ('chest', it)}
    # the torn flag: a waving cloth sheet with a ragged trailing edge
    nx, nz = 18, 10
    x0, x1, z0, z1 = -0.1495, -0.226, 1.905, 1.955
    cut = 1.0 - 0.18 * rng.uniform(0, 1, nz) ** 2 - 0.10 * (np.arange(nz) % 3 == 1)
    top, bot = np.zeros((nx, nz, 3)), np.zeros((nx, nz, 3))
    for i, u in enumerate(np.linspace(0, 1, nx)):
        for j, w in enumerate(np.linspace(0, 1, nz)):
            x = x0 + (x1 - x0) * u * cut[j]
            y = cy + 0.071 + 0.006 * math.sin(2 * np.pi * u * 1.6 + 0.8 * w) * u
            z = z1 + (z0 - z1) * w - 0.008 * u * u
            p = np.array([x, y, z])
            top[i, j] = p + np.array([0, -0.0007, 0])
            bot[i, j] = p + np.array([0, 0.0007, 0])
    V = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3)])
    o = nx * nz
    polys = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a0, b0, c0, d0 = i * nz + j, (i + 1) * nz + j, (i + 1) * nz + j + 1, i * nz + j + 1
            polys.append((a0, b0, c0, d0))
            polys.append((o + d0, o + c0, o + b0, o + a0))
    for i in range(nx - 1):
        for j in (0, nz - 1):
            a0, b0 = i * nz + j, (i + 1) * nz + j
            polys.append((a0, o + a0, o + b0, b0) if j == 0 else (b0, o + b0, o + a0, a0))
    for j in range(nz - 1):
        for i in (0, nx - 1):
            a0, d0 = i * nz + j, i * nz + j + 1
            polys.append((d0, o + d0, o + a0, a0) if i == 0 else (a0, o + a0, o + d0, d0))
    V, polys = orient_outward(V, polys)
    out['AntennaFlag'] = ('chest', [(V, polys, 'flag')])
    return out


def belt_items(rng):
    out = {}
    ph = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    path = np.stack([0.196 * np.cos(ph), 0.126 * np.sin(ph), np.full_like(ph, 1.035)], 1)
    nrm = unit_rows(np.stack([np.cos(ph) / 0.196, np.sin(ph) / 0.126, np.zeros_like(ph)], 1))
    pc, nc = np.concatenate([path, path[:1]]), np.concatenate([nrm, nrm[:1]])
    it = [(*band(pc, nc, 0.023, 0.0058, sides=14, ex=5.0, caps=(False, False)), 'leather')]
    for e in (-1, 1):
        q = pc + nc * 0.0055 + np.array([0.0, 0.0, e * 0.0185])
        V, P = dashes(q, nc, 0.0062, 0.0036, wdt=0.0010, hgt=0.0007)
        it.append((V, P, 'rthread'))
    # free tip, coming out from under the buckle frame and through a keeper, with holes
    a = np.linspace(-np.pi / 2 + 0.12, -np.pi / 2 + 0.40, 26)
    tp = np.stack([0.196 * np.cos(a), 0.126 * np.sin(a), np.full_like(a, 1.035)], 1)
    tn = unit_rows(np.stack([np.cos(a) / 0.196, np.sin(a) / 0.126, np.zeros_like(a)], 1))
    tp = tp + tn * 0.0108
    hw = 0.0205 * np.clip((a[-1] - a) / 0.05, 0.35, 1.0) ** 0.5
    it.append((*band(tp, tn, hw, 0.0042, sides=12, ex=4.0), 'leather'))
    for k in range(4):
        i = 6 + 4 * k
        it.append((*cyl(tp[i] + tn[i] * 0.0030, tp[i] + tn[i] * 0.0046, 0.0026, 0.0026, sides=10), 'rubber'))
    ak = -np.pi / 2 + 0.28
    ck = np.array([0.196 * math.cos(ak), 0.126 * math.sin(ak), 1.035])
    nk = unit(np.array([math.cos(ak) / 0.196, math.sin(ak) / 0.126, 0.0]))
    tk = unit(np.array([-0.196 * math.sin(ak), 0.126 * math.cos(ak), 0.0]))
    loop = [ck + nk * (0.0165 * math.cos(t)) + np.array([0, 0, 0.0265 * math.sin(t)]) for t in np.linspace(0, 2 * np.pi, 33)]
    loop = np.array(loop) + nk * 0.0065
    lrad = unit_rows(loop - (ck + nk * 0.0065))
    it.append((*band(loop, lrad, 0.0060, 0.0015, sides=8, caps=(False, False)), 'leather'))
    out['Belt'] = ('pelvis', it)
    # frame buckle with centre bar and prong, standing proud of the belt so the tip can pass under its bar
    c = np.array([0.0, -0.1428, 1.035])
    R = np.eye(3)
    bk = []
    for sx in (-1, 1):
        bk.append(rbox(c + np.array([0.025 * sx, 0.0, 0.0]), (0.0058, 0.0052, 0.054), 0.0018, n=1))
    for sz in (-1, 1):
        bk.append(rbox(c + np.array([0.0, 0.0, 0.0242 * sz]), (0.056, 0.0052, 0.0058), 0.0018, n=1))
    bk.append(rbox(c + np.array([0.004, 0.0, 0.0]), (0.0045, 0.0046, 0.050), 0.0016, n=1))
    bk.append(rbox(c + np.array([0.0145, -0.0030, 0.0]), (0.021, 0.0030, 0.0040), 0.0012, n=1))
    for sx in (-1, 1):
        bk.append(rbox(c + np.array([0.0 + 0.004, 0.0062, 0.0215 * sx]), (0.006, 0.0080, 0.004), 0.0012, n=1))
    out['BeltBuckle'] = ('pelvis', [(*merge(bk), 'steel')])
    for name, cc, sz, face in (('PouchR', (-0.135, -0.120, 1.005), (0.075, 0.050, 0.080), (0, -1, 0)),
                               ('PouchL', (0.115, -0.122, 1.000), (0.065, 0.045, 0.070), (0, -1, 0)),
                               ('PouchBack', (0.0, 0.132, 1.005), (0.130, 0.055, 0.090), (0, 1, 0)),
                               ('PouchSideR', (-0.198, -0.020, 1.0), (0.055, 0.075, 0.085), (-1, 0, 0))):
        out[name] = ('pelvis', pouch(np.array(cc), np.array(sz), np.array(face, float)))
    V, P = rbox((0.118, -0.126, 0.935), (0.108, 0.036, 0.170), 0.006, n=3)
    hol = [(V, P, 'leather')]
    u = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    pts = np.array([(0.118 + np.sign(math.cos(t)) * min(abs(math.cos(t)) ** 0.25, 1) * 0.0505, -0.1445,
                     0.935 + np.sign(math.sin(t)) * min(abs(math.sin(t)) ** 0.25, 1) * 0.0815) for t in u])
    pts = np.concatenate([pts, pts[:1]])
    hol.append((*tube(pts, 0.0024, sides=6, caps=(False, False), up=(0, 1, 0)), 'leather'))
    for sx in (-1, 1):
        hol.append((*ellipsoid((0.118 + 0.040 * sx, -0.1445, 1.008), (0.0032, 0.0016, 0.0032), 10, 6), 'steel'))
    out['Holster'] = ('pelvis', hol)
    # a combination wrench hanging from a loop
    out['WrenchShaft'] = ('pelvis', [(*rbox((-0.165, -0.095, 0.870), (0.016, 0.0055, 0.165), 0.0022, n=2), 'steel'),
                                      (*torus((-0.165, -0.095, 0.962), 0.0135, 0.0040, axis=(0, 1, 0), nu=24, nv=8), 'steel'),
                                      (*band(np.array([[-0.165, -0.103, 0.988], [-0.165, -0.098, 0.975], [-0.165, -0.095, 0.966]]),
                                             (0, 1, 0), 0.0055, 0.0012, sides=8), 'leather')])
    a = np.radians(np.linspace(-30, 210, 30))            # open end: an arc with the gap facing down
    jaw = np.stack([np.full_like(a, -0.165) + 0.0145 * np.cos(a), np.full_like(a, -0.095), 0.776 + 0.0145 * np.sin(a)], 1)
    out['WrenchHead'] = ('pelvis', [(*band(jaw, (0.0, 1.0, 0.0), 0.0045, 0.0028, sides=10), 'steel')])
    return out


def pouch(c, sz, face):
    """a leather pouch with a flap over the top and the outer face, and a snap"""
    it = []
    V, P = rbox(c, sz, 0.007, n=3)
    ax = int(np.argmax(np.abs(face)))
    sg = np.sign(face[ax])
    outer = (V[:, ax] - c[ax]) * sg > sz[ax] / 2 - 0.003
    other = [i for i in (0, 1) if i != ax][0]
    V[outer, ax] += sg * 0.0025 * np.clip(1.0 - ((V[outer, other] - c[other]) / (sz[other] / 2)) ** 2, 0, 1) * np.clip(1.0 - ((V[outer, 2] - c[2]) / (sz[2] / 2)) ** 2, 0, 1)
    it.append((V, P, 'leather'))
    top = c + np.array([0.0, 0.0, sz[2] / 2 + 0.0022])
    it.append((*rbox(top, (sz[0] + 0.005, sz[1] + 0.005, 0.0050), 0.0018, n=1), 'leather'))
    drop = c.copy()
    drop[ax] += sg * (sz[ax] / 2 + 0.0055)
    drop[2] = c[2] + sz[2] * 0.26
    dsz = sz + 0.005
    dsz[ax] = 0.0048
    dsz[2] = sz[2] * 0.50
    it.append((*rbox(drop, dsz, 0.0020, n=1), 'leather'))
    snap = drop.copy()
    snap[2] = c[2] + sz[2] * 0.10
    a = np.zeros(3)
    a[ax] = sg
    it.append((*cyl(snap + a * 0.0016, snap + a * 0.0040, 0.0055, 0.0052, sides=14), 'steel'))
    it.append((*ellipsoid(snap + a * 0.0040, (0.0040, 0.0040, 0.0040) * (1 - 0.6 * np.abs(a)), 12, 6), 'steel'))
    return it


def mech_arm_items():
    out = {}
    el, wr = J['el_R'], J['wr_R']
    a, b = lerp(el, wr, 0.42), wr + (wr - el) * 0.02
    d = unit(b - a)
    L = float(np.linalg.norm(b - a))
    Rb = basis(d, (0, 1, 0))
    X, Y = Rb[:, 0], Rb[:, 1]
    # machined core with ring grooves; a brass collar where the sleeve hem ends
    prof = [(0.0, 0.0), (0.045, 0.0)]
    for t, rr in ((0.12, 0.0466), (0.30, 0.0452), (0.55, 0.0434), (0.85, 0.0412)):
        z = t * L
        prof += [(rr, z - 0.0025), (rr - 0.0018, z - 0.0016), (rr - 0.0018, z + 0.0016), (rr, z + 0.0025)]
    prof += [(0.040, L), (0.0, L)]
    out['ArmCoreR'] = ('forearm.R', [(*lathe_on(prof, a, d, 32), 'steel'),
                                      (*lathe_on([(0.0471, 0.0090), (0.0512, 0.0098), (0.0520, 0.0115), (0.0520, 0.0128), (0.0508, 0.0144), (0.0466, 0.0150)], a, d, 36), 'brass')])
    # bolted plates (the housing sits a little lower than v02's cover so the brass collar shows between it and the sleeve hem)
    c1 = lerp(a, b, 0.40)
    cv = [(*rbox(c1, (0.094, 0.094, 0.090), 0.006, Rb, n=3), 'plate')]
    for f_ in (X, -Y):
        u_ = Y if f_ is X else X
        for su in (-1, 1):
            for sv in (-1, 1):
                p = c1 + f_ * 0.047 + u_ * su * 0.034 + d * sv * 0.032
                cv.append((*hexbolt(p, f_), 'steelDark'))
    out['ArmCoverR'] = ('forearm.R', cv)
    c2 = lerp(a, b, 0.72) + np.array([0.0, -0.030, 0.0])
    cv2 = [(*rbox(c2, (0.070, 0.050, 0.110), 0.005, Rb, n=3), 'plate')]
    for k in range(5):
        cv2.append((*rbox(c2 - Y * 0.0245 + d * (-0.030 + 0.015 * k), (0.046, 0.003, 0.0045), 0.0012, Rb, n=1), 'steelDark'))
    for su in (-1, 1):
        for sv in (-1, 1):
            cv2.append((*hexbolt(c2 - Y * 0.025 + X * su * 0.028 + d * sv * 0.047, -Y, 0.0034, 0.0022), 'steelDark'))
    out['ArmCover2R'] = ('forearm.R', cv2)
    # hydraulic rams: barrel, chrome rod, pivot lugs
    for k, off in enumerate((np.array([0.0, 0.040, 0.0]), np.array([0.034, 0.025, 0.0]))):
        p0, p1 = a + off, b + off
        dd = unit(p1 - p0)
        side = unit(np.cross(dd, off))
        out['PistonR%d' % k] = ('forearm.R', [(*cyl(lerp(p0, p1, 0.04), lerp(p0, p1, 0.60), 0.0092, 0.0092, sides=14), 'steelDark'),
                                              (*cyl(lerp(p0, p1, 0.56), lerp(p0, p1, 0.62), 0.0102, 0.0102, sides=14), 'steel'),
                                              (*cyl(lerp(p0, p1, 0.55), lerp(p0, p1, 0.95), 0.0046, 0.0046, sides=12), 'chrome'),
                                              (*cyl(p0 - side * 0.007, p0 + side * 0.007, 0.0062, 0.0062, sides=12), 'steel'),
                                              (*cyl(p1 - side * 0.007, p1 + side * 0.007, 0.0062, 0.0062, sides=12), 'steel')])
    for k in range(3):
        cc = lerp(a, b, 0.12 + 0.25 * k)
        rr = 0.049 - 0.0035 * k
        it = [(*torus(cc, rr, 0.0048, axis=d, nu=40, nv=8), 'steel'),
              (*rbox(cc + Y * (rr + 0.004), (0.016, 0.010, 0.012), 0.002, Rb, n=1), 'steel'),
              (*hexbolt(cc + Y * (rr + 0.009), Y, 0.0030, 0.0020), 'steelDark')]
        out['ArmBandR%d' % k] = ('forearm.R', it)
    cp = np.array([lerp(a, b, 0.0) + np.array([0.030, 0.032, 0.0]), lerp(a, b, 0.35) + np.array([0.058, 0.044, 0.0]),
                   lerp(a, b, 0.7) + np.array([0.045, 0.042, 0.0]), lerp(a, b, 1.0) + np.array([0.020, 0.030, 0.0])])
    cab = catmull(cp, 24)
    it = [(*tube(cab, 0.0065, sides=10, caps=(True, True), up=(1, 0, 0)), 'rubber')]
    for t in (0.25, 0.5, 0.75):
        i = int(t * (len(cab) - 1))
        it.append((*torus(cab[i], 0.0074, 0.0016, axis=unit(cab[i + 1] - cab[i - 1]), nu=16, nv=6), 'steel'))
    out['ArmCableR'] = ('forearm.R', it)
    # wrist: rotary housing with a pivot boss and Relic's brass caps on both sides; knuckle bar and finger pins capped in brass too
    wr_, R = hand_frame('R')

    def Lp(q):
        return L2D(wr_, R, q)

    xa = R[:, 0]
    ya = R[:, 1]                         # back of the hand: the pivot runs out through the outer side, where the cap shows
    w = [(*lathe_on([(0.0, -0.012), (0.044, -0.012), (0.045, -0.009), (0.045, 0.004), (0.0428, 0.0055), (0.0428, 0.0085), (0.0455, 0.010),
                     (0.046, 0.019), (0.044, 0.022), (0.0, 0.022)], wr_, R[:, 2], 36), 'steel')]
    for sx in (-1, 1):
        cb = Lp((0.0, 0.0, 0.004)) + ya * sx * 0.040
        w.append((*cyl(cb, cb + ya * sx * 0.0085, 0.0215, 0.0215, sides=28), 'steel'))
        w.append((*dome_cap(cb + ya * sx * 0.0085, ya * sx, 0.0195, 0.0068), 'brass'))
    out['MechWristR'] = ('hand.R', w)
    kb = [(*cyl(Lp((-0.040, 0.0, 0.088)), Lp((0.040, 0.0, 0.088)), 0.0108, 0.0108, sides=16), 'steel')]
    for sx in (-1, 1):
        kb.append((*dome_cap(Lp((0.040 * sx, 0.0, 0.088)), xa * sx, 0.0118, 0.0042, sides=20), 'brass'))
    out['MechKnuckleBarR'] = ('hand.R', kb)
    MJ = mech_joints_R()
    for nm in ('mi', 'mm', 'mr', 'mt'):
        jt = MJ[nm]
        pin = [(*cyl(jt[1] - xa * 0.0115, jt[1] + xa * 0.0115, 0.0088, 0.0088, sides=14), 'steelDark')]
        for sx in (-1, 1):
            pin.append((*dome_cap(jt[1] + xa * sx * 0.0115, xa * sx, 0.0080, 0.0028, sides=16), 'brass'))
        out['Mech%sPin' % nm] = ('%s1.R' % nm, pin)
    return out


def rigid_v03(ctx, rng):
    """{object name without the Mender_ prefix: (bone, [(V, polys, key)])} for every rigid part that gets new geometry"""
    out = {}
    for k, v in hat_items(rng).items():
        out[k] = ('head', v)
    for k, v in lamp_items().items():
        out[k] = ('head', v)
    out.update(shoulder_straps_v03(ctx))
    out.update(pack_items(rng))
    out.update(belt_items(rng))
    out.update(mech_arm_items())
    return out


# ================================================================== materials (all through style_mats.worn)
MAT_NAMES = {'coverall': 'Mender_Coverall', 'patch': 'Mender_Patch', 'pad': 'Mender_Pad', 'patch2': 'Mender_PatchOlive',
             'patch3': 'Mender_PatchMustard', 'thread': 'Mender_Thread', 'thread2': 'Mender_ThreadLight', 'smetal': 'Mender_SuitMetal'}


def mats_v03(suit, boots, hand):
    pal = house_palette()
    M = {}
    MUD = (0.042, 0.033, 0.023)          # Mender works in wet ruins: darker mud than the dry grey-brown default
    CL = dict(paint=False, metal=0.0, cloth=1.0, scratches=0.0, chips=0.0, spec=0.35)
    LE = dict(paint=False, metal=0.0, cloth=1.0, scratches=0.0, chips=0.0, spec=0.45)
    LB = dict(LE, spec=0.28)                 # boot and belt leather: a darker-looking matte grain
    # ---- coveralls (skinned: every material on the suit is rest-mapped with ob=suit)
    M['coverall'] = worn('Mender_Coverall', (0.170, 0.050, 0.014), (0.250, 0.085, 0.024), rough=0.88, ob=suit, wear=0.6, grime=1.45, rust=0.05,
                         dust=0.40, mud=2.2, mud_col=MUD, nscale=1.3, bump=0.10, weave=0.55, fade=0.30, fade_col=(0.28, 0.125, 0.055), caked=0.7,
                         folds=0.40, sheen=0.2, **CL)
    for k, nm, b1, b2, fc in (('patch', 'Mender_Patch', (0.022, 0.058, 0.056), (0.040, 0.088, 0.082), (0.10, 0.15, 0.135)),
                              ('patch2', 'Mender_PatchOlive', (0.040, 0.044, 0.024), (0.068, 0.072, 0.042), (0.12, 0.12, 0.08)),
                              ('patch3', 'Mender_PatchMustard', (0.170, 0.110, 0.022), (0.250, 0.165, 0.045), (0.28, 0.23, 0.13))):
        M[k] = worn(nm, b1, b2, rough=0.90, ob=suit, wear=0.8, grime=1.2, rust=0.0, dust=0.45, mud=1.0, mud_col=MUD, nscale=2.0, bump=0.14,
                    weave=0.65, fade=0.35, fade_col=fc, caked=0.5, folds=0.2, sheen=0.2, **CL)
    M['pad'] = worn('Mender_Pad', (0.026, 0.022, 0.018), (0.045, 0.039, 0.031), rough=0.86, ob=suit, wear=1.0, grime=1.2, rust=0.0, dust=0.45,
                    mud=1.2, mud_col=MUD, nscale=1.8, bump=0.20, weave=0.9, fade=0.15, fade_col=(0.11, 0.10, 0.085), caked=0.8, sheen=0.15, **CL)
    M['thread'] = worn('Mender_Thread', (0.046, 0.041, 0.031), (0.070, 0.062, 0.047), rough=0.85, ob=suit, wear=0.0, grime=0.9, rust=0.0, dust=0.35,
                       mud=1.0, mud_col=MUD, nscale=3.0, bump=0.05, **CL)
    M['thread2'] = worn('Mender_ThreadLight', (0.24, 0.21, 0.155), (0.32, 0.285, 0.21), rough=0.85, ob=suit, wear=0.0, grime=1.0, rust=0.0,
                        dust=0.35, mud=1.0, mud_col=MUD, nscale=3.0, bump=0.05, **CL)
    M['smetal'] = worn('Mender_SuitMetal', *pal['steel'], metal=1.0, rough=0.34, ob=suit, paint=False, wear=0.8, scratches=0.5, grime=1.0,
                       rust=0.45, dust=0.35, mud=1.0, mud_col=MUD, nscale=3.5)
    # ---- boots and the gloved left hand (skinned)
    M['boot0'] = worn('Mender_BootUpper', (0.075, 0.038, 0.016), (0.140, 0.074, 0.032), rough=0.66, ob=boots, wear=1.2, grime=0.55, rust=0.0,
                      dust=0.25, mud=0.45, mud_col=MUD, nscale=2.2, bump=0.30, caked=0.3, folds=0.25, **LB)
    M['boot1'] = worn('Mender_BootToeCap', (0.040, 0.022, 0.011), (0.075, 0.042, 0.020), rough=0.62, ob=boots, wear=1.3, grime=0.55, rust=0.0,
                      dust=0.25, mud=0.45, mud_col=MUD, nscale=2.2, bump=0.25, caked=0.3, **LB)
    M['boot2'] = worn('Mender_BootSole', (0.011, 0.011, 0.011), (0.024, 0.023, 0.021), rough=0.90, ob=boots, wear=0.4, grime=1.0, rust=0.0,
                      dust=0.25, mud=1.0, mud_col=MUD, nscale=2.2, bump=0.4, caked=0.9, cloth=1.0, paint=False, scratches=0.0, chips=0.0, spec=0.25)
    M['boot3'] = worn('Mender_BootLace', (0.085, 0.070, 0.045), (0.14, 0.118, 0.078), rough=0.9, ob=boots, wear=0.3, grime=1.4, rust=0.0,
                      dust=0.25, mud=0.3, mud_col=MUD, nscale=3.0, bump=0.2, weave=0.4, **CL)
    M['boot4'] = worn('Mender_BootEyelet', *pal['brass'], metal=0.9, rough=0.38, ob=boots, paint=False, wear=0.8, scratches=0.4, grime=1.2,
                      rust=0.0, dust=0.30, mud=0.6, mud_col=MUD, nscale=4.0)
    M['boot5'] = worn('Mender_BootLeatherDark', (0.040, 0.022, 0.011), (0.075, 0.042, 0.020), rough=0.62, ob=boots, wear=1.3, grime=0.55,
                      rust=0.0, dust=0.25, mud=0.45, mud_col=MUD, nscale=2.2, bump=0.25, caked=0.3, **LB)
    M['hskin'] = worn('Mender_HandSkin', (0.19, 0.105, 0.068), (0.26, 0.150, 0.098), rough=0.55, ob=hand, wear=0.3, grime=1.5, rust=0.0,
                      dust=0.30, mud=0.0, nscale=4.0, bump=0.10, **CL)
    M['hglove'] = worn('Mender_GloveSuede', (0.090, 0.056, 0.029), (0.155, 0.098, 0.054), rough=0.92, ob=hand, wear=1.0, grime=1.4, rust=0.0,
                       dust=0.40, mud=0.0, nscale=3.0, bump=0.35, folds=0.3, sheen=0.25, **CL)
    M['hglove2'] = worn('Mender_GlovePalm', (0.034, 0.021, 0.012), (0.062, 0.040, 0.023), rough=0.7, ob=hand, wear=1.2, grime=1.4, rust=0.0,
                        dust=0.30, mud=0.0, nscale=3.0, bump=0.3, **LE)
    M['hstrap'] = worn('Mender_GloveStrap', (0.030, 0.027, 0.022), (0.055, 0.050, 0.040), rough=0.8, ob=hand, wear=0.8, grime=1.2, rust=0.0,
                       dust=0.35, mud=0.0, nscale=3.0, weave=0.5, **CL)
    M['hnail'] = worn('Mender_Nail', (0.21, 0.14, 0.10), (0.27, 0.19, 0.14), rough=0.45, ob=hand, wear=0.2, grime=1.8, rust=0.0, dust=0.2,
                      mud=0.0, nscale=5.0, bump=0.05, **CL)
    # ---- rigid gear (Object coordinates = rest rig space)
    M['steel'] = worn('Mender_Steel', *pal['steel'], metal=1.0, rough=0.34, paint=False, wear=0.8, scratches=0.8, grime=1.2, rust=0.35, dust=0.45,
                      mud_col=MUD, nscale=2.2)
    M['steelDark'] = worn('Mender_SteelDark', *pal['darksteel'], metal=1.0, rough=0.50, paint=False, wear=0.9, scratches=0.6, grime=1.2, rust=0.30,
                          dust=0.45, mud_col=MUD, nscale=2.2)
    M['plate'] = worn('Mender_Plate', *pal['plate'], metal=1.0, rough=0.42, paint=False, wear=0.75, scratches=1.0, grime=1.4, rust=0.45, dust=0.5,
                      mud_col=MUD, nscale=2.2)
    M['brass'] = worn('Mender_Brass', *pal['brass'], metal=0.9, rough=0.36, paint=False, wear=0.8, scratches=0.5, rust=0.0, dust=0.4, nscale=2.6)
    M['chrome'] = worn('Mender_Chrome', *pal['chrome'], metal=1.0, rough=0.14, paint=False, wear=0.3, scratches=0.4, grime=0.7, rust=0.0, dust=0.25,
                       bump=0.15, nscale=2.6)
    M['rubber'] = worn('Mender_Rubber', *pal['rubber'], rough=0.82, paint=False, cloth=1.0, wear=0.15, scratches=0.0, chips=0.0, grime=0.7, rust=0.0,
                       dust=0.15, mud_col=MUD, nscale=2.0, spec=0.30)
    M['buckle'] = worn('Mender_Buckle', (0.020, 0.020, 0.021), (0.034, 0.034, 0.036), rough=0.45, under=pal['steel'][0], wear=1.3, chips=0.4,
                       scratches=0.6, grime=1.1, rust=0.4, dust=0.4, mud_col=MUD, nscale=3.5)
    M['strap'] = worn('Mender_Strap', (0.030, 0.030, 0.023), (0.052, 0.051, 0.040), rough=0.9, wear=0.8, grime=1.25, dust=0.4, rust=0.05,
                      mud_col=MUD, nscale=2.0, bump=0.12, weave=0.85, fade=0.15, fade_col=(0.10, 0.10, 0.08), folds=0.15, sheen=0.15, **CL)
    M['leather'] = worn('Mender_Leather', (0.072, 0.038, 0.017), (0.130, 0.072, 0.033), rough=0.64, wear=1.1, grime=1.3, dust=0.35, rust=0.0,
                        mud_col=MUD, nscale=2.2, bump=0.30, folds=0.2, **LB)
    M['rthread'] = worn('Mender_ThreadRigid', (0.22, 0.195, 0.145), (0.30, 0.27, 0.20), rough=0.85, wear=0.0, grime=1.0, rust=0.0, dust=0.4,
                        mud_col=MUD, nscale=3.0, bump=0.05, **CL)
    M['canvas'] = worn('Mender_PackCanvas', (0.036, 0.040, 0.023), (0.062, 0.067, 0.040), rough=0.9, wear=0.85, grime=1.3, dust=0.55, rust=0.05,
                       mud=0.9, mud_col=MUD, nscale=1.6, bump=0.15, weave=0.75, fade=0.25, fade_col=(0.11, 0.115, 0.08), caked=0.6, folds=0.35,
                       sheen=0.15, **CL)
    M['canvas2'] = worn('Mender_PackCanvas2', (0.075, 0.060, 0.034), (0.125, 0.100, 0.060), rough=0.9, wear=0.85, grime=1.3, dust=0.55, rust=0.05,
                        mud=0.9, mud_col=MUD, nscale=1.8, bump=0.15, weave=0.75, fade=0.25, fade_col=(0.19, 0.165, 0.115), caked=0.6, folds=0.3,
                        sheen=0.15, **CL)
    M['packbase'] = worn('Mender_PackBase', (0.026, 0.023, 0.018), (0.044, 0.039, 0.030), rough=0.85, wear=1.1, grime=1.3, dust=0.5, rust=0.0,
                         mud=1.0, mud_col=MUD, nscale=1.8, bump=0.2, weave=0.9, caked=0.8, **CL)
    M['blanket'] = worn('Mender_Blanket', (0.050, 0.046, 0.040), (0.080, 0.074, 0.064), rough=0.95, wear=0.6, grime=1.2, dust=0.6, rust=0.0,
                        mud_col=MUD, nscale=1.6, bump=0.25, weave=1.2, fade=0.2, fade_col=(0.13, 0.12, 0.10), folds=0.6, sheen=0.25, **CL)
    M['rope'] = worn('Mender_Rope', (0.105, 0.076, 0.040), (0.165, 0.125, 0.068), rough=0.92, wear=0.5, grime=1.3, dust=0.5, rust=0.0,
                     mud_col=MUD, nscale=3.0, bump=0.30, weave=0.5, sheen=0.25, **CL)
    M['mat'] = worn('Mender_Mat', (0.040, 0.052, 0.060), (0.070, 0.085, 0.095), rough=0.86, wear=0.9, grime=1.2, dust=0.6, rust=0.0, mud=0.8,
                    mud_col=MUD, nscale=2.0, bump=0.45, fade=0.2, fade_col=(0.12, 0.14, 0.15), **CL)
    M['scarf'] = worn('Mender_Scarf', (0.100, 0.018, 0.014), (0.165, 0.038, 0.026), rough=0.92, wear=0.6, grime=1.2, dust=0.35, rust=0.0,
                      mud_col=MUD, nscale=2.0, bump=0.12, weave=0.6, fade=0.2, fade_col=(0.20, 0.09, 0.075), folds=0.4, sheen=0.25, **CL)
    M['flag'] = worn('Mender_Flag', (0.12, 0.022, 0.016), (0.19, 0.045, 0.030), rough=0.9, wear=0.6, grime=1.0, dust=0.5, rust=0.0, nscale=3.0,
                     bump=0.1, weave=0.5, fade=0.6, fade_col=(0.27, 0.15, 0.12), sheen=0.25, **CL)
    M['hat'] = worn('Mender_Hat', (0.30, 0.165, 0.004), (0.38, 0.22, 0.010), rough=0.46, under=pal['steel'][0], wear=1.6, chips=0.7, scratches=0.9,
                    grime=1.2, rust=0.35, dust=0.28, nscale=2.6)
    M['tape'] = worn('Mender_Tape', (0.066, 0.066, 0.063), (0.098, 0.098, 0.094), rough=0.55, wear=0.6, grime=1.4, dust=0.5, rust=0.0, nscale=4.0,
                     bump=0.15, weave=0.5, **CL)
    M['filter'] = worn('Mender_Filter', (0.055, 0.058, 0.035), (0.090, 0.094, 0.058), rough=0.5, under=pal['steel'][0], wear=1.2, chips=0.4,
                       scratches=0.5, grime=1.2, rust=0.25, dust=0.35, nscale=3.0)
    M['lens'] = worn('Mender_Lens', (0.004, 0.005, 0.006), (0.010, 0.012, 0.014), rough=0.06, paint=False, wear=0.0, scratches=0.30, grime=0.45,
                     rust=0.0, dust=0.25, mud=0.0, spec=0.9, nscale=4.0)
    M['jerry'] = worn('Mender_Jerrycan', (0.038, 0.048, 0.029), (0.062, 0.074, 0.045), rough=0.55, under=pal['steel'][0], wear=1.0, chips=0.35,
                      scratches=0.6, grime=1.2, rust=0.6, dust=0.55, mud_col=MUD, nscale=2.0)
    M['cuff'] = worn('Mender_Cuff', (0.100, 0.032, 0.010), (0.150, 0.052, 0.018), rough=0.9, wear=0.7, grime=1.4, dust=0.4, rust=0.0, mud=1.8,
                     mud_col=MUD, nscale=2.0, bump=0.12, weave=0.6, caked=0.8, sheen=0.15, **CL)
    M['skin'] = worn('Mender_Skin', (0.20, 0.11, 0.072), (0.27, 0.155, 0.10), rough=0.55, wear=0.3, grime=1.3, rust=0.0, dust=0.3, mud=0.0,
                     nscale=4.0, bump=0.10, **CL)
    M['port'] = worn('Mender_RegistryPort', (0.70, 0.72, 0.75), (0.85, 0.87, 0.90), metal=1.0, rough=0.16, paint=False, wear=0.0, scratches=0.06,
                     grime=0.12, rust=0.0, dust=0.04, mud=0.0, bump=0.04, nscale=4.0)
    # ---- props: their own coordinates do not start at the floor, so the road-grime band is shifted (zoff)
    M['p_wood'] = worn('Mender_Wood', (0.075, 0.044, 0.021), (0.145, 0.088, 0.044), rough=0.62, wear=1.0, grime=1.4, dust=0.4, rust=0.0, mud=0.4,
                       mud_col=MUD, zoff=0.90, nscale=2.5, bump=0.35, **LE)
    M['p_steel'] = worn('Mender_PropSteel', *pal['steel'], metal=1.0, rough=0.30, paint=False, wear=1.0, scratches=1.0, grime=1.3, rust=0.45,
                        dust=0.4, mud=0.4, mud_col=MUD, zoff=0.90, nscale=2.6)
    M['p_paint'] = worn('Mender_ScannerPaint', (0.058, 0.062, 0.056), (0.088, 0.094, 0.086), rough=0.5, under=pal['steel'][0], wear=1.3, chips=0.5,
                        scratches=0.7, grime=1.2, rust=0.2, dust=0.4, mud=0.0, zoff=0.93, nscale=4.0)
    M['p_rubber'] = worn('Mender_PropRubber', *pal['rubber'], rough=0.8, paint=False, cloth=1.0, wear=0.3, scratches=0.0, chips=0.0, grime=0.8,
                         rust=0.0, dust=0.3, mud=0.0, zoff=0.93, nscale=4.0)
    # ---- light only where there is a lamp or a lens (unchanged)
    M['led'] = glow_mat('Mender_LED', (0.25, 0.65, 1.0), 6.0)
    M['lamp'] = glow_mat('Mender_Lamp', (1.0, 0.92, 0.75), 5.0)
    M['lamp2'] = glow_mat('Mender_Lantern', (1.0, 0.62, 0.25), 2.5)
    M['screen'] = glow_mat('Mender_Screen', (0.35, 1.0, 0.55), 2.2)
    return M


# ================================================================== writing into the existing datablocks
def smooth_all(me):
    try:
        me.shade_smooth()
    except Exception:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))


def write_geometry(me, V, polys, mi):
    me.clear_geometry()
    me.from_pydata((np.asarray(V, float) * SCV).tolist(), [], [tuple(int(i) for i in p) for p in polys])
    me.update()
    me.polygons.foreach_set("material_index", np.asarray(mi, dtype=np.int32))
    smooth_all(me)
    me.update()


def write_rigid(ob, items, M):
    keys, Vs, Ps, mi, off = [], [], [], [], 0
    for V, P, k in items:
        if k not in keys:
            keys.append(k)
        Vs.append(np.asarray(V, float))
        Ps += [tuple(int(i) + off for i in p) for p in P]
        mi += [keys.index(k)] * len(P)
        off += len(V)
    me = ob.data
    me.materials.clear()                 # clearing the slots resets the face material indices, so it must come first
    for k in keys:
        me.materials.append(M[k])
    write_geometry(me, np.concatenate(Vs), Ps, mi)


def set_weights(ob, W):
    for b, arr in W.items():
        vg = ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b)
        q = np.round(np.asarray(arr) * 32).astype(int)
        for lv in np.unique(q):
            if lv <= 0:
                continue
            vg.add([int(i) for i in np.where(q == lv)[0]], lv / 32.0, 'REPLACE')


def rest_parent_inverse(rig, bone):
    b = rig.data.bones[bone]
    return (b.matrix_local @ Matrix.Translation((0.0, b.length, 0.0))).inverted()


def new_part(name, bone, coll):
    """a new rigid part on a bone of the studio rig (and its twin on the cave rig, sharing the mesh)"""
    me = bpy.data.meshes.new("Mender_" + name)
    ob = bpy.data.objects.new("Mender_" + name, me)
    coll.objects.link(ob)
    rig = bpy.data.objects[RIGN]
    ob.parent = rig
    ob.parent_type = 'BONE'
    ob.parent_bone = bone
    ob.matrix_parent_inverse = rest_parent_inverse(rig, bone)
    crig = bpy.data.objects.get(CPRE + "Rig")
    if crig is not None:
        tw = bpy.data.objects.new(CPRE + name, me)
        bpy.data.collections["Mender_Cave"].objects.link(tw)
        tw.parent = crig
        tw.parent_type = 'BONE'
        tw.parent_bone = bone
        tw.matrix_parent_inverse = rest_parent_inverse(crig, bone)
    return ob


def stow_props_rest(rig):
    """pick on the pack, scanner in the holster (studio rig at rest at the origin), with analytic bone inverses"""
    scv = SCV
    B = np.array([-0.17, 0.338, 0.86]) * scv
    T = np.array([0.15, 0.338, 1.78]) * scv
    for name, bone, W in (("Mender_Pick", 'chest', frame4(B, basis(unit(T - B), (0, 1, 0)))),
                          ("Mender_Scanner", 'pelvis', frame4(np.array([0.118, -0.150, 0.925]) * scv, rot_axis((1, 0, 0), math.radians(-8.0))))):
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        ob.parent = rig
        ob.parent_type = 'BONE'
        ob.parent_bone = bone
        ob.matrix_parent_inverse = rest_parent_inverse(rig, bone)
        ob.matrix_basis = W


def restyle():
    log = {}
    st = bpy.data.scenes.get("Mender_Studio")
    orig_scene = None
    try:
        if st is not None and bpy.context.window_manager.windows:
            orig_scene = bpy.context.window_manager.windows[0].scene
            bpy.context.window_manager.windows[0].scene = st
    except Exception as e:
        log['window'] = repr(e)
    rig = bpy.data.objects[RIGN]
    rig.location, rig.rotation_euler = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    for pb in rig.pose.bones:
        pb.location, pb.rotation_quaternion, pb.rotation_euler, pb.scale = (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (1, 1, 1)
    stow_props_rest(rig)
    coll = bpy.data.collections[MC]
    # 1. the coverall
    ctx = Ctx()
    suit_base(ctx)
    suit_details(ctx)
    suit = bpy.data.objects["Mender_Suit"]
    V, polys, mats, W = suit_arrays(ctx.S)
    if not np.isfinite(V).all():
        raise RuntimeError("non-finite suit vertices")
    write_geometry(suit.data, V, polys, mats)
    set_weights(suit, W)
    log['suit'] = {"verts": len(suit.data.vertices), "polys": len(suit.data.polygons), "groups": len(suit.vertex_groups)}
    boots, hand = bpy.data.objects["Mender_Boots"], bpy.data.objects["Mender_HandL"]
    # 2. materials (after the suit, whose new rest bounding box the cloth maps use)
    M = mats_v03(suit, boots, hand)
    suit.data.materials.clear()
    for k in SUIT_SLOTS:
        suit.data.materials.append(M[k])
    suit.data.polygons.foreach_set("material_index", np.asarray(mats, dtype=np.int32))      # clear() reset them
    suit.data.update()
    log['suit_slots_used'] = sorted(set(int(m) for m in mats))
    # 3. rigid gear with new geometry
    parts = rigid_v03(ctx, np.random.default_rng(5))
    made, created = 0, []
    for name, (bone, items) in parts.items():
        if not items:
            continue
        ob = bpy.data.objects.get("Mender_" + name)
        if ob is None:
            ob = new_part(name, bone, coll)
            created.append(name)
        for V_, P_, k in items:
            if not np.isfinite(np.asarray(V_, float)).all():
                raise RuntimeError("non-finite vertices in " + name)
        write_rigid(ob, items, M)
        made += 1
    log['rigid_rebuilt'] = made
    log['rigid_created'] = created
    # 4. material swaps on parts whose geometry stays
    for nm, keys in (("GoggleRimL", ['brass']), ("GoggleRimR", ['brass']), ("MechPalmCoverR", ['plate']), ("LegCuffL", ['cuff']),
                     ("LegCuffR", ['cuff']), ("Pick", ['p_wood', 'p_steel']), ("Scanner", ['p_paint', 'screen', 'p_steel', 'p_rubber'])):
        ob = bpy.data.objects.get("Mender_" + nm)
        if ob is None:
            log.setdefault('missing', []).append(nm)
            continue
        if len(ob.data.materials) != len(keys):
            log.setdefault('slot_mismatch', []).append((nm, len(ob.data.materials)))
        for i, k in enumerate(keys[:len(ob.data.materials)]):
            ob.data.materials[i] = M[k]
    # 5. curvature attributes for the edge wear, per mesh (shared with the cave copy)
    for ob in coll.objects:
        if ob.type != 'MESH':
            continue
        if ob.name == "Mender_Suit":
            edge_attr(ob, lo=8.0, hi=70.0)
        elif ob.name in ("Mender_Boots", "Mender_HandL"):
            edge_attr(ob, lo=14.0, hi=70.0)
        else:
            edge_attr(ob, lo=12.0, hi=55.0)
    # 6. report
    used = set()
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and (ob.name.startswith("Mender_") or ob.name.startswith(CPRE)):
            for s_ in ob.material_slots:
                if s_.material:
                    used.add(s_.material.name)
    log['materials_used'] = len(used)
    log['old_unused'] = sorted(m.name for m in bpy.data.materials if m.name.startswith("Mender_") and m.name not in used)
    log['mesh_objects'] = sum(1 for o in coll.objects if o.type == 'MESH')
    log['verts_total'] = sum(len(o.data.vertices) for o in coll.objects if o.type == 'MESH')
    crig = bpy.data.objects.get(CPRE + "Rig")
    if crig is not None:
        cc = bpy.data.collections.get("Mender_Cave")
        log['cave_copy'] = {"objects": len(cc.objects), "suit_shares_mesh": bpy.data.objects[CPRE + "Suit"].data == suit.data}
    bpy.context.view_layer.update()
    pts = [np.array(o.matrix_world @ Vector(c)) for o in coll.objects if o.type == 'MESH' for c in o.bound_box]
    log['height_m'] = round(max(p[2] for p in pts), 3)
    if orig_scene is not None:
        bpy.context.window_manager.windows[0].scene = orig_scene        # the saved file opens on the scene it opened on
        log['window_scene'] = orig_scene.name
    return log


# ================================================================== studio, shots (the KillDroid v02 cast light, copied from killdroid_v02_build.py)
def gradient_world(sc, name, zenith, horizon, ground):
    w = bpy.data.worlds.new(name)
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sp = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['Generated'], sp.inputs['Vector'])
    cr = nt.nodes.new('ShaderNodeValToRGB')
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = -1.0
    mr.inputs['From Max'].default_value = 1.0
    nt.links.new(sp.outputs['Z'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], cr.inputs['Fac'])
    el = cr.color_ramp.elements
    el[0].position, el[0].color = 0.40, (*ground, 1.0)
    el[1].position, el[1].color = 1.00, (*zenith, 1.0)
    e2 = el.new(0.52)
    e2.color = (*horizon, 1.0)
    bg = nt.nodes.new('ShaderNodeBackground')
    nt.links.new(cr.outputs['Color'], bg.inputs['Color'])
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    sc.world = w


def cast_studio(colls):
    """kd2_studio(dark=False) + final_studio() of killdroid_v02_build.py: same backdrop, the same five area lights, the same gradient world"""
    sc = bpy.data.scenes.new("Cast_Studio")
    sc.render.engine = 'BLENDER_EEVEE'
    for a, v in (("use_raytracing", True), ("ray_tracing_method", 'SCREEN'), ("taa_render_samples", 48), ("shadow_ray_count", 2), ("shadow_step_count", 8)):
        try:
            setattr(sc.eevee, a, v)
        except Exception:
            pass
    sc.view_settings.view_transform = 'AgX'
    try:
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    for c in colls:
        sc.collection.children.link(c)
    setc = bpy.data.collections.new("Cast_Set")
    sc.collection.children.link(setc)
    prof = [(-16.0, 0.0), (0.0, 0.0), (4.5, 0.0)]
    for t in np.linspace(0.0, math.pi / 2, 14)[1:]:
        prof.append((4.5 + 3.5 * math.sin(t), 3.5 - 3.5 * math.cos(t)))
    prof += [(8.0, 9.0), (8.0, 16.0)]
    Vc, Pc = [], []
    for (yy, zz) in prof:
        Vc += [(-28.0, yy, zz), (28.0, yy, zz)]
    for i in range(len(prof) - 1):
        Pc.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
    me = bpy.data.meshes.new("Cast_Backdrop")
    me.from_pydata(Vc, [], Pc)
    me.update()
    for p_ in me.polygons:
        p_.use_smooth = True
    fo = bpy.data.objects.new("Cast_Backdrop", me)
    setc.objects.link(fo)
    fm, _nt, fpb = _new_mat("Cast_BackdropMat")
    fpb.inputs['Base Color'].default_value = (0.21, 0.21, 0.22, 1)
    fpb.inputs['Roughness'].default_value = 0.85
    me.materials.append(fm)
    tgt = (0.0, 0.0, 1.5)
    for nm, sz, en, loc, col in (("Cast_Key", 3.0, 1600, (-3.6, -4.6, 4.6), (1.0, 0.94, 0.86)), ("Cast_Fill", 4.0, 420, (4.2, -3.8, 2.2), (0.80, 0.88, 1.0)),
                                 ("Cast_Rim", 2.0, 1300, (2.8, 4.0, 4.0), (1.0, 0.90, 0.80)), ("Cast_Rim2", 2.0, 900, (-3.2, 3.6, 3.2), (0.85, 0.92, 1.0)),
                                 ("Cast_Top", 1.8, 380, (0.0, -0.6, 6.0), (1.0, 1.0, 1.0))):
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.energy, ld.size, ld.color = en, sz, col
        lo = bpy.data.objects.new(nm, ld)
        setc.objects.link(lo)
        lo.location = loc
        look_at(lo, tgt)
    cd = bpy.data.cameras.new("Cast_Cam")
    cd.clip_start, cd.clip_end = 0.05, 200.0
    cam = bpy.data.objects.new("Cast_Cam", cd)
    setc.objects.link(cam)
    sc.camera = cam
    gradient_world(sc, "Cast_WorldGrad", (0.55, 0.57, 0.60), (0.20, 0.205, 0.21), (0.03, 0.03, 0.032))
    sc.eevee.taa_render_samples = 64
    try:
        bpy.context.window_manager.windows[0].scene = sc
    except Exception:
        pass
    return sc, cam


def shot(sc, cam, name, loc, tgt, lens, res, samples=64):
    cam.location = Vector(loc)
    look_at(cam, tgt)
    cam.data.lens = lens
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.eevee.taa_render_samples = samples
    sc.render.filepath = os.path.join(REND, name)
    sc.frame_set(sc.frame_current)
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print("RENDERED", sc.render.filepath, flush=True)
    return sc.render.filepath


VIEWS = [("front", 0.0), ("q3", 35.0), ("side", 90.0), ("back", 200.0), ("q3left", -40.0)]
CLOSE = [("head", -25.0, (0.0, -0.02, 1.66), (0.10, -1.25, 0.06), 85.0),
         ("arm", 55.0, (-0.36, -0.03, 0.88), (-0.12, -1.15, 0.10), 85.0),
         ("legs", -15.0, (0.0, -0.02, 0.34), (0.05, -1.55, 0.16), 70.0),
         ("patch", -20.0, (0.14, -0.10, 0.62), (0.05, -0.85, 0.05), 85.0),
         ("pack", 200.0, (0.0, 0.20, 1.22), (0.0, -2.0, 0.15), 70.0)]


def turn(tag, only=None):
    rig = bpy.data.objects[RIGN]
    rig.location = (0.0, 0.0, 0.0)
    for pb in rig.pose.bones:
        pb.location, pb.rotation_quaternion, pb.rotation_euler, pb.scale = (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (1, 1, 1)
    stow_props_rest(rig)
    sc, cam = cast_studio([bpy.data.collections[MC]])
    for o in bpy.data.collections[MC].objects:
        if o.type == 'LIGHT':
            o.hide_render = True
    files = []
    for nm, deg in VIEWS:
        if only and nm not in only:
            continue
        rig.rotation_euler = (0.0, 0.0, math.radians(deg))
        bpy.context.view_layer.update()
        files.append(shot(sc, cam, "mender_%s_%s.png" % (tag, nm), (0.0, -6.2, 0.95), (0.0, 0.0, 0.86), 100, (760, 1000)))
    for nm, deg, p, off, lens in CLOSE:
        if only and nm not in only:
            continue
        rig.rotation_euler = (0.0, 0.0, math.radians(deg))
        bpy.context.view_layer.update()
        t = rotz(math.radians(deg)) @ (np.array(p) * SCV)
        files.append(shot(sc, cam, "mender_%s_cu_%s.png" % (tag, nm), tuple(t + np.array(off)), tuple(t), lens, (760, 760)))
    rig.rotation_euler = (0.0, 0.0, 0.0)
    print("TURN", tag, files, flush=True)
    return files


def relax_arms(rig_, out=0.22):
    for pb in rig_.pose.bones:
        if not pb.name.startswith("upperarm."):
            continue
        b = pb.bone
        sx = 1.0 if b.head_local.x > 0 else -1.0
        rest = b.matrix_local.to_3x3()
        cur = (b.tail_local - b.head_local).normalized()
        q = cur.rotation_difference(Vector((sx * out, 0.04, -1.0)).normalized())
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (rest.inverted() @ q.to_matrix() @ rest).to_quaternion()


def lineup():
    """killdroid_v02_build.lineup() with Mender v03: the cast at true scale under the same light (run on killdroid_v02.blend)"""
    kd = bpy.data.objects["KillDroid_Rig"]
    got = {}
    for path, cname in ((os.path.join(OUT_DIR, "fight_v07.blend"), "Relic_AI"), (os.path.join(OUT_DIR, "mender_v03.blend"), "Mender")):
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.collections = [n for n in src.collections if n == cname]
        got[cname] = dst.collections[0]
    sc, cam = cast_studio([bpy.data.collections["KillDroid"]] + list(got.values()))
    rr, mr = bpy.data.objects["Relic_AI_Rig"], bpy.data.objects["Mender_Rig"]
    for r_ in (rr, mr):
        if r_.animation_data:
            r_.animation_data_clear()
        for pb in r_.pose.bones:
            pb.rotation_mode = 'QUATERNION'
            pb.location, pb.rotation_quaternion, pb.scale = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    relax_arms(rr)
    for o in bpy.data.objects:
        if "normals_ref" in o.name:
            o.hide_render = True
        if o.type == 'LIGHT' and o.name.startswith("Mender_"):
            o.hide_render = True
    mr.location, mr.rotation_euler = (-2.75, 0.0, 0.0), (0.0, 0.0, 0.0)
    rr.location, rr.rotation_euler = (-1.35, 0.0, 0.0), (0.0, 0.0, 0.0)
    kd.location, kd.rotation_euler = (0.75, 0.0, 0.0), (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    files = [shot(sc, cam, "cast_lineup_v03_front.png", (-0.9, -12.5, 1.55), (-0.9, 0.0, 1.30), 70, (1600, 1000), 64)]
    for r_ in (mr, rr, kd):
        r_.rotation_euler = (0.0, 0.0, math.radians(30.0))
    bpy.context.view_layer.update()
    files.append(shot(sc, cam, "cast_lineup_v03_q3.png", (-0.9, -12.5, 1.55), (-0.9, 0.0, 1.30), 70, (1600, 1000), 64))
    # a closer pair: Mender beside Relic, same light
    for r_ in (mr, rr, kd):
        r_.rotation_euler = (0.0, 0.0, math.radians(15.0))
    bpy.context.view_layer.update()
    files.append(shot(sc, cam, "cast_lineup_v03_pair.png", (-2.05, -6.2, 1.15), (-2.05, 0.0, 1.00), 70, (1200, 1000), 64))
    print("LINEUP", files, flush=True)


def cave(tag="v03"):
    sc = bpy.data.scenes["Cave"]
    try:
        bpy.context.window_manager.windows[0].scene = sc
    except Exception:
        pass
    cc = bpy.data.objects.get("Cave_Controls")
    if cc is not None and "dug" in cc.keys():
        cc["dug"] = 1.0
        cc.update_tag()
    cd = bpy.data.cameras.new("MV3_Cam")
    cam = bpy.data.objects.new("MV3_Cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    files = []
    for nm, loc, tgt, lens in (("two_shot_dug", (-1.3, -0.9, 1.9), (0.4, 1.2, 1.1), 40.0), ("over_shoulder_dug", (1.0, -0.9, 1.7), (0.2, 1.3, 1.1), 40.0)):
        cam.location = Vector(loc)
        look_at(cam, tgt)
        cd.lens = lens
        cd.clip_start = 0.05
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1600, 900, 100
        sc.render.filepath = os.path.join(REND, "mender_%s_cave_%s.png" % (tag, nm))
        sc.frame_set(sc.frame_current)
        bpy.ops.render.render(write_still=True, scene=sc.name)
        files.append(sc.render.filepath)
        print("RENDERED", sc.render.filepath, flush=True)
    print("CAVE", files, flush=True)


# ================================================================== entry
if MODE in ("build", "dry"):
    src = os.path.basename(bpy.data.filepath)
    log = restyle()
    print("RESTYLE", log, flush=True)
    if MODE == "build":
        out = ARGS[1] if len(ARGS) > 1 else "mender_v03.blend"
        if os.path.basename(out) in PROTECTED:
            raise RuntimeError("refusing to overwrite " + out)
        path = out if os.path.isabs(out) else os.path.join(OUT_DIR, out)
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print("SAVED", path, "from", src, flush=True)
elif MODE == "turn":
    turn(ARGS[1] if len(ARGS) > 1 else "v03", set(ARGS[2].split(",")) if len(ARGS) > 2 else None)
elif MODE == "lineup":
    lineup()
elif MODE == "cave":
    cave(ARGS[1] if len(ARGS) > 1 else "v03")
print("DONE", MODE, flush=True)
