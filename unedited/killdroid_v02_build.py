# killdroid_v02_build.py : KillDroid v02, StarCapital's enforcer redesigned as its own character.
# v01 (killdroid_build.py) followed a concept image credited to Tsvetomir Georgiev too closely: the silhouette, red muscle with black
# plates, a visor bar with two red eyes, a three-clawed steel hand, banded legs and round pauldrons. v02 keeps the role (a 2.85 m enforcer,
# brute strength, one weapon arm: his right) and v01's skeleton: the same joints and bone names (root, pelvis, spine, chest, neck, head,
# clavicle/upperarm/forearm/hand, thigh/shin/foot/toe, ik_foot, ik_knee), so Motion Assist, kd_ctrl.py and the fight scripts keep working.
# Three directions:  A "Reclaimer"  corporate repossession hull (ivory shell, cobalt, slit visor, pile-driver ram)
#                    B "Breaker"    refitted demolition machine (olive paint, engine hump, work-lamp cluster, crusher jaws)
#                    C "Grown"      vat-grown body in a surgical frame (pale flesh, steel ribs and spine, one amber lens, cleaver blade)
# Run headless, one Blender at a time (run_gated.ps1 waits until the PC is free):
#   blender.exe -b --factory-startup --python killdroid_v02_build.py -- check A          build only, print stats (no render)
#   blender.exe -b --factory-startup --python killdroid_v02_build.py -- concept A        concept stills -> renders\kd2_concept_A_{hero,side,head,sil}.png
#   blender.exe -b killdroid_v01.blend --python killdroid_v02_build.py -- silv01         v01 silhouette for the concept board
import bpy, bmesh, math, os, sys, types
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

_g = globals()
_src = open(r"C:\ai\unedited\killdroid_build.py", encoding="utf-8").read()
_cut = _src.index('if __name__ == "__main__" or True:')
exec(compile(_src[:_cut], "killdroid_build.py", "exec"), _g)     # mender helpers, v01 skeleton (J), studio; v01's own build and save are NOT run

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = ARGS[0] if ARGS else "check"
KEY = ARGS[1].upper() if len(ARGS) > 1 else "A"
REND = os.path.join(OUT_DIR, "renders")
S_ = KD_SCALE


# ------------------------------------------------------------------ skeleton: v01's bones without the claw fingers
def bone_defs():
    B = [('root', (0, 0, 0), (0, -0.3, 0), None),
         ('pelvis', J['hip_c'], J['pelvis_t'], 'root'),
         ('spine', J['pelvis_t'], J['spine_t'], 'pelvis'),
         ('chest', J['spine_t'], J['chest_t'], 'spine'),
         ('neck', J['chest_t'], J['neck_t'], 'chest'),
         ('head', J['neck_t'], J['head_t'], 'neck')]
    for s, sx in (('L', 1.0), ('R', -1.0)):
        B += [('clavicle.' + s, J['clav_' + s], J['sh_' + s], 'chest'),
              ('upperarm.' + s, J['sh_' + s], J['el_' + s], 'clavicle.' + s),
              ('forearm.' + s, J['el_' + s], J['wr_' + s], 'upperarm.' + s),
              ('hand.' + s, J['wr_' + s], J['hd_' + s], 'forearm.' + s)]
        B += [('thigh.' + s, J['hip_' + s], J['kn_' + s], 'pelvis'),
              ('shin.' + s, J['kn_' + s], J['an_' + s], 'thigh.' + s),
              ('foot.' + s, J['an_' + s], J['ball_' + s], 'shin.' + s),
              ('toe.' + s, J['ball_' + s], J['toe_' + s], 'foot.' + s),
              ('ik_foot.' + s, J['an_' + s], J['ball_' + s], 'root'),
              ('ik_knee.' + s, np.array([J['kn_' + s][0], -1.2, J['kn_' + s][2]]), np.array([J['kn_' + s][0], -1.2, J['kn_' + s][2] + 0.1]), 'root')]
    return B + list(EXTRA_BONES)


EXTRA_BONES = []      # (name, head, tail, parent): weapon bones of the final design (the Breaker's crusher jaws)


# ------------------------------------------------------------------ shape helpers
def stube(P, rx, ry=None, n=2.0, sides=24, caps=(True, True), up=(1.0, 0.0, 0.0), phase=0.0):
    """tube() with superellipse sections: n=2 ellipse, n=3..5 rounded box (armour shells). rx along the start normal (from `up`)"""
    P = np.asarray(P, float)
    m = len(P)
    rx = np.broadcast_to(np.asarray(rx, float), (m,)).copy()
    ry = rx.copy() if ry is None else np.broadcast_to(np.asarray(ry, float), (m,)).copy()
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1)[:, None], 1e-9)
    up = np.asarray(up, float)
    N = up - T[0] * (up @ T[0])
    if np.linalg.norm(N) < 1e-6:
        N = np.cross(T[0], [0.0, 1.0, 0.0])
    N /= np.linalg.norm(N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False) + phase
    c, s = np.cos(ang), np.sin(ang)
    cx, sy = np.sign(c) * np.abs(c) ** (2.0 / n), np.sign(s) * np.abs(s) ** (2.0 / n)
    rings = []
    for i in range(m):
        if i > 0:
            N = N - T[i] * (N @ T[i])
            N /= np.linalg.norm(N)
        B = np.cross(T[i], N)
        rings.append(P[i] + np.outer(rx[i] * cx, N) + np.outer(ry[i] * sy, B))
    V = np.concatenate(rings)
    polys = [(i * sides + k, i * sides + (k + 1) % sides, (i + 1) * sides + (k + 1) % sides, (i + 1) * sides + k)
             for i in range(m - 1) for k in range(sides)]
    extra = []
    if caps[0]:
        ci = len(V) + len(extra)
        extra.append(P[0])
        polys += [(ci, (k + 1) % sides, k) for k in range(sides)]
    if caps[1]:
        ci = len(V) + len(extra)
        extra.append(P[-1])
        o = (m - 1) * sides
        polys += [(ci, o + k, o + (k + 1) % sides) for k in range(sides)]
    if extra:
        V = np.concatenate([V, np.array(extra)])
    return V, polys


def zshell(ZS, RX, RY, CY=0.0, n=3.0, sides=36, caps=(True, True), CX=0.0):
    ZS = np.asarray(ZS, float)
    CY = np.broadcast_to(np.asarray(CY, float), ZS.shape)
    CX = np.broadcast_to(np.asarray(CX, float), ZS.shape)
    return stube(np.stack([CX, CY, ZS], 1), RX, RY, n=n, sides=sides, caps=caps, up=(1.0, 0.0, 0.0))


def sfront(x, z, ZS, RX, RY, CY, n, back=False):
    """y of a zshell's front (or back) surface at (x, z)"""
    rx, ry, cy = np.interp(z, ZS, RX), np.interp(z, ZS, RY), np.interp(z, ZS, CY)
    q = max(1.0 - abs(x / rx) ** n, 0.0) ** (1.0 / n)
    return cy + ry * q if back else cy - ry * q


def fplate(fy, xb, xt, z0, z1, side=-1.0, lift=0.004, thick=0.03, inset=0.014, nx=10, nz=8):
    """a plate lying on the surface y = fy(x, z) (front: side -1, back: side +1); outline is a trapezoid from xb=(x0, x1) at z0 to xt at z1;
    the outer face is inset at the border so the edge reads as a chamfer"""
    def grid(ins, off):
        G = np.zeros((nz + 1, nx + 1, 3))
        for j in range(nz + 1):
            t = j / nz
            z = (z0 + ins) + ((z1 - ins) - (z0 + ins)) * t
            xa = xb[0] + (xt[0] - xb[0]) * t + ins
            xc = xb[1] + (xt[1] - xb[1]) * t - ins
            for i in range(nx + 1):
                x = xa + (xc - xa) * i / nx
                G[j, i] = (x, fy(x, z) + side * off, z)
        return G
    Gi, Go = grid(0.0, lift), grid(inset, lift + thick)
    ni = (nz + 1) * (nx + 1)
    V = np.concatenate([Gi.reshape(-1, 3), Go.reshape(-1, 3)])
    idx = lambda j, i: j * (nx + 1) + i
    polys = []
    for j in range(nz):
        for i in range(nx):
            q = (idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i))
            polys.append(tuple(ni + a for a in q))
            polys.append(q[::-1])
    loop = [idx(0, i) for i in range(nx)] + [idx(j, nx) for j in range(nz)] + [idx(nz, i) for i in range(nx, 0, -1)] + [idx(j, 0) for j in range(nz, 0, -1)]
    for k in range(len(loop)):
        a, b = loop[k], loop[(k + 1) % len(loop)]
        polys.append((ni + b, ni + a, a, b))
    return orient_outward(V, polys)


def fnormal(fy, x, z, side=-1.0, e=0.004):
    """outward normal of the surface y = fy(x, z) (front: side -1, back: side +1)"""
    dx = (fy(x + e, z) - fy(x - e, z)) / (2 * e)
    dz = (fy(x, z + e) - fy(x, z - e)) / (2 * e)
    return unit(np.array([-dx, 1.0, -dz]) * side)


def frame_on(normal, upv):
    n = unit(normal)
    u = np.asarray(upv, float)
    u = unit(u - n * (u @ n))
    return np.stack([np.cross(u, n), u, n], 1)


def text_vp(s, size, center, normal, upv, extrude=0.0025):
    """stencil text lying on a surface: reads left to right for a viewer facing the surface"""
    cu = bpy.data.curves.new("kd2_txt", 'FONT')
    cu.body = s
    cu.size = size
    cu.extrude = extrude
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    ob = bpy.data.objects.new("kd2_txt", cu)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    V = np.array([v.co[:] for v in me.vertices], float)
    P = [tuple(p.vertices) for p in me.polygons]
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(cu)
    bpy.data.meshes.remove(me)
    R = frame_on(normal, upv)
    return V @ R.T + np.asarray(center, float) + R[:, 2] * (extrude + 0.0015), P


def star_vp(center, normal, upv, r_out, r_in, depth=0.006, k=5):
    R = frame_on(normal, upv)
    pts = []
    for i in range(2 * k):
        a = math.pi / 2 + i * math.pi / k
        rr = r_out if i % 2 == 0 else r_in
        pts.append(np.asarray(center, float) + R[:, 0] * rr * math.cos(a) + R[:, 1] * rr * math.sin(a))
    pts = np.array(pts)
    M = 2 * k
    V = np.concatenate([pts, pts + R[:, 2] * depth, [np.asarray(center, float), np.asarray(center, float) + R[:, 2] * depth]])
    polys = []
    for i in range(M):
        j = (i + 1) % M
        polys += [(i, j, M + j, M + i), (2 * M, j, i), (2 * M + 1, M + i, M + j)]
    return orient_outward(V, polys)


def prism2d(outline, frame_o, ax_u, ax_v, ax_t, thick_fn):
    """a flat part from a 2D outline (u, v): fan-triangulated faces, thickness thick_fn(u, v) along ax_t (both sides)"""
    O = np.asarray(outline, float)
    c = O.mean(0)
    M = len(O)
    pt = lambda uv, sgn: frame_o + ax_u * uv[0] + ax_v * uv[1] + ax_t * (sgn * thick_fn(uv[0], uv[1]))
    V = [pt(p, 1) for p in O] + [pt(p, -1) for p in O] + [pt(c, 1), pt(c, -1)]
    polys = []
    for i in range(M):
        j = (i + 1) % M
        polys += [(i, j, M + j, M + i), (2 * M, j, i), (2 * M + 1, M + i, M + j)]
    return orient_outward(np.array(V), polys)


def resample(pts, n):
    """n points evenly along a polyline"""
    P = np.asarray(pts, float)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, L[-1], n)
    return np.stack([np.interp(t, L, P[:, k]) for k in range(3)], 1)


def sub(C, R, local):
    return np.asarray(C, float) + np.asarray(R) @ np.asarray(local, float)


# ------------------------------------------------------------------ materials
def hazard_mat(name, ca=(0.78, 0.50, 0.02), cb=(0.014, 0.014, 0.015), scale=4.2):
    m, nt, pb = _new_mat(name)
    h = _nh(nt)
    mp = h.mapping((1, 1, 1))
    wv = h.N.new('ShaderNodeTexWave')
    wv.wave_type = 'BANDS'
    wv.bands_direction = 'DIAGONAL'
    wv.inputs['Scale'].default_value = scale
    h.L.new(mp.outputs['Vector'], wv.inputs['Vector'])
    col = h.mixc(h.fm('GREATER_THAN', wv.outputs['Fac'], 0.5), ca, cb)
    wm = h.fm('MULTIPLY', h.ramp(h.noise(mp, 6.0, 5), 0.58, 0.70), 0.9, clamp=True)
    col = h.mixc(wm, col, (0.20, 0.20, 0.21))
    dm = h.fm('MULTIPLY', h.fm('MULTIPLY', _crev(h, 0.04), 2.0), h.ramp(h.noise(mp, 15.0, 4), 0.30, 0.62), clamp=True)
    col = h.mixc(dm, col, (0.04, 0.03, 0.02))
    h.L.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.5
    pb.inputs['Metallic'].default_value = 0.1
    return m


def flesh_mat(name, ob=None):
    """pale vat-grown skin: blue veins, bruise mottling, a little subsurface"""
    m, nt, pb = _new_mat(name)
    h = _nh(nt)
    mp = rest_mapping(h, ob) if ob is not None else h.mapping((1, 1, 1))
    col = h.mixc(h.ramp(h.noise(mp, 5.0, 4), 0.3, 0.7), (0.29, 0.245, 0.235), (0.40, 0.34, 0.32))
    nv = h.N.new('ShaderNodeTexNoise')
    nv.inputs['Scale'].default_value = 6.0
    nv.inputs['Detail'].default_value = 6.0
    nv.inputs['Roughness'].default_value = 0.5
    h.L.new(mp.outputs['Vector'], nv.inputs['Vector'])
    d = h.fm('ABSOLUTE', h.fm('SUBTRACT', nv.outputs['Fac'], 0.5))
    vein = h.fm('SUBTRACT', 1.0, h.fm('MULTIPLY', d, 48.0), clamp=True)
    col = h.mixc(h.fm('MULTIPLY', vein, 0.45), col, (0.13, 0.12, 0.22))
    br = h.fm('MULTIPLY', h.ramp(h.noise(mp, 2.2, 3), 0.60, 0.78), 0.6, clamp=True)
    col = h.mixc(br, col, (0.26, 0.15, 0.17))
    col = h.mixc(h.fm('MULTIPLY', _crev(h, 0.05), 0.9, clamp=True), col, (0.09, 0.05, 0.05))
    h.L.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.42
    try:
        pb.inputs['Subsurface Weight'].default_value = 0.22
        pb.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.25)
        pb.inputs['Subsurface Scale'].default_value = 0.03
    except Exception:
        pass
    bp = h.N.new('ShaderNodeBump')
    bp.inputs['Distance'].default_value = 0.003
    bp.inputs['Strength'].default_value = 0.3
    h.L.new(h.noise(mp, 90.0, 3), bp.inputs['Height'])
    h.L.new(bp.outputs['Normal'], pb.inputs['Normal'])
    return m


def common_mats(M):
    M['steel'] = pbr('KD2_Steel', (0.22, 0.22, 0.23), (0.36, 0.36, 0.38), metal=1.0, rough=0.32, nscale=8, bump=0.2, dirt=0.6)
    M['steelWorn'] = pbr('KD2_SteelWorn', (0.15, 0.15, 0.16), (0.28, 0.28, 0.29), metal=1.0, rough=0.46, nscale=10, bump=0.5, bscale=90, dirt=0.9,
                         wear_col=(0.22, 0.10, 0.05), wear=0.4)
    M['chrome'] = pbr('KD2_Chrome', (0.56, 0.56, 0.58), (0.72, 0.72, 0.74), metal=1.0, rough=0.14, nscale=8, bump=0.04, dirt=0.3)
    M['darksteel'] = pbr('KD2_DarkSteel', (0.045, 0.046, 0.05), (0.09, 0.09, 0.10), metal=1.0, rough=0.44, nscale=7, bump=0.3, dirt=0.8,
                         wear_col=(0.24, 0.24, 0.25), wear=0.35)
    M['rubber'] = pbr('KD2_Rubber', (0.010, 0.010, 0.012), (0.028, 0.028, 0.031), rough=0.72, nscale=7, bump=0.3, dirt=0.5)
    return M


# ================================================================== A : RECLAIMER
CORE_CH = {}


def skin_core(full_right=False, head=True):
    """plain inner body (under-suit / inner machinery) that the shells sit on"""
    S = Suit()
    chains = {'torso': ([J['hip_c'], J['pelvis_t'], J['spine_t'], J['chest_t'], J['neck_t'], J['head_t']], ['pelvis', 'spine', 'chest', 'neck', 'head'], 0.09)}
    for s in ('L', 'R'):
        chains['leg_' + s] = ([J['hip_' + s], J['kn_' + s], J['an_' + s]], ['thigh.' + s, 'shin.' + s], 0.10)
        chains['arm_' + s] = ([J['clav_' + s], J['sh_' + s], J['el_' + s], J['wr_' + s]], ['clavicle.' + s, 'upperarm.' + s, 'forearm.' + s], 0.09)
    CH.update(chains)

    def host(V, polys, region, mat=0):
        j, b, bl = chains[region]
        S.host(V, polys, mat, j, b, bl)

    host(*tube(np.stack([np.zeros_like(TZ), TCY, TZ], 1), TRX * 0.84, TRY * 0.86, sides=28, caps=(True, True), up=(1.0, 0.0, 0.0)), 'torso')
    host(*tube(np.array([[0, 0.0, 1.98], [0, 0.01, 2.12], [0, 0.015, 2.22]]), [0.17, 0.15, 0.14], sides=20, caps=(False, False)), 'torso')
    if head:
        host(*ellipsoid((0.0, 0.01, 2.32), (0.13, 0.15, 0.16), 18, 12), 'torso')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        P = np.array([[0.20, 0.0, 1.14], [0.22, -0.01, 0.92], [0.25, -0.03, 0.66], [0.26, -0.00, 0.42], [0.26, 0.03, 0.19]]) * np.array([sx, 1, 1])
        r = np.array([0.20, 0.19, 0.15, 0.14, 0.11])
        host(*tube(P, r, r * 1.04, sides=20, caps=(True, True), up=(1.0, 0.0, 0.0)), 'leg_' + s)
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        path = [J['clav_' + s], sh, lerp(sh, el, 0.5), el]
        rad = [0.12, 0.15, 0.14, 0.12]
        if s == 'L' or full_right:
            path += [lerp(el, wr, 0.5), wr]
            rad += [0.11, 0.095]
        else:
            path += [lerp(el, wr, 0.3)]
            rad += [0.11]
        host(*tube(np.array(path), np.array(rad), sides=18, caps=(True, True), up=(0.0, 1.0, 0.0)), 'arm_' + s)
    return S


def parts_A(S=None):
    P = []

    def add(name, vp, mk, bone, smooth=True):
        P.append((name, vp[0], vp[1], mk, bone, smooth))

    # --- torso: chest carapace, keel breastplates, abdomen plate, hip girdle, collar, back unit
    CZ = np.array([1.48, 1.58, 1.70, 1.82, 1.94, 2.03, 2.09])
    CRX = np.array([0.40, 0.46, 0.53, 0.57, 0.56, 0.47, 0.33])
    CRY = np.array([0.27, 0.30, 0.33, 0.345, 0.33, 0.28, 0.22])
    CCY = np.array([-0.01, -0.02, -0.03, -0.03, -0.02, 0.0, 0.02])
    add('ChestShell', zshell(CZ, CRX, CRY, CCY, n=3.0, sides=40), 'shell', 'chest')
    fy = lambda x, z: sfront(x, z, CZ, CRX, CRY, CCY, 3.0)
    by = lambda x, z: sfront(x, z, CZ, CRX, CRY, CCY, 3.0, back=True)
    add('BreastL', fplate(fy, (0.03, 0.34), (0.03, 0.44), 1.64, 1.97, lift=0.003, thick=0.032), 'shell', 'chest')
    add('BreastR', fplate(fy, (-0.34, -0.03), (-0.44, -0.03), 1.64, 1.97, lift=0.003, thick=0.032), 'shell', 'chest')
    add('Keel', fplate(fy, (-0.025, 0.025), (-0.025, 0.025), 1.60, 2.00, lift=0.002, thick=0.045, inset=0.008, nx=2), 'cobalt', 'chest')
    # StarCapital star and the unit number on the left breast
    for nm, x, z in (('Star', 0.20, 1.89), ('TxtUnit', 0.22, 1.77)):
        n_ = fnormal(fy, x, z)
        p = np.array([x, fy(x, z), z]) + n_ * 0.035
        if nm == 'Star':
            add(nm, star_vp(p, n_, (0, 0, 1), 0.050, 0.021), 'cobalt', 'chest', smooth=False)
        else:
            add(nm, text_vp("KD-09", 0.065, p, n_, (0, 0, 1)), 'cobalt', 'chest', smooth=False)
    add('AbShell', zshell([1.27, 1.35, 1.43, 1.51], [0.335, 0.345, 0.37, 0.40], [0.25, 0.255, 0.265, 0.28], [0.0, -0.005, -0.01, -0.012], n=2.8), 'shell', 'spine')
    add('Girdle', zshell([1.02, 1.10, 1.20, 1.29], [0.30, 0.38, 0.41, 0.38], [0.24, 0.27, 0.28, 0.27], n=3.0), 'shell', 'pelvis')
    add('Codplate', rbox((0.0, -0.275, 1.05), (0.26, 0.06, 0.24), 0.03, rot_x(0.10)), 'graphite', 'pelvis')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        Ct, Rt = np.array([sx * 0.39, -0.02, 1.03]), rot_y(-sx * 0.20)
        add('Tasset' + s, rbox(Ct, (0.09, 0.36, 0.28), 0.025, Rt), 'shell', 'pelvis')
        add('TassetEdge' + s, rbox(sub(Ct, Rt, (0.0, 0.0, -0.135)), (0.10, 0.37, 0.03), 0.01, Rt), 'cobalt', 'pelvis')
    z = 1.62
    rx_, ry_, cy_ = np.interp(z, CZ, CRX) + 0.004, np.interp(z, CZ, CRY) + 0.004, np.interp(z, CZ, CCY)
    add('ChestSeam', zshell([z - 0.007, z + 0.007], [rx_, rx_], [ry_, ry_], [cy_, cy_], n=3.0, sides=40, caps=(False, False)), 'graphite', 'chest')
    add('Collar', zshell([1.98, 2.07, 2.15], [0.29, 0.27, 0.245], [0.25, 0.235, 0.215], [0.01, 0.015, 0.02], n=2.6, caps=(False, False)), 'cobalt', 'chest')
    add('BackUnit', rbox((0.0, 0.40, 1.80), (0.56, 0.22, 0.50), 0.05), 'graphite', 'chest')
    for k in range(4):
        add('BackVent%d' % k, rbox((0.0, 0.512, 1.66 + 0.07 * k), (0.40, 0.012, 0.03), 0.005), 'rubber', 'chest')
    add('TxtBack', text_vp("STARCAPITAL", 0.060, (0.0, 0.508, 1.93), (0, 1, 0), (0, 0, 1)), 'cobalt', 'chest', smooth=False)
    # --- shoulders: solid angular blocks tilted down outward, a graphite rim under them, cobalt stripe, unit number on the left
    for s, sx in (('L', 1.0), ('R', -1.0)):
        Cp, Rp = np.array([sx * 0.76, 0.0, 1.96]), rot_y(sx * 0.30)
        add('Pauldron' + s, rbox(Cp, (0.44, 0.60, 0.32), 0.05, Rp, n=3), 'shell', 'clavicle.' + s)
        add('PauldronRim' + s, rbox(sub(Cp, Rp, (0.0, 0.0, -0.165)), (0.42, 0.57, 0.035), 0.012, Rp), 'graphite', 'clavicle.' + s)
        add('PauldronStripe' + s, rbox(sub(Cp, Rp, (sx * 0.222, 0.0, -0.095)), (0.012, 0.50, 0.05), 0.004, Rp), 'cobalt', 'clavicle.' + s)
        for k, yo in enumerate((-0.22, 0.22)):
            add('PauldronBolt%s%d' % (s, k), ellipsoid(sub(Cp, Rp, (sx * 0.222, yo, 0.10)), (0.012, 0.018, 0.018), 8, 6, R=Rp), 'graphite', 'clavicle.' + s)
        if s == 'L':
            add('TxtShoulder', text_vp("KD-09", 0.10, sub(Cp, Rp, (0.222, 0.0, 0.03)), Rp @ np.array([1.0, 0, 0]), Rp @ np.array([0, 0, 1.0])),
                'cobalt', 'clavicle.L', smooth=False)
    # --- arms
    for s, sx in (('L', 1.0), ('R', -1.0)):
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        add('UpperArm' + s, stube([lerp(sh, el, 0.20), lerp(sh, el, 0.55), lerp(sh, el, 0.86)], [0.205, 0.195, 0.17], [0.20, 0.19, 0.165],
                                  n=2.7, sides=28, up=(0, 1, 0)), 'shell', 'upperarm.' + s)
        add('ElbowJoint' + s, ellipsoid(el, (0.135, 0.135, 0.135), 18, 10), 'graphite', 'upperarm.' + s)
        add('ElbowCop' + s, rbox(el + np.array([sx * 0.02, 0.13, 0.0]), (0.20, 0.07, 0.20), 0.03, rot_x(-0.25)), 'shell', 'forearm.' + s)
    sh, el, wr = J['sh_L'], J['el_L'], J['wr_L']
    add('GauntletL', stube([lerp(el, wr, 0.10), lerp(el, wr, 0.50), lerp(el, wr, 0.93)], [0.17, 0.185, 0.155], [0.16, 0.17, 0.145], n=2.8, sides=28, up=(0, 1, 0)),
        'shell', 'forearm.L')
    add('CuffL', stube([lerp(el, wr, 0.82), lerp(el, wr, 0.97)], [0.168, 0.16], [0.158, 0.15], n=2.8, sides=28, caps=(False, False), up=(0, 1, 0)), 'cobalt', 'forearm.L')
    fist = wr + np.array([0.0, -0.01, -0.14])
    add('FistL', rbox(fist, (0.25, 0.22, 0.22), 0.06), 'graphite', 'hand.L')
    for k, xo in enumerate((-0.085, -0.028, 0.028, 0.085)):
        add('FingerL%d' % k, rbox(fist + np.array([xo, -0.085, -0.09]), (0.055, 0.08, 0.13), 0.022), 'graphite', 'hand.L')
    add('KnuckleGuardL', rbox(fist + np.array([0.0, -0.115, -0.02]), (0.25, 0.05, 0.08), 0.02), 'shell', 'hand.L')
    add('ThumbL', rbox(fist + np.array([-0.13, -0.05, -0.02]), (0.06, 0.09, 0.14), 0.025, rot_z(0.3)), 'graphite', 'hand.L')
    # --- the right arm: pile-driver ram (housing on the forearm, piston and head on the hand bone: translate hand.R to fire it)
    sh, el, wr, hd = J['sh_R'], J['el_R'], J['wr_R'], J['hd_R']
    d = unit(wr - el)
    Bd = basis(d, (0, 1, 0))
    a0, a1 = el + d * 0.12, wr + d * 0.04
    add('RamHousing', stube([a0, lerp(a0, a1, 0.5), a1], [0.205, 0.215, 0.205], n=2.0, sides=32, up=(0, 1, 0)), 'shell', 'forearm.R')
    add('RamCapRear', dome(a0, 0.20, 0.20, 0.08, -d), 'graphite', 'forearm.R')
    add('RamStripe', stube([lerp(a0, a1, 0.28), lerp(a0, a1, 0.40)], 0.221, n=2.0, sides=32, caps=(False, False), up=(0, 1, 0)), 'cobalt', 'forearm.R')
    for k in range(3):
        ang = 2 * math.pi * k / 3 + 0.6
        off = (Bd[:, 0] * math.cos(ang) + Bd[:, 1] * math.sin(ang)) * 0.236
        add('RamRod%d' % k, cyl(a0 + off + d * 0.03, a1 + off, 0.024, 0.024, sides=10), 'chrome', 'forearm.R')
    add('RamCollar', torus(a1, 0.205, 0.03, axis=d), 'steel', 'forearm.R')
    add('RamPiston', cyl(a1 - d * 0.02, hd, 0.10, 0.10, sides=20), 'chrome', 'hand.R')
    h0 = hd - d * 0.02
    add('RamHead', cyl(h0, h0 + d * 0.20, 0.25, 0.25, sides=8), 'graphite', 'hand.R', smooth=False)
    add('RamHazard', cyl(h0 + d * 0.035, h0 + d * 0.105, 0.257, 0.257, sides=8), 'hazard', 'hand.R', smooth=False)
    add('RamFace', cyl(h0 + d * 0.20, h0 + d * 0.235, 0.228, 0.205, sides=8), 'steelWorn', 'hand.R', smooth=False)
    # --- legs: smooth thigh and shin shells, an angular knee guard, sabatons
    for s, sx in (('L', 1.0), ('R', -1.0)):
        hip, kn, an, ball, toe = J['hip_' + s], J['kn_' + s], J['an_' + s], J['ball_' + s], J['toe_' + s]
        add('Thigh' + s, stube([lerp(hip, kn, 0.14), lerp(hip, kn, 0.5), lerp(hip, kn, 0.86)], [0.205, 0.215, 0.19], [0.235, 0.235, 0.20], n=2.8, sides=28),
            'shell', 'thigh.' + s)
        add('KneeJoint' + s, ellipsoid(kn, (0.15, 0.15, 0.14), 16, 10), 'graphite', 'shin.' + s)
        add('KneeGuard' + s, rbox(kn + np.array([0, -0.19, 0.03]), (0.27, 0.10, 0.25), 0.035, rot_x(0.18)), 'shell', 'shin.' + s)
        add('KneeRidge' + s, rbox(kn + np.array([0, -0.245, 0.04]), (0.05, 0.04, 0.21), 0.012, rot_x(0.18)), 'cobalt', 'shin.' + s)
        add('Shin' + s, stube([lerp(kn, an, 0.12), lerp(kn, an, 0.5), lerp(kn, an, 0.92)], [0.19, 0.18, 0.155], [0.205, 0.20, 0.17], n=2.6, sides=28),
            'shell', 'shin.' + s)
        add('ShinPlate' + s, rbox(lerp(kn, an, 0.47) + np.array([0, -0.205, 0]), (0.14, 0.05, 0.32), 0.025, rot_x(-0.05)), 'graphite', 'shin.' + s)
        add('BootUpper' + s, rbox(an + np.array([0, -0.06, -0.05]), (0.32, 0.40, 0.24), 0.06), 'graphite', 'foot.' + s)
        add('BootHeel' + s, rbox(an + np.array([0, 0.10, -0.09]), (0.30, 0.16, 0.18), 0.05), 'shell', 'foot.' + s)
        add('BootToe' + s, rbox(ball + np.array([0, -0.10, -0.015]), (0.33, 0.30, 0.13), 0.05, rot_x(0.06)), 'shell', 'toe.' + s)
        add('BootSole' + s, rbox(lerp(an, toe, 0.40) + np.array([0, 0, -0.105]), (0.35, 0.76, 0.06), 0.022), 'rubber', 'foot.' + s)
    # --- head: helmet, graphite face plate with ONE vertical glowing slit
    HZ = np.array([2.10, 2.17, 2.28, 2.40, 2.49, 2.545])
    add('Helmet', zshell(HZ, [0.150, 0.168, 0.176, 0.170, 0.140, 0.075], [0.168, 0.188, 0.196, 0.190, 0.158, 0.085],
                         [0.012, 0.012, 0.012, 0.015, 0.02, 0.025], n=2.5, sides=32), 'shell', 'head')
    add('FacePlate', rbox((0.0, -0.19, 2.30), (0.20, 0.05, 0.30), 0.02, rot_x(0.06)), 'graphite', 'head')
    add('Slit', rbox((0.0, -0.218, 2.31), (0.024, 0.012, 0.21), 0.004, rot_x(0.06)), 'glow', 'head')
    for sx in (-1.0, 1.0):
        add('SlitRail%d' % (sx > 0), rbox((sx * 0.024, -0.219, 2.31), (0.010, 0.016, 0.22), 0.003, rot_x(0.06)), 'steel', 'head')
        for k in range(3):
            add('CheekVent%d%d' % (sx > 0, k), rbox((sx * 0.07, -0.214, 2.19 + 0.026 * k), (0.05, 0.008, 0.012), 0.003), 'rubber', 'head')
        add('EarPlate%d' % (sx > 0), rbox((sx * 0.175, 0.01, 2.30), (0.04, 0.17, 0.13), 0.015), 'graphite', 'head')
        add('EarLed%d' % (sx > 0), rbox((sx * 0.196, 0.01, 2.30), (0.006, 0.12, 0.018), 0.002), 'cobalt', 'head')
    add('NeckSeal', tube(np.array([[0, 0.01, 2.03], [0, 0.012, 2.14]]), [0.18, 0.17], sides=22, caps=(False, False)), 'rubber', 'neck')
    return P


def mats_A(body):
    M = common_mats({})
    M['suit'] = pbr('KD2_UnderSuit', (0.020, 0.024, 0.030), (0.042, 0.048, 0.058), rough=0.75, nscale=9, bump=0.6, bscale=180, dirt=0.6,
                    dirt_col=(0.02, 0.02, 0.02), ob=body)
    M['shell'] = pbr('KD2_Shell', (0.46, 0.43, 0.37), (0.60, 0.57, 0.50), rough=0.40, nscale=5, bump=0.14, bscale=40, dirt=1.6,
                     dirt_col=(0.075, 0.062, 0.05), wear_col=(0.10, 0.10, 0.11), wear=0.5, spec=0.6)
    M['cobalt'] = pbr('KD2_Cobalt', (0.010, 0.030, 0.13), (0.018, 0.050, 0.20), rough=0.35, nscale=6, bump=0.1, dirt=0.8, wear_col=(0.16, 0.16, 0.17), wear=0.3)
    M['graphite'] = pbr('KD2_Graphite', (0.035, 0.037, 0.042), (0.06, 0.063, 0.07), metal=0.6, rough=0.42, nscale=6, bump=0.2, dirt=0.7,
                        wear_col=(0.22, 0.22, 0.23), wear=0.3)
    M['hazard'] = hazard_mat('KD2_Hazard')
    M['glow'] = glow_mat('KD2_Slit', (0.06, 0.45, 1.0), 4.0)
    return M


# ================================================================== B : BREAKER
def parts_B(S=None):
    P = []

    def add(name, vp, mk, bone, smooth=True):
        P.append((name, vp[0], vp[1], mk, bone, smooth))

    # --- torso: boxy cab frame with a radiator grille, engine hump with exhaust stacks, exposed waist column
    FZ = np.array([1.46, 1.58, 1.72, 1.86, 1.98, 2.06])
    FRX = np.array([0.42, 0.48, 0.54, 0.565, 0.53, 0.42])
    FRY = np.array([0.28, 0.31, 0.335, 0.345, 0.33, 0.27])
    FCY = np.array([0.0, -0.01, -0.02, -0.02, -0.01, 0.0])
    add('Frame', zshell(FZ, FRX, FRY, FCY, n=4.0, sides=40), 'paint', 'chest')
    fy = lambda x, z: sfront(x, z, FZ, FRX, FRY, FCY, 4.0)
    add('Grille', fplate(fy, (-0.22, 0.22), (-0.22, 0.22), 1.58, 1.92, lift=0.0, thick=0.03, inset=0.01), 'darksteel', 'chest')
    for k in range(7):
        x = -0.18 + 0.06 * k
        add('Slat%d' % k, rbox((x, fy(x, 1.75) - 0.045, 1.75), (0.022, 0.026, 0.30), 0.006), 'paint', 'chest')
    Ch, Rh = np.array([0.0, 0.26, 2.16]), rot_x(0.26)
    add('Hump', rbox(Ch, (1.04, 0.62, 0.56), 0.09, Rh, n=3), 'paint', 'chest')
    add('HumpVent', rbox(sub(Ch, Rh, (0.0, 0.05, 0.283)), (0.62, 0.34, 0.02), 0.008, Rh), 'darksteel', 'chest')
    for k in range(6):
        add('HumpVentSlat%d' % k, rbox(sub(Ch, Rh, (0.0, -0.08 + 0.052 * k, 0.293)), (0.58, 0.012, 0.012), 0.004, Rh), 'paint', 'chest')
    add('TxtHump', text_vp("STARCAPITAL", 0.075, sub(Ch, Rh, (0.0, 0.312, 0.0)), Rh @ np.array([0, 1.0, 0]), Rh @ np.array([0, 0, 1.0])),
        'stencil', 'chest', smooth=False)
    add('TxtHumpSide', text_vp("KD-B7", 0.13, sub(Ch, Rh, (0.522, 0.0, 0.0)), Rh @ np.array([1.0, 0, 0]), Rh @ np.array([0, 0, 1.0])),
        'stencil', 'chest', smooth=False)
    for s, sx in (('L', 1.0), ('R', -1.0)):
        b0, b1 = np.array([sx * 0.30, 0.44, 2.28]), np.array([sx * 0.33, 0.49, 2.57])
        add('Stack' + s, cyl(b0, b1, 0.07, 0.066, sides=16), 'darksteel', 'chest')
        add('StackBand' + s, cyl(lerp(b0, b1, 0.70), lerp(b0, b1, 0.82), 0.073, 0.072, sides=16), 'yellow', 'chest')
        add('StackLip' + s, torus(b1, 0.066, 0.012, axis=unit(b1 - b0)), 'rubber', 'chest')
    add('SpineColumn', cyl((0, 0.04, 1.27), (0, 0.04, 1.50), 0.13, 0.13, sides=16), 'darksteel', 'spine')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('WaistPiston' + s, cyl((sx * 0.21, -0.10, 1.25), (sx * 0.25, -0.13, 1.50), 0.034, 0.034, sides=10), 'chrome', 'spine')
        add('WaistCyl' + s, cyl((sx * 0.21, -0.10, 1.22), (sx * 0.22, -0.11, 1.34), 0.052, 0.052, sides=12), 'darksteel', 'pelvis')
    add('Pelvis', rbox((0.0, 0.02, 1.19), (0.62, 0.46, 0.26), 0.06, n=3), 'paint', 'pelvis')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('HipDrum' + s, cyl((sx * 0.12, 0.0, 1.16), (sx * 0.37, 0.0, 1.16), 0.15, 0.15, sides=8), 'darksteel', 'pelvis', smooth=False)
    # --- head: a squat sensor turret under the hump's overhang, six work lamps, vertical guard rods
    Ct, Rt = np.array([0.0, -0.19, 2.21]), rot_x(-0.06)
    add('Turret', rbox(Ct, (0.38, 0.30, 0.22), 0.04, Rt, n=3), 'paint', 'head')
    for k, (x, z, r) in enumerate(((-0.11, 2.255, 0.032), (0.0, 2.255, 0.040), (0.11, 2.255, 0.032), (-0.11, 2.17, 0.026), (0.0, 2.17, 0.030), (0.11, 2.17, 0.026))):
        add('LampCup%d' % k, cyl((x, -0.325, z), (x, -0.368, z), r + 0.013, r + 0.010, sides=16), 'darksteel', 'head')
        add('Lamp%d' % k, ellipsoid((x, -0.369, z), (r, 0.010, r), 14, 8), 'lamp', 'head')
    for x in (-0.055, 0.055):
        add('Guard%d' % (x > 0), cyl((x, -0.392, 2.125), (x, -0.392, 2.30), 0.008, 0.008, sides=8), 'steel', 'head')
        add('GuardFoot%d' % (x > 0), rbox((x, -0.37, 2.12), (0.03, 0.05, 0.02), 0.005), 'steel', 'head')
    add('TurretNeck', tube(np.array([[0, -0.11, 2.02], [0, -0.15, 2.12]]), [0.14, 0.13], sides=18, caps=(False, False)), 'rubber', 'neck')
    add('Antenna', cyl((0.15, -0.10, 2.30), (0.165, -0.07, 2.55), 0.006, 0.004, sides=6), 'steel', 'head')
    # --- shoulders: octagonal actuator drums with a painted end plate and bolts
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('ShoulderDrum' + s, cyl((sx * 0.46, 0.0, 1.90), (sx * 0.80, 0.0, 1.90), 0.21, 0.21, sides=8), 'darksteel', 'clavicle.' + s, smooth=False)
        add('ShoulderPlate' + s, cyl((sx * 0.80, 0.0, 1.90), (sx * 0.835, 0.0, 1.90), 0.19, 0.18, sides=8), 'paint', 'clavicle.' + s, smooth=False)
        for k in range(6):
            a = 2 * math.pi * k / 6
            c = np.array([sx * 0.835, 0.13 * math.cos(a), 1.90 + 0.13 * math.sin(a)])
            add('ShoulderBolt%s%d' % (s, k), cyl(c, c + np.array([sx * 0.02, 0, 0]), 0.016, 0.016, sides=6), 'steel', 'clavicle.' + s)
    # --- arms: box boom upper arm with a hydraulic cylinder, elbow drum, huge boxy forearms
    for s, sx in (('L', 1.0), ('R', -1.0)):
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        add('Boom' + s, stube([lerp(sh, el, 0.15), lerp(sh, el, 0.85)], 0.15, 0.17, n=5.0, sides=28, up=(0, 1, 0)), 'paint', 'upperarm.' + s)
        c0, c1 = sh + np.array([sx * 0.02, -0.22, -0.10]), lerp(sh, el, 0.70) + np.array([0.0, -0.20, 0.0])
        add('ArmCyl' + s, cyl(c0, c1, 0.065, 0.065, sides=14), 'darksteel', 'upperarm.' + s)
        add('ArmRod' + s, cyl(c1, el + np.array([0.0, -0.17, -0.04]), 0.036, 0.036, sides=10), 'chrome', 'upperarm.' + s)
        ex = np.array([1.0, 0, 0])
        add('ElbowDrum' + s, cyl(el - ex * 0.19, el + ex * 0.19, 0.135, 0.135, sides=8), 'darksteel', 'upperarm.' + s, smooth=False)
        add('Forearm' + s, stube([lerp(el, wr, 0.06), lerp(el, wr, 0.45), lerp(el, wr, 0.92)], [0.22, 0.28, 0.25], [0.20, 0.26, 0.23], n=3.5, sides=32, up=(0, 1, 0)),
            'paint', 'forearm.' + s)
        add('WearPlate' + s, rbox(lerp(el, wr, 0.50) + np.array([sx * 0.265, 0.0, 0.0]), (0.05, 0.30, 0.40), 0.015), 'steelWorn', 'forearm.' + s)
        hp = np.array([[sx * 0.48, 0.30, 2.05], [sx * 0.78, 0.22, 1.99], [sx * 0.92, 0.10, 1.72], [sx * 0.90, 0.07, 1.42], [sx * 0.84, 0.05, 1.22]])
        add('Hose' + s, tube(resample(hp, 24), 0.028, sides=10, caps=(True, True), up=(1, 0, 0)), 'rubber', 'upperarm.' + s)
    # left: a heavy mechanical fist
    wr = J['wr_L']
    fist = wr + np.array([0.0, -0.02, -0.13])
    add('FistBlockL', rbox(fist, (0.30, 0.26, 0.22), 0.05, n=3), 'paint', 'hand.L')
    for k, xo in enumerate((-0.10, -0.035, 0.035, 0.10)):
        add('FingerL%d' % k, rbox(fist + np.array([xo, -0.11, -0.10]), (0.06, 0.09, 0.15), 0.02), 'darksteel', 'hand.L')
    add('ThumbL', rbox(fist + np.array([-0.16, -0.06, -0.03]), (0.07, 0.10, 0.15), 0.025, rot_z(0.3)), 'darksteel', 'hand.L')
    # right: crusher jaws on a hinge block; the jaws open sideways (outer and inner jaw) so they read from the front
    el, wr, hd = J['el_R'], J['wr_R'], J['hd_R']
    d = unit(hd - wr)
    ey = np.array([0.0, 1.0, 0.0])
    add('JawBlock', rbox(wr + d * 0.07, (0.38, 0.30, 0.20), 0.04, basis(d, (0, 1, 0)), n=3), 'paint', 'hand.R')
    for sgn, nm in ((-1.0, 'Out'), (1.0, 'In')):
        piv = wr + d * 0.15 + np.array([sgn * 0.095, 0.0, 0.0])
        add('Pivot' + nm, cyl(piv - ey * 0.17, piv + ey * 0.17, 0.048, 0.048, sides=12), 'chrome', 'hand.R')
        rel = [(0.095, 0.15), (0.20, 0.27), (0.235, 0.41), (0.18, 0.53), (0.065, 0.60)]
        path = resample(np.array([wr + np.array([sgn * xx, 0.0, 0.0]) + d * zz for (xx, zz) in rel]), 9)
        rx = np.interp(np.linspace(0, 1, 9), [0, 0.5, 1], [0.15, 0.14, 0.10])
        ry = np.interp(np.linspace(0, 1, 9), [0, 0.5, 1], [0.055, 0.05, 0.03])
        add('Jaw' + nm, stube(path, rx, ry, n=4.0, sides=24, up=(0, 1, 0)), 'hazard', 'hand.R')
        for k in (3, 5, 7):
            T = unit(path[min(k + 1, 8)] - path[k - 1])
            inner = unit(np.cross(T, ey))
            if inner[0] * sgn > 0:
                inner = -inner
            for yo in (-0.075, 0.0, 0.075):
                base = path[k] + inner * (ry[k] * 0.8) + ey * yo
                add('Tooth%s%d%d' % (nm, k, int(yo * 100) + 8), cyl(base, base + inner * 0.085, 0.03, 0.0, sides=8), 'steelWorn', 'hand.R')
    # --- legs: box thigh frame with an exposed ram, knee block and pivot, box shin with a calf ram, wide foot pads
    for s, sx in (('L', 1.0), ('R', -1.0)):
        hip, kn, an, ball, toe = J['hip_' + s], J['kn_' + s], J['an_' + s], J['ball_' + s], J['toe_' + s]
        add('ThighFrame' + s, stube([lerp(hip, kn, 0.12), lerp(hip, kn, 0.85)], [0.20, 0.19], [0.22, 0.21], n=4.0, sides=28), 'paint', 'thigh.' + s)
        m0 = lerp(hip, kn, 0.55) + np.array([sx * 0.25, 0.0, 0.0])
        add('ThighCyl' + s, cyl(np.array([sx * 0.44, 0.02, 1.16]), m0, 0.07, 0.07, sides=14), 'darksteel', 'thigh.' + s)
        add('ThighRod' + s, cyl(m0, kn + np.array([sx * 0.23, 0.0, -0.04]), 0.04, 0.04, sides=10), 'chrome', 'thigh.' + s)
        add('KneeBlock' + s, rbox(kn + np.array([0.0, -0.17, 0.0]), (0.30, 0.16, 0.24), 0.04, n=3), 'paint', 'shin.' + s)
        add('KneePivot' + s, cyl(kn - ex * 0.24, kn + ex * 0.24, 0.12, 0.12, sides=8), 'darksteel', 'shin.' + s, smooth=False)
        add('ShinFrame' + s, stube([lerp(kn, an, 0.10), lerp(kn, an, 0.90)], [0.19, 0.16], [0.21, 0.17], n=4.0, sides=28), 'paint', 'shin.' + s)
        add('CalfCyl' + s, cyl(kn + np.array([0, 0.20, -0.08]), lerp(kn, an, 0.55) + np.array([0, 0.19, 0]), 0.055, 0.055, sides=12), 'darksteel', 'shin.' + s)
        add('CalfRod' + s, cyl(lerp(kn, an, 0.55) + np.array([0, 0.19, 0]), an + np.array([0, 0.17, 0.10]), 0.032, 0.032, sides=10), 'chrome', 'shin.' + s)
        add('AnkleBlock' + s, rbox(an + np.array([0, 0.0, -0.02]), (0.30, 0.30, 0.22), 0.04, n=3), 'darksteel', 'foot.' + s)
        add('FootPad' + s, rbox(lerp(an, toe, 0.36) + np.array([0, 0, -0.095]), (0.44, 0.74, 0.09), 0.035), 'darksteel', 'foot.' + s)
        add('ToePad' + s, rbox(ball + np.array([0, -0.12, -0.03]), (0.44, 0.28, 0.10), 0.035), 'paint', 'toe.' + s)
    return P


def mats_B(body):
    M = common_mats({})
    M['core'] = pbr('KD2_Core', (0.030, 0.030, 0.034), (0.06, 0.06, 0.065), metal=0.8, rough=0.5, nscale=7, bump=0.3, dirt=0.7, ob=body)
    M['paint'] = pbr('KD2_Olive', (0.085, 0.090, 0.058), (0.14, 0.15, 0.095), metal=0.2, rough=0.55, nscale=5, bump=0.25, bscale=50, dirt=1.2,
                     dirt_col=(0.05, 0.04, 0.03), wear_col=(0.20, 0.09, 0.035), wear=0.45)
    M['yellow'] = pbr('KD2_Yellow', (0.55, 0.38, 0.02), (0.70, 0.50, 0.04), rough=0.5, dirt=1.0, wear_col=(0.2, 0.2, 0.21), wear=0.5)
    M['stencil'] = pbr('KD2_Stencil', (0.52, 0.52, 0.49), (0.64, 0.64, 0.60), rough=0.6, dirt=0.9, wear_col=(0.10, 0.10, 0.07), wear=0.5)
    M['hazard'] = hazard_mat('KD2_HazardB')
    M['lamp'] = glow_mat('KD2_Lamp', (1.0, 0.93, 0.80), 12.0)
    return M


# ================================================================== C : GROWN
TZ_C = np.array([1.04, 1.16, 1.30, 1.45, 1.60, 1.74, 1.86, 1.96, 2.04, 2.10])
TRX_C = np.array([0.30, 0.32, 0.30, 0.30, 0.36, 0.44, 0.48, 0.42, 0.24, 0.16])
TRY_C = np.array([0.21, 0.23, 0.24, 0.26, 0.29, 0.31, 0.32, 0.30, 0.22, 0.17])
TCY_C = np.array([0.0, 0.0, -0.01, -0.015, -0.01, 0.015, 0.045, 0.075, 0.08, 0.05])
TORSO_C = []


def skin_C():
    S = Suit()
    chains = {'torso': ([J['hip_c'], J['pelvis_t'], J['spine_t'], J['chest_t'], J['neck_t'], J['head_t']], ['pelvis', 'spine', 'chest', 'neck', 'head'], 0.09)}
    for s in ('L', 'R'):
        chains['leg_' + s] = ([J['hip_' + s], J['kn_' + s], J['an_' + s]], ['thigh.' + s, 'shin.' + s], 0.10)
        chains['arm_' + s] = ([J['clav_' + s], J['sh_' + s], J['el_' + s], J['wr_' + s]], ['clavicle.' + s, 'upperarm.' + s, 'forearm.' + s], 0.09)
    CH.update(chains)
    TORSO_C.clear()

    def host(V, polys, region, mat=0, torso_ray=False):
        j, b, bl = chains[region]
        S.host(V, polys, mat, j, b, bl)
        if torso_ray:
            TORSO_C.append((np.asarray(V, float), polys))

    host(*tube(np.stack([np.zeros_like(TZ_C), TCY_C, TZ_C], 1), TRX_C, TRY_C, sides=30, caps=(True, True), up=(1.0, 0.0, 0.0)), 'torso', torso_ray=True)
    for sx in (-1.0, 1.0):
        host(*ellipsoid((0.19 * sx, -0.20, 1.78), (0.19, 0.08, 0.13), 20, 10, R=rot_y(-0.10 * sx)), 'torso', torso_ray=True)        # pecs (modest)
        host(*ellipsoid((0.38 * sx, 0.10, 1.70), (0.12, 0.14, 0.26), 16, 10, R=rot_y(0.10 * sx)), 'torso')                         # lats
        host(*ellipsoid((0.20 * sx, 0.11, 2.00), (0.24, 0.22, 0.14), 18, 10, R=rot_y(-0.45 * sx)), 'torso', torso_ray=True)        # big traps: the hunch
    host(*tube(np.array([[0, 0.06, 1.98], [0, 0.01, 2.10], [0, -0.03, 2.20]]), [0.19, 0.165, 0.15], sides=22, caps=(False, False)), 'torso')
    host(*ellipsoid((0.0, -0.025, 2.33), (0.140, 0.168, 0.165), 22, 14), 'torso')                                                 # skull
    host(*ellipsoid((0.0, -0.09, 2.21), (0.125, 0.115, 0.085), 18, 10), 'torso')                                                 # jaw
    host(*ellipsoid((0.0, -0.150, 2.355), (0.13, 0.05, 0.04), 16, 8), 'torso')                                                    # brow
    for s, sx in (('L', 1.0), ('R', -1.0)):
        P = np.array([[0.20, 0.0, 1.12], [0.22, -0.01, 0.92], [0.25, -0.03, 0.66], [0.26, -0.00, 0.42], [0.26, 0.03, 0.19]]) * np.array([sx, 1, 1])
        r = np.array([0.23, 0.215, 0.165, 0.165, 0.11])
        host(*tube(P, r, r * 1.04, sides=22, caps=(True, True), up=(1.0, 0.0, 0.0)), 'leg_' + s)
        host(*ellipsoid((0.26 * sx, 0.07, 0.46), (0.13, 0.14, 0.22), 14, 8), 'leg_' + s)
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        path = [sh] + [lerp(sh, el, u) for u in (0.25, 0.5, 0.8)] + [el]
        rad = [0.155, 0.17, 0.175, 0.15, 0.13]
        if s == 'L':
            path += [lerp(el, wr, u) for u in (0.25, 0.5, 0.8)] + [wr]
            rad += [0.145, 0.14, 0.12, 0.10]
        else:
            path += [lerp(el, wr, u) for u in (0.25, 0.5)]
            rad += [0.145, 0.14]
        host(*tube(np.array(path), np.array(rad), np.array(rad) * 1.03, sides=22, caps=(True, True), up=(0.0, 1.0, 0.0)), 'arm_' + s)
        host(*ellipsoid(sh + np.array([0.0, 0.0, 0.03]), (0.185, 0.18, 0.19), 18, 12), 'arm_' + s)
    # surgical seam (material 1): a stitched diagonal cut across the belly
    for u in np.linspace(0.0, 1.0, 13):
        x, z = -0.22 + 0.40 * u, 1.50 - 0.22 * u
        rx_, ry_, cy_ = np.interp(z, TZ_C, TRX_C), np.interp(z, TZ_C, TRY_C), np.interp(z, TZ_C, TCY_C)
        y = cy_ - ry_ * math.sqrt(max(1.0 - (x / rx_) ** 2, 0.0))
        host(*rbox((x, y - 0.003, z), (0.032, 0.010, 0.012), 0.003, rot_y(0.50)), 'torso', mat=1)
        host(*rbox((x, y - 0.005, z), (0.008, 0.008, 0.050), 0.002, rot_y(0.50)), 'torso', mat=1)
    return S


def torso_tree_C():
    V, P = merge(TORSO_C)
    return BVHTree.FromPolygons([tuple(v) for v in V], [tuple(p) for p in P])


def parts_C(S=None):
    P = []

    def add(name, vp, mk, bone, smooth=True):
        P.append((name, vp[0], vp[1], mk, bone, smooth))

    tree = torso_tree_C()

    def hit(x, z, front=True):
        p, n = surface_hit(tree, x, z, front)
        if p is None:
            return None, None
        n = unit(n)
        if (n[1] > 0) == front:
            n = -n
        return p, n

    # --- steel ribcage over the chest, a sternum bar
    for k, z in enumerate((1.60, 1.70, 1.80, 1.90)):
        for s, sx in (('L', 1.0), ('R', -1.0)):
            pts = []
            for x in np.linspace(0.06, 0.40, 8):
                p, n = hit(sx * x, z - 0.05 * (x / 0.40) ** 2, True)
                if p is not None:
                    pts.append(p + n * 0.030)
            if len(pts) >= 3:
                add('Rib%d%s' % (k, s), stube(resample(pts, 12), 0.026, 0.012, n=2.2, sides=10, up=(0, 0, 1)), 'steel', 'chest')
                end = pts[-1]
                add('RibBolt%d%s' % (k, s), ellipsoid(end, (0.022, 0.022, 0.022), 10, 6), 'darksteel', 'chest')
    p, n = hit(0.0, 1.76, True)
    add('Sternum', rbox(p + n * 0.035, (0.085, 0.035, 0.46), 0.012), 'steel', 'chest')
    for k in range(5):
        add('SternBolt%d' % k, ellipsoid(p + n * 0.055 + np.array([0, 0, -0.18 + 0.09 * k]), (0.014, 0.008, 0.014), 8, 6), 'darksteel', 'chest')
    # --- external spine: vertebrae with short dorsal fins down the back
    for k, z in enumerate(np.linspace(1.14, 2.02, 11)):
        p, n = hit(0.0, z, False)
        if p is None:
            continue
        bone = 'pelvis' if z < 1.30 else ('spine' if z < 1.56 else 'chest')
        c = p + n * 0.035
        add('Vert%d' % k, rbox(c, (0.11, 0.07, 0.065), 0.016), 'steel', bone)
        add('Fin%d' % k, rbox(c + n * 0.055 + np.array([0, 0, 0.01]), (0.028, 0.07, 0.05), 0.008), 'steel', bone)
    for k, (y, z) in enumerate(((0.205, 2.10), (0.175, 2.18))):
        add('NeckVert%d' % k, rbox((0.0, y, z), (0.09, 0.06, 0.06), 0.014), 'steel', 'neck')
    add('SkullSocket', cyl((0.0, 0.10, 2.25), (0.0, 0.19, 2.24), 0.055, 0.05, sides=16), 'darksteel', 'head')
    # --- the back canister and teal feed tubes
    add('Canister', cyl((0.0, 0.42, 1.60), (0.0, 0.42, 1.92), 0.10, 0.10, sides=18), 'steel', 'chest')
    add('CanisterCap', dome(np.array([0.0, 0.42, 1.92]), 0.10, 0.10, 0.05, (0, 0, 1)), 'darksteel', 'chest')
    for k, x in enumerate((-0.045, 0.045)):
        add('CanWin%d' % k, rbox((x, 0.515, 1.76), (0.03, 0.012, 0.22), 0.005), 'fluid', 'chest')
    add('FeedHead', tube(resample(np.array([[0, 0.45, 1.93], [0, 0.40, 2.06], [0, 0.29, 2.18], [0, 0.20, 2.245]]), 14), 0.022, sides=10, up=(1, 0, 0)), 'tube', 'neck')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        fp = np.array([[sx * 0.06, 0.44, 1.88], [sx * 0.30, 0.36, 2.02], [sx * 0.55, 0.20, 2.02], [sx * 0.70, 0.08, 1.96]])
        add('FeedTube' + s, tube(resample(fp, 14), 0.018, sides=10, up=(1, 0, 0)), 'tube', 'clavicle.' + s)
    # --- head: halo ring with pins, steel face plate with ONE amber lens, jaw cage
    hc = np.array([0.0, -0.025, 2.40])
    hax = unit((0.0, 0.25, 1.0))
    add('Halo', torus(hc, 0.235, 0.016, axis=hax, nu=40, nv=8), 'steel', 'head')
    Bh = basis(hax, (0, -1, 0))
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        rp = hc + (Bh[:, 0] * math.cos(a) + Bh[:, 1] * math.sin(a)) * 0.235
        add('HaloPin%d' % k, cyl(rp, rp + (hc - rp) * 0.33, 0.013, 0.009, sides=8), 'steel', 'head')
    # one lens replaces both eyes: a housing driven into the skull, a bolted flange, cheek plates
    lz = 2.325
    add('LensHousing', cyl((0.0, -0.15, lz), (0.0, -0.248, lz), 0.068, 0.062, sides=24), 'darksteel', 'head')
    add('LensFlange', torus((0.0, -0.200, lz), 0.080, 0.014, axis=(0, 1, 0), nu=32, nv=8), 'steel', 'head')
    for k in range(6):
        a = math.pi / 6 + k * math.pi / 3
        add('FlangeBolt%d' % k, ellipsoid((0.096 * math.cos(a), -0.196, lz + 0.096 * math.sin(a)), (0.011, 0.008, 0.011), 8, 6), 'darksteel', 'head')
    add('Lens', ellipsoid((0.0, -0.249, lz), (0.050, 0.012, 0.050), 20, 10), 'lens', 'head')
    add('Iris', torus((0.0, -0.252, lz), 0.053, 0.006, axis=(0, 1, 0)), 'steel', 'head')
    for sx in (-1.0, 1.0):
        add('CheekPlate%d' % (sx > 0), rbox((sx * 0.105, -0.155, 2.245), (0.06, 0.035, 0.085), 0.01, rot_z(sx * 0.55)), 'steel', 'head')
    add('PinLensHousing', cyl((0.10, -0.18, 2.29), (0.10, -0.205, 2.29), 0.016, 0.015, sides=12), 'darksteel', 'head')
    add('PinLens', ellipsoid((0.10, -0.206, 2.29), (0.011, 0.004, 0.011), 10, 6), 'lens', 'head')
    jp = np.array([[-0.135, -0.03, 2.29], [-0.13, -0.14, 2.17], [-0.06, -0.20, 2.125], [0.0, -0.21, 2.12], [0.06, -0.20, 2.125], [0.13, -0.14, 2.17], [0.135, -0.03, 2.29]])
    add('JawCage', tube(resample(jp, 22), 0.013, sides=8, up=(0, 0, 1)), 'steel', 'head')
    for x in (-0.045, 0.045):
        add('JawBar%d' % (x > 0), cyl((x, -0.205, 2.125), (x, -0.185, 2.215), 0.009, 0.009, sides=8), 'steel', 'head')
    # --- shoulders: bolted joint implants
    for s, sx in (('L', 1.0), ('R', -1.0)):
        sh = J['sh_' + s]
        c0, c1 = sh + np.array([sx * 0.15, 0.0, 0.06]), sh + np.array([sx * 0.215, 0.0, 0.065])
        add('Implant' + s, cyl(c0, c1, 0.115, 0.105, sides=8), 'steel', 'upperarm.' + s, smooth=False)
        for k in range(6):
            a = 2 * math.pi * k / 6
            c = c1 + np.array([0.0, 0.075 * math.cos(a), 0.075 * math.sin(a)])
            add('ImplantBolt%s%d' % (s, k), cyl(c, c + np.array([sx * 0.016, 0, 0]), 0.013, 0.013, sides=6), 'darksteel', 'upperarm.' + s)
    # --- right arm: cleaver blade on a steel mount; left: flesh fist with a steel knuckle plate
    el, wr, hd = J['el_R'], J['wr_R'], J['hd_R']
    d = unit(wr - el)
    add('BladeMount', stube([lerp(el, wr, 0.40), lerp(el, wr, 0.70), wr + d * 0.03], [0.15, 0.165, 0.15], n=3.0, sides=28, up=(0, 1, 0)), 'steel', 'forearm.R')
    add('MountClamp', torus(lerp(el, wr, 0.40), 0.152, 0.020, axis=d), 'darksteel', 'forearm.R')
    dh = unit(hd - wr)
    f0 = unit(np.array([0.0, -1.0, 0.0]) - dh * (-dh[1]))
    t0 = unit(np.cross(dh, f0))
    tw = math.radians(25.0)                       # edge forward and a little inward, so the flat of the blade faces front-out
    f = f0 * math.cos(tw) - t0 * math.sin(tw)
    t = unit(np.cross(dh, f))
    add('BladeRoot', rbox(wr + dh * 0.05, (0.17, 0.17, 0.14), 0.02, basis(dh, f), n=2), 'darksteel', 'hand.R')
    outline = [(0.02, -0.08), (0.52, -0.08), (0.63, -0.05), (0.68, 0.04), (0.65, 0.16), (0.57, 0.28), (0.10, 0.26), (0.03, 0.13)]
    outline = [tuple(p) for p in resample(np.array([[u, v, 0.0] for (u, v) in outline + [outline[0]]]), 44)[:-1, :2]]
    add('Blade', prism2d(outline, wr + dh * 0.06, dh, f, t, lambda u, v: 0.020 - 0.018 * max(0.0, min(1.0, (v + 0.02) / 0.29))), 'blade', 'hand.R', smooth=False)
    for k, off in enumerate((-0.05, 0.05)):
        add('BladeRod%d' % k, cyl(lerp(el, wr, 0.72) + f * (-0.13) + t * off, wr + dh * 0.20 + f * (-0.06) + t * off, 0.016, 0.016, sides=8), 'chrome', 'hand.R')
    wl = J['wr_L']
    fist = wl + np.array([0.0, -0.01, -0.13])
    add('FistL', rbox(fist, (0.23, 0.20, 0.21), 0.06), 'fleshR', 'hand.L')
    for k, xo in enumerate((-0.078, -0.026, 0.026, 0.078)):
        add('FingerL%d' % k, rbox(fist + np.array([xo, -0.08, -0.09]), (0.05, 0.075, 0.12), 0.022), 'fleshR', 'hand.L')
    add('ThumbL', rbox(fist + np.array([-0.12, -0.05, -0.02]), (0.055, 0.085, 0.13), 0.024, rot_z(0.3)), 'fleshR', 'hand.L')
    add('KnucklePlateL', rbox(fist + np.array([0.0, -0.125, -0.06]), (0.22, 0.03, 0.07), 0.012), 'steel', 'hand.L')
    # --- pelvis: a steel girdle and surgical canvas flaps
    add('Girdle', zshell([1.08, 1.16], [0.335, 0.345], [0.245, 0.255], [0.0, 0.0], n=2.4, caps=(False, False)), 'darksteel', 'pelvis')
    add('FlapF', rbox((0.0, -0.26, 0.96), (0.34, 0.025, 0.30), 0.01, rot_x(0.12)), 'canvas', 'pelvis')
    add('FlapB', rbox((0.0, 0.26, 0.96), (0.36, 0.025, 0.30), 0.01, rot_x(-0.12)), 'canvas', 'pelvis')
    # --- legs: orthopaedic braces (outer rods, knee hinge), steel sabots
    for s, sx in (('L', 1.0), ('R', -1.0)):
        kn, an, ball, toe = J['kn_' + s], J['an_' + s], J['ball_' + s], J['toe_' + s]
        k0 = np.array([sx * 0.47, -0.03, 0.66])
        add('BraceThigh' + s, cyl(np.array([sx * 0.47, 0.0, 1.08]), k0 + np.array([0, 0, 0.05]), 0.020, 0.020, sides=8), 'steel', 'thigh.' + s)
        add('BraceClampT' + s, rbox(np.array([sx * 0.455, -0.01, 0.90]), (0.04, 0.16, 0.06), 0.01), 'darksteel', 'thigh.' + s)
        add('BraceHinge' + s, cyl(k0 - np.array([sx * 0.02, 0, 0]), k0 + np.array([sx * 0.03, 0, 0]), 0.065, 0.065, sides=16), 'steel', 'shin.' + s)
        add('BraceShin' + s, cyl(k0 - np.array([0, 0, 0.05]), np.array([sx * 0.40, 0.03, 0.27]), 0.018, 0.018, sides=8), 'steel', 'shin.' + s)
        add('BraceClampS' + s, rbox(np.array([sx * 0.43, 0.0, 0.46]), (0.04, 0.15, 0.06), 0.01), 'darksteel', 'shin.' + s)
        add('SabotHeel' + s, rbox(an + np.array([0, 0.04, -0.08]), (0.28, 0.30, 0.22), 0.09, n=3), 'darksteel', 'foot.' + s)
        add('SabotToe' + s, rbox(ball + np.array([0, -0.08, -0.02]), (0.30, 0.34, 0.13), 0.06, n=3), 'steel', 'toe.' + s)
        add('SabotSole' + s, rbox(lerp(an, toe, 0.42) + np.array([0, 0, -0.10]), (0.32, 0.74, 0.05), 0.02), 'rubber', 'foot.' + s)
    return P


def mats_C(body):
    M = common_mats({})
    M['flesh'] = flesh_mat('KD2_Flesh', body)
    M['fleshR'] = flesh_mat('KD2_FleshRigid', None)
    M['scar'] = pbr('KD2_Scar', (0.16, 0.06, 0.08), (0.24, 0.09, 0.11), rough=0.5, nscale=20, bump=0.6, ob=body)
    M['steel'] = pbr('KD2_SteelBrushed', (0.40, 0.40, 0.42), (0.52, 0.52, 0.54), metal=1.0, rough=0.28, nscale=8, bump=0.15, dirt=0.7, dirt_col=(0.05, 0.04, 0.035))
    M['blade'] = pbr('KD2_Blade', (0.55, 0.55, 0.57), (0.68, 0.68, 0.70), metal=1.0, rough=0.30, nscale=12, bump=0.12, dirt=0.5, wear_col=(0.25, 0.12, 0.08), wear=0.3)
    M['tube'] = pbr('KD2_TealTube', (0.02, 0.16, 0.16), (0.04, 0.24, 0.23), rough=0.4, dirt=0.5)
    M['fluid'] = glow_mat('KD2_Fluid', (0.10, 0.90, 0.75), 2.0)
    M['lens'] = glow_mat('KD2_Lens', (1.0, 0.50, 0.06), 7.0)
    M['canvas'] = pbr('KD2_Canvas', (0.12, 0.13, 0.12), (0.20, 0.21, 0.19), rough=0.9, nscale=5, bump=0.5, bscale=120, dirt=1.0)
    return M


DIRS = {
    'A': dict(title="Reclaimer", skin=lambda: skin_core(False), parts=parts_A, mats=mats_A, body=['suit'], head_z=2.31),
    'B': dict(title="Breaker", skin=lambda: skin_core(True, head=False), parts=parts_B, mats=mats_B, body=['core'], head_z=2.21),
    'C': dict(title="Grown", skin=skin_C, parts=parts_C, mats=mats_C, body=['flesh', 'scar'], head_z=2.31),
}


# ================================================================== FINAL: the Breaker, realistic, in Relic's house style (style_mats.py)
exec(compile(open(r"C:\ai\unedited\style_mats.py", encoding="utf-8").read(), "style_mats.py", "exec"), _g)
EX, EY, EZ = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
JAW = {}


def spline(pts, n):
    """Catmull-Rom curve through pts, about n points"""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return resample(P, n)
    Q = np.concatenate([[2 * P[0] - P[1]], P, [2 * P[-1] - P[-2]]])
    segs = len(P) - 1
    k = max(2, n // segs)
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        for t in np.linspace(0.0, 1.0, k, endpoint=(i == segs - 1)):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def ngon(p0, p1, r0, r1=None, k=8):
    """a k-sided prism with a flat side on top, from p0 to p1"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    up = (0.0, 0.0, 1.0) if abs(unit(p1 - p0)[2]) < 0.9 else (0.0, 1.0, 0.0)
    return stube([p0, p1], [r0, r0 if r1 is None else r1], n=2.0, sides=k, caps=(True, True), up=up, phase=math.pi / k)


def jaw_geometry():
    """the two crusher jaws: bone -> (pivot, tip, path, side). claw1a.R = outer jaw (away from the body), claw2a.R = inner jaw"""
    wr, hd = J['wr_R'], J['hd_R']
    d = unit(hd - wr)
    rel = [(0.095, 0.15), (0.20, 0.27), (0.235, 0.41), (0.18, 0.53), (0.065, 0.60)]
    out = {}
    for sgn, bone in ((-1.0, 'claw1a.R'), (1.0, 'claw2a.R')):
        path = spline([wr + EX * (sgn * xx) + d * zz for (xx, zz) in rel], 13)
        out[bone] = (path[0].copy(), path[-1].copy(), path, sgn)
    return out


def parts_B2():
    """the realistic Breaker: (parts, pistons, lamps). parts are (name, V, polys, material, bone, smooth) in the design frame"""
    P, PIST, LAMPS = [], [], []

    def add(name, vp, mk, bone, smooth=True):
        P.append((name, vp[0], vp[1], mk, bone, smooth))

    def piston(name, a, bone_a, b, bone_b, rb, rr):
        PIST.append((name, np.asarray(a, float), bone_a, np.asarray(b, float), bone_b, rb, rr))

    def bolt(c, n, bone, r=0.011, h=0.009, mk='steel'):
        n, c = unit(n), np.asarray(c, float)
        add('Bolt', stube([c - n * 0.003, c + n * h], r, n=2.0, sides=6, up=(0, 0, 1) if abs(n[2]) < 0.9 else (0, 1, 0), phase=math.pi / 6), mk, bone, smooth=False)

    def bolts(pts, n, bone, r=0.011, mk='steel'):
        for c in pts:
            bolt(c, n, bone, r, mk=mk)

    def ring_pts(c, axis, R, k, phase=0.0):
        B_ = basis(axis, (0, 0, 1) if abs(unit(axis)[2]) < 0.9 else (0, 1, 0))
        return [np.asarray(c, float) + (B_[:, 0] * math.cos(phase + 2 * math.pi * i / k) + B_[:, 1] * math.sin(phase + 2 * math.pi * i / k)) * R for i in range(k)]

    def hose(name, pts, r, bone, n=26):
        path = spline(pts, n)
        add(name, tube(path, r, sides=10, caps=(True, True), up=(1, 0, 0)), 'rubber', bone)
        for p, q in ((path[0], path[1]), (path[-1], path[-2])):
            dd = unit(q - p)
            add(name + 'Fit', stube([p - dd * 0.004, p + dd * 0.035], r * 1.4, n=2.0, sides=6, up=(0, 0, 1) if abs(dd[2]) < 0.9 else (0, 1, 0)),
                'steel', bone, smooth=False)

    def plate(name, c, size, R, mk, bone, r=0.008, bolt_n=None, inset=0.03):
        """a plate with four corner bolts on the face that points along bolt_n"""
        add(name, rbox(c, size, r, R, n=3), mk, bone)
        if bolt_n is not None:
            R_ = np.eye(3) if R is None else np.asarray(R)
            n = unit(bolt_n)
            ln = R_.T @ n
            ax = int(np.argmax(np.abs(ln)))
            o1, o2 = [i for i in range(3) if i != ax]
            pts = []
            for a in (-1, 1):
                for b in (-1, 1):
                    l = np.zeros(3)
                    l[ax] = np.sign(ln[ax]) * size[ax] / 2
                    l[o1] = a * (size[o1] / 2 - inset)
                    l[o2] = b * (size[o2] / 2 - inset)
                    pts.append(sub(c, R_, l))
            bolts(pts, n, bone)

    # ====================== torso: dark core frame, bolted olive armour, radiator grille
    CZ = np.array([1.46, 1.58, 1.72, 1.86, 1.98, 2.06])
    CRX = np.array([0.40, 0.46, 0.52, 0.545, 0.51, 0.40])
    CRY = np.array([0.27, 0.30, 0.32, 0.33, 0.31, 0.25])
    CCY = np.array([0.0, -0.01, -0.02, -0.02, -0.01, 0.0])
    add('ChestCore', zshell(CZ, CRX, CRY, CCY, n=5.0, sides=56), 'olive2', 'chest')
    fy = lambda x, z: sfront(x, z, CZ, CRX, CRY, CCY, 5.0)
    add('ChestPlate', fplate(fy, (-0.37, 0.37), (-0.44, 0.44), 1.50, 1.97, lift=0.004, thick=0.045, inset=0.016, nx=16, nz=10), 'olive', 'chest')
    yp = fy(0.0, 1.75) - 0.049
    for z in (1.592, 1.908):
        add('GrilleBarH', rbox((0.0, yp - 0.016, z), (0.46, 0.032, 0.036), 0.008, n=3), 'olive2', 'chest')
    for x in (-0.212, 0.212):
        add('GrilleBarV', rbox((x, yp - 0.016, 1.75), (0.036, 0.032, 0.28), 0.008, n=3), 'olive2', 'chest')
    add('GrilleBack', rbox((0.0, yp - 0.003, 1.75), (0.40, 0.006, 0.29), 0.002), 'darksteel', 'chest')
    for k in range(9):
        add('GrilleFin', rbox((-0.16 + 0.04 * k, yp - 0.016, 1.75), (0.010, 0.026, 0.282), 0.003), 'plate', 'chest')
    bolts([(x, fy(x, 1.54) - 0.049, 1.54) for x in (-0.33, 0.33)] + [(x, fy(x, 1.93) - 0.049, 1.93) for x in (-0.40, 0.40)], (0, -1, 0), 'chest', r=0.013)
    dp = np.array([0.29, fy(0.29, 1.555) - 0.051, 1.555])
    add('DangerPlate', rbox(dp, (0.13, 0.005, 0.06), 0.002), 'yellow', 'chest')
    add('TxtDanger', text_vp("DANGER", 0.030, dp + np.array([0.0, -0.003, 0.0]), (0, -1, 0), (0, 0, 1)), 'black', 'chest', smooth=False)
    # ====================== engine hump: heat-sink fins, louvres, lifting eyes, stencils, exhaust stacks with rain caps
    Ch, Rh = np.array([0.0, 0.26, 2.16]), rot_x(0.26)
    add('Hump', rbox(Ch, (1.00, 0.60, 0.54), 0.025, Rh, n=3), 'olive', 'chest')
    add('FinBase', rbox(sub(Ch, Rh, (0.0, 0.0, 0.274)), (0.84, 0.46, 0.008), 0.003, Rh), 'darksteel', 'chest')
    for k in range(9):
        add('Fin', rbox(sub(Ch, Rh, (0.0, -0.20 + 0.05 * k, 0.300)), (0.80, 0.008, 0.050), 0.002, Rh), 'darksteel', 'chest')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('LouvrePanel' + s, rbox(sub(Ch, Rh, (sx * 0.502, 0.02, -0.07)), (0.008, 0.42, 0.27), 0.003, Rh), 'darksteel', 'chest')
        for k in range(6):
            add('Louvre' + s, rbox(sub(Ch, Rh, (sx * 0.513, 0.02, -0.185 + 0.045 * k)), (0.010, 0.40, 0.050), 0.003, Rh @ rot_y(sx * 0.5)), 'olive2', 'chest')
        add('LiftEye' + s, torus(sub(Ch, Rh, (sx * 0.34, 0.14, 0.318)), 0.042, 0.011, axis=Rh @ EX, nu=24, nv=8), 'steel', 'chest')
        add('LiftEyeBase' + s, rbox(sub(Ch, Rh, (sx * 0.34, 0.14, 0.281)), (0.03, 0.11, 0.02), 0.004, Rh), 'steel', 'chest')
        add('HumpBracket' + s, rbox((sx * 0.30, 0.30, 1.86), (0.04, 0.20, 0.22), 0.008), 'olive2', 'chest')
        b0, b1 = np.array([sx * 0.30, 0.44, 2.22]), np.array([sx * 0.33, 0.49, 2.50])
        ax = unit(b1 - b0)
        add('Stack' + s, cyl(b0, lerp(b0, b1, 0.80), 0.068, 0.066, sides=24), 'darksteel', 'chest')
        add('StackTip' + s, cyl(lerp(b0, b1, 0.80), b1, 0.066, 0.064, sides=24, caps=False), 'soot', 'chest')
        add('StackRim' + s, torus(b1, 0.064, 0.006, axis=ax, nu=24, nv=6), 'soot', 'chest')
        for u in (0.18, 0.55):
            c = lerp(b0, b1, u)
            add('StackClamp' + s, cyl(c - ax * 0.018, c + ax * 0.018, 0.077, 0.077, sides=24), 'olive2', 'chest')
        Vc, Pc = cyl(b1 + ax * 0.004, b1 + ax * 0.010, 0.074, 0.074, sides=24)
        hinge = b1 + np.array([0.0, 0.074, 0.0])
        add('RainCap' + s, ((Vc - hinge) @ rot_x(-0.55).T + hinge, Pc), 'darksteel', 'chest')
    add('TxtSide', text_vp("KD-B7", 0.10, sub(Ch, Rh, (0.503, 0.02, 0.15)), Rh @ EX, Rh @ EZ), 'stencil', 'chest', smooth=False)
    add('FrontHatch', rbox(sub(Ch, Rh, (0.30, -0.305, 0.05)), (0.26, 0.010, 0.22), 0.004, Rh), 'olive2', 'chest')
    bolts([sub(Ch, Rh, (0.30 + x, -0.311, 0.05 + z)) for x in (-0.10, 0.10) for z in (-0.08, 0.08)], Rh @ (-EY), 'chest', r=0.010)
    add('BackPanel', rbox(sub(Ch, Rh, (0.0, 0.304, -0.03)), (0.74, 0.012, 0.40), 0.004, Rh), 'olive2', 'chest')
    bolts([sub(Ch, Rh, (x, 0.311, z)) for x in (-0.34, 0.34) for z in (-0.20, 0.14)], Rh @ EY, 'chest', r=0.012)
    add('TxtBack', text_vp("STARCAPITAL", 0.072, sub(Ch, Rh, (0.0, 0.311, 0.06)), Rh @ EY, Rh @ EZ), 'stencil', 'chest', smooth=False)
    add('TxtBack2', text_vp("ASSET RECOVERY", 0.040, sub(Ch, Rh, (0.0, 0.311, -0.02)), Rh @ EY, Rh @ EZ), 'stencil', 'chest', smooth=False)
    # ====================== head: turret under the hump, six work lamps behind a wire guard, sensor pod, corrugated rubber neck
    Ct, Rt = np.array([0.0, -0.19, 2.21]), rot_x(-0.06)
    add('Turret', rbox(Ct, (0.40, 0.30, 0.23), 0.02, Rt, n=3), 'olive', 'head')
    add('TurretFace', rbox(sub(Ct, Rt, (0.0, -0.153, -0.005)), (0.36, 0.010, 0.19), 0.003, Rt), 'darksteel', 'head')
    fwd = Rt @ np.array([0.0, -1.0, 0.0])
    for k, (lx, lz, r) in enumerate(((-0.115, 0.042, 0.032), (0.0, 0.042, 0.040), (0.115, 0.042, 0.032),
                                     (-0.115, -0.048, 0.026), (0.0, -0.048, 0.030), (0.115, -0.048, 0.026))):
        c0 = sub(Ct, Rt, (lx, -0.157, lz))
        c1 = c0 + fwd * 0.040
        add('LampCup', cyl(c0, c1, r + 0.012, r + 0.010, sides=24), 'darksteel', 'head')
        add('LampSeal', torus(c1, r + 0.007, 0.0045, axis=fwd, nu=24, nv=6), 'rubber', 'head')
        add('LampLens', ellipsoid(c1 - fwd * 0.003, (r, 0.009, r), 18, 8, R=Rt), 'lamp', 'head')
        LAMPS.append((c1 + fwd * 0.012, fwd))
    gy = -0.222
    for x in (-0.175, -0.058, 0.058, 0.175):
        add('GuardV', cyl(sub(Ct, Rt, (x, gy, -0.100)), sub(Ct, Rt, (x, gy, 0.092)), 0.0065, 0.0065, sides=8), 'steel', 'head')
    for z in (-0.100, 0.092):
        add('GuardH', cyl(sub(Ct, Rt, (-0.175, gy, z)), sub(Ct, Rt, (0.175, gy, z)), 0.0065, 0.0065, sides=8), 'steel', 'head')
        for x in (-0.175, 0.175):
            add('GuardPost', cyl(sub(Ct, Rt, (x, -0.152, z)), sub(Ct, Rt, (x, gy, z)), 0.008, 0.008, sides=8), 'steel', 'head')
    add('SensorPod', cyl(sub(Ct, Rt, (0.125, -0.07, 0.140)), sub(Ct, Rt, (0.125, -0.15, 0.140)), 0.024, 0.024, sides=20), 'darksteel', 'head')
    add('SensorLens', ellipsoid(sub(Ct, Rt, (0.125, -0.151, 0.140)), (0.016, 0.006, 0.016), 14, 8), 'lensdark', 'head')
    add('SensorMount', rbox(sub(Ct, Rt, (0.125, -0.09, 0.122)), (0.04, 0.07, 0.016), 0.004, Rt), 'darksteel', 'head')
    add('AntennaBase', cyl(sub(Ct, Rt, (-0.14, 0.08, 0.115)), sub(Ct, Rt, (-0.14, 0.08, 0.145)), 0.016, 0.014, sides=12), 'darksteel', 'head')
    add('Antenna', cyl(sub(Ct, Rt, (-0.14, 0.08, 0.145)), sub(Ct, Rt, (-0.13, 0.10, 0.32)), 0.004, 0.0025, sides=6), 'steel', 'head')
    for sx in (-1.0, 1.0):
        bolts([sub(Ct, Rt, (sx * 0.201, y, z)) for y in (-0.08, 0.06) for z in (-0.06, 0.06)], Rt @ (EX * sx), 'head', r=0.010)
    zz = np.linspace(0.0, 1.0, 22)
    path = np.array([lerp(np.array([0.0, -0.11, 2.00]), np.array([0.0, -0.16, 2.11]), t) for t in zz])
    add('NeckBellows', tube(path, 0.13 + 0.008 * np.cos(zz * 2 * np.pi * 5), sides=24, caps=(False, False), up=(1, 0, 0)), 'rubber', 'neck')
    # ====================== waist: spine column, slewing ring, tilt rams, cable run
    add('SpineColumn', cyl((0, 0.03, 1.27), (0, 0.03, 1.47), 0.15, 0.15, sides=28), 'darksteel', 'spine')
    add('SlewRing', cyl((0, 0.03, 1.40), (0, 0.03, 1.45), 0.30, 0.30, sides=48), 'darksteel', 'spine')
    add('SlewLip', torus((0, 0.03, 1.452), 0.27, 0.010, axis=(0, 0, 1), nu=48, nv=6), 'steel', 'spine')
    bolts(ring_pts((0, 0.03, 1.452), (0, 0, 1), 0.285, 18), (0, 0, 1), 'spine', r=0.010)
    for s, sx in (('L', 1.0), ('R', -1.0)):
        piston('Waist' + s, (sx * 0.30, -0.19, 1.322), 'pelvis', (sx * 0.30, -0.20, 1.462), 'chest', 0.040, 0.024)
    for k, x in enumerate((-0.10, -0.035, 0.035, 0.10)):
        hose('WaistCable%d' % k, [(x, 0.20, 1.30), (x * 1.1, 0.30, 1.39), (x, 0.25, 1.49)], 0.016, 'spine')
    # ====================== pelvis: cast block, skid plate, hip actuators with brass caps
    add('PelvisBlock', rbox((0.0, 0.02, 1.19), (0.60, 0.44, 0.26), 0.022, n=3), 'olive', 'pelvis')
    plate('SkidPlate', np.array([0.0, -0.228, 1.115]), (0.44, 0.035, 0.22), rot_x(0.35), 'plate', 'pelvis', bolt_n=rot_x(0.35) @ np.array([0, -1.0, 0]))
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('HipDrum' + s, ngon((sx * 0.10, 0.0, 1.16), (sx * 0.37, 0.0, 1.16), 0.15), 'darksteel', 'pelvis', smooth=False)
        add('HipCap' + s, cyl((sx * 0.37, 0.0, 1.16), (sx * 0.395, 0.0, 1.16), 0.092, 0.086, sides=32), 'brass', 'pelvis')
        bolts(ring_pts((sx * 0.37, 0.0, 1.16), (sx, 0, 0), 0.122, 8, math.pi / 8), (sx, 0, 0), 'pelvis', r=0.011)
    # ====================== shoulders: octagonal actuator drums, brass caps, hoses from the hump
    for s, sx in (('L', 1.0), ('R', -1.0)):
        cb = 'clavicle.' + s
        add('ShoulderDrum' + s, ngon((sx * 0.46, 0.0, 1.90), (sx * 0.78, 0.0, 1.90), 0.20), 'darksteel', cb, smooth=False)
        add('ShoulderPlate' + s, ngon((sx * 0.78, 0.0, 1.90), (sx * 0.81, 0.0, 1.90), 0.186, 0.180), 'olive', cb, smooth=False)
        add('ShoulderCap' + s, cyl((sx * 0.81, 0.0, 1.90), (sx * 0.845, 0.0, 1.90), 0.095, 0.088, sides=32), 'brass', cb)
        bolts(ring_pts((sx * 0.81, 0.0, 1.90), (sx, 0, 0), 0.142, 8, math.pi / 8), (sx, 0, 0), cb, r=0.013)
        add('ShoulderMount' + s, rbox((sx * 0.40, 0.0, 2.035), (0.22, 0.30, 0.09), 0.012, n=3), 'olive2', cb)
        hose('HoseShoulderA' + s, [(sx * 0.40, 0.44, 2.04), (sx * 0.62, 0.32, 2.10), (sx * 0.76, 0.12, 2.06)], 0.026, cb)
        hose('HoseShoulderB' + s, [(sx * 0.34, 0.45, 1.97), (sx * 0.60, 0.33, 1.99), (sx * 0.72, 0.16, 1.98)], 0.021, cb)
    # ====================== arms: box booms with lightening holes, elbow pin and clevis, elbow ram, forearm housings
    for s, sx in (('L', 1.0), ('R', -1.0)):
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        ub, fb = 'upperarm.' + s, 'forearm.' + s
        da = unit(el - sh)
        out = sx * unit(np.cross(da, EY))
        add('Boom' + s, stube([lerp(sh, el, 0.14), lerp(sh, el, 0.86)], 0.13, 0.15, n=6.0, sides=44, up=(0, 1, 0)), 'olive', ub)
        for u in (0.40, 0.66):
            c = lerp(sh, el, u)
            add('Hole' + s, cyl(c + out * 0.146, c + out * 0.153, 0.046, 0.046, sides=24), 'darksteel', ub)
            add('HoleRim' + s, torus(c + out * 0.152, 0.049, 0.006, axis=out, nu=24, nv=6), 'olive2', ub)
        hose('HoseBoom' + s, [sh + np.array([sx * 0.10, 0.12, 0.02]), lerp(sh, el, 0.5) + np.array([sx * 0.06, 0.165, 0.0]), el + np.array([sx * 0.02, 0.19, 0.04])], 0.022, ub)
        add('ElbowPin' + s, cyl(el - EX * 0.17, el + EX * 0.17, 0.064, 0.064, sides=24), 'steel', ub)
        for sg in (-1.0, 1.0):
            add('ElbowCap' + s, cyl(el + EX * sg * 0.17, el + EX * sg * 0.195, 0.076, 0.068, sides=32), 'brass', ub)
            add('Clevis' + s, rbox(el + EX * sg * 0.138 + np.array([0.0, 0.0, -0.07]), (0.034, 0.21, 0.27), 0.008, n=3), 'olive2', fb)
        a = sh + np.array([0.0, -0.17, -0.075])
        b = lerp(el, wr, 0.20) + np.array([0.0, -0.255, 0.0])
        add('RamLugA' + s, rbox(a + np.array([0.0, 0.028, 0.0]), (0.07, 0.06, 0.08), 0.008), 'olive2', ub)
        add('RamLugB' + s, rbox(b + np.array([0.0, 0.030, 0.0]), (0.075, 0.06, 0.085), 0.008), 'olive2', fb)
        piston('Elbow' + s, a, ub, b, fb, 0.056, 0.033)
        add('Forearm' + s, stube([lerp(el, wr, 0.08), lerp(el, wr, 0.50), lerp(el, wr, 0.94)], [0.21, 0.26, 0.24], [0.19, 0.24, 0.22], n=4.5, sides=48, up=(0, 1, 0)),
            'olive', fb)
        for u in (0.27, 0.76):                   # panel seams: the housing is two castings and a cap, not one blob
            rd = float(np.interp(u, [0.08, 0.50, 0.94], [0.21, 0.26, 0.24])) + 0.003
            rw = float(np.interp(u, [0.08, 0.50, 0.94], [0.19, 0.24, 0.22])) + 0.003
            add('ForeSeam' + s, stube([lerp(el, wr, u - 0.008), lerp(el, wr, u + 0.008)], rd, rw, n=4.5, sides=48, caps=(False, False), up=(0, 1, 0)), 'darksteel', fb)
        c = lerp(el, wr, 0.52)
        plate('WearPlate' + s, c + np.array([sx * 0.252, 0.0, 0.0]), (0.03, 0.30, 0.38), None, 'darksteel', fb, bolt_n=(sx, 0, 0))
        plate('ImpactPlate' + s, c + np.array([0.0, -0.272, 0.0]), (0.26, 0.03, 0.36), None, 'plate', fb, bolt_n=(0, -1, 0))
        hose('HoseForeA' + s, [el + np.array([sx * 0.07, 0.19, -0.04]), lerp(el, wr, 0.5) + np.array([sx * 0.09, 0.275, 0.0]), wr + np.array([sx * 0.07, 0.235, 0.06])], 0.021, fb)
        hose('HoseForeB' + s, [el + np.array([-sx * 0.02, 0.19, -0.04]), lerp(el, wr, 0.5) + np.array([-sx * 0.01, 0.28, 0.0]), wr + np.array([-sx * 0.02, 0.24, 0.06])], 0.018, fb)
        add('WristRing' + s, cyl(lerp(el, wr, 0.93), wr + unit(wr - el) * 0.03, 0.205, 0.205, sides=40), 'darksteel', fb)
    # ====================== right: the crusher (rotator, frame, two jaws on their own bones, jaw rams in front, teeth)
    el, wr, hd = J['el_R'], J['wr_R'], J['hd_R']
    d = unit(hd - wr)
    Rd = basis(d, (0, 1, 0))
    add('Rotator', cyl(wr + d * 0.03, wr + d * 0.07, 0.195, 0.195, sides=40), 'darksteel', 'hand.R')
    bolts(ring_pts(wr + d * 0.07, d, 0.165, 12), d, 'hand.R', r=0.010)
    add('CrusherFrame', rbox(wr + d * 0.125, (0.44, 0.30, 0.12), 0.018, Rd, n=3), 'olive', 'hand.R')
    pc_ = wr + d * 0.125 + EY * (-0.153)
    add('CrushPlate', rbox(pc_, (0.17, 0.005, 0.07), 0.002, Rd), 'yellow', 'hand.R')
    add('TxtCrush', text_vp("CRUSH POINT", 0.020, pc_ + EY * (-0.003), (0, -1, 0), -d), 'black', 'hand.R', smooth=False)
    for bone, (piv, tip, path, sgn) in JAW.items():
        add('PivotPin', cyl(piv - EY * 0.165, piv + EY * 0.165, 0.046, 0.046, sides=24), 'steel', 'hand.R')
        for sg in (-1.0, 1.0):
            add('PivotCap', cyl(piv + EY * sg * 0.165, piv + EY * sg * 0.186, 0.058, 0.052, sides=28), 'brass', 'hand.R')
        m = len(path)
        t = np.linspace(0.0, 1.0, m)
        rxj = np.interp(t, [0.0, 0.4, 1.0], [0.135, 0.13, 0.085])
        ryj = np.interp(t, [0.0, 0.4, 1.0], [0.058, 0.052, 0.030])
        add('Jaw', stube(path, rxj, ryj, n=5.0, sides=36, up=(0, 1, 0)), 'hazard', bone)
        T = np.gradient(path, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None]
        O = np.cross(T, EY)
        O *= np.where((O @ EX) * sgn > 0, 1.0, -1.0)[:, None]
        add('JawRib', stube((path + O * (ryj[:, None] + 0.010))[1:-1], 0.026, 0.014, n=4.0, sides=12, up=(0, 1, 0)), 'olive2', bone)
        for k in (5, 7, 9):
            for yo in (-0.06, 0.06):
                base = path[k] - O[k] * (ryj[k] * 0.9) + EY * yo
                add('Tooth', stube([base, base - O[k] * 0.075], [0.027, 0.004], n=2.0, sides=4, up=(0, 1, 0), phase=math.pi / 4), 'steel', bone, smooth=False)
        for k in (3, 8):
            for yo in (-0.08, 0.08):
                bolt(path[k] + O[k] * (ryj[k] + 0.002) + EY * yo, O[k], bone, r=0.012)
        a = wr + d * 0.07 + EX * (sgn * 0.13) + EY * (-0.175)
        b = path[6] + EX * (sgn * 0.02) + EY * (-0.175)
        add('JawRamLugA', rbox(a + EY * 0.03, (0.05, 0.06, 0.06), 0.006), 'olive2', 'hand.R')
        add('JawRamLugB', rbox(b + EY * 0.03, (0.05, 0.06, 0.06), 0.006), 'olive2', bone)
        piston('Jaw' + ('Out' if sgn < 0 else 'In'), a, 'hand.R', b, bone, 0.036, 0.021)
    # ====================== left: a heavy mechanical fist
    wl, hl = J['wr_L'], J['hd_L']
    dl = unit(hl - wl)
    fc = wl + np.array([0.0, -0.02, -0.13])
    add('FistHousing', rbox(fc + np.array([0.0, 0.025, 0.03]), (0.30, 0.24, 0.20), 0.02, n=3), 'olive', 'hand.L')
    kp = fc + np.array([0.0, -0.128, -0.04])
    add('KnucklePlate', rbox(kp, (0.31, 0.034, 0.11), 0.01, rot_x(0.15), n=3), 'plate', 'hand.L')
    bolts([kp + np.array([x, -0.018, 0.0]) for x in (-0.11, 0.0, 0.11)], (0, -1, 0), 'hand.L', r=0.011)
    for k, xo in enumerate((-0.105, -0.035, 0.035, 0.105)):
        add('FingerA', rbox(fc + np.array([xo, -0.095, -0.105]), (0.064, 0.10, 0.11), 0.012, rot_x(0.5), n=3), 'darksteel', 'hand.L')
        add('FingerB', rbox(fc + np.array([xo, -0.035, -0.165]), (0.060, 0.09, 0.075), 0.012, rot_x(1.25), n=3), 'darksteel', 'hand.L')
    add('FingerPin', cyl(fc + np.array([-0.15, -0.11, -0.06]), fc + np.array([0.15, -0.11, -0.06]), 0.017, 0.017, sides=16), 'brass', 'hand.L')
    add('ThumbA', rbox(fc + np.array([-0.165, -0.06, -0.03]), (0.07, 0.10, 0.14), 0.012, rot_z(0.3), n=3), 'darksteel', 'hand.L')
    add('ThumbB', rbox(fc + np.array([-0.15, -0.115, -0.10]), (0.06, 0.08, 0.08), 0.012, rot_z(0.3) @ rot_x(0.6), n=3), 'darksteel', 'hand.L')
    add('WristRingL', cyl(lerp(J['el_L'], wl, 0.93), wl + dl * 0.03, 0.18, 0.18, sides=40), 'darksteel', 'forearm.L')
    # ====================== legs: box beams, hip ram outside, knee drum and guard, knee ram behind, Achilles ram, tread feet
    for s, sx in (('L', 1.0), ('R', -1.0)):
        hip, kn, an, ball, toe = J['hip_' + s], J['kn_' + s], J['an_' + s], J['ball_' + s], J['toe_' + s]
        tb, sb, fb, ob_ = 'thigh.' + s, 'shin.' + s, 'foot.' + s, 'toe.' + s
        add('Thigh' + s, stube([lerp(hip, kn, 0.12), lerp(hip, kn, 0.86)], [0.19, 0.17], [0.21, 0.19], n=6.0, sides=44), 'olive', tb)
        c = lerp(hip, kn, 0.40)
        plate('ThighPlate' + s, c + np.array([0.0, -(0.205 + 0.015), 0.0]), (0.22, 0.03, 0.30), None, 'olive2', tb, bolt_n=(0, -1, 0))
        a, b = np.array([sx * 0.45, 0.04, 1.33]), np.array([sx * 0.45, -0.02, 0.86])
        add('HipRamBracket' + s, rbox(np.array([sx * 0.38, 0.04, 1.33]), (0.16, 0.10, 0.06), 0.008, n=3), 'olive2', 'pelvis')
        add('HipRamLug' + s, rbox(np.array([sx * 0.43, -0.02, 0.86]), (0.06, 0.07, 0.08), 0.008), 'olive2', tb)
        piston('Hip' + s, a, 'pelvis', b, tb, 0.05, 0.03)
        add('KneeDrum' + s, ngon(kn - EX * 0.20, kn + EX * 0.20, 0.125), 'darksteel', sb, smooth=False)
        for sg in (-1.0, 1.0):
            add('KneeCap' + s, cyl(kn + EX * sg * 0.20, kn + EX * sg * 0.222, 0.078, 0.070, sides=28), 'brass', sb)
        add('KneeGuard' + s, rbox(kn + np.array([0.0, -0.175, 0.035]), (0.28, 0.06, 0.24), 0.012, rot_x(0.15), n=3), 'olive', sb)
        bolts([kn + np.array([x, -0.207, 0.035 + z]) for x in (-0.10, 0.10) for z in (-0.08, 0.08)], (0, -1, 0), sb, r=0.011)
        a, b = lerp(hip, kn, 0.30) + np.array([0.0, 0.215, 0.0]), lerp(kn, an, 0.34) + np.array([0.0, 0.205, 0.0])
        add('KneeRamLugA' + s, rbox(a - np.array([0, 0.03, 0]), (0.07, 0.06, 0.07), 0.008), 'olive2', tb)
        add('KneeRamLugB' + s, rbox(b - np.array([0, 0.03, 0]), (0.07, 0.06, 0.07), 0.008), 'olive2', sb)
        piston('Knee' + s, a, tb, b, sb, 0.055, 0.032)
        hose('HoseThigh' + s, [np.array([sx * 0.18, 0.235, 1.17]), lerp(hip, kn, 0.18) + np.array([sx * 0.04, 0.25, 0.0]), lerp(hip, kn, 0.28) + np.array([sx * 0.05, 0.25, -0.03])], 0.018, tb)
        add('Shin' + s, stube([lerp(kn, an, 0.12), lerp(kn, an, 0.88)], [0.17, 0.15], [0.19, 0.16], n=6.0, sides=44), 'olive', sb)
        c = lerp(kn, an, 0.50)
        plate('ShinPlate' + s, c + np.array([0.0, -(0.175 + 0.015), 0.0]), (0.20, 0.03, 0.34), None, 'plate', sb, bolt_n=(0, -1, 0))
        add('AnkleBlock' + s, rbox(an + np.array([0.0, 0.0, 0.01]), (0.26, 0.24, 0.17), 0.015, n=3), 'darksteel', fb)
        add('AnklePin' + s, cyl(an - EX * 0.165, an + EX * 0.165, 0.048, 0.048, sides=24), 'steel', sb)
        for sg in (-1.0, 1.0):
            add('AnkleCap' + s, cyl(an + EX * sg * 0.165, an + EX * sg * 0.185, 0.058, 0.052, sides=28), 'brass', sb)
        a, b = lerp(kn, an, 0.55) + np.array([0.0, 0.195, 0.0]), an + np.array([0.0, 0.165, -0.085])
        add('AnkleRamLugA' + s, rbox(a - np.array([0, 0.025, 0]), (0.06, 0.05, 0.06), 0.006), 'olive2', sb)
        add('AnkleRamLugB' + s, rbox(b - np.array([0, 0.0, 0.02]), (0.06, 0.06, 0.05), 0.006), 'olive2', fb)
        piston('Ankle' + s, a, sb, b, fb, 0.042, 0.025)
        pc = np.array([an[0], -0.02, 0.065])
        add('FootPad' + s, rbox(pc, (0.44, 0.40, 0.07), 0.012, n=3), 'darksteel', fb)
        add('FootCover' + s, rbox(pc + np.array([0.0, 0.02, 0.062]), (0.36, 0.32, 0.06), 0.015, n=3), 'olive', fb)
        for k in range(3):
            add('Grouser' + s, rbox(np.array([an[0], 0.12 - 0.12 * k, 0.016]), (0.44, 0.03, 0.032), 0.004), 'darksteel', fb)
        bolts([pc + np.array([x, y, 0.093]) for x in (-0.15, 0.15) for y in (-0.10, 0.12)], (0, 0, 1), fb, r=0.011)
        tc = np.array([ball[0], -0.35, 0.065])
        add('ToePad' + s, rbox(tc, (0.44, 0.26, 0.07), 0.012, n=3), 'darksteel', ob_)
        add('ToeCover' + s, rbox(tc + np.array([0.0, 0.02, 0.055]), (0.36, 0.20, 0.05), 0.015, n=3), 'olive', ob_)
        add('ToeGrouser' + s, rbox(np.array([ball[0], -0.42, 0.016]), (0.44, 0.03, 0.032), 0.004), 'darksteel', ob_)
    return P, PIST, LAMPS


def mats_B2(body):
    pal = house_palette()
    olive = ((0.060, 0.065, 0.040), (0.098, 0.104, 0.064))
    M = {}
    M['olive'] = worn('KD_Olive', *olive, rough=0.62, under=pal['steel'][0], wear=1.0, chips=0.35, rust=0.35, dust=0.8)
    M['olive2'] = worn('KD_OliveDark', (0.045, 0.049, 0.031), (0.072, 0.077, 0.048), rough=0.64, under=pal['steel'][0], wear=1.1, chips=0.4)
    M['steel'] = worn('KD_Steel', *pal['steel'], metal=1.0, rough=0.32, paint=False, wear=0.8, scratches=0.8, grime=1.2, rust=0.25)
    M['plate'] = worn('KD_SteelPlate', *pal['plate'], metal=1.0, rough=0.45, paint=False, wear=0.7, scratches=1.0, grime=1.4, rust=0.5, dust=0.8)
    M['darksteel'] = worn('KD_DarkSteel', *pal['darksteel'], metal=1.0, rough=0.52, paint=False, wear=0.9, scratches=0.6, grime=1.2, rust=0.30)
    M['brass'] = worn('KD_Brass', *pal['brass'], metal=0.9, rough=0.38, paint=False, wear=0.8, scratches=0.5, rust=0.0, dust=0.6)
    M['chrome'] = worn('KD_Chrome', *pal['chrome'], metal=1.0, rough=0.12, paint=False, wear=0.3, scratches=0.3, grime=0.6, rust=0.0, dust=0.3, bump=0.15)
    M['rubber'] = worn('KD_Rubber', *pal['rubber'], rough=0.80, paint=False, wear=0.0, scratches=0.0, grime=0.6, rust=0.0, dust=0.35)
    M['hazard'] = worn('KD_Hazard', *pal['hazard'], rough=0.50, under=pal['steel'][0], wear=1.3, chips=0.5, rust=0.4,
                       stripes=(pal['hazard'][0], pal['hazard'][1], 4.2))
    M['stencil'] = worn('KD_Stencil', *pal['stencil'], rough=0.60, under=olive[0], wear=1.4, chips=0.6, scratches=0.3, rust=0.2)
    M['yellow'] = worn('KD_Yellow', (0.55, 0.36, 0.010), (0.66, 0.45, 0.02), rough=0.5, under=pal['steel'][0], wear=1.2, chips=0.4)
    M['black'] = worn('KD_PaintBlack', (0.010, 0.010, 0.011), (0.020, 0.020, 0.022), rough=0.55, under=(0.55, 0.36, 0.010), wear=0.6, scratches=0.2)
    M['soot'] = worn('KD_Soot', (0.018, 0.016, 0.014), (0.045, 0.036, 0.030), metal=0.3, rough=0.80, paint=False, wear=0.0, scratches=0.0, rust=0.5, dust=0.2)
    M['core'] = worn('KD_Core', (0.035, 0.035, 0.038), (0.060, 0.060, 0.065), metal=0.8, rough=0.50, ob=body, paint=False, wear=0.5)
    M['lensdark'] = worn('KD_LensDark', (0.004, 0.005, 0.006), (0.010, 0.012, 0.014), rough=0.06, paint=False, wear=0.0, scratches=0.15, grime=0.3,
                         rust=0.0, dust=0.15, mud=0.0, spec=0.9)
    M['lamp'] = glow_mat('KD_Lamp', (1.0, 0.92, 0.78), 14.0)
    return M


def bone_parent(objs, rig, bones):
    """parent each object to its bone and keep it where it is. The inverse comes from the bone's evaluated matrix (at its tail, as
    Blender parents to bones), never from the object's own matrix, which a constraint (the rams' Damped Track) may already have turned"""
    bpy.context.view_layer.update()
    for ob, bone in zip(objs, bones):
        pb = rig.pose.bones[bone]
        Pb = rig.matrix_world @ pb.matrix @ Matrix.Translation((0.0, pb.bone.length, 0.0))
        ob.parent = rig
        ob.parent_type = 'BONE'
        ob.parent_bone = bone
        ob.matrix_parent_inverse = Pb.inverted()
    bpy.context.view_layer.update()


def local_mesh(name, items, coll, M):
    """a mesh object from [(V, polys, material key)] in local coordinates (already in metres)"""
    me = bpy.data.meshes.new(name)
    keys = []
    Vs, Ps, mi, off = [], [], [], 0
    for V, polys, mk in items:
        if mk not in keys:
            keys.append(mk)
        Vs.append(np.asarray(V, float))
        Ps += [tuple(int(i) + off for i in p) for p in polys]
        mi += [keys.index(mk)] * len(polys)
        off += len(V)
    me.from_pydata(np.concatenate(Vs).tolist(), [], Ps)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
        p.material_index = mi[p.index]
    for mk in keys:
        me.materials.append(M[mk])
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def make_pistons(PIST, coll, rig, M):
    """hydraulic rams that stay connected: the barrel rides bone A and aims at the rod's pivot, the rod rides bone B and aims back"""
    objs, bones = [], []
    for name, a, bone_a, b, bone_b, rb, rr in PIST:
        A, B_ = Vector(tuple(a * SC)), Vector(tuple(b * SC))
        L = (B_ - A).length
        rb, rr = rb * KD_SCALE, rr * KD_SCALE
        fb_, fr_ = (0.74, 0.62) if name.startswith(("Ankle", "Waist")) else (0.62, 0.52)     # these stretch most (deep crouch, waist twist)
        Lb, Lr = fb_ * L, fr_ * L
        o = np.zeros(3)
        barrel = [(*cyl(o, (0, Lb, 0), rb, rb, sides=24), 'darksteel'),
                  (*cyl((0, Lb - 0.035, 0), (0, Lb + 0.004, 0), rb * 1.12, rb * 1.12, sides=24), 'steel'),
                  (*cyl((-rb * 0.95, 0, 0), (rb * 0.95, 0, 0), rb * 0.72, rb * 0.72, sides=16), 'steel'),
                  (*cyl((0, Lb * 0.22, rb * 0.9), (0, Lb * 0.22, rb * 1.35), rb * 0.20, rb * 0.20, sides=8), 'steel'),
                  (*cyl((0, Lb * 0.80, rb * 0.9), (0, Lb * 0.80, rb * 1.35), rb * 0.20, rb * 0.20, sides=8), 'steel')]
        rod = [(*cyl(o, (0, Lr, 0), rr, rr, sides=20), 'chrome'),
               (*cyl((-rr * 1.5, 0, 0), (rr * 1.5, 0, 0), rr * 1.3, rr * 1.3, sides=16), 'steel')]
        ob_b = local_mesh("KillDroid_Ram%s_Barrel" % name, barrel, coll, M)
        ob_r = local_mesh("KillDroid_Ram%s_Rod" % name, rod, coll, M)
        tA = bpy.data.objects.new("KillDroid_Ram%s_PivotA" % name, None)
        tB = bpy.data.objects.new("KillDroid_Ram%s_PivotB" % name, None)
        for t_, p_ in ((tA, A), (tB, B_)):
            coll.objects.link(t_)
            t_.empty_display_type = 'SPHERE'
            t_.empty_display_size = 0.03
            t_.location = p_
        for ob, p_, aim in ((ob_b, A, B_ - A), (ob_r, B_, A - B_)):
            ob.location = p_
            ob.rotation_mode = 'QUATERNION'
            ob.rotation_quaternion = Vector((0.0, 1.0, 0.0)).rotation_difference(aim.normalized())
        objs += [ob_b, ob_r, tA, tB]
        bones += [bone_a, bone_b, bone_a, bone_b]
        for ob, tgt in ((ob_b, tB), (ob_r, tA)):
            c = ob.constraints.new('DAMPED_TRACK')
            c.target = tgt
            c.track_axis = 'TRACK_Y'
    bone_parent(objs, rig, bones)
    return [o for o in objs if o.type == 'MESH']


def lamp_driver(owner, path, rig, scale):
    fc = owner.driver_add(path)
    drv = fc.driver
    drv.type = 'SCRIPTED'
    v = drv.variables.new()
    v.name = 'p'
    v.type = 'SINGLE_PROP'
    v.targets[0].id_type = 'OBJECT'
    v.targets[0].id = rig
    v.targets[0].data_path = '["lamp_power"]'
    drv.expression = 'p * %.3f' % scale
    drv.is_valid = True             # a driver evaluated while still half set up is flagged invalid and then stays at 0 until reset


def build_final():
    global EXTRA_BONES
    JAW.clear()
    JAW.update(jaw_geometry())
    EXTRA_BONES = [(b, v[0], v[1], 'hand.R') for b, v in JAW.items()]
    for nm in ("Cube", "Light", "Camera"):
        o = bpy.data.objects.get(nm)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    if bpy.data.collections.get(MC):
        wipe(MC)
    coll = _coll(MC)
    rig = make_rig(coll)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for b in JAW:
        rig.data.edit_bones[b].align_roll(Vector((1.0, 0.0, 0.0)))     # rotate a jaw bone about its local X to open / close it
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in rig.pose.bones:
        pb.rotation_mode = 'XYZ'
    setup_ik(rig)
    S = skin_core(True, head=False)
    body = make_skinned("KillDroid_Body", S, coll, rig)
    M = mats_B2(body)
    body.data.materials.append(M['core'])
    parts, pistons, lamps = parts_B2()
    groups = {}
    for name, V, Pp, mk, bone, smooth in parts:
        groups.setdefault((bone, mk, smooth), []).append((V, Pp))
    objs, bones = [], []
    for (bone, mk, smooth), items in sorted(groups.items()):
        V, Pp = merge(items)
        ob = make_obj("KillDroid_%s_%s%s" % (bone, mk, "" if smooth else "_flat"), V, Pp, coll, smooth=smooth)
        ob.data.materials.append(M[mk])
        objs.append(ob)
        bones.append(bone)
    bone_parent(objs, rig, bones)
    rams = make_pistons(pistons, coll, rig, M)
    for ob in [body] + objs + rams:
        edge_attr(ob)
    # the six work lamps cast real light; rig["lamp_power"] (0..1) dims the lamps and the lenses together
    rig["lamp_power"] = 1.0
    try:
        rig.id_properties_ui("lamp_power").update(min=0.0, max=1.0, description="turret work lamps (0 = dead, the beat where Relic kills the camera rig)")
    except Exception:
        pass
    rig.update_tag()                     # the evaluated copy of the rig must carry the new property before any driver reads it
    bpy.context.view_layer.update()
    lamp_objs = []
    for k, (p, f) in enumerate(lamps):
        ld = bpy.data.lights.new("KillDroid_Lamp%d" % k, 'SPOT')
        ld.energy, ld.color = 60.0, (1.0, 0.92, 0.80)
        ld.spot_size, ld.spot_blend, ld.shadow_soft_size = math.radians(55.0), 0.35, 0.02
        lo = bpy.data.objects.new("KillDroid_Lamp%d" % k, ld)
        coll.objects.link(lo)
        lo.location = Vector(tuple(p * SC))
        lo.rotation_mode = 'QUATERNION'
        aim = Vector(tuple(f)) - Vector((0.0, 0.0, 0.14))
        lo.rotation_quaternion = Vector((0.0, 0.0, -1.0)).rotation_difference(aim.normalized())
        lamp_driver(ld, "energy", rig, 60.0)
        lamp_objs.append(lo)
    bone_parent(lamp_objs, rig, ['head'] * len(lamp_objs))
    try:
        lamp_driver(M['lamp'].node_tree, 'nodes["Principled BSDF"].inputs["Emission Strength"].default_value', rig, 14.0)
    except Exception as e:
        print("LAMP DRIVER (material) failed:", e)
    for idb in [lo.data for lo in lamp_objs] + [M['lamp'].node_tree]:
        for fc in idb.animation_data.drivers:
            fc.driver.is_valid = True
        idb.update_tag()
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in [body] + objs + rams for c in o.bound_box]
    nverts = sum(len(o.data.vertices) for o in [body] + objs + rams)
    return rig, {"bones": len(rig.data.bones), "objects": len(coll.objects), "meshes": 1 + len(objs) + len(rams), "verts": nverts,
                 "rams": len(pistons), "lamps": len(lamp_objs), "height_m": round(max(p.z for p in pts), 3), "floor_m": round(min(p.z for p in pts), 3),
                 "width_m": round(max(p.x for p in pts) - min(p.x for p in pts), 3)}


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


def final_studio(dark=False):
    sc, cam = kd2_studio(dark)
    if dark:
        gradient_world(sc, "KD2_WorldDark", (0.10, 0.11, 0.12), (0.035, 0.037, 0.040), (0.010, 0.010, 0.011))
    else:
        gradient_world(sc, "KD2_WorldGrad", (0.55, 0.57, 0.60), (0.20, 0.205, 0.21), (0.03, 0.03, 0.032))
    sc.eevee.taa_render_samples = 64
    return sc, cam


def spin_shot(sc, cam, rig, name, deg, loc, tgt, lens, res, samples=64):
    rig.rotation_euler = (0.0, 0.0, math.radians(deg))
    bpy.context.view_layer.update()
    return shot(sc, cam, name, tuple(np.array(loc) * S_), tuple(np.array(tgt) * S_), lens, res, samples)


def final_renders(rig):
    for s_ in bpy.data.scenes:
        s_.frame_set(1)                     # evaluate the lamp drivers before the first render
    sc, cam = final_studio(False)
    out = []
    for nm, deg in (("front", 0.0), ("q3", 35.0), ("side", 90.0), ("back", 215.0)):
        out.append(spin_shot(sc, cam, rig, "killdroid_v02_%s.png" % nm, deg, (0.0, -9.4, 1.42), (0.0, 0.0, 1.33), 100, (760, 1000)))
    out.append(spin_shot(sc, cam, rig, "killdroid_v02_head.png", 18.0, (0.0, -2.55, 2.32), (0.0, -0.12, 2.20), 70, (760, 760)))
    a = math.radians(55.0)
    jc = np.array([-0.80, -0.08, 0.50])
    jw = np.array([jc[0] * math.cos(a) - jc[1] * math.sin(a), jc[0] * math.sin(a) + jc[1] * math.cos(a), jc[2]])
    out.append(spin_shot(sc, cam, rig, "killdroid_v02_crusher.png", 55.0, tuple(jw + np.array([0.0, -2.1, 0.30])), tuple(jw + np.array([0.0, 0.0, 0.05])), 55, (760, 760)))
    rig.rotation_euler = (0.0, 0.0, 0.0)
    scd, camd = final_studio(True)
    out.append(spin_shot(scd, camd, rig, "killdroid_v02_vsref_render.png", 14.0, (0.0, -6.2, 1.00), (0.0, 0.0, 1.36), 72, (800, 1008), 80))
    rig.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return out


def ram_report(rig):
    """for every ram: how far the barrel is off its aim, and how much barrel and rod still overlap (negative = they came apart)"""
    rows = []
    for ob in bpy.data.objects:
        if not (ob.name.startswith("KillDroid_Ram") and ob.name.endswith("_Barrel")):
            continue
        rod = bpy.data.objects[ob.name.replace("_Barrel", "_Rod")]
        A, B_ = ob.matrix_world.translation, rod.matrix_world.translation
        lb = max(v.co.y for v in ob.data.vertices)
        lr = max(v.co.y for v in rod.data.vertices)
        ax = (ob.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
        d = B_ - A
        aim = math.degrees(ax.angle(d.normalized())) if d.length > 1e-6 else 0.0
        rows.append((ob.name[13:-7], round(aim, 2), round(lb + lr - d.length, 3), round(max(0.0, lr - d.length), 3)))
    return rows


def verify_v02():
    rig = bpy.data.objects[RIGN]
    # 1. the skeleton: every v01 bone except the claw fingers must be identical
    with bpy.data.libraries.load(os.path.join(OUT_DIR, "killdroid_v01.blend"), link=False) as (src, dst):
        dst.armatures = [n for n in src.armatures if n == "KillDroid_Rig"]
    a01 = dst.armatures[0]
    shared = [b.name for b in a01.bones if not b.name.startswith("claw")]
    missing = [n for n in shared if n not in rig.data.bones]
    worst = max(max((a01.bones[n].head_local - rig.data.bones[n].head_local).length, (a01.bones[n].tail_local - rig.data.bones[n].tail_local).length)
                for n in shared if n not in missing)
    print("VERIFY bones: v01 %d, v02 %d, shared %d, missing %s, worst head/tail offset %.6f m" % (len(a01.bones), len(rig.data.bones), len(shared), missing, worst))
    print("VERIFY v02 bones:", ", ".join(b.name for b in rig.data.bones))
    # 2. Motion Assist (the installed extension) reads the rig and builds its mass model
    try:
        import importlib
        ap = importlib.import_module("bl_ext.user_default.motion_assist.autopose")
        bd = importlib.import_module("bl_ext.user_default.motion_assist.body")
        m = ap.rig_map(rig)
        parts = bd.character_parts(rig)
        body = bd.build_body(rig)
        print("VERIFY Motion Assist: rig_map %d joints, %d parts (%d skinned, %d on bones), mass model %s" % (
            len(m), len(parts), sum(1 for p in parts if p[1] == 'SKIN'), sum(1 for p in parts if p[1] == 'BONE'), type(body).__name__))
    except Exception as e:
        print("VERIFY Motion Assist FAILED", repr(e))
    # 3. kd_ctrl.py drives it the way fight_anim.py does; the rams must stay aimed and connected in strong poses
    exec(compile(open(r"C:\ai\unedited\kd_ctrl.py", encoding="utf-8").read(), "kd_ctrl.py", "exec"), _g)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    kd_add_arm_ik(rig)
    POSES = [("guard", dict(hips=(0.0, -0.04, 1.10), yaw=0.0, lean=(0.18, 0.0), wL=(0.42, -0.62, 1.40), wR=(-0.48, -0.70, 1.30),
                            aL=(0.34, 0.10, 0.19), aR=(-0.30, -0.30, 0.19)), 0.0),
             ("wind-up", dict(hips=(0.0, 0.02, 1.12), yaw=0.25, lean=(-0.12, 0.05), wL=(0.62, -0.30, 1.05), wR=(-0.62, 0.30, 2.05),
                              aL=(0.30, -0.15, 0.19), aR=(-0.30, 0.20, 0.19)), 0.0),
             ("crouch", dict(hips=(0.0, -0.10, 0.98), yaw=0.0, lean=(0.40, 0.0), wL=(0.48, -0.75, 0.70), wR=(-0.55, -0.80, 0.62),
                             aL=(0.40, 0.05, 0.19), aR=(-0.40, 0.05, 0.19)), 0.0),
             ("strike, jaws open", dict(hips=(0.05, -0.28, 1.06), yaw=-0.20, lean=(0.22, 0.04), wL=(0.58, 0.05, 1.00), wR=(-0.40, -1.10, 1.25),
                                        aL=(0.28, -0.62, 0.19), aR=(-0.28, 0.32, 0.19)), 0.5)]
    sc, cam = final_studio(False)
    files = []
    for k, (nm, P, jaw) in enumerate(POSES):
        P = {kk: (tuple(np.array(v) * S_) if isinstance(v, tuple) and len(v) == 3 else v) for kk, v in P.items()}
        kd_apply(rig, 1, P, key=False)
        rig.pose.bones['claw1a.R'].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), -jaw)
        rig.pose.bones['claw2a.R'].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), jaw)
        bpy.context.view_layer.update()
        wr_err = max((rig.matrix_world @ rig.pose.bones['forearm.' + s].tail - Vector(P['w' + s])).length for s in 'LR')
        an_err = max((rig.matrix_world @ rig.pose.bones['shin.' + s].tail - Vector(P['a' + s])).length for s in 'LR')
        rows = ram_report(rig)
        print("VERIFY pose %-18s wrist err %.3f m, ankle err %.3f m, worst ram aim %.2f deg, smallest ram overlap %.3f m (%s), worst rod poke-through %.3f m" % (
            nm, wr_err, an_err, max(r[1] for r in rows), min(r[2] for r in rows), min(rows, key=lambda r: r[2])[0], max(r[3] for r in rows)))
        files.append(shot(sc, cam, "killdroid_v02_pose%d.png" % k, (-3.4 * S_, -6.6 * S_, 1.6 * S_), (0.0, -0.2 * S_, 1.15 * S_), 62, (640, 800), 32))
    print("VERIFY renders", files)


def relax_arms(rig_, out=0.22):
    """swing a T-posed rig's upper arms down to the sides (rest-relative, about the shoulder)"""
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
    """the cast at true scale under the same light: Mender, Relic, the KillDroid"""
    kd = bpy.data.objects[RIGN]
    got = {}
    for path, cname in ((os.path.join(OUT_DIR, "fight_v07.blend"), "Relic_AI"), (os.path.join(OUT_DIR, "relic_v07_mender.blend"), "Mender")):
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.collections = [n for n in src.collections if n == cname]
        got[cname] = dst.collections[0]
    sc, cam = final_studio(False)
    for c in got.values():
        sc.collection.children.link(c)
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
    mr.location, mr.rotation_euler = (-2.75, 0.0, 0.0), (0.0, 0.0, 0.0)
    rr.location, rr.rotation_euler = (-1.35, 0.0, 0.0), (0.0, 0.0, 0.0)
    kd.location, kd.rotation_euler = (0.75, 0.0, 0.0), (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    files = [shot(sc, cam, "cast_lineup_v02_front.png", (-0.9, -12.5, 1.55), (-0.9, 0.0, 1.30), 70, (1600, 1000), 64)]
    for r_ in (mr, rr, kd):
        r_.rotation_euler = (0.0, 0.0, math.radians(30.0))
    bpy.context.view_layer.update()
    files.append(shot(sc, cam, "cast_lineup_v02_q3.png", (-0.9, -12.5, 1.55), (-0.9, 0.0, 1.30), 70, (1600, 1000), 64))
    print("LINEUP", files)


# ------------------------------------------------------------------ build
def build_v2(key):
    D = DIRS[key]
    for nm in ("Cube", "Light", "Camera"):
        o = bpy.data.objects.get(nm)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    if bpy.data.collections.get(MC):
        wipe(MC)
    coll = _coll(MC)
    rig = make_rig(coll)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'XYZ'
    setup_ik(rig)
    S = D['skin']()
    body = make_skinned("KillDroid_Body", S, coll, rig)
    M = D['mats'](body)
    for slot in D['body']:
        body.data.materials.append(M[slot])
    objs = []
    for name, Vp, Pp, mk, bone, smooth in D['parts'](S):
        ob = make_obj("KillDroid_" + name, Vp, Pp, coll, smooth=smooth)
        ob.data.materials.append(M[mk])
        ob.parent = rig
        ob.parent_type = 'BONE'
        ob.parent_bone = bone
        objs.append(ob)
    bpy.context.view_layer.update()
    for ob in objs:
        ob.matrix_parent_inverse = ob.matrix_world.inverted()
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in [body] + objs for c in o.bound_box]
    return {"direction": key, "body_verts": len(body.data.vertices), "rigid_parts": len(objs), "bones": len(rig.data.bones),
            "height_m": round(max(p.z for p in pts), 3), "floor_m": round(min(p.z for p in pts), 3),
            "width_m": round(max(p.x for p in pts) - min(p.x for p in pts), 3), "depth_m": round(max(p.y for p in pts) - min(p.y for p in pts), 3)}


# ------------------------------------------------------------------ studio, shots, silhouettes
def kd2_studio(dark=False):
    sc = bpy.data.scenes.new("KD2_Studio")
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
    w = bpy.data.worlds.new("KD2_World")
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.045, 0.047, 0.052, 1.0) if dark else (0.16, 0.165, 0.175, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    sc.world = w
    sc.collection.children.link(bpy.data.collections[MC])
    setc = bpy.data.collections.new("KD2_Set")
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
    me = bpy.data.meshes.new("KD2_Backdrop")
    me.from_pydata(Vc, [], Pc)
    me.update()
    for p_ in me.polygons:
        p_.use_smooth = True
    fo = bpy.data.objects.new("KD2_Backdrop", me)
    setc.objects.link(fo)
    fm, _nt, fpb = _new_mat("KD2_BackdropMat")
    fpb.inputs['Base Color'].default_value = (0.05, 0.05, 0.055, 1) if dark else (0.21, 0.21, 0.22, 1)
    fpb.inputs['Roughness'].default_value = 0.85
    me.materials.append(fm)
    tgt = (0.0, 0.0, 1.5)
    for nm, sz, en, loc, col in (("KD2_Key", 3.0, 1600, (-3.6, -4.6, 4.6), (1.0, 0.94, 0.86)), ("KD2_Fill", 4.0, 420, (4.2, -3.8, 2.2), (0.80, 0.88, 1.0)),
                                 ("KD2_Rim", 2.0, 1300, (2.8, 4.0, 4.0), (1.0, 0.90, 0.80)), ("KD2_Rim2", 2.0, 900, (-3.2, 3.6, 3.2), (0.85, 0.92, 1.0)),
                                 ("KD2_Top", 1.8, 380, (0.0, -0.6, 6.0), (1.0, 1.0, 1.0))):
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.energy, ld.size, ld.color = en * (0.55 if dark else 1.0), sz, col
        lo = bpy.data.objects.new(nm, ld)
        setc.objects.link(lo)
        lo.location = loc
        look_at(lo, tgt)
    cd = bpy.data.cameras.new("KD2_Cam")
    cd.clip_start, cd.clip_end = 0.05, 200.0
    cam = bpy.data.objects.new("KD2_Cam", cd)
    setc.objects.link(cam)
    sc.camera = cam
    return sc, cam


def shot(sc, cam, name, loc, tgt, lens, res, samples=48):
    cam.location = Vector(loc)
    look_at(cam, tgt)
    cam.data.lens = lens
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.eevee.taa_render_samples = samples
    sc.render.filepath = os.path.join(REND, name)
    sc.frame_set(sc.frame_current)                  # evaluates drivers (the lamp power) before the render
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print("RENDERED", sc.render.filepath)
    return sc.render.filepath


def silhouette(name, res=(480, 560), oscale=3.35):
    sc = bpy.data.scenes.new("KD2_Sil")
    sc.collection.children.link(bpy.data.collections[MC])
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'FLAT'
    sh.color_type = 'SINGLE'
    sh.single_color = (0.0, 0.0, 0.0)
    for a in ('show_cavity', 'show_shadows', 'show_object_outline', 'show_specular_highlight'):
        try:
            setattr(sh, a, False)
        except Exception:
            pass
    sc.view_settings.view_transform = 'Standard'
    w = bpy.data.worlds.new("KD2_SilWorld")
    w.color = (1.0, 1.0, 1.0)
    sc.world = w
    cd = bpy.data.cameras.new("KD2_SilCam")
    cd.type = 'ORTHO'
    cd.ortho_scale = oscale
    cam = bpy.data.objects.new("KD2_SilCam", cd)
    sc.collection.objects.link(cam)
    cam.location = (0.0, -14.0, oscale / 2 - 0.08)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.filepath = os.path.join(REND, name)
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print("RENDERED", sc.render.filepath)


def concept(key):
    info = build_v2(key)
    print("BUILT", info)
    sc, cam = kd2_studio()
    hz = DIRS[key]['head_z']
    shot(sc, cam, "kd2_concept_%s_hero.png" % key, (-2.4 * S_, -7.0 * S_, 1.55 * S_), (0.0, 0.0, 1.30 * S_), 75, (720, 1000))
    rig = bpy.data.objects[RIGN]
    rig.rotation_euler = (0.0, 0.0, math.pi / 2)          # his right side (the weapon arm) to the camera; the backdrop stays the same
    bpy.context.view_layer.update()
    shot(sc, cam, "kd2_concept_%s_side.png" % key, (0.0, -7.6 * S_, 1.50 * S_), (0.0, 0.0, 1.30 * S_), 75, (720, 1000), 32)
    rig.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    shot(sc, cam, "kd2_concept_%s_head.png" % key, (-0.80 * S_, -2.25 * S_, (hz + 0.14) * S_), (0.0, -0.05 * S_, (hz + 0.01) * S_), 70, (640, 640))
    silhouette("kd2_concept_%s_sil.png" % key)
    return info


if MODE == "check":
    print("BUILT", build_v2(KEY))
elif MODE == "concept":
    concept(KEY)
elif MODE == "silv01":
    silhouette("kd2_concept_v01_sil.png")
elif MODE == "verify":
    verify_v02()
elif MODE == "lineup":
    lineup()
elif MODE == "finalcheck":
    rig_, info_ = build_final()
    print("BUILT", info_)
elif MODE == "final":
    rig_, info_ = build_final()
    print("BUILT", info_)
    V02 = os.path.join(OUT_DIR, "killdroid_v02.blend")
    bpy.ops.wm.save_as_mainfile(filepath=V02)
    print("SAVED", V02)
    print("RENDERED", final_renders(rig_))
    bpy.ops.wm.save_as_mainfile(filepath=V02)
    print("SAVED", V02)
print("DONE", MODE, KEY)
