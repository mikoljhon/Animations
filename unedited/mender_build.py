# mender_build.py : procedural Mender (the scavenger mechanic of UNEDITED), skinned suit + rigid gear, rig with leg IK.
# Design frame: feet at z=0, facing -Y, +X = his left. Everything is authored at 1.74 m and scaled by SC at creation.
# Exec in Blender, then build_mender(). Idempotent: it wipes the "Mender" collection first.
import bpy, bmesh, math, os, types
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

OUT_DIR = r"C:\ai\unedited"
SC = np.array([0.99, 0.99, 0.90])    # design frame (1.74 m, slim) -> a stockier 1.57 m (1.61 m with the hat)
CYBORG_ARM = True             # scavenged mechanical right forearm and gripper
RIGN = "Mender_Rig"
MC = "Mender"


# ------------------------------------------------------------------ geometry primitives (numpy)
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def basis(z, y_hint=(0.0, 1.0, 0.0)):
    """rotation matrix with columns (x, y, z): z along `z`, y as close to y_hint as possible"""
    z = unit(z)
    y = np.asarray(y_hint, float)
    y = y - z * (y @ z)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([1.0, 0.0, 0.0]) - z * z[0]
    y = unit(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def tube(P, rx, ry=None, sides=18, caps=(True, True), up=(1.0, 0.0, 0.0)):
    """swept elliptical tube along path P (n,3); radii arrays per path point; returns V, polys (quads + cap triangles)"""
    P = np.asarray(P, float)
    n = len(P)
    rx = np.broadcast_to(np.asarray(rx, float), (n,)).copy()
    ry = rx.copy() if ry is None else np.broadcast_to(np.asarray(ry, float), (n,)).copy()
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1)[:, None], 1e-9)
    up = np.asarray(up, float)
    N = up - T[0] * (up @ T[0])
    if np.linalg.norm(N) < 1e-6:
        N = np.cross(T[0], [0.0, 1.0, 0.0])
    N /= np.linalg.norm(N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = []
    for i in range(n):
        if i > 0:
            N = N - T[i] * (N @ T[i])
            N /= np.linalg.norm(N)
        B = np.cross(T[i], N)
        rings.append(P[i] + np.outer(rx[i] * np.cos(ang), N) + np.outer(ry[i] * np.sin(ang), B))
    V = np.concatenate(rings)
    polys = []
    for i in range(n - 1):
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            c = (i + 1) * sides + (k + 1) % sides
            d = (i + 1) * sides + k
            polys.append((a, b, c, d))
    extra = []
    if caps[0]:
        ci = len(V) + len(extra)
        extra.append(P[0])
        for k in range(sides):
            polys.append((ci, (k + 1) % sides, k))
    if caps[1]:
        ci = len(V) + len(extra)
        extra.append(P[-1])
        o = (n - 1) * sides
        for k in range(sides):
            polys.append((ci, o + k, o + (k + 1) % sides))
    if extra:
        V = np.concatenate([V, np.array(extra)])
    return V, polys


def cyl(p0, p1, r0, r1=None, sides=14, caps=True):
    r1 = r0 if r1 is None else r1
    return tube([p0, p1], [r0, r1], sides=sides, caps=(caps, caps), up=(0.0, 0.0, 1.0))


def _axis_grid(H, r, n):
    rho = min(r / H, 0.95)
    a = [-1.0, -1 + rho / 3, -1 + 2 * rho / 3, -1 + rho]
    inner = list(np.linspace(-1 + rho, 1 - rho, n + 1))[1:-1]
    b = [1 - rho, 1 - 2 * rho / 3, 1 - rho / 3, 1.0]
    return np.array(sorted(set(round(float(x), 6) for x in a + inner + b)))


def rbox(center, size, r, R=None, n=2):
    """rounded box; size = full extents, R = 3x3 rotation (local -> design frame)"""
    H = np.asarray(size, float) / 2.0
    r = float(min(r, H.min() * 0.98))
    grids = [_axis_grid(H[i], r, n) for i in range(3)]
    pts, V, polys = {}, [], []

    def vid(p):
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        i = pts.get(key)
        if i is None:
            i = len(V)
            pts[key] = i
            V.append(p)
        return i

    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        gb, gc = grids[b], grids[c]
        for s in (-1.0, 1.0):
            ids = np.zeros((len(gb), len(gc)), dtype=int)
            for i, u in enumerate(gb):
                for j, v in enumerate(gc):
                    p = [0.0, 0.0, 0.0]
                    p[a], p[b], p[c] = s, u, v
                    ids[i, j] = vid(p)
            for i in range(len(gb) - 1):
                for j in range(len(gc) - 1):
                    q = (ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1])
                    polys.append(q if s > 0 else q[::-1])
    V = np.array(V) * H
    q = np.clip(V, -(H - r), H - r)
    d = V - q
    nrm = np.linalg.norm(d, axis=1)
    m = nrm > 1e-9
    V[m] = q[m] + r * d[m] / nrm[m][:, None]
    if R is not None:
        V = V @ np.asarray(R).T
    return V + np.asarray(center, float), polys


def box(center, size, R=None):
    """plain 8-vertex box (for stitches and other tiny parts)"""
    h = np.asarray(size, float) / 2.0
    V = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    polys = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    if R is not None:
        V = V @ np.asarray(R).T
    return V + np.asarray(center, float), polys


def lathe(profile, sides=20):
    """surface of revolution about local Z; profile = [(radius, z), ...] bottom to top; radius 0 makes a pole"""
    prof = np.asarray(profile, float)
    V, idx = [], []
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for (r_, z_) in prof:
        if r_ < 1e-9:
            idx.append([len(V)])
            V.append((0.0, 0.0, z_))
        else:
            idx.append([len(V) + k for k in range(sides)])
            V.extend((r_ * math.cos(a), r_ * math.sin(a), z_) for a in ang)
    polys = []
    for i in range(len(prof) - 1):
        A, B = idx[i], idx[i + 1]
        for k in range(sides):
            k2 = (k + 1) % sides
            if len(A) == 1 and len(B) == 1:
                continue
            if len(A) == 1:
                polys.append((A[0], B[k2], B[k]))
            elif len(B) == 1:
                polys.append((A[k], A[k2], B[0]))
            else:
                polys.append((A[k], A[k2], B[k2], B[k]))
    return np.array(V), polys


def place(V, center=(0, 0, 0), R=None, scale=None):
    V = np.asarray(V, float)
    if scale is not None:
        V = V * np.asarray(scale, float)
    if R is not None:
        V = V @ np.asarray(R).T
    return V + np.asarray(center, float)


def ellipsoid(center, radii, nu=18, nv=10, R=None):
    prof = [(math.sin(math.pi * k / nv), -math.cos(math.pi * k / nv)) for k in range(nv + 1)]
    prof[0] = (0.0, -1.0)
    prof[-1] = (0.0, 1.0)
    V, polys = lathe(prof, nu)
    return place(V, center, R, scale=radii), polys


def torus(center, Rm, rm, axis=(0, 0, 1), nu=28, nv=10, y_hint=(0, 1, 0)):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    V = []
    for a in u:
        for b in v:
            rr = Rm + rm * math.cos(b)
            V.append((rr * math.cos(a), rr * math.sin(a), rm * math.sin(b)))
    polys = [(i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv)
             for i in range(nu) for j in range(nv)]
    return place(np.array(V), center, basis(axis, y_hint)), polys


def lerp(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


def merge(items):
    """items = [(V, polys), ...] -> V, polys"""
    Vs, Ps, off = [], [], 0
    for V, polys in items:
        Vs.append(V)
        Ps.extend(tuple(int(i) + off for i in p) for p in polys)
        off += len(V)
    return np.concatenate(Vs), Ps


# ------------------------------------------------------------------ skeleton
def joint_table():
    J = {}
    J['hip_c'] = np.array([0, 0, 0.94])
    J['pelvis_t'] = np.array([0, 0, 1.04])
    J['spine_t'] = np.array([0, 0, 1.23])
    J['chest_t'] = np.array([0, 0, 1.43])
    J['neck_t'] = np.array([0, 0, 1.51])
    J['head_t'] = np.array([0, 0, 1.76])
    for s, sx in (('L', 1.0), ('R', -1.0)):
        J['clav_' + s] = np.array([0.05 * sx, 0, 1.41])
        J['sh_' + s] = np.array([0.205 * sx, 0, 1.39])
        J['el_' + s] = np.array([0.325 * sx, 0.012, 1.12])
        J['wr_' + s] = np.array([0.37 * sx, -0.02, 0.87])
        J['hd_' + s] = np.array([0.39 * sx, -0.03, 0.71])
        J['hip_' + s] = np.array([0.095 * sx, 0, 0.94])
        J['kn_' + s] = np.array([0.105 * sx, -0.015, 0.50])
        J['an_' + s] = np.array([0.11 * sx, 0.015, 0.11])
        J['ball_' + s] = np.array([0.11 * sx, -0.13, 0.05])
        J['toe_' + s] = np.array([0.11 * sx, -0.24, 0.05])
    return J


J = joint_table()


def hand_dir(s):
    return unit(J['hd_' + s] - J['wr_' + s])


def bone_defs():
    B = [('root', (0, 0, 0), (0, -0.25, 0), None),
         ('pelvis', J['hip_c'], J['pelvis_t'], 'root'),
         ('spine', J['pelvis_t'], J['spine_t'], 'pelvis'),
         ('chest', J['spine_t'], J['chest_t'], 'spine'),
         ('neck', J['chest_t'], J['neck_t'], 'chest'),
         ('head', J['neck_t'], J['head_t'], 'neck')]
    for s, sx in (('L', 1.0), ('R', -1.0)):
        wr, dh = J['wr_' + s], hand_dir(s)
        B += [('clavicle.' + s, J['clav_' + s], J['sh_' + s], 'chest'),
              ('upperarm.' + s, J['sh_' + s], J['el_' + s], 'clavicle.' + s),
              ('forearm.' + s, J['el_' + s], J['wr_' + s], 'upperarm.' + s),
              ('hand.' + s, wr, hand_end(s), 'forearm.' + s)]
        if s == 'L':
            for nm, jt in hand_joints_L().items():
                for k in range(len(jt) - 1):
                    par = 'hand.L' if k == 0 else '%s%d.L' % (nm, k)
                    B.append(('%s%d.L' % (nm, k + 1), jt[k], jt[k + 1], par))
        else:
            for nm, jt in mech_joints_R().items():
                for k in range(len(jt) - 1):
                    par = 'hand.R' if k == 0 else '%s%d.R' % (nm, k)
                    B.append(('%s%d.R' % (nm, k + 1), jt[k], jt[k + 1], par))
        B += [('thigh.' + s, J['hip_' + s], J['kn_' + s], 'pelvis'),
              ('shin.' + s, J['kn_' + s], J['an_' + s], 'thigh.' + s),
              ('foot.' + s, J['an_' + s], J['ball_' + s], 'shin.' + s),
              ('toe.' + s, J['ball_' + s], J['toe_' + s], 'foot.' + s),
              ('ik_foot.' + s, J['an_' + s], J['ball_' + s], 'root'),
              ('ik_knee.' + s, np.array([J['kn_' + s][0], -0.9, J['kn_' + s][2]]), np.array([J['kn_' + s][0], -0.9, J['kn_' + s][2] + 0.1]), 'root')]
    return B


def make_rig(coll):
    arm = bpy.data.armatures.new(RIGN)
    rig = bpy.data.objects.new(RIGN, arm)
    coll.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = {}
    for name, h, t, par in bone_defs():
        b = arm.edit_bones.new(name)
        b.head = tuple(float(x) for x in np.asarray(h, float) * SC)
        b.tail = tuple(float(x) for x in np.asarray(t, float) * SC)
        eb[name] = b
        if par:
            b.parent = eb[par]
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.display_type = 'OCTAHEDRAL'
    rig.show_in_front = True
    return rig


def setup_ik(rig):
    for s in ('L', 'R'):
        pb = rig.pose.bones['shin.' + s]
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        ik = pb.constraints.new('IK')
        ik.target = rig
        ik.subtarget = 'ik_foot.' + s
        ik.pole_target = rig
        ik.pole_subtarget = 'ik_knee.' + s
        ik.pole_angle = math.radians(-90)
        ik.chain_count = 2
        fb = rig.pose.bones['foot.' + s]
        for c in list(fb.constraints):
            fb.constraints.remove(c)
        cr = fb.constraints.new('COPY_ROTATION')
        cr.target = rig
        cr.subtarget = 'ik_foot.' + s


# ------------------------------------------------------------------ weights
def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def chain_weights(P, joints, bones, blend):
    """weights for points P along a joint chain; bones[k] runs joints[k] -> joints[k+1]"""
    P = np.asarray(P, float)
    Jn = np.array([np.asarray(j, float) for j in joints])
    seg = Jn[1:] - Jn[:-1]
    L = np.linalg.norm(seg, axis=1)
    S = np.concatenate([[0.0], np.cumsum(L)])
    best_s = np.zeros(len(P))
    best_d = np.full(len(P), 1e9)
    for k in range(len(seg)):
        d = seg[k] / L[k]
        t = np.clip((P - Jn[k]) @ d, 0.0, L[k])
        q = Jn[k] + t[:, None] * d
        dist = np.linalg.norm(P - q, axis=1)
        m = dist < best_d
        best_d[m] = dist[m]
        best_s[m] = S[k] + t[m]
    w, n = {}, len(bones)
    for k, b in enumerate(bones):
        up = _sstep((best_s - (S[k] - blend)) / (2 * blend)) if k > 0 else np.ones(len(P))
        down = 1.0 - _sstep((best_s - (S[k + 1] - blend)) / (2 * blend)) if k < n - 1 else np.ones(len(P))
        w[b] = up * down
    tot = sum(w.values())
    return {b: w[b] / np.maximum(tot, 1e-6) for b in w}


# ------------------------------------------------------------------ the suit (one skinned mesh) and the rigid gear
class Suit:
    def __init__(self):
        self.V, self.polys, self.mat, self.w = [], [], [], []

    def add(self, V, polys, mat, w):
        self.V.append(np.asarray(V, float))
        self.polys.append(list(polys))
        self.mat.append(mat)
        self.w.append(w)

    def host(self, V, polys, mat, joints, bones, blend=0.06):
        self.add(V, polys, mat, chain_weights(V, joints, bones, blend))


CH = {}   # joint chains per body region, filled in build_suit


def build_suit():
    S = Suit()
    chains = {'torso': ([J['hip_c'], J['pelvis_t'], J['spine_t'], J['chest_t'], J['neck_t']], ['pelvis', 'spine', 'chest', 'neck'], 0.07)}
    for s in ('L', 'R'):
        chains['leg_' + s] = ([J['hip_' + s], J['kn_' + s], J['an_' + s]], ['thigh.' + s, 'shin.' + s], 0.07)
        chains['arm_' + s] = ([J['clav_' + s], J['sh_' + s], J['el_' + s], J['wr_' + s]],
                              ['clavicle.' + s, 'upperarm.' + s, 'forearm.' + s], 0.06)
    CH.update(chains)

    def host(V, polys, mat, region):
        j, b, bl = chains[region]
        S.host(V, polys, mat, j, b, bl)

    # torso: baggy coverall, belt line, collar
    z = np.array([0.80, 0.87, 0.94, 1.04, 1.16, 1.29, 1.38, 1.44, 1.48, 1.51])
    rx = np.array([0.150, 0.188, 0.194, 0.178, 0.186, 0.208, 0.224, 0.178, 0.098, 0.090])
    ry = np.array([0.100, 0.116, 0.120, 0.110, 0.113, 0.127, 0.127, 0.103, 0.082, 0.078])
    cy = np.array([0.0, 0.0, 0.0, -0.002, -0.008, -0.013, -0.008, 0.0, 0.006, 0.008])
    Vt, Pt = tube(np.stack([np.zeros_like(z), cy, z], 1), rx, ry, sides=24, caps=(True, True), up=(1.0, 0.0, 0.0))
    host(Vt, Pt, 0, 'torso')
    suit_tris = [(V0, P0) for V0, P0 in [(Vt, Pt)]]
    # legs
    legV = {}
    for s, sx in (('L', 1.0), ('R', -1.0)):
        P = np.array([[0.095, 0.0, 0.97], [0.098, -0.003, 0.80], [0.103, -0.010, 0.63], [0.105, -0.015, 0.50],
                      [0.107, -0.005, 0.38], [0.109, 0.008, 0.28], [0.110, 0.014, 0.215], [0.110, 0.015, 0.172]]) * np.array([sx, 1, 1])
        t = np.linspace(0, 1, len(P))
        r = np.array([0.101, 0.099, 0.092, 0.085, 0.077, 0.072, 0.076, 0.082])
        r = r * (1 + 0.025 * np.sin(t * 17 + (0.0 if s == 'L' else 1.3)))
        V, polys = tube(P, r, r * 1.02, sides=20, caps=(False, False), up=(1.0, 0.0, 0.0))
        host(V, polys, 0, 'leg_' + s)
        legV[s] = (V, polys)
    # arms: sleeve to the wrist (right sleeve stops at mid forearm when the arm is mechanical)
    for s, sx in (('L', 1.0), ('R', -1.0)):
        sh, el, wr = J['sh_' + s], J['el_' + s], J['wr_' + s]
        pts = [sh + np.array([-0.02 * sx, 0, 0.005])] + [lerp(sh, el, u) for u in (0.18, 0.45, 0.75)] + [el]
        rad = [0.046, 0.063, 0.061, 0.056, 0.057]
        fa = (0.25, 0.55, 0.80)
        far = (0.054, 0.051, 0.0545)
        if s == 'R' and CYBORG_ARM:
            fa, far = (0.22, 0.45), (0.054, 0.060)
        pts += [lerp(el, wr, u) for u in fa]
        rad += list(far)
        V, polys = tube(np.array(pts), np.array(rad), sides=18, caps=(True, True), up=(0.0, 1.0, 0.0))
        host(V, polys, 0, 'arm_' + s)
    return S, chains


def surface_hit(tree, x, z, from_front=True):
    o = Vector((x, -1.0 if from_front else 1.0, z))
    d = Vector((0, 1.0, 0)) if from_front else Vector((0, -1.0, 0))
    h = tree.ray_cast(o, d)
    return (np.array(h[0]), np.array(h[1])) if h[0] is not None else (None, None)


def build_suit_details(S):
    """pads, patches, pockets, zipper and collar bits; placed by ray-casting onto the finished tubes"""
    V0, P0 = merge([(v, p) for v, p in zip(S.V, S.polys)])
    tree = BVHTree.FromPolygons([tuple(v) for v in V0], [tuple(p) for p in P0])

    def hostw(V, region):
        j, b, bl = CH[region]
        return chain_weights(V, j, b, bl)

    def onto(x, z, region, size, thick, mat, r=0.004, from_front=True, y_hint=(0, 0, 1), lift=0.0, border=None, rot=0.0):
        p, n = surface_hit(tree, x, z, from_front)
        if p is None:
            return
        n = unit(n)
        if (n[1] > 0) == from_front:
            n = -n
        R = basis(n, y_hint)
        if rot:
            c_, s_ = math.cos(rot), math.sin(rot)
            R = R @ np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        items = []
        if border is not None:
            Vb, Pb = rbox(p + n * (0.001 + lift), (size[0] + border, size[1] + border, thick * 0.7), r, R)
            S.add(Vb, Pb, 2, hostw(Vb, region))
        V, polys = rbox(p + n * (thick / 2 + 0.001 + lift + (0.002 if border else 0.0)), (size[0], size[1], thick), r, R)
        S.add(V, polys, mat, hostw(V, region))
        if border is not None and thick < 0.02:
            # hand stitching: little dark dashes just inside the edge
            sx_, sy_ = size
            inset = 0.0065
            top = p + n * (thick + 0.0045 + lift + 0.002)
            dashes = []
            for yy in (-(sy_ / 2 - inset), (sy_ / 2 - inset)):
                for xx in np.arange(-sx_ / 2 + 0.013, sx_ / 2 - 0.009, 0.0155):
                    dashes.append((xx, yy, 0))
            for xx in (-(sx_ / 2 - inset), (sx_ / 2 - inset)):
                for yy in np.arange(-sy_ / 2 + 0.013, sy_ / 2 - 0.009, 0.0155):
                    dashes.append((xx, yy, 1))
            dv, dp = [], []
            for (xx, yy, o) in dashes:
                c = top + R[:, 0] * xx + R[:, 1] * yy
                szd = (0.0085, 0.0024, 0.0020) if o == 0 else (0.0024, 0.0085, 0.0020)
                Vd, Pd = box(c, szd, R)
                dv.append((Vd, Pd))
            if dv:
                Vm, Pm = merge(dv)
                S.add(Vm, Pm, 2, hostw(Vm, region))

    # patches: material 1 teal canvas, 3 olive-grey, 4 mustard; dark stitched borders (material 2); each a little crooked
    onto(0.115, 0.66, 'leg_L', (0.080, 0.100), 0.007, 1, rot=0.15, border=0.013)
    onto(-0.100, 0.32, 'leg_R', (0.070, 0.090), 0.007, 3, rot=-0.20, border=0.013)
    onto(0.230, 1.34, 'torso', (0.075, 0.070), 0.007, 1, rot=0.25, border=0.012)
    onto(-0.289, 1.20, 'arm_R', (0.060, 0.085), 0.007, 4, rot=0.10, border=0.011)
    onto(0.352, 0.99, 'arm_L', (0.050, 0.070), 0.007, 3, rot=-0.15, border=0.010)
    onto(-0.140, 1.10, 'torso', (0.085, 0.065), 0.007, 4, rot=0.05, border=0.012)
    onto(0.085, 0.93, 'torso', (0.090, 0.090), 0.007, 1, rot=-0.12, from_front=False, border=0.012)
    onto(-0.105, 0.72, 'leg_R', (0.080, 0.095), 0.007, 3, rot=0.20, from_front=False, border=0.012)
    onto(0.060, 1.16, 'torso', (0.070, 0.060), 0.006, 3, rot=-0.3, border=0.010)
    # a bulging cargo pocket on the outer left thigh (coverall cloth), with a dark flap
    onto(0.190, 0.60, 'leg_L', (0.085, 0.120), 0.042, 0, r=0.014)
    onto(0.190, 0.665, 'leg_L', (0.090, 0.040), 0.050, 2, r=0.010)
    # knee pads (material 2: dark canvas / leather)
    for s, sx in (('L', 1.0), ('R', -1.0)):
        kn = J['kn_' + s]
        onto(kn[0], kn[2] + 0.015, 'leg_' + s, (0.125, 0.150), 0.034, 2, r=0.016, y_hint=(0, 0, 1))
    # elbow pad on the left arm
    p, n = surface_hit(tree, J['el_L'][0] + 0.03, J['el_L'][2] + 0.0, from_front=False)
    if p is not None:
        n = unit(n)
        if n[1] < 0:
            n = -n
        R = basis(n, (0, 0, 1))
        V, polys = rbox(p + n * 0.012, (0.09, 0.10, 0.030), 0.013, R)
        S.add(V, polys, 2, hostw(V, 'arm_L'))
    # zipper (dark teeth) down the front of the torso
    p1, _ = surface_hit(tree, 0.0, 1.38, True)
    p2, _ = surface_hit(tree, 0.0, 1.0, True)
    if p1 is not None and p2 is not None:
        V, polys = rbox((p1 + p2) / 2 + np.array([0, -0.004, 0]), (0.012, 0.007, abs(p1[2] - p2[2])), 0.003)
        S.add(V, polys, 2, hostw(V, 'torso'))
    return S


def suit_arrays(S):
    V, polys, mats, off = [], [], [], 0
    for v, p, m in zip(S.V, S.polys, S.mat):
        V.append(v)
        polys.extend(tuple(int(i) + off for i in q) for q in p)
        mats.extend([m] * len(p))
        off += len(v)
    V = np.concatenate(V)
    bones = set()
    for w in S.w:
        bones |= set(w.keys())
    W = {b: np.zeros(len(V)) for b in bones}
    off = 0
    for v, w in zip(S.V, S.w):
        for b, arr in w.items():
            W[b][off:off + len(v)] = arr
        off += len(v)
    return V, polys, mats, W


# ------------------------------------------------------------------ rigid gear (each part = (name, V, polys, material key, bone))
def head_gear():
    parts = []
    hc = np.array([0.0, -0.005, 1.63])
    # skin: head, ears, neck
    V, p = ellipsoid(hc, (0.092, 0.105, 0.115), 22, 14)
    parts.append(('Head', V, p, 'skin', 'head'))
    for sx in (1, -1):
        V, p = ellipsoid(hc + np.array([0.092 * sx, 0.01, -0.005]), (0.012, 0.022, 0.032), 12, 8)
        parts.append(('Ear%s' % ('L' if sx > 0 else 'R'), V, p, 'skin', 'head'))
    V, p = cyl((0, 0.0, 1.46), (0, 0.0, 1.58), 0.052, 0.050, 16)
    parts.append(('Neck', V, p, 'skin', 'neck'))
    # hard hat
    prof = [(0.134, 0.0), (0.136, 0.010), (0.132, 0.040), (0.122, 0.072), (0.102, 0.097), (0.070, 0.116), (0.034, 0.128), (0.0, 0.132)]
    V, p = lathe(prof, 28)
    V = place(V, (0.0, -0.003, 1.652), scale=(1.0, 1.10, 1.0))
    parts.append(('Hat', V, p, 'hat', 'head'))
    V, p = torus((0.0, -0.003, 1.655), 0.134, 0.0085, nu=32, nv=8)
    V[:, 1] = (V[:, 1] + 0.003) * 1.10 - 0.003
    parts.append(('HatRim', V, p, 'hat', 'head'))
    a8 = math.radians(8.0)
    Rx8 = np.array([[1, 0, 0], [0, math.cos(a8), -math.sin(a8)], [0, math.sin(a8), math.cos(a8)]])
    V, p = rbox((0.0, -0.158, 1.667), (0.205, 0.080, 0.013), 0.005, Rx8)
    parts.append(('HatBrim', V, p, 'hat', 'head'))
    V, p = rbox((0.0, 0.150, 1.658), (0.17, 0.055, 0.011), 0.005)
    parts.append(('HatNapeFlap', V, p, 'hat', 'head'))
    # ridge hugging the dome: a flattened tube following the dome profile from front to back
    r_prof = np.array([0.134, 0.136, 0.132, 0.122, 0.102, 0.070, 0.034, 0.0])
    z_prof = np.array([0.0, 0.010, 0.040, 0.072, 0.097, 0.116, 0.128, 0.132])
    ys = np.linspace(-0.130, 0.130, 26)
    zs = 1.652 + np.interp(np.abs(ys) / 1.10, r_prof[::-1], z_prof[::-1]) + 0.004
    path = np.stack([np.zeros_like(ys), ys - 0.003, zs], 1)
    V, p = tube(path, 0.017, 0.0075, sides=8, caps=(True, True), up=(1.0, 0.0, 0.0))
    parts.append(('HatRidge', V, p, 'hat', 'head'))
    # headlamp
    V, p = cyl((0.0, -0.140, 1.728), (0.0, -0.188, 1.728), 0.034, 0.036, 16)
    parts.append(('LampBody', V, p, 'steel', 'head'))
    V, p = cyl((0.0, -0.188, 1.728), (0.0, -0.1925, 1.728), 0.027, 0.027, 16)
    parts.append(('LampLens', V, p, 'lamp', 'head'))
    V, p = rbox((0.0, -0.132, 1.716), (0.060, 0.045, 0.050), 0.008)
    parts.append(('LampMount', V, p, 'steel', 'head'))
    # goggles
    for sx in (1, -1):
        e = np.array([0.041 * sx, -0.102, 1.645])
        V, p = cyl(e + np.array([0, 0.018, 0]), e + np.array([0, -0.020, 0]), 0.037, 0.036, 18)
        parts.append(('GoggleHousing%s' % ('L' if sx > 0 else 'R'), V, p, 'rubber', 'head'))
        V, p = torus(e + np.array([0, -0.0205, 0]), 0.036, 0.0055, axis=(0, 1, 0), nu=24, nv=8)
        parts.append(('GoggleRim%s' % ('L' if sx > 0 else 'R'), V, p, 'steel', 'head'))
        V, p = ellipsoid(e + np.array([0, -0.0195, 0]), (0.031, 0.013, 0.031), 18, 10)
        parts.append(('GoggleLens%s' % ('L' if sx > 0 else 'R'), V, p, 'lens', 'head'))
    V, p = rbox((0, -0.108, 1.650), (0.034, 0.022, 0.014), 0.005)
    parts.append(('GoggleBridge', V, p, 'rubber', 'head'))
    ph = np.linspace(0.62, 2 * np.pi - 0.62, 30)
    path = np.stack([0.103 * np.sin(ph), -0.116 * np.cos(ph) - 0.003, np.full_like(ph, 1.646)], 1)
    V, p = tube(path, 0.015, 0.0045, sides=8, caps=(True, True), up=(0.0, 0.0, 1.0))
    parts.append(('GoggleStrap', V, p, 'strap', 'head'))
    # respirator: mask, two filter canisters, valve, straps, hose
    V, p = rbox((0.0, -0.100, 1.560), (0.116, 0.080, 0.094), 0.032, n=3)
    parts.append(('MaskBody', V, p, 'rubber', 'head'))
    for sx in (1, -1):
        d = unit((0.55 * sx, -0.85, 0.0))
        c0 = np.array([0.072 * sx, -0.100, 1.552])
        V, p = cyl(c0, c0 + d * 0.052, 0.033, 0.031, 18)
        parts.append(('Canister%s' % ('L' if sx > 0 else 'R'), V, p, 'filter', 'head'))
        V, p = torus(c0 + d * 0.052, 0.027, 0.0055, axis=d, nu=24, nv=8)
        parts.append(('CanisterRing%s' % ('L' if sx > 0 else 'R'), V, p, 'steel', 'head'))
        V, p = cyl(c0 - d * 0.004, c0 + d * 0.012, 0.037, 0.037, 18)
        parts.append(('CanisterBase%s' % ('L' if sx > 0 else 'R'), V, p, 'steel', 'head'))
    V, p = cyl((0.0, -0.138, 1.552), (0.0, -0.157, 1.552), 0.014, 0.012, 12)
    parts.append(('MaskValve', V, p, 'steel', 'head'))
    ph = np.linspace(0.78, 2 * np.pi - 0.78, 26)
    path = np.stack([0.100 * np.sin(ph), -0.112 * np.cos(ph) - 0.003, np.full_like(ph, 1.585)], 1)
    V, p = tube(path, 0.011, 0.0045, sides=8, caps=(True, True), up=(0.0, 0.0, 1.0))
    parts.append(('MaskStrap', V, p, 'strap', 'head'))
    # registry restore port behind the left ear: the one clean thing on him
    V, p = rbox((0.099, 0.034, 1.606), (0.010, 0.042, 0.030), 0.003)
    parts.append(('RegistryPort', V, p, 'port', 'head'))
    V, p = cyl((0.1055, 0.026, 1.612), (0.1075, 0.026, 1.612), 0.0042, 0.0042, 10)
    parts.append(('RegistryLED', V, p, 'led', 'head'))
    return parts


def hose_and_cable():
    parts = []
    # corrugated breathing hose: right canister -> over the shoulder -> pack
    pts = np.array([[-0.082, -0.115, 1.548], [-0.108, -0.100, 1.505], [-0.125, -0.060, 1.470], [-0.138, 0.020, 1.455],
                    [-0.135, 0.110, 1.430], [-0.100, 0.175, 1.400], [-0.050, 0.205, 1.360]])
    from_ = np.linspace(0, 1, 90)
    seg = np.array([np.interp(from_, np.linspace(0, 1, len(pts)), pts[:, k]) for k in range(3)]).T
    r = 0.0125 * (1 + 0.14 * np.sin(from_ * 2 * np.pi * 22))
    V, p = tube(seg, r, sides=10, caps=(True, True), up=(0, 0, 1))
    parts.append(('Hose', V, p, 'rubber', 'chest'))
    # headlamp cable down the right side of the hat to the pack
    pts = np.array([[0.020, -0.130, 1.712], [0.100, -0.085, 1.705], [0.128, 0.020, 1.672], [0.115, 0.110, 1.630], [0.085, 0.160, 1.570],
                    [0.070, 0.190, 1.480], [0.050, 0.205, 1.390]])
    from_ = np.linspace(0, 1, 60)
    seg = np.array([np.interp(from_, np.linspace(0, 1, len(pts)), pts[:, k]) for k in range(3)]).T
    V, p = tube(seg, 0.0055, sides=7, caps=(True, True), up=(0, 0, 1))
    parts.append(('LampCable', V, p, 'rubber', 'head'))
    return parts


def scarf():
    parts = []
    V, p = torus((0, 0.0, 1.495), 0.074, 0.026, axis=(0, 0, 1), nu=30, nv=10)
    V[:, 1] *= 0.95
    parts.append(('Scarf', V, p, 'scarf', 'neck'))
    path = np.array([[0.045, 0.060, 1.49], [0.050, 0.085, 1.42], [0.052, 0.098, 1.33], [0.056, 0.104, 1.25], [0.060, 0.108, 1.19]])
    V, p = tube(path, np.array([0.030, 0.034, 0.036, 0.034, 0.028]), np.array([0.010, 0.010, 0.010, 0.010, 0.009]), sides=10, up=(1, 0, 0))
    parts.append(('ScarfTail', V, p, 'scarf', 'chest'))
    return parts


def backpack():
    parts = []
    cx, cy = 0.0, 0.205
    V, p = rbox((cx, cy, 1.215), (0.350, 0.190, 0.470), 0.035, n=3)
    parts.append(('PackBody', V, p, 'canvas', 'chest'))
    V, p = rbox((cx, cy + 0.012, 1.215), (0.300, 0.185, 0.400), 0.03, n=2)
    parts.append(('PackFlap', V, p, 'canvas2', 'chest'))
    V, p = cyl((-0.17, cy, 1.505), (0.17, cy, 1.505), 0.072, 0.072, 20)
    parts.append(('PackRoll', V, p, 'canvas', 'chest'))
    V, p = rbox((cx, cy, 0.94), (0.300, 0.135, 0.120), 0.03, n=2)
    parts.append(('PackBottom', V, p, 'canvas2', 'chest'))
    # compression straps across the back of the pack, with buckles
    for k, zz in enumerate((1.04, 1.36)):
        V, p = rbox((cx, cy + 0.103, zz), (0.320, 0.012, 0.028), 0.004)
        parts.append(('PackBackStrap%d' % k, V, p, 'strap', 'chest'))
        V, p = rbox((0.090, cy + 0.111, zz), (0.044, 0.014, 0.040), 0.005)
        parts.append(('PackBuckle%d' % k, V, p, 'steel', 'chest'))
    for sx in (1, -1):
        V, p = rbox((0.205 * sx, cy - 0.005, 1.10), (0.085, 0.120, 0.230), 0.025, n=2)
        parts.append(('PackPocket%s' % ('L' if sx > 0 else 'R'), V, p, 'canvas2', 'chest'))
        V, p = rbox((0.10 * sx, cy + 0.096, 1.20), (0.026, 0.012, 0.44), 0.004)
        parts.append(('PackStrap%s' % ('L' if sx > 0 else 'R'), V, p, 'strap', 'chest'))
        # frame bars poking up past the roll
        V, p = cyl((0.125 * sx, cy + 0.085, 1.10), (0.125 * sx, cy + 0.085, 1.60), 0.009, 0.009, 8)
        parts.append(('PackBar%s' % ('L' if sx > 0 else 'R'), V, p, 'steel', 'chest'))
    V, p = rbox((0.0, -0.138, 1.30), (0.150, 0.012, 0.022), 0.004)
    parts.append(('ChestStrap', V, p, 'strap', 'chest'))
    # lashed cargo: rope coil, rolled mat, lantern
    for k in range(3):
        V, p = torus((-0.215, cy + 0.015, 1.02 + 0.034 * k), 0.082, 0.019, axis=(1, 0, 0), nu=26, nv=8)
        parts.append(('Rope%d' % k, V, p, 'rope', 'chest'))
    V, p = cyl((-0.18, cy + 0.125, 0.835), (0.18, cy + 0.125, 0.835), 0.052, 0.052, 18)
    parts.append(('RolledMat', V, p, 'mat', 'chest'))
    V, p = cyl((0.205, cy + 0.02, 0.80), (0.205, cy + 0.02, 0.92), 0.040, 0.040, 14)
    parts.append(('Lantern', V, p, 'steel', 'chest'))
    V, p = cyl((0.205, cy + 0.02, 0.832), (0.205, cy + 0.02, 0.888), 0.034, 0.034, 14)
    parts.append(('LanternGlass', V, p, 'lamp2', 'chest'))
    # a battered jerrycan strapped on the left side, with a handle and a cap
    V, p = rbox((0.255, cy + 0.015, 1.215), (0.070, 0.150, 0.190), 0.016, n=2)
    parts.append(('Jerrycan', V, p, 'paint', 'chest'))
    V, p = cyl((0.255, cy + 0.015, 1.305), (0.255, cy + 0.015, 1.330), 0.016, 0.016, 10)
    parts.append(('JerryCap', V, p, 'steel', 'chest'))
    V, p = rbox((0.255, cy + 0.075, 1.25), (0.020, 0.012, 0.075), 0.004)
    parts.append(('JerryHandle', V, p, 'steel', 'chest'))
    for k, zz in enumerate((1.155, 1.275)):
        V, p = rbox((0.255, cy + 0.015, zz), (0.082, 0.160, 0.016), 0.005)
        parts.append(('JerryStrap%d' % k, V, p, 'strap', 'chest'))
    # whip antenna with a faded flag, and a hanging cable loop
    V, p = cyl((-0.145, cy + 0.04, 1.55), (-0.150, cy + 0.07, 2.02), 0.0045, 0.0025, 6)
    parts.append(('Antenna', V, p, 'steel', 'chest'))
    V, p = rbox((-0.185, cy + 0.071, 1.93), (0.080, 0.004, 0.050), 0.0015)
    parts.append(('AntennaFlag', V, p, 'scarf', 'chest'))
    cp = np.array([[0.10, cy + 0.10, 1.50], [0.14, cy + 0.14, 1.40], [0.12, cy + 0.17, 1.30], [0.07, cy + 0.15, 1.24]])
    V, p = tube(cp, 0.012, sides=8, caps=(True, True), up=(1, 0, 0))
    parts.append(('PackCable', V, p, 'rubber', 'chest'))
    return parts


def shoulder_straps(tree):
    """pack straps that lie ON the suit: ray-cast down over the shoulder, then onto the chest"""
    parts = []
    off = 0.0068
    for sx in (1, -1):
        pts = []
        zlast = 1.45
        for yv in np.linspace(0.150, -0.060, 9):
            h = tree.ray_cast(Vector((0.105 * sx, yv, 2.2)), Vector((0, 0, -1.0)))
            if h[0] is not None:
                zlast = h[0].z
            pts.append((0.105 * sx, yv, zlast + off))
        for k, zv in enumerate(np.linspace(1.395, 1.06, 10)):
            xv = (0.105 + (0.085 - 0.105) * (1.395 - zv) / 0.335) * sx
            h = tree.ray_cast(Vector((xv, -1.0, zv)), Vector((0, 1.0, 0)))
            if h[0] is not None:
                pts.append((xv, h[0].y - off, zv))
        V, p = tube(np.array(pts), 0.030, 0.0065, sides=8, caps=(True, True), up=(1, 0, 0))
        parts.append(('ShoulderStrap%s' % ('L' if sx > 0 else 'R'), V, p, 'strap', 'chest'))
    return parts


def belt_and_pouches():
    parts = []
    ph = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    path = np.stack([0.196 * np.cos(ph), 0.126 * np.sin(ph), np.full_like(ph, 1.035)], 1)
    path = np.concatenate([path, path[:1]])
    V, p = tube(path, 0.023, 0.0065, sides=8, caps=(False, False), up=(0, 0, 1))
    parts.append(('Belt', V, p, 'leather', 'pelvis'))
    V, p = rbox((0.0, -0.131, 1.035), (0.045, 0.012, 0.040), 0.005)
    parts.append(('BeltBuckle', V, p, 'steel', 'pelvis'))
    for name, c, sz in (('PouchR', (-0.135, -0.120, 1.005), (0.075, 0.050, 0.080)), ('PouchL', (0.115, -0.122, 1.000), (0.065, 0.045, 0.070)),
                        ('PouchBack', (0.0, 0.132, 1.005), (0.130, 0.055, 0.090)), ('PouchSideR', (-0.198, -0.020, 1.0), (0.055, 0.075, 0.085))):
        V, p = rbox(c, sz, 0.014, n=2)
        parts.append((name, V, p, 'leather', 'pelvis'))
    V, p = rbox((0.118, -0.126, 0.935), (0.108, 0.036, 0.170), 0.012, n=2)
    parts.append(('Holster', V, p, 'leather', 'pelvis'))
    # tool loops: wrench and hammer hanging from the belt
    V, p = cyl((-0.165, -0.095, 0.99), (-0.165, -0.095, 0.77), 0.011, 0.011, 8)
    parts.append(('WrenchShaft', V, p, 'steel', 'pelvis'))
    V, p = rbox((-0.165, -0.095, 0.745), (0.044, 0.016, 0.060), 0.006)
    parts.append(('WrenchHead', V, p, 'steel', 'pelvis'))
    return parts


def boots():
    parts = []
    for s, sx in (('L', 1.0), ('R', -1.0)):
        an, ball = J['an_' + s], J['ball_' + s]
        V, p = cyl((0.110 * sx, 0.012, 0.045), (0.110 * sx, 0.012, 0.21), 0.079, 0.080, 18)
        parts.append(('BootShaft' + s, V, p, 'leatherB', 'foot.' + s))
        V, p = rbox((0.110 * sx, -0.092, 0.072), (0.134, 0.305, 0.110), 0.045, n=3)
        parts.append(('BootFoot' + s, V, p, 'leatherB', 'foot.' + s))
        V, p = rbox((0.110 * sx, -0.090, 0.024), (0.144, 0.326, 0.044), 0.018, n=3)
        parts.append(('BootSole' + s, V, p, 'rubber', 'foot.' + s))
        V, p = rbox((0.110 * sx, -0.212, 0.064), (0.128, 0.088, 0.084), 0.036, n=2)
        parts.append(('BootToe' + s, V, p, 'steelDark', 'foot.' + s))
        V, p = torus((0.110 * sx, 0.012, 0.205), 0.080, 0.0085, axis=(0, 0, 1), nu=26, nv=8)
        parts.append(('BootTop' + s, V, p, 'strap', 'foot.' + s))
        for k in range(4):
            z = 0.058 + 0.027 * k
            V, p = cyl((0.110 * sx - 0.045, -0.068 - 0.014 * k + 0.0, z + 0.06 * 0), (0.110 * sx + 0.045, -0.068 - 0.014 * k, z + 0.0), 0.0045, 0.0045, 6)
            parts.append(('Lace%s%d' % (s, k), V, p, 'strap', 'foot.' + s))
    return parts


def hands_and_arm():
    parts = []
    # left: fingerless work glove (mitten blocks so it can grip)
    s = 'L'
    wr, dh = J['wr_' + s], hand_dir(s)
    Rh = basis(dh, (1, 0, 0))
    V, p = rbox(wr + dh * 0.040, (0.092, 0.047, 0.092), 0.016, Rh, n=2)
    parts.append(('PalmL', V, p, 'glove', 'hand.L'))
    V, p = rbox(wr + dh * 0.1035, (0.088, 0.043, 0.062), 0.016, Rh, n=2)
    parts.append(('Fingers1L', V, p, 'glove', 'fingers1.L'))
    V, p = rbox(wr + dh * 0.1525, (0.086, 0.039, 0.054), 0.016, Rh, n=2)
    parts.append(('Fingers2L', V, p, 'glove', 'fingers2.L'))
    tp = np.array([wr + dh * 0.04 + np.array([0.0, -0.035, 0.0]), wr + dh * 0.075 + np.array([0.0, -0.052, 0.0]),
                   wr + dh * 0.110 + np.array([0.0, -0.066, 0.0])])
    V, p = tube(tp, np.array([0.022, 0.020, 0.017]), sides=10, caps=(True, True), up=(1, 0, 0))
    parts.append(('ThumbL', V, p, 'glove', 'thumb.L'))
    # glove cuff
    V, p = cyl(wr - dh * 0.012, wr + dh * 0.025, 0.056, 0.052, 16)
    parts.append(('CuffL', V, p, 'glove', 'forearm.L'))
    # right: scavenged mechanical forearm and gripper
    s = 'R'
    el, wr, dh = J['el_' + s], J['wr_' + s], hand_dir(s)
    a, b = lerp(el, wr, 0.42), wr + (wr - el) * 0.02
    d = unit(b - a)
    V, p = cyl(a, b, 0.047, 0.040, 18)
    parts.append(('ArmCoreR', V, p, 'steel', 'forearm.R'))
    V, p = rbox(lerp(a, b, 0.35), (0.092, 0.092, 0.095), 0.016, basis(d, (0, 1, 0)), n=2)
    parts.append(('ArmCoverR', V, p, 'paint', 'forearm.R'))
    V, p = rbox(lerp(a, b, 0.72) + np.array([0.0, -0.030, 0.0]), (0.070, 0.050, 0.110), 0.012, basis(d, (0, 1, 0)), n=2)
    parts.append(('ArmCover2R', V, p, 'paint', 'forearm.R'))
    for k, off in enumerate((np.array([0.0, 0.040, 0.0]), np.array([0.034, 0.025, 0.0]))):
        V, p = cyl(a + off, b + off + np.array([0, 0, 0.0]), 0.008, 0.008, 8)
        parts.append(('PistonR%d' % k, V, p, 'steel', 'forearm.R'))
    for k in range(3):
        V, p = torus(lerp(a, b, 0.12 + 0.25 * k), 0.049 - 0.0035 * k, 0.0048, axis=d, nu=22, nv=6)
        parts.append(('ArmBandR%d' % k, V, p, 'steel', 'forearm.R'))
    # cable loop along the arm
    cp = np.array([lerp(a, b, 0.0) + np.array([0.030, 0.032, 0.0]), lerp(a, b, 0.35) + np.array([0.058, 0.044, 0.0]),
                   lerp(a, b, 0.7) + np.array([0.045, 0.042, 0.0]), lerp(a, b, 1.0) + np.array([0.020, 0.030, 0.0])])
    V, p = tube(cp, 0.0065, sides=7, caps=(True, True), up=(1, 0, 0))
    parts.append(('ArmCableR', V, p, 'rubber', 'forearm.R'))
    # gripper: wrist cuff, palm drum, two jaws
    V, p = cyl(wr - dh * 0.010, wr + dh * 0.080, 0.043, 0.047, 18)
    parts.append(('GripPalmR', V, p, 'paint', 'hand.R'))
    V, p = torus(wr + dh * 0.080, 0.045, 0.006, axis=dh, nu=22, nv=6)
    parts.append(('GripRingR', V, p, 'steel', 'hand.R'))
    Rj = basis(dh, (0, 1, 0))
    for nm, oy, bone in (('GripJaw1R', -0.026, 'jaw1.R'), ('GripJaw2R', 0.026, 'jaw2.R')):
        V, p = rbox(wr + dh * 0.125 + np.array([0, oy, 0]), (0.052, 0.014, 0.094), 0.005, Rj, n=2)
        parts.append((nm, V, p, 'steel', bone))
        V, p = rbox(wr + dh * 0.172 + np.array([0, oy * 0.60, 0]), (0.050, 0.020, 0.024), 0.006, Rj, n=2)
        parts.append((nm + 'Tip', V, p, 'steel', bone))
    return parts


def props():
    """the pick and the scanner as separate objects, parented to the rig root (the pose script moves them)"""
    parts = []
    # pick: wooden handle along local Z (0 .. 0.95), steel head at the top
    V, p = tube([(0, 0, 0.0), (0, 0, 0.5), (0, 0, 0.95)], [0.019, 0.016, 0.018], sides=12, caps=(True, True), up=(1, 0, 0))
    parts.append(('PickHandle', V, p, 'wood', None))
    V, p = rbox((0, 0, 0.95), (0.050, 0.044, 0.060), 0.008)
    parts.append(('PickHub', V, p, 'steel', None))
    arc = np.array([[0.02, 0, 0.95], [0.09, 0, 0.955], [0.16, 0, 0.935], [0.225, 0, 0.895], [0.27, 0, 0.855]])
    V, p = tube(arc, np.array([0.022, 0.018, 0.013, 0.008, 0.002]), sides=10, caps=(False, True), up=(0, 1, 0))
    parts.append(('PickSpike', V, p, 'steel', None))
    V, p = rbox((-0.095, 0, 0.945), (0.100, 0.012, 0.070), 0.004)
    parts.append(('PickAdze', V, p, 'steel', None))
    # scanner: boxy handheld with a screen and antenna (local Z up, screen toward -Y)
    V, p = rbox((0, 0, 0.0), (0.095, 0.050, 0.150), 0.012, n=2)
    parts.append(('ScannerBody', V, p, 'paint', None))
    V, p = rbox((0, -0.0265, 0.020), (0.068, 0.008, 0.062), 0.003, n=2)
    parts.append(('ScannerScreen', V, p, 'screen', None))
    V, p = cyl((0.032, 0.0, 0.074), (0.040, 0.0, 0.250), 0.004, 0.003, 6)
    parts.append(('ScannerAntenna', V, p, 'steel', None))
    V, p = rbox((0, 0.032, -0.070), (0.050, 0.020, 0.055), 0.008, n=2)
    parts.append(('ScannerGrip', V, p, 'rubber', None))
    return parts


# ------------------------------------------------------------------ v02: realistic lace-up boots and five-finger hands
def orient_outward(V, polys):
    """reverse every polygon if the closed surface has negative volume"""
    V = np.asarray(V, float)
    vol = 0.0
    for p in polys:
        a = V[p[0]]
        for k in range(1, len(p) - 1):
            vol += float(a @ np.cross(V[p[k]], V[p[k + 1]]))
    if vol < 0:
        polys = [tuple(reversed(p)) for p in polys]
    return V, polys


def loft(rings, caps=(True, True)):
    n, M = len(rings), len(rings[0])
    V = np.concatenate(rings)
    polys = []
    for i in range(n - 1):
        for k in range(M):
            polys.append((i * M + k, i * M + (k + 1) % M, (i + 1) * M + (k + 1) % M, (i + 1) * M + k))
    extra = []
    if caps[0]:
        ci = len(V) + len(extra)
        extra.append(rings[0].mean(0))
        for k in range(M):
            polys.append((ci, (k + 1) % M, k))
    if caps[1]:
        ci = len(V) + len(extra)
        extra.append(rings[-1].mean(0))
        o = (n - 1) * M
        for k in range(M):
            polys.append((ci, o + k, o + (k + 1) % M))
    if extra:
        V = np.concatenate([V, np.array(extra)])
    return orient_outward(V, polys)


def mirror_x(V, polys):
    V2 = np.asarray(V, float).copy()
    V2[:, 0] *= -1.0
    return V2, [tuple(reversed(p)) for p in polys]


BOOT_XC = 0.110
BOOT_Y = np.array([0.070, 0.066, 0.055, 0.035, 0.010, -0.015, -0.036, -0.045, -0.060, -0.085, -0.115, -0.150, -0.185, -0.215, -0.233, -0.243, -0.248])
BOOT_W = np.array([0.034, 0.041, 0.048, 0.051, 0.052, 0.052, 0.052, 0.052, 0.053, 0.055, 0.057, 0.056, 0.050, 0.040, 0.027, 0.013, 0.004])
BOOT_ZT = np.array([0.125, 0.150, 0.190, 0.208, 0.216, 0.216, 0.212, 0.188, 0.158, 0.128, 0.105, 0.088, 0.077, 0.069, 0.060, 0.050, 0.044])


def boot_ring(y, w, zb, zt, M=28, n_top=2.3, n_bot=5.5):
    zc, hh = 0.5 * (zb + zt), 0.5 * (zt - zb)
    phi = np.linspace(0, 2 * np.pi, M, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    n = np.where(s >= 0, n_top, n_bot)
    x = BOOT_XC + w * np.sign(c) * np.abs(c) ** (2.0 / n)
    z = zc + hh * np.sign(s) * np.abs(s) ** (2.0 / n)
    return np.stack([x, np.full(M, y), z], 1)


def boot_surface(y, xoff):
    """height of the leather upper at lateral offset xoff and length position y (for placing eyelets and laces)"""
    w = float(np.interp(-y, -BOOT_Y, BOOT_W))
    zt = float(np.interp(-y, -BOOT_Y, BOOT_ZT))
    zc, hh = 0.5 * (0.040 + zt), 0.5 * (zt - 0.040)
    c = min(abs(xoff) / w, 0.999) ** (2.3 / 2.0)
    s = math.sqrt(max(1.0 - c * c, 0.0))
    return zc + hh * s ** (2.0 / 2.3)


def boot_parts():
    """one left boot: (V, polys, material slot). slots: 0 leather, 1 toe cap, 2 sole, 3 lace, 4 eyelet, 5 dark leather"""
    parts = []
    rings = [boot_ring(y, w, 0.040, zt) for y, w, zt in zip(BOOT_Y, BOOT_W, BOOT_ZT)]
    V, p = loft(rings)
    parts.append((V, p, 0))
    # reinforced toe cap: a thin dark-leather overlay on the front of the upper
    idx = [i for i, y in enumerate(BOOT_Y) if y <= -0.178]
    V, p = loft([boot_ring(BOOT_Y[i], BOOT_W[i] + 0.0030, 0.0405, BOOT_ZT[i] + 0.0030) for i in idx])
    parts.append((V, p, 5))
    # heel counter: a stiffer dark-leather piece wrapped round the back of the boot
    hidx = [i for i, y in enumerate(BOOT_Y) if y >= 0.020]
    V, p = loft([boot_ring(BOOT_Y[i], BOOT_W[i] + 0.0030, 0.0405, min(BOOT_ZT[i], 0.100) + 0.0030) for i in hidx])
    parts.append((V, p, 5))
    # sole: mid slab, welt, then a shaped heel and a shaped forefoot outsole following the foot outline
    V, p = loft([boot_ring(y, w + 0.0080, 0.017, 0.042, n_top=6.0, n_bot=6.0) for y, w in zip(BOOT_Y, BOOT_W)])
    parts.append((V, p, 2))
    V, p = loft([boot_ring(y, w + 0.0036, 0.039, 0.048, n_top=4.0, n_bot=4.0) for y, w in zip(BOOT_Y, BOOT_W)])
    parts.append((V, p, 5))
    hi = [i for i, y in enumerate(BOOT_Y) if y >= 0.012]
    V, p = loft([boot_ring(BOOT_Y[i], BOOT_W[i] + 0.0065, 0.0, 0.021, n_top=5.0, n_bot=5.0) for i in hi])
    parts.append((V, p, 2))
    fi = [i for i, y in enumerate(BOOT_Y) if y <= -0.040]
    V, p = loft([boot_ring(BOOT_Y[i], BOOT_W[i] + 0.0075, 0.0, 0.021, n_top=5.0, n_bot=5.0) for i in fi])
    parts.append((V, p, 2))
    # tongue: a padded strip along the instep
    ty = np.linspace(-0.040, -0.150, 12)
    path = np.array([[BOOT_XC, y, boot_surface(y, 0.0) + 0.0035] for y in ty])
    V, p = tube(path, 0.0285, 0.0045, sides=10, caps=(True, True), up=(1.0, 0.0, 0.0))
    parts.append((V, p, 5))
    # eyelets and criss-cross laces
    ys = [-0.056, -0.077, -0.098, -0.119, -0.140]
    pos = {}
    for yy in ys:
        for sx in (-1, 1):
            xo = sx * 0.031
            z = boot_surface(yy, xo)
            dy = -0.008
            dz = boot_surface(yy - 0.004, xo) - boot_surface(yy + 0.004, xo)
            n = unit((0.0, dz, -dy))
            V, p = torus((BOOT_XC + xo, yy, z + 0.0012), 0.0046, 0.0014, axis=n, nu=14, nv=6)
            parts.append((V, p, 4))
            pos[(yy, sx)] = np.array([BOOT_XC + xo, yy, z + 0.0035])
    for i in range(len(ys) - 1):
        for sx in (-1, 1):
            a, b = pos[(ys[i], sx)], pos[(ys[i + 1], -sx)]
            ym = 0.5 * (ys[i] + ys[i + 1])
            mid = np.array([BOOT_XC, ym, boot_surface(ym, 0.0) + 0.0095])
            path = np.array([a, 0.5 * (a + mid) + [0, 0, 0.001], mid, 0.5 * (b + mid) + [0, 0, 0.001], b])
            V, p = tube(path, 0.0024, sides=6, caps=(True, True), up=(0.0, 0.0, 1.0))
            parts.append((V, p, 3))
    zt_ = boot_surface(-0.050, 0.0) + 0.008
    for sx in (-1, 1):
        V, p = torus((BOOT_XC + sx * 0.012, -0.050, zt_), 0.0105, 0.0024, axis=(0.0, 0.5, 1.0), nu=16, nv=6)
        parts.append((V, p, 3))
    return parts


def build_boots_skin():
    S = Suit()
    for s, sx in (('L', 1.0), ('R', -1.0)):
        joints = [J['kn_' + s], J['an_' + s], J['ball_' + s], J['toe_' + s]]
        bones = ['shin.' + s, 'foot.' + s, 'toe.' + s]
        for V, p, mat in boot_parts():
            V = np.asarray(V, float).copy()
            V[:, 0] = BOOT_XC + (V[:, 0] - BOOT_XC) * 1.07        # a little chunkier than a city boot
            V[:, 1] = 0.015 + (V[:, 1] - 0.015) * 1.07
            V[:, 2] = V[:, 2] * 1.04
            if sx < 0:
                V, p = mirror_x(V, p)
            S.host(V, p, mat, joints, bones, 0.045)
    return S


PALM_L = 0.098
FINGERS = {'index': (-0.0310, (0.040, 0.024, 0.020), (0.0098, 0.0090, 0.0080, 0.0068), -3.0),
           'middle': (-0.0105, (0.043, 0.027, 0.021), (0.0100, 0.0092, 0.0082, 0.0070), 0.0),
           'ring': (0.0105, (0.040, 0.025, 0.020), (0.0095, 0.0088, 0.0078, 0.0066), 2.0),
           'pinky': (0.0300, (0.032, 0.019, 0.017), (0.0085, 0.0078, 0.0070, 0.0060), 6.0)}
CURL0 = (8.0, 18.0, 26.0)      # rest curl of the relaxed hand (degrees, per segment direction)
THUMB_PTS = [(-0.034, -0.010, 0.022), (-0.056, -0.020, 0.058), (-0.066, -0.026, 0.088), (-0.068, -0.027, 0.112)]
THUMB_R = (0.0128, 0.0118, 0.0102, 0.0088)


def hand_frame(s):
    """origin (wrist) and rotation of the hand: z along the fingers, y = back of the hand, x across (thumb on -x for L, +x for R)"""
    return J['wr_' + s], basis(hand_dir(s), (1.0, 0.0, 0.0) if s == 'L' else (-1.0, 0.0, 0.0))


def L2D(origin, R, p):
    return np.asarray(origin, float) + np.asarray(R) @ np.asarray(p, float)


def hand_end(s):
    wr, R = hand_frame(s)
    return L2D(wr, R, (0.0, 0.0, PALM_L if s == 'L' else 0.088))


def finger_local_joints(x, lens, splay):
    sp = math.radians(splay)
    p = np.array([x, 0.0, PALM_L])
    pts = [p.copy()]
    for k in range(3):
        c = math.radians(CURL0[k])
        d = np.array([math.sin(sp) * math.cos(c), -math.cos(sp) * math.sin(c), math.cos(sp) * math.cos(c)])
        p = p + d * lens[k]
        pts.append(p.copy())
    return pts


def hand_joints_L():
    wr, R = hand_frame('L')
    out = {}
    for nm, (x, lens, rad, sp) in FINGERS.items():
        out[nm] = [L2D(wr, R, q) for q in finger_local_joints(x, lens, sp)]
    out['thumb'] = [L2D(wr, R, q) for q in THUMB_PTS]
    return out


def mech_local_joints():
    out = {}
    for nm, x in (('mi', 0.026), ('mm', 0.0), ('mr', -0.026)):
        pts = [np.array([x, 0.0, 0.088])]
        for ln, cdeg in ((0.046, 10.0), (0.040, 26.0)):
            c = math.radians(cdeg)
            pts.append(pts[-1] + np.array([0.0, -math.sin(c), math.cos(c)]) * ln)
        out[nm] = pts
    t0 = np.array([0.046, -0.004, 0.034])
    d1 = unit([0.55, -0.15, 0.82])
    d2 = unit([0.50, -0.45, 0.74])
    out['mt'] = [t0, t0 + d1 * 0.045, t0 + d1 * 0.045 + d2 * 0.036]
    return out


def mech_joints_R():
    wr, R = hand_frame('R')
    return {nm: [L2D(wr, R, q) for q in pts] for nm, pts in mech_local_joints().items()}


def finger_tube(joints, radii, ellip=0.9, sides=10):
    pts, rad = [], []
    for k in range(3):
        a, b = joints[k], joints[k + 1]
        ts = (0.0, 0.33, 0.66) if k < 2 else (0.0, 0.33, 0.66, 1.0)
        for t in ts:
            pts.append(a + (b - a) * t)
            rad.append(radii[k] + (radii[k + 1] - radii[k]) * t)
    d = unit(joints[3] - joints[2])
    pts += [joints[3] + d * 0.0025, joints[3] + d * 0.0042]
    rad += [radii[3] * 0.82, radii[3] * 0.45]
    rad = np.array(rad)
    for i in (3, 6):
        rad[i] *= 1.07
    return np.array(pts), rad


def build_hand_skin():
    """the human left hand in a fingerless work glove: skin (0), glove (1), palm leather (2), strap (3), nails (4)"""
    S = Suit()
    wr, R = hand_frame('L')
    HJ = hand_joints_L()
    pal_j = [J['el_L'], wr, hand_end('L')]
    pal_b = ['forearm.L', 'hand.L']

    def Lp(p):
        return L2D(wr, R, p)

    def palmhost(V, P, mat, blend=0.035):
        S.host(V, P, mat, pal_j, pal_b, blend)

    V, P = rbox(Lp((0, 0, 0.049)), (0.086, 0.030, 0.098), 0.011, R, n=2)
    palmhost(V, P, 0)
    path = np.array([Lp((0, 0, z)) for z in np.linspace(-0.030, 0.012, 5)])
    V, P = tube(path, np.full(5, 0.029), np.full(5, 0.019), sides=14, caps=(False, False), up=R[:, 0])
    palmhost(V, P, 0)
    V, P = ellipsoid(Lp((-0.030, -0.010, 0.040)), (0.020, 0.012, 0.032), 12, 8, R)
    palmhost(V, P, 0)
    V, P = ellipsoid(Lp((0.032, -0.008, 0.048)), (0.014, 0.011, 0.030), 12, 8, R)
    palmhost(V, P, 0)
    # fingers and thumb (skin) with nails
    for nm in ('index', 'middle', 'ring', 'pinky', 'thumb'):
        jt = HJ[nm]
        rad = THUMB_R if nm == 'thumb' else FINGERS[nm][2]
        pth, rr = finger_tube(jt, rad, ellip=0.88 if nm == 'thumb' else 0.9)
        V, P = tube(pth, rr, rr * (0.88 if nm == 'thumb' else 0.9), sides=10, caps=(False, True), up=R[:, 0])
        bones = ['%s%d.L' % (nm, k) for k in (1, 2, 3)]
        S.host(V, P, 0, jt, bones, 0.006)
        dseg = unit(jt[3] - jt[2])
        dors = R[:, 1] if nm != 'thumb' else unit(0.6 * R[:, 1] - 0.8 * R[:, 0])
        c = jt[2] + (jt[3] - jt[2]) * 0.55 + dors * (rad[2] * 0.9 + 0.0004)
        V, P = ellipsoid(c, (0.0042, 0.0012, 0.0080), 10, 6, basis(dseg, dors))
        S.host(V, P, 4, jt, bones, 0.006)
    # glove: a shell that tapers from the wrist to the knuckles
    gz = [-0.030, -0.012, 0.020, 0.060, 0.095, 0.1015]
    gx = [0.0385, 0.0375, 0.0400, 0.0455, 0.0470, 0.0460]
    gy = [0.0205, 0.0190, 0.0178, 0.0195, 0.0185, 0.0150]
    ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    cc, ss = np.cos(ph), np.sin(ph)
    nn = 3.2
    rings = []
    for z_, xw, yh in zip(gz, gx, gy):
        xs = xw * np.sign(cc) * np.abs(cc) ** (2.0 / nn)
        ys = 0.0006 + yh * np.sign(ss) * np.abs(ss) ** (2.0 / nn)
        rings.append(np.array([Lp((x_, y_, z_)) for x_, y_ in zip(xs, ys)]))
    V, P = loft(rings)
    palmhost(V, P, 1)
    V, P = rbox(Lp((0, -0.0201, 0.053)), (0.064, 0.0045, 0.062), 0.002, R, n=1)
    palmhost(V, P, 2)
    for nm in ('index', 'middle', 'ring', 'pinky'):
        jt = HJ[nm]
        rad = FINGERS[nm][2]
        a = jt[0] - unit(jt[1] - jt[0]) * 0.004
        b = jt[0] + (jt[1] - jt[0]) * 0.72
        pts = np.array([a + (b - a) * t for t in np.linspace(0, 1, 5)])
        rr = np.array([rad[0] + 0.0036 - 0.0003 * k for k in range(5)])
        V, P = tube(pts, rr, rr * 0.92, sides=10, caps=(False, False), up=R[:, 0])
        bones = ['%s%d.L' % (nm, k) for k in (1, 2, 3)]
        S.host(V, P, 1, jt, bones, 0.006)
        V, P = torus(b, rr[-1] - 0.0004, 0.0024, axis=unit(jt[1] - jt[0]), nu=14, nv=6)
        S.host(V, P, 1, jt, bones, 0.006)
    jt = HJ['thumb']
    tpts = np.array([jt[0], lerp(jt[0], jt[1], 0.5), jt[1], lerp(jt[1], jt[2], 0.30), lerp(jt[1], jt[2], 0.55)])
    trr = np.array([THUMB_R[0] + 0.0036, THUMB_R[0] + 0.0034, THUMB_R[1] + 0.0033, THUMB_R[1] + 0.0030, THUMB_R[2] + 0.0028])
    V, P = tube(tpts, trr, trr * 0.9, sides=10, caps=(False, False), up=R[:, 0])
    S.host(V, P, 1, jt, ['thumb1.L', 'thumb2.L', 'thumb3.L'], 0.007)
    V, P = torus(tpts[-1], trr[-1] - 0.0004, 0.0024, axis=unit(tpts[-1] - tpts[-2]), nu=14, nv=6)
    S.host(V, P, 1, jt, ['thumb1.L', 'thumb2.L', 'thumb3.L'], 0.007)
    # cuff over the sleeve end, strap and velcro tab
    cpath = np.array([Lp((0, 0, z)) for z in np.linspace(-0.056, 0.004, 5)])
    crad = np.array([0.0588, 0.0575, 0.0545, 0.0490, 0.0440])
    V, P = tube(cpath, crad, crad * 0.96, sides=20, caps=(False, False), up=R[:, 0])
    palmhost(V, P, 1, 0.03)
    V, P = torus(Lp((0, 0, -0.034)), 0.0578, 0.0030, axis=R[:, 2], nu=26, nv=6)
    palmhost(V, P, 3, 0.03)
    V, P = rbox(Lp((0, 0.0600, -0.034)), (0.030, 0.0042, 0.020), 0.0015, R, n=1)
    palmhost(V, P, 3, 0.03)
    return S


def mech_arm_parts():
    """scavenged mechanical right forearm and hand (rigid parts, each on its own bone)"""
    parts = []
    s = 'R'
    el, wr = J['el_R'], J['wr_R']
    a, b = lerp(el, wr, 0.42), wr + (wr - el) * 0.02
    d = unit(b - a)
    V, p = cyl(a, b, 0.047, 0.040, 18)
    parts.append(('ArmCoreR', V, p, 'steel', 'forearm.R'))
    V, p = rbox(lerp(a, b, 0.35), (0.092, 0.092, 0.095), 0.016, basis(d, (0, 1, 0)), n=2)
    parts.append(('ArmCoverR', V, p, 'paint', 'forearm.R'))
    V, p = rbox(lerp(a, b, 0.72) + np.array([0.0, -0.030, 0.0]), (0.070, 0.050, 0.110), 0.012, basis(d, (0, 1, 0)), n=2)
    parts.append(('ArmCover2R', V, p, 'paint', 'forearm.R'))
    for k, off in enumerate((np.array([0.0, 0.040, 0.0]), np.array([0.034, 0.025, 0.0]))):
        V, p = cyl(a + off, b + off, 0.008, 0.008, 8)
        parts.append(('PistonR%d' % k, V, p, 'steel', 'forearm.R'))
    for k in range(3):
        V, p = torus(lerp(a, b, 0.12 + 0.25 * k), 0.049 - 0.0035 * k, 0.0048, axis=d, nu=22, nv=6)
        parts.append(('ArmBandR%d' % k, V, p, 'steel', 'forearm.R'))
    cp = np.array([lerp(a, b, 0.0) + np.array([0.030, 0.032, 0.0]), lerp(a, b, 0.35) + np.array([0.058, 0.044, 0.0]),
                   lerp(a, b, 0.7) + np.array([0.045, 0.042, 0.0]), lerp(a, b, 1.0) + np.array([0.020, 0.030, 0.0])])
    V, p = tube(cp, 0.0065, sides=7, caps=(True, True), up=(1, 0, 0))
    parts.append(('ArmCableR', V, p, 'rubber', 'forearm.R'))
    # hand
    wr_, R = hand_frame('R')
    MJ = mech_joints_R()

    def Lp(q):
        return L2D(wr_, R, q)

    V, p = cyl(Lp((0, 0, -0.012)), Lp((0, 0, 0.022)), 0.044, 0.046, 18)
    parts.append(('MechWristR', V, p, 'steel', 'hand.R'))
    V, p = rbox(Lp((0, 0, 0.056)), (0.088, 0.040, 0.074), 0.010, R, n=2)
    parts.append(('MechPalmR', V, p, 'steelDark', 'hand.R'))
    V, p = rbox(Lp((0, 0.0235, 0.056)), (0.076, 0.010, 0.062), 0.004, R, n=2)
    parts.append(('MechPalmCoverR', V, p, 'paint', 'hand.R'))
    for k, (rx, rz) in enumerate(((-0.030, 0.030), (0.030, 0.030), (-0.030, 0.080), (0.030, 0.080))):
        V, p = ellipsoid(Lp((rx, 0.0295, rz)), (0.0042, 0.0030, 0.0042), 8, 6, R)
        parts.append(('RivetR%d' % k, V, p, 'steel', 'hand.R'))
    V, p = cyl(Lp((-0.040, 0.0, 0.088)), Lp((0.040, 0.0, 0.088)), 0.0108, 0.0108, 12)
    parts.append(('MechKnuckleBarR', V, p, 'steel', 'hand.R'))
    for nm in ('mi', 'mm', 'mr', 'mt'):
        jt = MJ[nm]
        b1, b2 = '%s1.R' % nm, '%s2.R' % nm
        d1, d2 = unit(jt[1] - jt[0]), unit(jt[2] - jt[1])
        l1, l2 = np.linalg.norm(jt[1] - jt[0]), np.linalg.norm(jt[2] - jt[1])
        w1, w2 = (0.021, 0.019) if nm == 'mt' else (0.019, 0.017)
        V, p = rbox((jt[0] + jt[1]) / 2, (w1, w1, l1 + 0.004), 0.005, basis(d1, R[:, 1]), n=1)
        parts.append(('Mech%sSeg1' % nm, V, p, 'steel', b1))
        V, p = cyl(jt[1] - R[:, 0] * 0.0115, jt[1] + R[:, 0] * 0.0115, 0.0088, 0.0088, 10)
        parts.append(('Mech%sPin' % nm, V, p, 'steelDark', b1))
        V, p = rbox((jt[1] + jt[2]) / 2, (w2, w2, l2 + 0.004), 0.005, basis(d2, R[:, 1]), n=1)
        parts.append(('Mech%sSeg2' % nm, V, p, 'steel', b2))
        V, p = rbox(jt[2] - d2 * 0.010 - R[:, 1] * (w2 / 2 + 0.001), (0.0155, 0.0055, 0.022), 0.002, basis(d2, R[:, 1]), n=1)
        parts.append(('Mech%sPad' % nm, V, p, 'rubber', b2))
        if nm != 'mt':
            V, p = cyl(jt[0] + R[:, 1] * 0.015 - d1 * 0.012, jt[1] + R[:, 1] * 0.0125 - d1 * 0.004, 0.0034, 0.0034, 8)
            parts.append(('Mech%sPiston' % nm, V, p, 'steel', b1))
    for k, x in enumerate((-0.018, 0.018)):
        cp = np.array([Lp((x * 0.6, 0.026, -0.030)), Lp((x, 0.030, 0.020)), Lp((x * 1.2, 0.030, 0.062)), Lp((x * 1.4, 0.026, 0.086))])
        V, p = tube(cp, 0.0055, sides=7, caps=(True, True), up=(1, 0, 0))
        parts.append(('MechCable%d' % k, V, p, 'rubber', 'hand.R'))
    return parts
