# Motion Assist: sample an animation and judge it the way a physics check would.
# Per frame: centre of mass, its velocity and acceleration, angular momentum about it, which probe points touch the
# ground (fulcrum points), the support polygon and the zero moment point (where the ground must push for the motion to
# be possible). Frames with no contact are flight: there the centre of mass must follow a parabola and the angular
# momentum must stay constant.

import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import body as bodymod

CACHE = {}

FLIGHT, OK, OFF_BALANCE, SLIP, PULL = 0, 1, 2, 3, 4
STATUS_NAMES = {FLIGHT: "in the air", OK: "balanced", OFF_BALANCE: "off balance", SLIP: "would slip", PULL: "pulled down"}


def fps_of(scene):
    return scene.render.fps / scene.render.fps_base


def frame_range(scene, s):
    if s.use_scene_range:
        return scene.frame_start, scene.frame_end
    return s.frame_start, max(s.frame_end, s.frame_start + 2)


def orthonormal(A):
    U, _S, Vt = np.linalg.svd(A)
    return U @ Vt


def rotvec(R):
    """rotation vectors of a stack of rotation matrices (n, 3, 3)"""
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1) / 2.0
    s = np.sin(ang)
    k = np.where(s > 1e-8, ang / np.maximum(s, 1e-8), 1.0)
    return w * k[:, None]


def hull2d(pts):
    P = sorted(set((round(float(x), 5), round(float(y), 5)) for x, y in pts))
    if len(P) <= 2:
        return np.array(P, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _seg_dist(p, a, b):
    ab = b - a
    t = 0.0 if ab.dot(ab) < 1e-12 else min(max((p - a).dot(ab) / ab.dot(ab), 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def hull_dist(H, p):
    """0 inside the convex polygon H (counter-clockwise), else the distance to it"""
    p = np.asarray(p, dtype=float)
    if len(H) == 0:
        return math.inf
    if len(H) == 1:
        return float(np.linalg.norm(p - H[0]))
    if len(H) == 2:
        return _seg_dist(p, H[0], H[1])
    inside = True
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            inside = False
            break
    if inside:
        return 0.0
    return min(_seg_dist(p, H[i], H[(i + 1) % len(H)]) for i in range(len(H)))


def _overlaps(ob, lo, hi):
    M = ob.matrix_world
    C = np.array([tuple(M @ Vector(c)) for c in ob.bound_box])
    return bool(np.all(C.max(axis=0) >= lo) and np.all(C.min(axis=0) <= hi))


class Ground:
    """signed distance from points to the scene's surfaces near the character, or to a flat floor"""

    def __init__(self, scene, exclude, mode='SCENE', height=0.0, lo=None, hi=None, ignore=()):
        self.mode, self.h, self.tree, self.objects = mode, height, None, []
        if mode != 'SCENE':
            return
        deps = bpy.context.evaluated_depsgraph_get()
        V, T, base = [], [], 0
        for ob in scene.objects:
            if ob.type != 'MESH' or ob.name in exclude or bodymod.ignored(ob, ignore) or ob.name.startswith("MA_"):
                continue
            if not ob.visible_get() or ob.hide_render:
                continue
            if lo is not None and not _overlaps(ob, lo, hi):
                continue
            ev = ob.evaluated_get(deps)
            me = ev.to_mesh()
            try:
                me.calc_loop_triangles()
                n, t = len(me.vertices), len(me.loop_triangles)
                if n == 0 or t == 0:
                    continue
                co = np.empty(n * 3)
                me.vertices.foreach_get("co", co)
                M = np.array(ob.matrix_world, dtype=float)
                V.append(co.reshape(n, 3) @ M[:3, :3].T + M[:3, 3])
                tri = np.empty(t * 3, dtype=np.int64)
                me.loop_triangles.foreach_get("vertices", tri)
                T.append(tri.reshape(t, 3) + base)
                base += n
                self.objects.append(ob.name)
            finally:
                ev.to_mesh_clear()
        if V:
            self.tree = BVHTree.FromPolygons(np.vstack(V).tolist(), np.vstack(T).tolist(), all_triangles=True)
        else:
            self.mode = 'PLANE'

    def query(self, P, maxd=0.3):
        P = np.asarray(P, dtype=float)
        if self.mode == 'PLANE':
            Q = P.copy()
            Q[:, 2] = self.h
            return P[:, 2] - self.h, Q
        D = np.full(len(P), np.inf)
        Q = np.full((len(P), 3), np.nan)
        fn = self.tree.find_nearest
        for i, p in enumerate(P):
            v = Vector(p)
            loc, nrm, _idx, dist = fn(v, maxd)
            if loc is None:
                continue
            if nrm.z < -0.5:
                nrm = -nrm
            D[i] = dist if (v - loc).dot(nrm) >= 0.0 else -dist
            Q[i] = tuple(loc)
        return D, Q


def _ancestors(b):
    out = []
    while b is not None:
        out.append(b.name)
        b = b.parent
    return out


def rig_info(rig, body):
    """what the fixes move: the master control (moves everything), the hips (body bone) and the IK chains"""
    bones = rig.data.bones
    tops = [b.name for b in bones if b.parent is None]
    chains = []
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type == 'IK' and not c.mute and c.target == rig and c.subtarget in rig.pose.bones:
                chain = [pb.name]
                b = pb.bone
                while b.parent is not None and (c.chain_count == 0 or len(chain) < c.chain_count):
                    b = b.parent
                    chain.append(b.name)
                chain.reverse()
                chains.append({"end": pb.name, "target": c.subtarget, "chain": chain,
                               "below": [d.name for d in pb.bone.children_recursive],
                               "length": float(sum(bones[n].length for n in chain))})
    common = None
    for n in body.names:
        a = _ancestors(bones[n])
        common = a if common is None else [x for x in common if x in a]
    hips = common[0] if common else None
    master = tops[0] if len(tops) == 1 else None
    return {"tops": tops, "master": master, "body": hips, "chains": chains}


def control_bones(info):
    names = list(info["tops"])
    if info["body"]:
        names.append(info["body"])
    for ch in info["chains"]:
        names.append(ch["target"])
        names.append(ch["chain"][0])
    return list(dict.fromkeys(names))


def analyze(scene, rig, s):
    body = bodymod.get_body(rig)
    if body is None:
        raise RuntimeError("build the body model first")
    info = rig_info(rig, body)
    ctrl = control_bones(info)
    f0, f1 = frame_range(scene, s)
    dt = 1.0 / fps_of(scene)
    frames = list(range(f0 - 2, f1 + 3))
    F = len(frames)
    W = np.empty((F, len(body.names), 4, 4))
    CW = {n: np.empty((F, 4, 4)) for n in ctrl}           # control bones: world matrix
    CB = {n: np.empty((F, 4, 4)) for n in ctrl}           # control bones: local (basis) matrix
    cur = scene.frame_current
    try:
        for k, f in enumerate(frames):
            scene.frame_set(f)
            W[k] = body.mats(rig)
            Mw = rig.matrix_world
            for n in ctrl:
                pb = rig.pose.bones[n]
                CW[n][k] = np.array(Mw @ pb.matrix, dtype=float)
                CB[n][k] = np.array(pb.matrix_basis, dtype=float)
    finally:
        scene.frame_set(cur)

    X = np.einsum("fnij,nj->fni", W[:, :, :3, :3], body.c) + W[:, :, :3, 3]
    com = (body.m[None, :, None] * X).sum(axis=1) / body.M
    Wp = W[:, body.pbone]
    P = np.einsum("fpij,pj->fpi", Wp[:, :, :3, :3], body.plocal) + Wp[:, :, :3, 3]

    vel = np.zeros((F, 3))
    acc = np.zeros((F, 3))
    vel[1:-1] = (com[2:] - com[:-2]) / (2 * dt)
    acc[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) / (dt * dt)

    R = orthonormal(W[:, :, :3, :3].reshape(-1, 3, 3)).reshape(F, -1, 3, 3)
    L = np.zeros((F, 3))
    Itot = np.tile(np.eye(3), (F, 1, 1))
    for k in range(1, F - 1):
        v = (X[k + 1] - X[k - 1]) / (2 * dt)
        vc = (com[k + 1] - com[k - 1]) / (2 * dt)
        w = rotvec(R[k + 1] @ np.transpose(R[k - 1], (0, 2, 1))) / (2 * dt)
        Iw = R[k] @ body.I @ np.transpose(R[k], (0, 2, 1))
        r = X[k] - com[k]
        L[k] = (np.einsum("nij,nj->ni", Iw, w) + body.m[:, None] * np.cross(r, v - vc)).sum(axis=0)
        rr = np.einsum("ni,ni->n", r, r)
        Itot[k] = Iw.sum(axis=0) + (body.m[:, None, None] * (rr[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", r, r))).sum(axis=0)
    Ldot = np.zeros((F, 3))
    Ldot[2:-2] = (L[3:-1] - L[1:-3]) / (2 * dt)

    inner = slice(2, F - 2)
    N = f1 - f0 + 1
    Pin = P[inner]
    speed = np.minimum(np.linalg.norm(P[2:F - 2] - P[1:F - 3], axis=2), np.linalg.norm(P[3:F - 1] - P[2:F - 2], axis=2)) / dt
    ign = bodymod.parse_ignore(s.ignore)
    exclude = {ob.name for ob, _how in bodymod.character_parts(rig, ign)} | {rig.name}
    flat = Pin.reshape(-1, 3)
    ground = Ground(scene, exclude, s.ground, s.ground_height, flat.min(axis=0) - 1.0, flat.max(axis=0) + 1.0, ign)
    D = np.empty((N, Pin.shape[1]))
    Q = np.empty((N, Pin.shape[1], 3))
    for i in range(N):
        D[i], Q[i] = ground.query(Pin[i])
    touch = (D <= s.contact_radius) & (D >= -0.3) & (speed <= s.max_speed)

    g = s.gravity
    status = np.zeros(N, dtype=int)
    zmp = np.full((N, 2), np.nan)
    zg = np.full(N, np.nan)
    margin = np.full(N, np.nan)
    hulls = []
    for i in range(N):
        k = i + 2
        t = touch[i]
        if not t.any():
            status[i] = FLIGHT
            hulls.append(None)
            continue
        pts = Pin[i][t]
        z_g = float(np.nanmean(Q[i][t][:, 2]))
        zg[i] = z_g
        H = hull2d(pts[:, :2])
        hulls.append(H)
        a, c = acc[k], com[k]
        Fz = a[2] + g
        if Fz < 0.05 * g:
            status[i] = PULL
            continue
        h = c[2] - z_g
        px = c[0] - h * a[0] / Fz - Ldot[k][1] / (body.M * Fz)
        py = c[1] - h * a[1] / Fz + Ldot[k][0] / (body.M * Fz)
        zmp[i] = (px, py)
        margin[i] = hull_dist(H, (px, py))
        if math.hypot(a[0], a[1]) > s.friction * Fz:
            status[i] = SLIP
        elif margin[i] > s.zmp_tolerance:
            status[i] = OFF_BALANCE
        else:
            status[i] = OK

    flights = []
    i = 0
    while i < N:
        if status[i] == FLIGHT:
            j = i
            while j + 1 < N and status[j + 1] == FLIGHT:
                j += 1
            if i > 0 and j < N - 1:
                flights.append((f0 + i - 1, f0 + j + 1))
            i = j + 1
        else:
            i += 1

    report = []
    for fa, fb in flights:
        ka, kb = fa - f0 + 2, fb - f0 + 2
        T = (fb - fa) * dt
        cz = com[ka:kb + 1, 2]
        apex = float(cz.max() - max(cz[0], cz[-1]))
        v = (cz[-1] - cz[0]) / T + g * T / 2.0
        top = cz[0] + v * v / (2.0 * g) if g > 0 and 0.0 < v / g < T else max(cz[0], cz[-1])
        need = float(top - max(cz[0], cz[-1]))
        Ls = L[ka + 1:kb]
        drift = float(np.linalg.norm(Ls - Ls.mean(axis=0), axis=1).max() / max(np.linalg.norm(Ls.mean(axis=0)), 1.0)) if len(Ls) else 0.0
        report.append({"from": fa, "to": fb, "frames": fb - fa, "apex": apex, "physical_apex": need, "spin_drift": drift})

    data = dict(f0=f0, f1=f1, dt=dt, frames=frames, info=info, com=com[inner], vel=vel[inner], acc=acc[inner],
                L=L[inner], com_full=com, L_full=L, I_full=Itot, vel_full=vel, W=W, CW=CW, CB=CB, P=Pin, D=D, Q=Q,
                touch=touch, status=status, zmp=zmp, zg=zg, margin=margin, hulls=hulls, flights=flights,
                jumps=report, ground=ground.mode, ground_objects=ground.objects)
    old = CACHE.get(rig.name)
    if old is not None and old.get("keep_before") and old["f0"] == f0 and old["f1"] == f1:
        data["com_before"] = old.get("com_before")
        data["keep_before"] = True
    CACHE[rig.name] = data
    rig["ma_summary"] = summary(data)
    return data


def summary(data):
    st = data["status"]
    parts = ["%d frames" % len(st)]
    if data["jumps"]:
        js = []
        for j in data["jumps"]:
            js.append("f%d-%d (%d fr, apex %.0f cm, physics says %.0f cm)" % (j["from"], j["to"], j["frames"], j["apex"] * 100, j["physical_apex"] * 100))
        parts.append("jumps: " + "; ".join(js))
    else:
        parts.append("no jumps")
    for code in (OFF_BALANCE, SLIP, PULL):
        n = int((st == code).sum())
        if n:
            parts.append("%s on %d frames" % (STATUS_NAMES[code], n))
    return " | ".join(parts)
