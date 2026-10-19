# Motion Assist: overlap (secondary motion).
# Each chosen bone's tip follows where the animation puts it through a spring-damper (second-order dynamics:
# frequency = stiffness, damping, response), so it lags, overshoots and settles; the bone is then aimed at the filtered
# tip. Bones are processed parent first, so a chain whips. The result is keyed on every frame of the range.

import math

import numpy as np
from mathutils import Matrix, Vector

from . import body as bodymod
from . import limits
from .analysis import fps_of


class Spring:
    def __init__(self, f, z, r, x0):
        self.k1 = z / (math.pi * f)
        self.k2 = 1.0 / ((2.0 * math.pi * f) ** 2)
        self.k3 = r * z / (2.0 * math.pi * f)
        self.xp = np.array(x0, dtype=float)
        self.y = self.xp.copy()
        self.yd = np.zeros_like(self.xp)

    def step(self, T, x):
        x = np.asarray(x, dtype=float)
        xd = (x - self.xp) / T
        self.xp = x.copy()
        k2 = max(self.k2, T * T / 2.0 + T * self.k1 / 2.0, T * self.k1)
        self.y = self.y + T * self.yd
        self.yd = self.yd + T * (x + self.k3 * xd - self.y - self.k1 * self.yd) / k2
        return self.y


def _depth(b):
    d = 0
    while b.parent is not None:
        b = b.parent
        d += 1
    return d


def bake_overlap(scene, rig, names, f0, f1, freq=2.5, damp=0.45, resp=0.0, strength=1.0, substeps=4, use_limits=True):
    lims = limits.rig_limits(rig) if use_limits else {}
    bad = [n for n in names if rig.pose.bones[n].constraints]
    names = [n for n in names if n not in bad]
    if not names:
        raise RuntimeError("pick bones without constraints (IK-driven bones are skipped)")
    sel = set(names)
    order = sorted(names, key=lambda n: _depth(rig.data.bones[n]))
    frames = list(range(f0, f1 + 1))
    basis = {n: [] for n in names}
    parents = {rig.pose.bones[n].parent.name for n in names if rig.pose.bones[n].parent is not None} - sel
    anchor = {p: [] for p in parents}
    rigw = []
    cur = scene.frame_current
    try:
        for f in frames:
            scene.frame_set(f)
            Mw = rig.matrix_world.copy()
            rigw.append(Mw)
            for n in names:
                basis[n].append(rig.pose.bones[n].matrix_basis.copy())
            for p in parents:
                anchor[p].append(Mw @ rig.pose.bones[p].matrix)
    finally:
        scene.frame_set(cur)
    rel = {}
    for n in names:
        b = rig.data.bones[n]
        rel[n] = (b.parent.matrix_local.inverted() @ b.matrix_local) if b.parent else b.matrix_local.copy()

    def parent_world(n, i, new):
        b = rig.data.bones[n]
        if b.parent is None:
            return rigw[i]
        pn = b.parent.name
        return new[pn][i] if pn in new else anchor[pn][i]

    T = 1.0 / fps_of(scene)
    new = {}
    for n in order:
        L = rig.data.bones[n].length
        out = []
        spring = None
        prev = None
        for i in range(len(frames)):
            Wd = parent_world(n, i, new) @ rel[n] @ basis[n][i]          # the pose if the bone followed its parent rigidly
            head = Wd.translation.copy()
            tip = np.array(Wd @ Vector((0.0, L, 0.0)))
            if spring is None:
                spring = Spring(freq, damp, resp, tip)
                y = tip
            else:
                for k in range(1, substeps + 1):
                    y = spring.step(T / substeps, prev + (tip - prev) * (k / substeps))
            prev = tip
            d0 = (Vector(tuple(tip)) - head).normalized()
            d1 = (Vector(tuple(y)) - head).normalized()
            if strength < 1.0:
                d1 = d0.slerp(d1, strength)
            q = d0.rotation_difference(d1)
            Wn = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ Wd
            lim = lims.get(n)
            if lim is not None:
                # the swing must not push the joint where it cannot go (an elbow sideways, a wrist rolled over)
                Wb = parent_world(n, i, new) @ rel[n]
                loc, ql, _s = (Wb.inverted() @ Wn).decompose()
                Wn = Wb @ Matrix.Translation(loc) @ limits.clamp_rotation(ql, lim).to_matrix().to_4x4()
            out.append(Wn)
        new[n] = out

    for n in names:
        pb = rig.pose.bones[n]
        prev_e = prev_q = None
        for i, f in enumerate(frames):
            Bn = (parent_world(n, i, new) @ rel[n]).inverted() @ new[n][i]
            _loc, q, _sca = Bn.decompose()
            if pb.rotation_mode == 'QUATERNION':
                if prev_q is not None and prev_q.dot(q) < 0:
                    q.negate()
                pb.rotation_quaternion = q
                prev_q = q.copy()
                pb.keyframe_insert("rotation_quaternion", frame=f, group=n)
            elif pb.rotation_mode == 'AXIS_ANGLE':
                ax, ang = q.to_axis_angle()
                pb.rotation_axis_angle = (ang, ax.x, ax.y, ax.z)
                pb.keyframe_insert("rotation_axis_angle", frame=f, group=n)
            else:
                e = q.to_euler(pb.rotation_mode, prev_e) if prev_e is not None else q.to_euler(pb.rotation_mode)
                pb.rotation_euler = e
                prev_e = e.copy()
                pb.keyframe_insert("rotation_euler", frame=f, group=n)
    return {"bones": names, "skipped": bad}


# ---------------------------------------------------------------------------------------- follow-through for IK and gear

def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _spring_track(X, T, freq, damp, resp, free, blend=3, substeps=4, strength=1.0):
    """second-order follow of the series X (n, k) on the frames where `free` is True; elsewhere the output is X itself.
    When a part comes free the spring starts from where it was with the speed it had, so it carries on with the motion
    (follow-through); over the last `blend` frames of a free stretch it settles back onto X."""
    n = len(X)
    Y = X.copy()
    i = 0
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and free[j + 1]:
            j += 1
        a = max(i - 1, 0)
        spring = Spring(freq, damp, resp, X[a])
        if a > 0:
            spring.yd = (X[a] - X[a - 1]) / T
            spring.xp = X[a] - spring.yd * T
        prev = X[a]
        for k in range(i, j + 1):
            for sub in range(1, substeps + 1):
                y = spring.step(T / substeps, prev + (X[k] - prev) * (sub / substeps))
            prev = X[k]
            w = 1.0 if j + 1 >= n else _smooth((j + 1 - k) / max(blend, 1))
            Y[k] = X[k] + (y - X[k]) * (w * strength)
        i = j + 1
    return Y


def _key_matrix(pb, M, frame, prev):
    loc, q, _s = M.decompose()
    pb.location = loc
    pb.keyframe_insert("location", frame=frame, group=pb.name)
    if pb.rotation_mode == 'QUATERNION':
        if prev[0] is not None and prev[0].dot(q) < 0:
            q.negate()
        pb.rotation_quaternion = q
        prev[0] = q.copy()
        pb.keyframe_insert("rotation_quaternion", frame=frame, group=pb.name)
    elif pb.rotation_mode == 'AXIS_ANGLE':
        ax, ang = q.to_axis_angle()
        pb.rotation_axis_angle = (ang, ax.x, ax.y, ax.z)
        pb.keyframe_insert("rotation_axis_angle", frame=frame, group=pb.name)
    else:
        e = q.to_euler(pb.rotation_mode, prev[1]) if prev[1] is not None else q.to_euler(pb.rotation_mode)
        pb.rotation_euler = e
        prev[1] = e.copy()
        pb.keyframe_insert("rotation_euler", frame=frame, group=pb.name)


def bake_leg_follow(scene, rig, chains, f0, f1, free, freq=5.0, damp=0.35, resp=3.0, strength=1.0, blend=3):
    """follow-through for IK legs in the air. The knee's bend goes through a spring that carries the motion it had and
    overshoots: the push carries on into a straight leg through the toes, then the knee folds faster and further than
    keyed (the heel flicks up) and opens again to reach for the ground. The leg keeps its keyed direction; frames on the
    ground stay exactly as they were; the IK target is moved along the hip-ankle line.
    chains: rig_info chains of the legs; free: {target: bool per frame of f0..f1}"""
    frames = list(range(f0, f1 + 1))
    rec = {ch["target"]: {"H": [], "K": [], "A": [], "W": [], "par": []} for ch in chains}
    cur = scene.frame_current
    try:
        for f in frames:
            scene.frame_set(f)
            Mw = rig.matrix_world
            P = rig.pose.bones
            for ch in chains:
                upper, lower = ch["chain"][0], ch["chain"][-1]
                r = rec[ch["target"]]
                r["H"].append(np.array(Mw @ P[upper].head, dtype=float))
                r["K"].append(np.array(Mw @ P[lower].head, dtype=float))
                r["A"].append(np.array(Mw @ P[lower].tail, dtype=float))
                pb = P[ch["target"]]
                r["W"].append(np.array(Mw @ pb.matrix, dtype=float))
                r["par"].append(np.array(Mw @ pb.parent.matrix, dtype=float) if pb.parent else np.array(Mw, dtype=float))
    finally:
        scene.frame_set(cur)
    T = 1.0 / fps_of(scene)
    out = {}
    for ch in chains:
        t = ch["target"]
        r = {k: np.array(v) for k, v in rec[t].items()}
        m = np.asarray(free[t], dtype=bool)
        if not m.any():
            continue
        a = float(np.mean(np.linalg.norm(r["K"] - r["H"], axis=1)))
        b = float(np.mean(np.linalg.norm(r["A"] - r["K"], axis=1)))
        u, v = r["K"] - r["H"], r["A"] - r["K"]
        bend = np.arccos(np.clip(np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1.0, 1.0))
        bend2 = np.clip(_spring_track(bend[:, None], T, freq, damp, resp, m, blend, strength=strength)[:, 0], 0.0, math.radians(150.0))
        reach = np.sqrt(a * a + b * b + 2.0 * a * b * np.cos(bend2))
        line = r["A"] - r["H"]
        line /= np.maximum(np.linalg.norm(line, axis=1, keepdims=True), 1e-9)
        A2 = np.where(m[:, None], r["H"] + line * reach[:, None], r["A"])
        pb = rig.pose.bones[t]
        bone = pb.bone
        rest = np.array((bone.parent.matrix_local.inverted() @ bone.matrix_local) if bone.parent else bone.matrix_local, dtype=float)
        prev = [None, None]
        for i, f in enumerate(frames):
            W = r["W"][i].copy()
            W[:3, 3] = A2[i]
            _key_matrix(pb, Matrix((np.linalg.inv(r["par"][i] @ rest) @ W).tolist()), f, prev)
        out[t] = round(math.degrees(float(np.abs(bend2 - bend)[m].max())), 1)
    return out


def gear_groups(rig):
    """meshes parented to the rig's bones and marked as gear (custom property ma_gear = group name), grouped"""
    groups = {}
    for ob in rig.children:
        g = ob.get("ma_gear")
        if ob.type == 'MESH' and g:
            groups.setdefault(str(g), []).append(ob)
    return groups


def _pendulum(P, Rm, rhat, L, f, z, gs, gain, T, sub=8, clamp=0.75):
    """damped pendulum about a moving pivot: P (n,3) pivot positions, Rm (n,3,3) object-to-world rotations; returns the
    swing as rotation vectors (n,3) in the object's frame"""
    n = len(P)
    acc = np.zeros_like(P)
    acc[1:-1] = (P[2:] - 2 * P[1:-1] + P[:-2]) / (T * T)
    acc[0], acc[-1] = acc[1], acc[-2]
    acc = np.clip(acc, -45.0, 45.0)
    for k in range(3):
        acc[1:-1, k] = 0.25 * acc[:-2, k] + 0.5 * acc[1:-1, k] + 0.25 * acc[2:, k]
    w0 = 2.0 * math.pi * f
    h = T / sub
    th, om = np.zeros(3), np.zeros(3)
    out = np.zeros((n, 3))
    g = np.array([0.0, 0.0, -9.81]) * gs
    for i in range(n):
        out[i] = th
        j = min(i + 1, n - 1)
        for s in range(sub):
            force = Rm[i].T @ (g - (acc[i] + (acc[j] - acc[i]) * (s / sub)) * gain)
            ang = float(np.linalg.norm(th))
            if ang > 1e-9:
                K = th / ang
                rh = rhat * math.cos(ang) + np.cross(K, rhat) * math.sin(ang) + K * (K @ rhat) * (1 - math.cos(ang))
            else:
                rh = rhat
            om = om + (np.cross(rh, force) / L - (w0 ** 2) * th - 2.0 * z * w0 * om) * h
            th = th + om * h
            m = float(np.linalg.norm(th))
            if m > clamp:
                th *= clamp / m
                om *= 0.5
    return out


def bake_gear(scene, rig, f0, f1, amount=1.0):
    """follow-through for gear: each group of marked meshes swings about its attachment point like a damped pendulum,
    pushed by gravity and by the body's accelerations. Per group, custom properties on its first mesh: ma_gear_freq (Hz),
    ma_gear_damp, ma_gear_gravity (0 = stiff piece, 1 = hangs freely), ma_gear_pivot ('top' or 'bottom' end) and
    ma_gear_swing (its biggest swing in degrees over the range)."""
    groups = gear_groups(rig)
    if not groups:
        return {}
    frames = list(range(f0, f1 + 1))
    for obs in groups.values():
        for ob in obs:
            if "ma_gear_base" in ob:
                ob.matrix_basis = Matrix([list(ob["ma_gear_base"])[r * 4:r * 4 + 4] for r in range(4)])
            else:
                ob["ma_gear_base"] = [float(v) for row in ob.matrix_basis for v in row]
            ob.animation_data_clear()
    mats = {g: [] for g in groups}
    cur = scene.frame_current
    try:
        for f in frames:
            scene.frame_set(f)
            for g, obs in groups.items():
                mats[g].append(obs[0].matrix_world.copy())
    finally:
        scene.frame_set(cur)
    T = 1.0 / fps_of(scene)
    report = {}
    for g, obs in groups.items():
        lead = obs[0]
        f = float(lead.get("ma_gear_freq", 1.5))
        z = float(lead.get("ma_gear_damp", 0.25))
        gs = float(lead.get("ma_gear_gravity", 0.3))
        end = str(lead.get("ma_gear_pivot", "top"))
        inv = lead.matrix_world.inverted()
        pts = []
        for ob in obs:
            co = np.empty(len(ob.data.vertices) * 3)
            ob.data.vertices.foreach_get("co", co)
            M = np.array(inv @ ob.matrix_world, dtype=float)
            pts.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        pts = np.concatenate(pts)
        W0 = np.array(lead.matrix_world.to_3x3(), dtype=float)
        height = (pts @ W0.T)[:, 2]
        cut = np.percentile(height, 92 if end == "top" else 8)
        piv = pts[height >= cut].mean(axis=0) if end == "top" else pts[height <= cut].mean(axis=0)
        r = pts.mean(axis=0) - piv
        L = max(float(np.linalg.norm(r)), 0.05)
        P = np.array([tuple(m @ Vector(tuple(piv))) for m in mats[g]])
        Rm = np.array([np.array(m.to_3x3().normalized(), dtype=float) for m in mats[g]])
        # how hard the body's accelerations throw the piece: scaled until its biggest swing is the one asked for
        target = math.radians(float(lead.get("ma_gear_swing", 25.0)) * amount)
        gain = 0.6
        for _it in range(5):
            th = _pendulum(P, Rm, r / L, L, f, z, gs, gain, T)
            peak = float(np.linalg.norm(th, axis=1).max())
            if target <= 0.0 or abs(peak - target) < 0.05 * target:
                break
            gain *= (target / max(peak, 1e-3)) if peak < 0.7 else 0.5
        report[g] = round(math.degrees(float(np.linalg.norm(th, axis=1).max())), 1)
        pv = Vector(tuple(piv))
        for ob in obs:
            base = ob.matrix_basis.copy()
            to_lead = (lead.matrix_world.inverted() @ ob.matrix_world)
            ob.rotation_mode = 'QUATERNION'
            prevq = None
            for i, fr in enumerate(frames):
                ang = float(np.linalg.norm(th[i]))
                Rot = Matrix.Identity(4) if ang < 1e-8 else Matrix.Rotation(ang, 4, Vector(tuple(th[i] / ang)))
                swing = Matrix.Translation(pv) @ Rot @ Matrix.Translation(-pv)          # in the lead mesh's frame
                Bn = base @ to_lead.inverted() @ swing @ to_lead
                loc, q, _s = Bn.decompose()
                if prevq is not None and prevq.dot(q) < 0:
                    q.negate()
                prevq = q.copy()
                ob.location = loc
                ob.rotation_quaternion = q
                ob.keyframe_insert("location", frame=fr)
                ob.keyframe_insert("rotation_quaternion", frame=fr)
    return report


# the whole body follows through from the hips outward: bones, frequency (Hz), damping, strength
PRESETS = [
    (("spine", "chest"), 5.0, 0.65, 0.35),
    (("clavicle.L", "clavicle.R"), 6.0, 0.7, 0.3),
    (("neck",), 7.0, 0.75, 0.3),
    (("head",), 6.0, 0.7, 0.35),
    (("forearm.L", "forearm.R"), 6.0, 0.6, 0.5),
    (("hand.L", "hand.R"), 7.0, 0.55, 0.55),
]


def follow_through(scene, rig, data, amount=1.0, use_limits=True):
    """follow-through everywhere: spine, shoulders, head and arms lag and settle, legs in the air carry the push into a
    straight leg, fold (the heel flicks up) and reach for the ground, and marked gear swings. `data` is an analysis of
    the range (for the contacts)."""
    f0, f1 = data["f0"], data["f1"]
    done = []
    for bones, f, z, k in PRESETS:
        names = [b for b in bones if b in rig.pose.bones and not rig.pose.bones[b].constraints]
        if names:
            bake_overlap(scene, rig, names, f0, f1, freq=f, damp=z, resp=0.0, strength=min(1.0, k * amount), use_limits=use_limits)
            done += names
    info = data["info"]
    legs = [ch for ch in info["chains"] if "foot" in ch["target"].lower()]
    feet = {}
    if legs and info["body"]:
        body = bodymod.get_body(rig)
        probe_bones = np.array(body.names)[body.pbone]
        free = {ch["target"]: ~data["touch"][:, np.isin(probe_bones, ch["below"])].any(axis=1) for ch in legs}
        feet = bake_leg_follow(scene, rig, legs, f0, f1, free, strength=min(1.0, amount))
    gear = bake_gear(scene, rig, f0, f1, amount)
    return {"bones": done, "feet": feet, "gear": gear}
