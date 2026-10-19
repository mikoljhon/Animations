# Motion Assist: physics fixes that rewrite the animation.
# ballistic_fix: in the air the centre of mass follows the parabola set by the take-off and landing frames, and the
#   angular momentum stays constant (the whole body turns about its centre of mass to cancel any extra spin); the hips
#   get a short push before take-off and a give after landing so the speed matches the flight (legs never over-stretch).
# foot_lock: points that touch the ground stay where they first touched (no sliding), released smoothly on lift-off.

import math

import numpy as np
from mathutils import Matrix

from . import body as bodymod
from .analysis import _ancestors


def _T(v):
    M = np.eye(4)
    M[:3, 3] = v
    return M


def _R(rv):
    th = float(np.linalg.norm(rv))
    M = np.eye(4)
    if th < 1e-12:
        return M
    k = np.asarray(rv) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M[:3, :3] = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return M


def key_basis(rig, name, Bs, f0, location=True, rotation=True):
    """key a bone's local transform on every frame from f0 (Bs: 4x4 basis matrices)"""
    pb = rig.pose.bones[name]
    prev_e = prev_q = None
    for i, B in enumerate(Bs):
        f = f0 + i
        loc, q, _sca = Matrix(np.asarray(B).tolist()).decompose()
        if location:
            pb.location = loc
            pb.keyframe_insert("location", frame=f, group=name)
        if rotation:
            mode = pb.rotation_mode
            if mode == 'QUATERNION':
                if prev_q is not None and prev_q.dot(q) < 0:
                    q.negate()
                pb.rotation_quaternion = q
                prev_q = q.copy()
                pb.keyframe_insert("rotation_quaternion", frame=f, group=name)
            elif mode == 'AXIS_ANGLE':
                ax, ang = q.to_axis_angle()
                pb.rotation_axis_angle = (ang, ax.x, ax.y, ax.z)
                pb.keyframe_insert("rotation_axis_angle", frame=f, group=name)
            else:
                e = q.to_euler(mode, prev_e) if prev_e is not None else q.to_euler(mode)
                pb.rotation_euler = e
                prev_e = e.copy()
                pb.keyframe_insert("rotation_euler", frame=f, group=name)


def _rebased(data, name, f0, N, world_edit):
    """new basis matrices for a control bone after world_edit(i, W) changes its world matrix on frame f0 + i"""
    out = []
    for i in range(N):
        k = i + 2
        W = data["CW"][name][k]
        Q = W @ np.linalg.inv(data["CB"][name][k])
        out.append(np.linalg.inv(Q) @ world_edit(i, W))
    return out


def hips_gain(rig, body, info):
    """how much of the body moves when the hips move (IK chains keep their planted ends in place)"""
    bones = rig.data.bones
    frac = {}
    for ch in info["chains"]:
        Lc = max(ch["length"], 1e-6)
        a = 0.0
        for n in ch["chain"]:
            if n in body.names:
                t = (a + float(body.c[body.names.index(n)][1])) / Lc
                frac[n] = 1.0 - min(max(t, 0.0), 1.0)
            a += bones[n].length
        for n in ch["below"]:
            frac[n] = 0.0
    hips = info["body"]
    moved = sum(body.m[i] * frac.get(n, 1.0) for i, n in enumerate(body.names) if hips in _ancestors(bones[n]))
    return moved / body.M


def _reach_scale(a, d, Lmax):
    """largest l in [0, 1] with |a + l d| <= Lmax"""
    if np.linalg.norm(a + d) <= Lmax:
        return 1.0
    if np.linalg.norm(a) >= Lmax:
        return 0.0
    dd, ad, aa = d.dot(d), a.dot(d), a.dot(a)
    disc = ad * ad - dd * (aa - Lmax * Lmax)
    if dd < 1e-12 or disc < 0:
        return 0.0
    return float(min(max((-ad + math.sqrt(disc)) / dd, 0.0), 1.0))


def _planted(data, body, ch, i):
    names = np.array(body.names)[body.pbone]
    sel = np.isin(names, ch["below"])
    return bool(data["touch"][i][sel].any())


def ballistic_fix(scene, rig, s, data, parts=("shift", "spin", "push")):
    body = bodymod.get_body(rig)
    info = data["info"]
    if not data["flights"]:
        raise RuntimeError("no jumps found: no frames without ground contact between two contacts")
    f0, f1, dt = data["f0"], data["f1"], data["dt"]
    N = f1 - f0 + 1
    com = data["com_full"]
    g = np.array([0.0, 0.0, -s.gravity])
    shift = np.zeros((N, 3))
    spin = np.zeros((N, 3))
    push = np.zeros((N, 3))
    notes = []
    for fa, fb in data["flights"]:
        ka, kb = fa - f0 + 2, fb - f0 + 2
        T = (fb - fa) * dt
        ca, cb = com[ka], com[kb]
        v0 = (cb - ca) / T - 0.5 * g * T
        if "shift" in parts:
            for f in range(fa + 1, fb):
                tau = (f - fa) * dt
                shift[f - f0] = ca + v0 * tau + 0.5 * g * tau * tau - com[f - f0 + 2]
        turned = 0.0
        if "spin" in parts and s.fix_spin and fb - fa >= 3:
            # the whole body turns about its centre of mass so the angular momentum stays at Lref; Lref is the
            # inertia-weighted mean, which makes the extra turn add up to zero by the landing frame
            ks = list(range(ka + 1, kb))
            Ls = data["L_full"][ks]
            Iinv = np.linalg.inv(data["I_full"][ks])
            Lref = np.linalg.solve(Iinv.sum(axis=0), np.einsum("kij,kj->i", Iinv, Ls))
            th = np.zeros(3)
            wprev = np.zeros(3)
            for j, k in enumerate(ks):
                w = -(Iinv[j] @ (Ls[j] - Lref))
                th = th + 0.5 * (w + wprev) * dt
                wprev = w
                spin[k - 2] = th
                turned = max(turned, float(np.linalg.norm(th)))
        K = s.comp_frames if "push" in parts else 0
        if K > 0:
            vL = (com[ka] - com[ka - 1]) / dt
            for j in range(K + 1):
                f = fa - K + j
                if f0 <= f <= f1:
                    u = j / K
                    push[f - f0] += (u ** 3 - u ** 2) * (v0 - vL) * K * dt
            vR = (com[kb + 1] - com[kb]) / dt
            ve = v0 + g * T
            for j in range(K + 1):
                f = fb + j
                if f0 <= f <= f1:
                    u = j / K
                    push[f - f0] += (u ** 3 - 2 * u ** 2 + u) * (ve - vR) * K * dt
        notes.append("f%d-%d: moved up to %.1f cm, turned up to %.1f deg" % (
            fa, fb, 100 * float(np.linalg.norm(shift[fa - f0 + 1:fb - f0], axis=1).max(initial=0.0)), math.degrees(turned)))

    # the hips carry the push and the give; legs must still reach their planted targets
    hips = info["body"]
    gain = max(hips_gain(rig, body, info), 0.2)
    hip = push / gain
    scale = max(rig.matrix_world.to_scale())
    free = {}
    for ch in info["chains"]:
        root, tgt = ch["chain"][0], ch["target"]
        if hips not in _ancestors(rig.data.bones[root]):
            continue
        Lmax = 0.998 * ch["length"] * scale
        free[tgt] = np.zeros(N, dtype=bool)
        for i in range(N):
            if not hip[i].any():
                continue
            if _planted(data, body, ch, i):
                k = i + 2
                a = data["CW"][root][k][:3, 3] - data["CW"][tgt][k][:3, 3]
                hip[i] *= _reach_scale(a, hip[i], Lmax)
            else:
                free[tgt][i] = True

    movers = [info["master"]] if info["master"] else info["tops"]
    turn = float(np.abs(spin).max()) > 1e-7
    if float(np.abs(shift).max()) > 1e-7 or turn:
        def edit(i, W):
            c = com[i + 2] + shift[i]
            return _T(c) @ _R(spin[i]) @ _T(-c) @ _T(shift[i]) @ W
        for name in movers:
            key_basis(rig, name, _rebased(data, name, f0, N, edit), f0, location=True, rotation=turn)
    if hips and float(np.abs(hip).max()) > 1e-7:
        key_basis(rig, hips, _rebased(data, hips, f0, N, lambda i, W: _T(hip[i]) @ W), f0, location=True, rotation=False)
        for tgt, mask in free.items():
            if mask.any():
                key_basis(rig, tgt, _rebased(data, tgt, f0, N, lambda i, W, m=mask: _T(hip[i]) @ W if m[i] else W), f0,
                          location=True, rotation=False)
    notes.append("push/give on the hips up to %.1f cm (hips move %.0f%% of the mass)" % (
        100 * float(np.linalg.norm(hip, axis=1).max(initial=0.0)), 100 * gain))
    return notes


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def lean_and_land(scene, rig, s, data, max_lean=16.0, max_reach=0.45, tilt=0.4):
    """Take-off: the body leans forward over the toes so the push that launches the jump points through the centre of
    mass (lean angle = forward push / upward push, from the flight's starting speed and how long the push lasts); it
    straightens up again in the air. Landing: the feet reach ahead of the body so the force that stops the forward speed
    points through the centre of mass too, and after touching down the body carries forward over the planted feet.
    The lean comes mostly from the legs: the hips swing forward about the toes, the torso tips only `tilt` of the angle."""
    info = data["info"]
    hips = info["body"]
    legs = [ch for ch in info["chains"] if "foot" in ch["target"].lower()] or info["chains"]
    if not data["flights"] or hips is None:
        return []
    f0, f1, dt = data["f0"], data["f1"], data["dt"]
    N = f1 - f0 + 1
    com = data["com_full"]
    g = s.gravity
    gvec = np.array([0.0, 0.0, -g])
    up = np.array([0.0, 0.0, 1.0])
    leg = max([ch["length"] for ch in legs] or [1.0]) * max(rig.matrix_world.to_scale())
    lean = np.zeros((N, 3))
    pivot = np.zeros((N, 3))
    hip_shift = np.zeros((N, 3))
    foot_shift = np.zeros((N, 3))
    notes = []
    for fa, fb in data["flights"]:
        ka, kb = fa - f0 + 2, fb - f0 + 2
        T = (fb - fa) * dt
        v0 = (com[kb] - com[ka]) / T - 0.5 * gvec * T
        ve = v0 + gvec * T
        vh = v0 * np.array([1.0, 1.0, 0.0])
        if np.linalg.norm(vh) < 0.05:
            continue
        u = vh / np.linalg.norm(vh)
        axis = np.cross(up, u)
        axis /= np.linalg.norm(axis)
        # take-off: the push runs from the lowest point of the crouch to the last frame on the ground
        fp = fa
        while fp - 1 >= f0 and com[fp - 1 - f0 + 2][2] < com[fp - f0 + 2][2] - 1e-4:
            fp -= 1
        Tp = max((fa - fp) * dt, 2 * dt)
        want = min(math.atan2(np.linalg.norm(vh) / Tp, max(v0[2] / Tp + g, 0.2 * g)), math.radians(max_lean))
        t = data["touch"][fa - f0]
        extra = 0.0
        if t.any():
            p = data["Q"][fa - f0][t].mean(axis=0)
            c = com[ka]
            extra = want - math.atan2(float((c - p) @ u), max(c[2] - p[2], 0.1))
            if extra > math.radians(1.0):
                for f in range(fp, fb):
                    if f <= fa:
                        w = _smooth((f - fp) / max(fa - fp, 1))
                    else:
                        w = 1.0 - _smooth((f - fa) / max(fb - fa, 1.0))
                    lean[f - f0] = axis * (extra * w)
                    pivot[f - f0] = p
        # landing: the stop runs from touch-down to the lowest point of the absorb
        fe = fb
        while fe + 1 <= f1 and com[fe + 1 - f0 + 2][2] < com[fe - f0 + 2][2] - 1e-4:
            fe += 1
        Ta = max((fe - fb) * dt, 3 * dt)
        stop_h = np.linalg.norm(ve * np.array([1.0, 1.0, 0.0])) / Ta
        stop_v = max(-ve[2] / Ta + g, 0.2 * g)
        t = data["touch"][fb - f0]
        delta = 0.0
        if t.any():
            p = data["Q"][fb - f0][t].mean(axis=0)
            cb = com[kb]
            ahead = min(max(cb[2] - p[2], 0.1) * stop_h / stop_v, max_reach * leg)
            delta = ahead - float((p - cb) @ u)
            if delta > 0.01:
                fm = (fa + fb) // 2
                for f in range(fm, f1 + 1):
                    foot_shift[f - f0] += u * delta * (_smooth((f - fm) / max(fb - fm, 1)) if f < fb else 1.0)
                fend = min(f1, fe + (fe - fb) + 2)
                for f in range(fb, f1 + 1):
                    hip_shift[f - f0] += u * delta * _smooth((f - fb) / max(fend - fb, 1))
        notes.append("f%d-%d: leaned %.0f deg into the push, feet land %.0f cm further ahead" % (
            fa, fb, math.degrees(max(extra, 0.0)), 100 * max(delta, 0.0)))

    body_move = [np.eye(4) for _ in range(N)]
    if np.abs(lean).max() > 1e-6 or np.abs(hip_shift).max() > 1e-6:
        def edit(i, W):
            M = _T(pivot[i]) @ _R(lean[i]) @ _T(-pivot[i]) @ W
            h = M[:3, 3].copy()
            out = _T(hip_shift[i]) @ _T(h) @ _R(-(1.0 - tilt) * lean[i]) @ _T(-h) @ M
            body_move[i] = out @ np.linalg.inv(W)
            return out
        key_basis(rig, hips, _rebased(data, hips, f0, N, edit), f0, location=True, rotation=True)
    # in the air the feet move with the body (the legs keep their shape); on landing they also reach ahead
    air = np.zeros(N, dtype=bool)
    for fa, fb in data["flights"]:
        air[max(fa + 1 - f0, 0):max(fb - f0, 0)] = True
    if np.abs(foot_shift).max() > 1e-6 or any(air[i] and not np.allclose(body_move[i], np.eye(4)) for i in range(N)):
        for ch in legs:
            key_basis(rig, ch["target"], _rebased(data, ch["target"], f0, N,
                                                  lambda i, W: _T(foot_shift[i]) @ (body_move[i] @ W if air[i] else W)),
                      f0, location=True, rotation=True)
    return notes


def fix_jumps(scene, rig, s, spin_passes=2, lean=True):
    """lean into the push and reach the feet ahead on landing (lean), the ballistic fix, then a couple of spin-only
    passes on the residual; returns notes and the new analysis"""
    from . import analysis
    d = analysis.analyze(scene, rig, s)
    before = d["com"].copy()
    notes = []
    if lean:
        for k in range(2):          # the second pass tops the lean up to what the push needs
            n = lean_and_land(scene, rig, s, d)
            notes += n if k == 0 else []
            d = analysis.analyze(scene, rig, s)
    notes += ballistic_fix(scene, rig, s, d)
    for _ in range(spin_passes):
        d = analysis.analyze(scene, rig, s)
        if not d["flights"]:
            break
        ballistic_fix(scene, rig, s, d, parts=("spin",))
    d = analysis.analyze(scene, rig, s)
    d["keep_before"] = True
    d["com_before"] = before
    return notes, d


def foot_lock(scene, rig, s, data):
    body = bodymod.get_body(rig)
    info = data["info"]
    f0, f1 = data["f0"], data["f1"]
    N = f1 - f0 + 1
    names = np.array(body.names)[body.pbone]
    notes = []
    B = s.lock_blend
    for ch in info["chains"]:
        idx = np.nonzero(np.isin(names, ch["below"]))[0]
        if len(idx) == 0:
            continue
        T = data["touch"][:, idx]
        Pc = data["P"][:, idx]
        corr = np.zeros((N, 3))
        in_run = T.any(axis=1)
        anchor = {}
        for i in range(N):
            if not in_run[i]:
                anchor = {}
                continue
            t = np.nonzero(T[i])[0]
            known = [j for j in t if j in anchor]
            c = np.mean([anchor[j] - Pc[i, j] for j in known], axis=0) if known else np.zeros(3)
            corr[i] = c
            for j in t:
                if j not in anchor:
                    anchor[j] = Pc[i, j] + c
        if B > 0:
            for i in range(1, N):
                if in_run[i - 1] and not in_run[i]:
                    c_end = corr[i - 1].copy()
                    for j in range(B):
                        if i + j >= N or in_run[i + j]:
                            break
                        u = (j + 1) / (B + 1)
                        corr[i + j] = c_end * (1 - u * u * (3 - 2 * u))
        worst = float(np.linalg.norm(corr, axis=1).max(initial=0.0))
        if worst < 1e-5:
            notes.append("%s: already planted" % ch["target"])
            continue
        tgt = ch["target"]
        key_basis(rig, tgt, _rebased(data, tgt, f0, N, lambda i, W, c=corr: _T(c[i]) @ W), f0, location=True, rotation=False)
        notes.append("%s: removed up to %.1f cm of sliding" % (tgt, 100 * worst))
    return notes
