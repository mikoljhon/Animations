# Motion Assist: grips. A hand that holds a prop (a pick, a hammer, a rod) has to stay on it however the body moves, but
# it is free to roll about the prop's axis (fingers wrap a shaft the same way at any roll) and to sit a little further up
# or down it. Here the hold is recorded (the hand's pose relative to the prop on every frame), the roll and the place on
# the shaft that keep the arm and wrist most natural are chosen (smooth in time), and the result is a list of absolute
# hand targets that Fix Joints reaches, so the arms can be brought inside their limits, or the body can move, without the
# hands ever letting go.

import json
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import autopose, limits


def prop_axis(ob):
    """the long axis of an object in its own space: (a point on it, unit direction, half length)"""
    bb = np.array([tuple(c) for c in ob.bound_box], dtype=float)
    lo, hi = bb.min(axis=0), bb.max(axis=0)
    i = int(np.argmax(hi - lo))
    d = np.zeros(3)
    d[i] = 1.0
    return Vector(tuple((lo + hi) * 0.5)), Vector(tuple(d)), float((hi - lo)[i] * 0.5)


def _closest(p0, u, p1, v):
    """parameters of the closest points of the lines p0 + s u and p1 + t v (unit directions)"""
    w = p0 - p1
    b, d, e = u.dot(v), u.dot(w), v.dot(w)
    den = 1.0 - b * b
    if den < 1e-9:
        return 0.0, e
    return (b * e - d) / den, (e - b * d) / den


def _m(rows):
    return Matrix([list(r) for r in rows])


def record(scene, rig, f0, f1, props, reach=0.10):
    """how each hand holds its prop. props: {"L": object, "R": object}. For every frame: the hand's pose relative to the
    prop (4x4, in the prop's space), the grip point on the prop's axis, and whether the hand is really on it (its palm axis
    passes within `reach` metres of the prop's axis, between the prop's ends: a fist wrapped round a shaft is about 8 cm
    from the hand bone's axis). Plain lists, so it can be stored as JSON."""
    out = {}
    cur = scene.frame_current
    try:
        for side, ob in props.items():
            pc, pd, half = prop_axis(ob)
            pb = rig.pose.bones["hand." + side]
            frames = {}
            for f in range(f0, f1 + 1):
                scene.frame_set(f)
                Pm = ob.matrix_world
                Hm = rig.matrix_world @ pb.matrix
                W, h = Hm.translation, Hm.to_3x3().col[1].normalized()
                p_w, d_w = Pm @ pc, (Pm.to_3x3() @ pd).normalized()
                half_w = (Pm.to_3x3() @ (pd * half)).length
                s, t = _closest(p_w, d_w, W, h)
                s = min(max(s, -half_w), half_w)
                t = min(max(t, 0.0), pb.length)
                dist = float(((p_w + d_w * s) - (W + h * t)).length)
                grip = Pm.inverted() @ (p_w + d_w * s)
                rel = Pm.inverted() @ Hm
                frames[f] = {"rel": [list(r) for r in rel], "g": list(grip), "on": bool(dist <= reach), "dist": round(dist, 4)}
            out[side] = {"object": ob.name, "axis": list(pd), "frames": frames}
    finally:
        scene.frame_set(cur)
    return out


def hand_at(prop, rig, rec_side, f, phi, slide=0.0):
    """(wrist position, hand rotation) in armature space for the recorded hold at frame f, rolled phi radians about the
    prop's axis and slid `slide` along it (prop units); the prop is read as it is on the current frame"""
    fr = rec_side["frames"][f]
    g, a = Vector(fr["g"]), Vector(rec_side["axis"])
    M = Matrix.Translation(a * slide) @ Matrix.Translation(g) @ Matrix.Rotation(phi, 4, a) @ Matrix.Translation(-g)
    H = rig.matrix_world.inverted() @ (prop.matrix_world @ M @ _m(fr["rel"]))
    return H.translation.copy(), H.to_3x3().normalized()


def _wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _side_costs(scene, rig, ob, r, side, frames, grid, slide, lam, comfort, leans, steps=12):
    """cost of every roll on every held frame (len(frames), len(grid)) for a hand slid `slide` along the prop; leans: the
    torso's forward lean on each frame, radians (the shoulder and elbow go with it). Each roll is tried with the arm solver
    itself (the elbow search with the wrist's limits), so what is judged is what Fix Joints can really do."""
    B, P = rig.data.bones, rig.pose.bones
    lims = limits.rig_limits(rig)
    fa_n, ha_n, up_n = "forearm." + side, "hand." + side, "upperarm." + side
    a_len, b_len = B[up_n].length, B[fa_n].length
    m = autopose.rig_map(rig)
    eyes = autopose.eye_points(rig, m)
    C = np.zeros((len(frames), len(grid)))
    for k, f in enumerate(frames):
        scene.frame_set(f)
        T = autopose.joints_now(rig, m, eyes)
        th = leans.get(f, 0.0) if leans else 0.0
        F = autopose.torso_fit(rig, T, th)
        F.follow("clavicle." + side)
        S = F.unrotated(up_n).translation
        E_cur = autopose.lean_matrix(T, th) @ P[fa_n].head
        for j, phi in enumerate(grid):
            W, H = hand_at(ob, rig, r, f, float(phi), slide)
            best = autopose._limb_search(F, lims, up_n, fa_n, S, W, E_cur - S, steps, ha_n, H)
            ex = best[4]
            bend = H.col[1].angle(W - E_cur) if (W - E_cur).length > 1e-6 else 0.0
            over = max(0.0, (W - S).length - comfort * (a_len + b_len))
            C[k, j] = (ex / 10.0) ** 2 + 0.5 * bend * bend + lam * phi * phi + 300.0 * over * over
    return C


def _viterbi(C, trans):
    V = np.zeros_like(C)
    back = np.zeros(C.shape, dtype=int)
    V[0] = C[0]
    for k in range(1, len(C)):
        tot = V[k - 1][:, None] + trans
        back[k] = tot.argmin(axis=0)
        V[k] = C[k] + tot.min(axis=0)
    j = int(V[-1].argmin())
    best = float(V[-1][j])
    idx = [j]
    for k in range(len(C) - 1, 0, -1):
        j = int(back[k][j])
        idx.append(j)
    return idx[::-1], best


def solve_grip(scene, rig, rec, props, step=10.0, lam=0.25, kappa=4.0, comfort=0.95, slides=(-0.12, -0.06, 0.0, 0.06, 0.12),
               leans=None, frame_step=3):
    """how each hand should hold its prop: {side: {"slide": metres along the prop, "roll": {frame: radians}}}. Every roll is
    tried with the real arm solution (the elbow nearest to where it is now). A roll costs: how far the arm and the wrist
    would be outside their limits (the wrist's swing, and the turn about the forearm that forearm and wrist cannot supply
    together), a little for the wrist's bend and for the roll itself, and a price for reaching with the arm nearly straight.
    A Viterbi search keeps the roll smooth in time; the slide is the one place along the shaft (a few centimetres either
    way) with the lowest total cost. The search looks at every `frame_step`th held frame and fills in between. leans: {frame: radians}, the torso lean that will be applied (see reach_lean)."""
    out = {}
    grid = np.radians(np.arange(-180.0, 180.0, step))
    trans = kappa * _wrap(grid[:, None] - grid[None, :]) ** 2
    cur = scene.frame_current
    try:
        for side, r in rec.items():
            ob = props[side]
            held = sorted(f for f, v in r["frames"].items() if v["on"])
            if not held:
                continue
            frames = held[::max(1, frame_step)]
            if frames[-1] != held[-1]:
                frames.append(held[-1])
            best = None
            trail = {}
            for sl in slides:
                C = _side_costs(scene, rig, ob, r, side, frames, grid, sl, lam, comfort, leans)
                idx, cost = _viterbi(C, trans)
                trail[round(sl, 3)] = round(cost / len(frames), 3)
                cost += 2.0 * len(frames) * (sl / 0.08) ** 2 * 0.01           # prefer where the hand was
                if best is None or cost < best[0]:
                    best = (cost, sl, idx)
            _c, sl, idx = best
            phi = np.unwrap(grid[np.array(idx)])
            phi = np.interp(held, frames, phi)                  # fill in the frames in between
            if len(phi) >= 5:                                   # and smooth the steps of the grid out
                pad = np.concatenate([[phi[0]] * 4, phi, [phi[-1]] * 4])
                phi = np.convolve(pad, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]) / 25.0, mode="valid")
            out[side] = {"slide": float(sl), "roll": {f: float(p) for f, p in zip(held, phi)}, "cost_by_slide": trail}
    finally:
        scene.frame_set(cur)
    return out


def refine(scene, rig, rec, props, grips, leans=None, span=30.0, n=7, frame_step=2, lam=0.25, kappa=4.0, comfort=0.95):
    """settle the rolls with the real arm solver. The coarse search (solve_grip) only estimates what Fix Joints can do, so
    around its roll on every `frame_step`th held frame a few candidate rolls (span degrees either way, n of them) are each
    solved for real and scored by how far the solved hand is from where the grip wants it (orientation and position), the
    wrist's bend, the roll itself, and a stretched arm; a Viterbi search keeps the chosen roll smooth in time. Returns grips
    in the same form as solve_grip."""
    out = {}
    B = rig.data.bones
    cur = scene.frame_current
    try:
        for side, g in grips.items():
            ob, r = props[side], rec[side]
            held = sorted(g["roll"])
            frames = held[::max(1, frame_step)]
            if frames[-1] != held[-1]:
                frames.append(held[-1])
            offs = np.radians(np.linspace(-span, span, n))
            reach = B["upperarm." + side].length + B["forearm." + side].length
            C = np.zeros((len(frames), n))
            PH = np.zeros((len(frames), n))
            for k, f in enumerate(frames):
                scene.frame_set(f)
                for j, off in enumerate(offs):
                    phi = g["roll"][f] + float(off)
                    W, H = hand_at(ob, rig, r, f, phi, g["slide"])
                    F = autopose.fix_pose(rig, parts=("arms",), hold={side: (W, H)}, lean=(leans or {}).get(f))
                    A, Af = F.A["hand." + side], F.A["forearm." + side]
                    rot_err = math.degrees(A.to_quaternion().rotation_difference(H.to_quaternion()).angle)
                    pos_err = (A.translation - W).length
                    bend = A.col[1].xyz.angle(Af.col[1].xyz)
                    over = max(0.0, (W - F.unrotated("upperarm." + side).translation).length - comfort * reach)
                    C[k, j] = (rot_err / 8.0) ** 2 + (pos_err / 0.01) ** 2 + 0.5 * bend * bend + lam * phi * phi + 300.0 * over * over
                    PH[k, j] = phi
            V = np.zeros_like(C)
            back = np.zeros(C.shape, dtype=int)
            V[0] = C[0]
            for k in range(1, len(frames)):
                tot = V[k - 1][:, None] + kappa * (PH[k][None, :] - PH[k - 1][:, None]) ** 2
                back[k] = tot.argmin(axis=0)
                V[k] = C[k] + tot.min(axis=0)
            j = int(V[-1].argmin())
            idx = [j]
            for k in range(len(frames) - 1, 0, -1):
                j = int(back[k][j])
                idx.append(j)
            idx = idx[::-1]
            phi = np.array([PH[k, i] for k, i in enumerate(idx)])
            phi = np.interp(held, frames, phi)
            if len(phi) >= 5:
                pad = np.concatenate([[phi[0]] * 2, phi, [phi[-1]] * 2])
                phi = np.convolve(pad, np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0, mode="valid")
            out[side] = {"slide": g["slide"], "roll": {f: float(p) for f, p in zip(held, phi)}}
    finally:
        scene.frame_set(cur)
    return out


def targets(scene, rig, rec, props, grips, ramp=9):
    """{frame: {side: (wrist, hand rotation)}} in armature space: where every recorded hold, rolled and slid as chosen, puts
    the hand with the prop where it is on each frame. After the last held frame the hand lets go: its own animation carries
    on, with the difference the grip made at the last held frame fading out over `ramp` frames (so it eases into the free
    motion). The held frames come from the recorded hold; the frames after the release come from the hand's own animation
    as it is now, so any follow-through on the free arms has to be baked before this is called."""
    out = {}
    cur = scene.frame_current
    try:
        for side, g in grips.items():
            ob = props[side]
            pb = rig.pose.bones["hand." + side]
            frames = sorted(g["roll"])
            for f in frames:
                scene.frame_set(f)
                out.setdefault(f, {})[side] = hand_at(ob, rig, rec[side], f, g["roll"][f], g["slide"])
            last = frames[-1]
            scene.frame_set(last)
            O_l = pb.matrix.copy()
            W_l, H_l = out[last][side]
            dR = H_l @ O_l.to_3x3().inverted()
            dt = W_l - O_l.translation
            for i in range(1, ramp + 1):
                f = last + i
                scene.frame_set(f)
                O = pb.matrix.copy()
                x = i / (ramp + 1.0)
                w = 1.0 - x * x * x * (x * (6.0 * x - 15.0) + 10.0)          # smootherstep: no jump in speed at either end
                q = O.to_quaternion().slerp((dR @ O.to_3x3()).to_quaternion(), w)
                out.setdefault(f, {})[side] = (O.translation + dt * w, q.to_matrix())
    finally:
        scene.frame_set(cur)
    return out


def plain(rec, frames=None):
    """grips that keep the hold exactly as recorded (no roll, no slide) on the held frames: the starting point for the
    torso's lean"""
    return {side: {"slide": 0.0, "roll": {f: 0.0 for f, v in r["frames"].items() if v["on"]}} for side, r in rec.items()}


def plan(scene, rig, f0, f1, props, rounds=2):
    """the whole grip job for Fix Joints. props: {"L": object, "R": object} (either or both). Records how the hands hold the
    props, finds how far the torso should lean for comfortable arms, searches the roll and place on the shaft that keep the
    wrists natural, settles them with the real arm solver, and returns (holds, leans) for autopose.fix_animation."""
    rec = record(scene, rig, f0, f1, props)
    holds = targets(scene, rig, rec, props, plain(rec))
    leans = autopose.reach_lean(scene, rig, f0, f1, holds)
    grips = solve_grip(scene, rig, rec, props, leans=leans)
    for _ in range(max(1, rounds)):
        grips = refine(scene, rig, rec, props, grips, leans=leans)
        grips = refine(scene, rig, rec, props, grips, leans=leans, span=12.0, n=5, frame_step=1)
        holds = targets(scene, rig, rec, props, grips)
        new = autopose.reach_lean(scene, rig, f0, f1, holds)
        change = max((abs(new.get(f, 0.0) - leans.get(f, 0.0)) for f in set(new) | set(leans)), default=0.0)
        leans = new
        if change < math.radians(1.5):
            break
    return holds, leans


def save(path, rec, grips, holds=None):
    d = {"record": rec, "grips": {s: {"slide": g["slide"], "roll": {str(f): v for f, v in g["roll"].items()}} for s, g in grips.items()}}
    if holds is not None:
        d["holds"] = {str(f): {s: [list(w), [list(r) for r in R]] for s, (w, R) in v.items()} for f, v in holds.items()}
    with open(path, "w") as fh:
        json.dump(d, fh)


def load(path):
    """(record, grips, holds or None)"""
    with open(path) as fh:
        d = json.load(fh)
    rec = {s: {"object": v["object"], "axis": v["axis"], "frames": {int(f): fr for f, fr in v["frames"].items()}} for s, v in d["record"].items()}
    grips = {s: {"slide": g["slide"], "roll": {int(f): v for f, v in g["roll"].items()}} for s, g in d["grips"].items()}
    holds = None
    if "holds" in d:
        holds = {int(f): {s: (Vector(w), Matrix([list(r) for r in R])) for s, (w, R) in v.items()} for f, v in d["holds"].items()}
    return rec, grips, holds
