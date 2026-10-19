# Motion Assist: AutoPose.
# A small network trained on the CMU motion capture library (C:\ai\mocap\train_autopose.py) predicts a natural position
# for every joint from a few given ones: by default the hips, neck, wrists and ankles. The rig is then fitted to that
# prediction with its own bone lengths: the torso follows the predicted spine curve, two-bone IK puts each hand exactly
# on its goal with the elbow where the network put it, and IK legs get their foot and knee targets.

import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import body as bodymod
from . import limits

MODELS = {"hips": os.path.join(os.path.dirname(__file__), "models", "autopose.npz"),
          "feet": os.path.join(os.path.dirname(__file__), "models", "autopose_feet.npz")}
CORE = ("hips", "neck", "wrist_L", "wrist_R", "ankle_L", "ankle_R")
ELBOW_TURN = 20.0      # degrees an elbow's swivel may turn from one frame to the next in Fix Joints (keeps the solution from flipping)
GOAL_PREFIX = "MA_Goal_"
_nets = {}
_last = {"clamped": {}}


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _layer_norm(x, w, b):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * w + b


class Net:
    """the trained network, run with numpy"""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        g = lambda k: d[k.replace(".", "__")].astype(np.float32)
        self.joints = [str(j) for j in d["joints"]]
        self.ix = {n: i for i, n in enumerate(self.joints)}
        self.inp = (g("inp.weight"), g("inp.bias"))
        self.blocks = [tuple(g("blocks.%d.%s" % (i, k)) for k in ("ln.weight", "ln.bias", "a.weight", "a.bias", "b.weight", "b.bias"))
                       for i in range(int(d["blocks"]))]
        self.ln = (g("ln.weight"), g("ln.bias"))
        self.out = (g("out.weight"), g("out.bias"))
        self.report = json.loads(str(d["report"]))

    def __call__(self, P, M):
        """P (n, J, 3) joints in the hips frame (leg lengths), M (n, J) given flags -> every joint (n, J, 3)"""
        x = np.concatenate([(P * M[..., None]).reshape(len(P), -1), M], axis=1).astype(np.float32)
        h = _silu(x @ self.inp[0].T + self.inp[1])
        for lw, lb, aw, ab, bw, bb in self.blocks:
            h = h + _silu(_layer_norm(h, lw, lb) @ aw.T + ab) @ bw.T + bb
        y = _silu(_layer_norm(h, *self.ln)) @ self.out[0].T + self.out[1]
        return y.reshape(len(P), -1, 3)


def get_net(anchor="hips"):
    """the hips-anchored network (the hips are given) or the feet-anchored one (it also places the hips)"""
    if anchor not in _nets:
        if not os.path.exists(MODELS[anchor]):
            raise RuntimeError("AutoPose model not found: " + MODELS[anchor])
        _nets[anchor] = Net(MODELS[anchor])
    return _nets[anchor]


def has_model(anchor):
    return os.path.exists(MODELS[anchor])


# ---------------------------------------------------------------------------------------------------------- the rig

def rig_map(rig):
    """canonical joint -> (bone, 'head' or 'tail'); works for rigs named like Mender's and Relic's"""
    B = rig.data.bones

    def pick(*names):
        return next((n for n in names if n in B), None)

    need = ["pelvis", "spine", "chest", "neck", "head"]
    for s in ("L", "R"):
        need += [p + "." + s for p in ("clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe")]
    missing = [n for n in need if n not in B]
    if missing:
        raise RuntimeError("AutoPose needs these bones: " + ", ".join(missing))
    m = {"hips": ("pelvis", "head"), "spine": ("spine", "head"), "chest": ("chest", "head"), "neck": ("neck", "head"),
         "head": ("head", "head")}
    for s in ("L", "R"):
        m["clav_" + s] = ("clavicle." + s, "head")
        m["shoulder_" + s] = ("upperarm." + s, "head")
        m["elbow_" + s] = ("forearm." + s, "head")
        m["wrist_" + s] = ("hand." + s, "head")
        k = pick("middle1." + s, "middle_1." + s, "mm1." + s)
        m["knuckle_" + s] = (k, "head") if k else ("hand." + s, "tail")
        t = pick("thumb2." + s, "thumb_2." + s, "mt2." + s)
        if t:
            m["thumb_" + s] = (t, "head")
        m["hip_" + s] = ("thigh." + s, "head")
        m["knee_" + s] = ("shin." + s, "head")
        m["ankle_" + s] = ("foot." + s, "head")
        m["toe_" + s] = ("toe." + s, "tail")
    return m


def _rest_point(rig, m, j):
    b, end = m[j]
    bone = rig.data.bones[b]
    return (bone.head_local if end == "head" else bone.tail_local).copy()


def eye_points(rig, m):
    """eyes as points in the head bone's own frame, placed from the rest pose (rigs rarely have eye bones)"""
    hb = rig.data.bones["head"]
    h0, h1 = hb.head_local, hb.tail_local
    L = (h1 - h0).length
    fwd = _rest_point(rig, m, "toe_L") - _rest_point(rig, m, "ankle_L")
    fwd.z = 0.0
    fwd.normalize()
    left = (_rest_point(rig, m, "hip_L") - _rest_point(rig, m, "hip_R")).normalized()
    mid = h0 + (h1 - h0) * 0.42 + fwd * (0.45 * L)
    inv = hb.matrix_local.inverted()
    return {"eye_L": inv @ (mid + left * 0.15 * L), "eye_R": inv @ (mid - left * 0.15 * L)}


def joints_now(rig, m, eyes):
    """armature-space positions of every mapped joint in the current pose"""
    P = rig.pose.bones
    out = {j: (P[b].head if end == "head" else P[b].tail).copy() for j, (b, end) in m.items()}
    for j, loc in eyes.items():
        out[j] = P["head"].matrix @ loc
    return out


def joints_rest(rig, m, eyes):
    out = {j: _rest_point(rig, m, j) for j in m}
    hb = rig.data.bones["head"]
    for j, loc in eyes.items():
        out[j] = hb.matrix_local @ loc
    return out


def leg_length(rig):
    B = rig.data.bones
    return 0.5 * (B["thigh.L"].length + B["shin.L"].length + B["thigh.R"].length + B["shin.R"].length)


# ------------------------------------------------------------------------------------------------------- predicting

def heading(J):
    """hips frame columns: left, back, up (armature space, Z up); the character faces -back"""
    left = J["hip_L"] - J["hip_R"]
    left.z = 0.0
    left = left.normalized() if left.length > 1e-6 else Vector((1, 0, 0))
    up = Vector((0, 0, 1))
    back = up.cross(left)
    return Matrix((left, back, up)).transposed()


def predict(rig, given, use_net=True):
    """given: {joint: armature-space position}. Returns armature-space positions for every joint."""
    m = rig_map(rig)
    eyes = eye_points(rig, m)
    cur = joints_now(rig, m, eyes)
    hips = given.get("hips", cur["hips"])
    R = heading(cur)
    L = leg_length(rig)
    if not use_net:
        out = {j: hips + (p - cur["hips"]) for j, p in cur.items()}
        out.update(given)
        return out, m, eyes, R
    feet = "hips" not in given and "ankle_L" in given and "ankle_R" in given and has_model("feet")
    origin = (given["ankle_L"] + given["ankle_R"]) * 0.5 if feet else hips
    net = get_net("feet" if feet else "hips")
    P = np.zeros((1, len(net.joints), 3), dtype=np.float32)
    M = np.zeros((1, len(net.joints)), dtype=np.float32)
    Rt = R.transposed()
    for j, p in given.items():
        if j in net.ix:
            P[0, net.ix[j]] = tuple(Rt @ (p - origin) / L)
            M[0, net.ix[j]] = 1.0
    if not feet:
        M[0, net.ix["hips"]] = 1.0
    Y = net(P, M)[0]
    out = {j: origin + R @ (Vector(tuple(float(v) for v in Y[i])) * L) for j, i in net.ix.items()}
    out.update(given)
    return out, m, eyes, R


# ---------------------------------------------------------------------------------------------------------- fitting

def _frame(x, y):
    """orthonormal frame with its first axis along x and its second as close to y as possible (columns)"""
    x = x.normalized()
    y = (y - x * y.dot(x)).normalized()
    return Matrix((x, y, x.cross(y))).transposed()


def _polyline_at(pts, frac):
    seg = [(b - a).length for a, b in zip(pts, pts[1:])]
    t = min(max(frac, 0.0), 1.0) * sum(seg)
    for i, s in enumerate(seg):
        if t <= s or i == len(seg) - 1:
            return pts[i].lerp(pts[i + 1], min(t / s, 1.0) if s > 1e-9 else 0.0)
        t -= s
    return pts[-1].copy()


class Fitter:
    """poses bones in armature space top-down without touching the depsgraph (so it also works on animated rigs)"""

    def __init__(self, rig):
        self.rig, self.B, self.P = rig, rig.data.bones, rig.pose.bones
        self.A = {}
        self.clamped = {}

    def rel(self, b):
        bone = self.B[b]
        return bone.parent.matrix_local.inverted() @ bone.matrix_local if bone.parent else bone.matrix_local.copy()

    def parent_A(self, b):
        p = self.B[b].parent
        if p is None:
            return Matrix.Identity(4)
        return self.A[p.name] if p.name in self.A else self.P[p.name].matrix.copy()

    def unrotated(self, b, loc=None):
        loc = self.P[b].location if loc is None else loc
        return self.parent_A(b) @ self.rel(b) @ Matrix.Translation(loc)

    def follow(self, b):
        """keep the bone's own rotation and let it ride its (moved) parent"""
        self.A[b] = self.parent_A(b) @ self.rel(b) @ self.P[b].matrix_basis
        return self.A[b]

    def aim(self, b, direction, loc=None):
        A0 = self.unrotated(b, loc)
        y = A0.col[1].xyz.normalized()
        if direction.length < 1e-9:
            self.A[b] = A0
            return A0
        q = y.rotation_difference(direction.normalized())
        h = A0.translation.copy()
        self.A[b] = Matrix.Translation(h) @ q.to_matrix().to_4x4() @ Matrix.Translation(-h) @ A0
        return self.A[b]

    def orient(self, b, rot3, head=None):
        A0 = self.unrotated(b)
        h = A0.translation.copy() if head is None else head
        M = rot3.to_4x4()
        M.translation = h
        self.A[b] = M
        return M

    def tail(self, b):
        return self.A[b] @ Vector((0.0, self.B[b].length, 0.0))

    def local_q(self, b):
        """the bone's rotation as decided so far, in its own rest frame relative to its parent"""
        return (self.unrotated(b).inverted() @ self.A[b]).to_quaternion()

    def set_local(self, b, q):
        self.A[b] = self.unrotated(b) @ q.to_matrix().to_4x4()
        return self.A[b]

    def limit(self, b, lims):
        """pull the bone back inside its joint limits (how far it had to move is kept in self.clamped, degrees)"""
        lim = lims.get(b)
        if lim is not None and b in self.A:
            q = self.local_q(b)
            c = limits.clamp_rotation(q, lim)
            moved = math.degrees(q.rotation_difference(c).angle)
            if moved > 0.05:
                self.clamped[b] = max(self.clamped.get(b, 0.0), moved)
            self.set_local(b, c)

    def basis(self, b):
        return (self.parent_A(b) @ self.rel(b)).inverted() @ self.A[b]

    def values(self):
        """the local channels this fit decided: {bone: (location or None, rotation or None)}"""
        out = {}
        for b in self.A:
            loc, q, _s = self.basis(b).decompose()
            moves = b == "pelvis" or b.startswith(("ik_foot", "ik_knee"))
            out[b] = (loc if moves else None, None if b.startswith("ik_knee") else q)
        return out

    def write(self, key=False, frame=None):
        write_values(self.rig, self.values(), key, frame)


def write_values(rig, values, key=False, frame=None):
    for b, (loc, q) in values.items():
        pb = rig.pose.bones[b]
        if loc is not None:
            pb.location = loc
            if key:
                pb.keyframe_insert("location", frame=frame, group=b)
        if q is None:
            continue
        q = q.copy()
        if pb.rotation_mode == 'QUATERNION':
            if pb.rotation_quaternion.dot(q) < 0:
                q.negate()
            pb.rotation_quaternion = q
            path = "rotation_quaternion"
        elif pb.rotation_mode == 'AXIS_ANGLE':
            ax, ang = q.to_axis_angle()
            pb.rotation_axis_angle = (ang, ax.x, ax.y, ax.z)
            path = "rotation_axis_angle"
        else:
            pb.rotation_euler = q.to_euler(pb.rotation_mode, pb.rotation_euler)
            path = "rotation_euler"
        if key:
            pb.keyframe_insert(path, frame=frame, group=b)


def _two_bone(S, W, a, b, hint):
    """elbow (or knee) position for a chain of lengths a, b from S reaching for W, bending toward hint"""
    d_vec = W - S
    d = d_vec.length
    if d < 1e-6:
        return S + (hint - S).normalized() * a
    u = d_vec / d
    d = min(max(d, abs(a - b) + 1e-4), a + b - 1e-4)
    v = (hint - S) - u * (hint - S).dot(u)
    if v.length < 1e-6:
        v = u.orthogonal()
    v.normalize()
    x = (a * a - b * b + d * d) / (2 * d)
    h = math.sqrt(max(a * a - x * x, 0.0))
    return S + u * x + v * h


def _ik_target(rig, chain_end):
    """the IK target and pole bones driving a leg, from the rig's IK constraint"""
    pb = rig.pose.bones.get(chain_end)
    if pb is None:
        return None, None
    for c in pb.constraints:
        if c.type == 'IK' and c.target == rig and c.subtarget:
            return c.subtarget, (c.pole_subtarget if c.pole_target == rig else None)
    return None, None


def _limb_end(T, given, j_root, j_mid, j_end, start, length):
    """a goal is kept exactly; a predicted end is rescaled to this rig's limb length, so the bend matches the prediction"""
    if j_end in given:
        return T[j_end]
    pred = (T[j_mid] - T[j_root]).length + (T[j_end] - T[j_mid]).length
    return start + (T[j_end] - T[j_root]) * (length / max(pred, 1e-6))


def _swivel_points(S, W, a, b, hint_dir, steps):
    """the circle of possible middle-joint positions for a two-bone chain from S to W, starting nearest the hint;
    yields (position, how far it is from the hint in degrees)"""
    d_vec = W - S
    d = max(d_vec.length, 1e-6)
    u = d_vec / d
    dd = min(max(d, abs(a - b) + 1e-4), a + b - 1e-4)
    x = (a * a - b * b + dd * dd) / (2 * dd)
    h = math.sqrt(max(a * a - x * x, 0.0))
    v0 = hint_dir - u * hint_dir.dot(u)
    v0 = v0.normalized() if v0.length > 1e-6 else u.orthogonal().normalized()
    if h < 1e-5:
        yield S + u * a, 0.0
        return
    for i in range(steps):
        ang = 2.0 * math.pi * i / steps
        dev = math.degrees(min(ang, 2.0 * math.pi - ang))
        yield S + u * x + (Quaternion(u, ang) @ v0) * h, dev


def _arm_pose(F, lims, upper, lower, S, W, E):
    """upper bone aimed at E and rolled so the lower bone's hinge lies across the bend; the lower bone's bend toward W.
    Returns (upper rotation, lower bend in radians, lower's off-plane angle in radians, how far outside the limits in
    degrees)"""
    U3 = F.unrotated(upper).to_3x3()
    e_u = (E - S).normalized()
    R1 = U3.col[1].xyz.normalized().rotation_difference(e_u).to_matrix() @ U3
    lim_u, lim_l = lims.get(upper), lims[lower]
    rel_l = F.rel(lower).to_3x3()
    f_dir = (W - E).normalized()
    for _ in range(2):
        R_f0 = R1 @ rel_l
        h = (R_f0 @ lim_l.hinge).normalized()
        want = (R_f0 @ limits.Y).cross(f_dir)
        hp, wp = h - e_u * h.dot(e_u), want - e_u * want.dot(e_u)
        if hp.length < 1e-6 or wp.length < 1e-6:
            break
        R1 = Quaternion(e_u, limits._signed_angle(hp.normalized(), wp.normalized(), e_u)).to_matrix() @ R1
    ex = limits.excess((U3.inverted() @ R1).to_quaternion(), lim_u) if lim_u is not None else 0.0
    d = (R1 @ rel_l).inverted() @ (W - E)
    phi = limits.hinge_angle(d.normalized(), lim_l.hinge) if d.length > 1e-9 else 0.0
    psi = math.asin(max(-1.0, min(1.0, d.normalized().dot(lim_l.hinge)))) if d.length > 1e-9 else 0.0
    lo, hi = math.radians(lim_l.ranges[0]), math.radians(lim_l.ranges[1])
    sl = math.radians(lim_l.slack)
    ex += math.degrees(max(lo - phi, phi - hi, 0.0) + max(abs(psi) - sl, 0.0))
    return R1, phi, max(-sl, min(sl, psi)), ex


def _wrist_excess(F, lims, lower, hand, R1, phi, psi, hand_rot):
    """how far (degrees) a hand held at `hand_rot` would be outside what the wrist and the forearm can do together, when the
    arm is posed with its upper bone at R1 and its elbow bent phi (off-plane psi): the wrist's swing counts as it is, the
    roll about the forearm is shared between forearm and wrist (their twist ranges added)"""
    lim_h, lim_f = lims.get(hand), lims.get(lower)
    if lim_h is None or lim_f is None or hand_rot is None:
        return 0.0
    B = F.B
    hinge = lim_f.hinge
    rot = limits.Y.rotation_difference((Quaternion(hinge, phi) @ limits.Y) * math.cos(psi) + hinge * math.sin(psi))
    Rf = R1 @ F.rel(lower).to_3x3() @ rot.to_matrix()
    rel_rest = B[lower].matrix_local.to_3x3().inverted() @ B[hand].matrix_local.to_3x3()
    q = (rel_rest.inverted() @ (Rf.inverted() @ hand_rot)).to_quaternion()
    qr = lim_h.ref.inverted() @ q if lim_h.ref is not None else q
    cap = abs(lim_f.ranges[-1]) + abs(lim_h.ranges[-1])
    twist = abs(math.degrees(limits.twist_angle(limits.split(qr)[1])))
    return limits.excess(limits.swing_only(q, lim_h), lim_h) + max(0.0, twist - cap)


def _limb_search(F, lims, upper, lower, S, target, hint_dir, steps=36, hand=None, hand_rot=None, dev_weight=0.05,
                 prev_dir=None, max_turn=None):
    """the best elbow (or knee) swivel for a two-bone chain from S to target: (cost, upper rotation, bend, off-plane angle,
    degrees outside the limits). With hand and hand_rot the wrist counts too: the elbow goes where the hand can be held as
    it should. prev_dir: the elbow's direction from the shoulder on the previous frame; the swivel may turn at most
    max_turn degrees from it (so the solution cannot flip from one frame to the next, and still follows the hint)."""
    B = F.B
    a, b = B[upper].length, B[lower].length
    best = None
    max_turn = ELBOW_TURN if max_turn is None else max_turn
    axis = (target - S).normalized() if (target - S).length > 1e-9 else None
    prev = None
    if prev_dir is not None and axis is not None:
        prev = prev_dir - axis * prev_dir.dot(axis)
        prev = prev.normalized() if prev.length > 1e-6 else None
    for E, dev in _swivel_points(S, target, a, b, hint_dir, steps):
        if prev is not None:
            c = (E - S) - axis * (E - S).dot(axis)
            if c.length > 1e-6 and math.degrees(c.angle(prev)) > max_turn:
                continue
        R1, phi, psi, ex = _arm_pose(F, lims, upper, lower, S, target, E)
        if hand_rot is not None:
            ex += _wrist_excess(F, lims, lower, hand, R1, phi, psi, hand_rot)
        cost = 10.0 * ex + dev_weight * dev
        if best is None or cost < best[0]:
            best = (cost, R1, phi, psi, ex)
    if best is None and prev is not None:          # nothing within reach of the previous swivel: search freely
        return _limb_search(F, lims, upper, lower, S, target, hint_dir, steps, hand, hand_rot, dev_weight)
    return best


def _limb(F, lims, upper, lower, target, hint_dir, steps=36, hand=None, hand_rot=None, dev_weight=0.05, prev_dir=None):
    """two-bone IK that respects the joints. The middle joint (elbow, knee) is tried all the way around the line from the
    root to the target (its swivel); the position that keeps every joint inside its limits (the wrist too, when the hand's
    rotation is given) and stays closest to the hint wins. The upper bone rolls so the hinge lies across the bend, the lower
    bone bends about its hinge only."""
    B = F.B
    S = F.unrotated(upper).translation
    a, b = B[upper].length, B[lower].length
    if lims.get(lower) is None:
        E = _two_bone(S, target, a, b, S + hint_dir)
        F.aim(upper, E - S)
        F.limit(upper, lims)
        F.aim(lower, target - F.unrotated(lower).translation)
        F.limit(lower, lims)
        return
    _cost, R1, phi, psi, _ex = _limb_search(F, lims, upper, lower, S, target, hint_dir, steps, hand, hand_rot, dev_weight, prev_dir)
    M = R1.to_4x4()
    M.translation = S
    F.A[upper] = M
    F.limit(upper, lims)
    hinge = lims[lower].hinge
    F.set_local(lower, limits.Y.rotation_difference((Quaternion(hinge, phi) @ limits.Y) * math.cos(psi) + hinge * math.sin(psi)))
    F.limit(lower, lims)


def _hand(F, lims, hand, lower, target_rot):
    """orient the hand (or the foot of an FK leg); a roll beyond the wrist's range goes into the bone above (the forearm
    is where the hand really turns)"""
    U = F.unrotated(hand)
    q = (U.to_3x3().inverted() @ target_rot).to_quaternion()
    lim = lims.get(hand)
    if lim is not None and lower in F.A:
        t = limits.twist_angle(limits.split(lim.ref.inverted() @ q if lim.ref is not None else q)[1])
        lo, hi = math.radians(lim.ranges[-2]), math.radians(lim.ranges[-1])
        extra = t - min(max(t, lo), hi)
        if abs(extra) > 1e-4:
            F.set_local(lower, F.local_q(lower) @ Quaternion(limits.Y, extra))
            F.limit(lower, lims)
            U = F.unrotated(hand)
            q = (U.to_3x3().inverted() @ target_rot).to_quaternion()
    F.set_local(hand, q)
    F.limit(hand, lims)


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _shin_frame(rig, lims, s, hip, K, ankle, knee_dir):
    """the shin's armature-space rotation the IK will produce: along knee -> ankle with its hinge across the bend"""
    t = (K - hip).normalized()
    y = (ankle - K).normalized()
    n = t.cross(y)
    if n.length < 1e-6:
        n = knee_dir.cross(t)
    n = (n - y * n.dot(y)).normalized()
    lim = lims.get("shin." + s)
    h = lim.hinge if lim is not None else Vector((1.0, 0.0, 0.0))
    local = Matrix((h, limits.Y, h.cross(limits.Y))).transposed()
    world = Matrix((n, y, n.cross(y))).transposed()
    return world @ local.transposed()


def _thigh_rotation(F, lims, s, hip, K, ankle):
    """the thigh's armature-space rotation the IK will produce for a knee at K (along hip -> knee, hinge across the bend)"""
    B = F.B
    th, sh = "thigh." + s, "shin." + s
    y_t = (K - hip).normalized()
    n = y_t.cross((ankle - K).normalized())
    if n.length < 1e-6:
        n = y_t.orthogonal()
    n = (n - y_t * n.dot(y_t)).normalized()
    h_s = lims[sh].hinge if sh in lims else Vector((1.0, 0.0, 0.0))
    h_t = B[th].matrix_local.to_3x3().inverted() @ B[sh].matrix_local.to_3x3() @ h_s
    h_t.y = 0.0
    h_t.normalize()
    return Matrix((n, y_t, n.cross(y_t))).transposed() @ Matrix((h_t, limits.Y, h_t.cross(limits.Y)))


ANKLE_RELIEF = 2.0     # degrees: a planted foot further outside the ankle's range than this is turned back inside it


def _foot_low(rig, F, s, ankle, rot3):
    """lowest probe point (height along the rig's up) of the foot and toe if the foot sat at `ankle` with rotation `rot3`;
    None without a body model"""
    body = bodymod.get_body(rig)
    if body is None:
        return None
    B = rig.data.bones
    ft, to = "foot." + s, "toe." + s
    Wf = rot3.to_4x4()
    Wf.translation = ankle
    mats = {ft: Wf}
    if to in B and B[to].parent is not None and B[to].parent.name == ft:
        mats[to] = Wf @ (B[ft].matrix_local.inverted() @ B[to].matrix_local) @ F.P[to].matrix_basis
    up = np.array(rig.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, 1.0)), dtype=float)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    low = None
    for name, W in mats.items():
        if name not in body.names:
            continue
        sel = body.plocal[body.pbone == body.names.index(name)]
        if len(sel) == 0:
            continue
        M = np.array(W, dtype=float)
        h = float(((sel @ M[:3, :3].T + M[:3, 3]) @ up).min())
        low = h if low is None else min(low, h)
    return low


def _ik_leg(F, lims, rig, s, hip, ankle, T, tgt, pole, rest, use_limits, foot_rot=None, steps=36):
    """an IK leg: foot target and knee pole. The foot comes first: kept as it is (foot_rot, a planted foot) or taken from
    the network with its turn kept within what hip and ankle can do together. Knees track the toes: the knee is tried all
    the way around the hip-ankle line and the direction closest to the foot's (or the network's bend) that keeps hip
    and ankle inside their limits wins. Only a foot from the network is then clamped against the shin it gets. A planted
    foot that the ankle could not really do (rolled onto its side, shin leaning over it further than an ankle bends) is
    turned back into range about the ball of the foot, so the toes stay where they are, a rolled boot stands on its sole
    again, and the lowest point of the foot keeps its height."""
    B = rig.data.bones
    L = leg_length(rig)
    th, sh, ft = "thigh." + s, "shin." + s, "foot." + s
    a, b = B[th].length, B[sh].length
    fwd_rest, _left = limits.body_axes(rig)
    fwd = (F.A["pelvis"].to_3x3() @ B["pelvis"].matrix_local.to_3x3().inverted()) @ fwd_rest
    foot_rest = B[ft].matrix_local.to_3x3()
    rest_toe = (rest["toe_" + s] - rest["ankle_" + s]).normalized()
    if foot_rot is None:
        toe = T["toe_" + s] - T["ankle_" + s]
        if use_limits:
            toe = limits.foot_direction(Vector((0.0, 0.0, 1.0)), fwd, toe, limits.HIP_TWIST + limits.ANKLE_TWIST)
        Fr = _frame(rest_toe, rest["knee_" + s] - rest["ankle_" + s])
        foot = _frame(toe, T["knee_" + s] - T["ankle_" + s]) @ Fr.transposed() @ foot_rest
    else:
        foot = foot_rot
    rel_f = B[sh].matrix_local.to_3x3().inverted() @ foot_rest
    lim_t, lim_f = lims.get(th), lims.get(ft)
    U_t = F.unrotated(th).to_3x3()
    bend = T["knee_" + s] - (T["hip_" + s] + T["ankle_" + s]) * 0.5

    def ankle_excess(K, k, ank, ftr):
        if lim_f is None:
            return 0.0, None
        U = _shin_frame(rig, lims, s, hip, K, ank, k) @ rel_f
        q = (U.inverted() @ ftr).to_quaternion()
        return limits.excess(q, lim_f), (U, q)

    def search(ank, ftr):
        axis = (ank - hip).normalized()
        toe_now = ftr @ (foot_rest.inverted() @ rest_toe)
        ref = toe_now - axis * toe_now.dot(axis)
        if ref.length < 1e-6 or ref.dot(fwd) < -0.2:
            ref = fwd - axis * fwd.dot(axis)
        ref.normalize()
        weight = _smooth((bend.length - 0.01 * L) / (0.06 * L)) if foot_rot is None else 0.0
        want = limits.knee_direction(axis, ref, bend, weight, limits.ANKLE_TWIST) if use_limits else (
            bend if bend.length > 1e-4 else ref)
        best = None
        for K, dev in (_swivel_points(hip, ank, a, b, want, steps) if lim_t is not None else [(_two_bone(hip, ank, a, b, hip + want), 0.0)]):
            k = K - (hip + ank) * 0.5
            k = k - axis * k.dot(axis)
            k = k.normalized() if k.length > 1e-6 else want
            ex = limits.excess((U_t.inverted() @ _thigh_rotation(F, lims, s, hip, K, ank)).to_quaternion(), lim_t) if lim_t else 0.0
            ex += ankle_excess(K, k, ank, ftr)[0]
            cost = 10.0 * ex + 0.2 * dev
            if best is None or cost < best[0]:
                best = (cost, k, K)
        return best

    best = search(ankle, foot)
    if foot_rot is not None and use_limits and lim_f is not None:
        ankle0, relieved = ankle, False
        Lf = B[ft].length
        for _ in range(3):
            _c, k, K = best
            ex, uq = ankle_excess(K, k, ankle, foot)
            if ex < ANKLE_RELIEF:
                break
            U, q = uq
            qr = lim_f.ref.inverted() @ q if lim_f.ref is not None else q
            tw = limits.twist_angle(limits.split(qr)[1])
            c = limits.clamp_rotation(q, lim_f)
            if abs(tw) > math.radians(max(abs(lim_f.ranges[-2]), abs(lim_f.ranges[-1]))):
                # rolled over: stand the foot on its sole again instead of leaving it at the edge of the roll range
                sw = limits.split(lim_f.ref.inverted() @ c if lim_f.ref is not None else c)[0]
                c = lim_f.ref @ sw if lim_f.ref is not None else sw
            new = U @ c.to_matrix()
            ball = ankle + foot.col[1].xyz.normalized() * Lf
            ankle = ball + (new @ foot.inverted()) @ (ankle - ball)
            foot = new
            relieved = True
            best = search(ankle, foot)
        if relieved:
            z0, z1 = _foot_low(rig, F, s, ankle0, foot_rot), _foot_low(rig, F, s, ankle, foot)
            if z0 is not None and z1 is not None and abs(z0 - z1) > 1e-4:
                up = (rig.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, 1.0))).normalized()
                ankle = ankle + up * (z0 - z1)
                best = search(ankle, foot)
    _cost, k, K = best
    if foot_rot is None and lim_f is not None:
        U, q = ankle_excess(K, k, ankle, foot)[1]
        foot = U @ limits.clamp_rotation(q, lim_f).to_matrix()
    F.orient(tgt, foot @ (foot_rest.inverted() @ B[tgt].matrix_local.to_3x3()), head=ankle)
    if pole:
        M = F.unrotated(pole).copy()
        M.translation = (hip + ankle) * 0.5 + k.normalized() * (0.75 * L)
        F.A[pole] = M
    return k


def fit(rig, T, m, eyes, given=(), pin_pelvis=False, ik_only=False, heading_frame=None, use_limits=True):
    """pose the rig so it matches the joint positions T (armature space) with its own proportions, inside its joint
    limits. ik_only: plain IK for comparison (hips move, hands and feet reach their goals, the rest keeps its rotation).
    heading_frame: the hips frame the prediction was made in; the hips keep its facing (the network sets their tilt and
    roll only), otherwise small turns would build up frame after frame"""
    F = Fitter(rig)
    rest = joints_rest(rig, m, eyes)
    B = rig.data.bones
    lims = limits.rig_limits(rig) if (use_limits and not ik_only) else {}

    # hips: position from the goal, orientation from the predicted hip joints and lower spine
    if pin_pelvis or ik_only:
        R = F.P["pelvis"].matrix.to_3x3().normalized()
    else:
        across = T["hip_L"] - T["hip_R"]
        if heading_frame is not None:
            c = heading_frame.transposed() @ across
            across = heading_frame @ Vector((math.hypot(c.x, c.y), 0.0, c.z))
        Ft = _frame(across, T["spine"] - T["hips"])
        Fr = _frame(rest["hip_L"] - rest["hip_R"], rest["spine"] - rest["hips"])
        R = Ft @ Fr.transposed() @ B["pelvis"].matrix_local.to_3x3()
    M = R.to_4x4()
    M.translation = T["hips"]
    F.A["pelvis"] = M

    # torso: rig joints sit on the predicted hips-spine-chest-neck-head curve at the same share of its length
    curve = [T[j] for j in ("hips", "spine", "chest", "neck", "head")]
    chain = ["spine", "chest", "neck"]
    pts = [B["pelvis"].head_local] + [B[b].head_local for b in chain] + [B["head"].head_local]
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + (b - a).length)
    on = [_polyline_at(curve, c / cum[-1]) for c in cum]
    for k, b in enumerate(chain):
        if ik_only:
            F.follow(b)
            continue
        if b == "chest":
            # the chest also turns with the predicted shoulder line, so a twist of the upper body is kept
            Ft = _frame(on[k + 2] - on[k + 1], T["clav_L"] - T["clav_R"])
            Fr = _frame(B["chest"].tail_local - B["chest"].head_local, rest["clav_L"] - rest["clav_R"])
            F.orient("chest", Ft @ Fr.transposed() @ B["chest"].matrix_local.to_3x3())
        else:
            F.aim(b, on[k + 2] - on[k + 1])
        F.limit(b, lims)
    if ik_only:
        F.follow("head")
    else:
        Ft = _frame(T["eye_L"] - T["eye_R"], (T["eye_L"] + T["eye_R"]) * 0.5 - T["head"])
        Fr = _frame(rest["eye_L"] - rest["eye_R"], (rest["eye_L"] + rest["eye_R"]) * 0.5 - rest["head"])
        H = Ft @ Fr.transposed() @ B["head"].matrix_local.to_3x3()
        if lims:
            # people look with neck and head together: the neck takes part of the turn
            qh = (F.unrotated("head").to_3x3().inverted() @ H).to_quaternion()
            rel = F.rel("head").to_3x3().to_quaternion()
            F.set_local("neck", F.local_q("neck") @ rel @ Quaternion().slerp(qh, 0.4) @ rel.inverted())
            F.limit("neck", lims)
        F.orient("head", H)
        F.limit("head", lims)

    # arms: clavicle toward the predicted shoulder, then two-bone IK to the wrist with the predicted elbow as the hint
    for s in ("L", "R"):
        if ik_only:
            F.follow("clavicle." + s)
        else:
            F.aim("clavicle." + s, T["shoulder_" + s] - T["clav_" + s])
            F.limit("clavicle." + s, lims)
        S = F.unrotated("upperarm." + s).translation
        a, b = B["upperarm." + s].length, B["forearm." + s].length
        W = _limb_end(T, given, "shoulder_" + s, "elbow_" + s, "wrist_" + s, S, a + b)
        hint = T["elbow_" + s] - T["shoulder_" + s]
        if ik_only:
            E = _two_bone(S, W, a, b, S + hint)
            F.aim("upperarm." + s, E - S)
            F.aim("forearm." + s, W - F.unrotated("forearm." + s).translation)
            F.follow("hand." + s)
            continue
        _limb(F, lims, "upperarm." + s, "forearm." + s, W, hint)
        if "thumb_" + s in T and "thumb_" + s in m:
            Ft = _frame(T["knuckle_" + s] - T["wrist_" + s], T["thumb_" + s] - T["wrist_" + s])
            Fr = _frame(rest["knuckle_" + s] - rest["wrist_" + s], rest["thumb_" + s] - rest["wrist_" + s])
            _hand(F, lims, "hand." + s, "forearm." + s, Ft @ Fr.transposed() @ B["hand." + s].matrix_local.to_3x3())
        else:
            F.aim("hand." + s, T["knuckle_" + s] - T["wrist_" + s])
            F.limit("hand." + s, lims)

    # legs: IK rigs get their foot target and knee pole; FK legs are solved like the arms
    for s in ("L", "R"):
        tgt, pole = _ik_target(rig, "shin." + s)
        hip = F.A["pelvis"] @ (B["pelvis"].matrix_local.inverted() @ B["thigh." + s].head_local)
        a, b = B["thigh." + s].length, B["shin." + s].length
        ankle = _limb_end(T, given, "hip_" + s, "knee_" + s, "ankle_" + s, hip, a + b)
        if tgt and ik_only:
            M = F.P[tgt].matrix.copy()
            M.translation = ankle
            F.A[tgt] = M
        elif tgt:
            _ik_leg(F, lims, rig, s, hip, ankle, T, tgt, pole, rest, use_limits)
        else:
            fwd_rest, _left = limits.body_axes(rig)
            fwd = (F.A["pelvis"].to_3x3() @ B["pelvis"].matrix_local.to_3x3().inverted()) @ fwd_rest
            axis = (ankle - hip).normalized()
            bend = T["knee_" + s] - (T["hip_" + s] + T["ankle_" + s]) * 0.5
            k = limits.knee_direction(axis, fwd, bend, 1.0) if lims else bend
            _limb(F, lims, "thigh." + s, "shin." + s, ankle, k)
            Ft = _frame(T["toe_" + s] - T["ankle_" + s], T["knee_" + s] - T["ankle_" + s])
            Fr = _frame(rest["toe_" + s] - rest["ankle_" + s], rest["knee_" + s] - rest["ankle_" + s])
            _hand(F, lims, "foot." + s, "shin." + s, Ft @ Fr.transposed() @ B["foot." + s].matrix_local.to_3x3())
    return F


def _leg_reach_scale(rig, F, given, shift):
    """largest share (0..1) of a hips shift that keeps both hip joints within leg's reach of their planted ankles"""
    B = rig.data.bones
    lam = 1.0
    for s in ("L", "R"):
        if "ankle_" + s not in given:
            continue
        hip = F.A["pelvis"] @ (B["pelvis"].matrix_local.inverted() @ B["thigh." + s].head_local)
        a = hip - given["ankle_" + s]
        Lmax = 0.995 * (B["thigh." + s].length + B["shin." + s].length)
        if (a + shift * lam).length <= Lmax:
            continue
        if a.length >= Lmax:
            return 0.0
        dd, ad, aa = shift.dot(shift), a.dot(shift), a.dot(a)
        disc = ad * ad - dd * (aa - Lmax * Lmax)
        lam = min(lam, max(0.0, (-ad + math.sqrt(max(disc, 0.0))) / dd)) if dd > 1e-12 else 0.0
    return lam


PARTS = ("spine", "arms", "legs")


def lean_matrix(T, theta):
    """armature-space matrix of a forward lean of the torso by theta radians: a turn about the hips' side-to-side axis
    through the base of the spine"""
    h = T["spine"]
    return Matrix.Translation(h) @ Matrix.Rotation(theta, 4, heading(T).col[0].xyz) @ Matrix.Translation(-h)


def torso_fit(rig, T, lean=0.0, lims=None):
    """a Fitter with the pelvis, spine, chest, neck and head following the current pose, the torso leaning `lean` radians
    further forward (about the base of the spine); limits applied when `lims` is given (the head keeps its facing and
    shares a turn it cannot make alone with the neck)"""
    F = Fitter(rig)
    F.follow("pelvis")
    F.follow("spine")
    if abs(lean) > 1e-6:
        F.A["spine"] = lean_matrix(T, lean) @ F.A["spine"]
    if lims:
        F.limit("spine", lims)
    for b in ("chest", "neck"):
        F.follow(b)
        if lims:
            F.limit(b, lims)
    # the head keeps the way it faces; a turn beyond what the head can do alone is shared with the neck
    head = F.P["head"].matrix.to_3x3().normalized()
    if abs(lean) > 1e-6:
        head = lean_matrix(T, lean).to_3x3() @ head
    if lims:
        _hand(F, lims, "head", "neck", head)
    else:
        F.follow("head")
    return F


def fix_pose(rig, parts=None, hold=None, lean=None, elbow_hint=None):
    """move the current pose inside the joint limits while hands and feet stay put: spine, neck and head are clamped,
    each arm is re-solved to the same wrist (elbow kept as close as the limits allow), and IK legs get a knee pole and
    a foot turn a real leg can do. parts: which of PARTS to fix (all by default); hold: {side: (wrist position, hand
    rotation)} to reach instead of where the hand is now (a hand that holds a prop, see grip.py); lean: radians the torso
    leans further forward, so the arms get some slack (the hips and feet stay put); elbow_hint: {side: elbow position
    relative to its shoulder} from the previous frame; the elbow's swivel may turn only so far from it, so the solution
    cannot flip between frames (the solved elbows are left in F.elbow)"""
    parts = PARTS if parts is None else parts
    m = rig_map(rig)
    eyes = eye_points(rig, m)
    T = joints_now(rig, m, eyes)
    rest = joints_rest(rig, m, eyes)
    lims = limits.rig_limits(rig)
    B = rig.data.bones
    if "spine" in parts or lean:
        F = torso_fit(rig, T, lean or 0.0, lims)
    else:
        F = Fitter(rig)
        F.follow("pelvis")
    F.elbow = {}
    if "arms" in parts:
        for s in ("L", "R"):
            F.follow("clavicle." + s)
            F.limit("clavicle." + s, lims)
            wrist, hand = T["wrist_" + s], F.P["hand." + s].matrix.to_3x3().normalized()
            if hold and s in hold:
                wrist, hand = hold[s]
            _limb(F, lims, "upperarm." + s, "forearm." + s, wrist, T["elbow_" + s] - T["shoulder_" + s], hand="hand." + s,
                  hand_rot=hand, prev_dir=elbow_hint.get(s) if elbow_hint else None)
            _hand(F, lims, "hand." + s, "forearm." + s, hand)
            F.elbow[s] = F.A["forearm." + s].translation - F.A["upperarm." + s].translation
    if "legs" in parts:
        for s in ("L", "R"):
            tgt, pole = _ik_target(rig, "shin." + s)
            if tgt:
                hip = F.A["pelvis"] @ (B["pelvis"].matrix_local.inverted() @ B["thigh." + s].head_local)
                foot = F.P["foot." + s].matrix.to_3x3().normalized()
                _ik_leg(F, lims, rig, s, hip, F.P[tgt].head.copy(), T, tgt, pole, rest, True, foot_rot=foot)
            else:
                for b in ("thigh." + s, "shin." + s, "foot." + s):
                    F.follow(b)
                    F.limit(b, lims)
    return F


def reach_lean(scene, rig, f0, f1, holds, comfort=0.92, ramp=9, max_lean=math.radians(18.0), window=9):
    """how far the torso should lean forward on each frame so that every hand that holds a prop reaches it with the arm a
    little bent (wrist no further from its shoulder than `comfort` of the arm's length): {frame: radians}. A stiff straight
    arm cannot also keep the wrist natural. Smoothed in time, and eased out over `ramp` frames after the last held frame
    (the hands let go there)."""
    B = rig.data.bones
    reach = {s: comfort * (B["upperarm." + s].length + B["forearm." + s].length) for s in ("L", "R")}
    frames = sorted(f for f in holds if f0 <= f <= f1)
    if not frames:
        return {}
    last_held = frames[-1] - ramp
    m = rig_map(rig)
    eyes = eye_points(rig, m)
    steps = np.radians(np.arange(0.0, math.degrees(max_lean) + 0.5, 1.0))
    raw = {}
    cur = scene.frame_current
    try:
        for f in frames:
            if f > last_held:
                continue
            scene.frame_set(f)
            T = joints_now(rig, m, eyes)
            best = None
            for th in steps:
                Ml = lean_matrix(T, float(th))
                over = max(((W - Ml @ T["shoulder_" + s]).length - reach[s] for s, (W, _H) in holds[f].items()), default=0.0)
                if over <= 0.0:
                    best = (0.0, float(th))
                    break
                if best is None or over < best[0]:
                    best = (over, float(th))
            raw[f] = best[1]
    finally:
        scene.frame_set(cur)
    held = sorted(raw)
    if not held:
        return {}
    X = np.array([raw[f] for f in held])
    k = window // 2
    pad = np.concatenate([np.repeat(X[:1], k), X, np.repeat(X[-1:], k)])
    Xs = np.convolve(pad, np.ones(window) / window, mode="valid")
    out = {f: float(Xs[i]) for i, f in enumerate(held)}
    end = out[held[-1]]
    for i in range(1, ramp + 1):
        x = i / (ramp + 1.0)
        out[held[-1] + i] = end * (1.0 - x * x * x * (x * (6.0 * x - 15.0) + 10.0))
    return out


def fix_animation(scene, rig, f0, f1, parts=None, holds=None, reach=0.92, leans=None):
    """Fix Joints on every frame of the range. All frames are solved from the untouched animation first, then keyed, so
    new keys never change what the next frame starts from. parts: which of PARTS to fix; holds: {frame: {side: (wrist,
    hand rotation)}} for hands that hold a prop (see grip.py): the arms reach those exactly, and with `reach` set the torso
    leans a little toward the prop wherever an arm would otherwise be stretched straight. Returns how far each joint had to
    move at most (degrees). leans: {frame: radians} to use instead of working them out (see reach_lean)."""
    cur = scene.frame_current
    frames, worst = [], {}
    if leans is None:
        leans = reach_lean(scene, rig, f0, f1, holds, reach) if (holds and reach) else {}
    hints = None
    try:
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            F = fix_pose(rig, parts, holds.get(f) if holds else None, leans.get(f), hints)
            hints = dict(F.elbow) if F.elbow else None
            frames.append((f, F.values()))
            for b, v in F.clamped.items():
                worst[b] = max(worst.get(b, 0.0), v)
    finally:
        scene.frame_set(cur)
    for f, vals in frames:
        write_values(rig, vals, key=True, frame=f)
    return worst


def check_animation(scene, rig, f0, f1, tolerance=2.0):
    """frames where a joint is outside its limits: {bone: (frames over, worst degrees, worst frame)}"""
    lims = limits.rig_limits(rig)
    B, P = rig.data.bones, rig.pose.bones
    out = {}
    cur = scene.frame_current
    try:
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            for b, lim in lims.items():
                rel_pose = P[lim.parent].matrix.to_3x3().normalized().inverted() @ P[b].matrix.to_3x3().normalized()
                rel_rest = B[lim.parent].matrix_local.to_3x3().inverted() @ B[b].matrix_local.to_3x3()
                e = limits.excess((rel_rest.inverted() @ rel_pose).to_quaternion(), lim)
                if e > tolerance:
                    n, w, wf = out.get(b, (0, 0.0, f))
                    out[b] = (n + 1, max(w, e), f if e > w else wf)
    finally:
        scene.frame_set(cur)
    return out


def solve(rig, goals=None, keep=CORE, pin_pelvis=False, use_net=True, key=False, frame=None, update=True, auto_hips=False,
          use_limits=True):
    """AutoPose the rig. goals: {joint: world position}; joints in `keep` without a goal stay where they are now.
    auto_hips: let the feet-anchored network place the hips and upper body (only hand and feet goals count; needs both
    feet given). Returns predicted world positions."""
    if update:
        bpy.context.view_layer.update()
    m = rig_map(rig)
    eyes = eye_points(rig, m)
    cur = joints_now(rig, m, eyes)
    inv = rig.matrix_world.inverted()
    given = {j: cur[j].copy() for j in keep if j in cur}
    for j, p in (goals or {}).items():
        given[j] = inv @ Vector(p)
    free_hips = auto_hips and use_net and has_model("feet") and "ankle_L" in given and "ankle_R" in given
    if free_hips:
        # the body is free: hips and neck follow from the hands and feet
        given.pop("hips", None)
        given.pop("neck", None)
    T, m, eyes, R = predict(rig, given, use_net=use_net)
    F = fit(rig, T, m, eyes, given=set(given), pin_pelvis=pin_pelvis, ik_only=not use_net, heading_frame=R, use_limits=use_limits)
    if free_hips:
        # the network places the hips for its own proportions; when a hand goal is out of this body's reach, move the
        # hips toward it (as far as the legs reach the planted feet) and pose again with the hips given
        for _ in range(3):
            miss = [given[w] - F.A["hand." + s].translation for w, s in (("wrist_L", "L"), ("wrist_R", "R")) if w in given]
            if not miss or max(v.length for v in miss) < 0.01:
                break
            shift = sum(miss, Vector()) * (0.9 / len(miss))
            shift *= _leg_reach_scale(rig, F, given, shift)
            if shift.length < 1e-3:
                break
            given["hips"] = T["hips"] + shift
            T, m, eyes, R = predict(rig, given, use_net=use_net)
            F = fit(rig, T, m, eyes, given=set(given), pin_pelvis=pin_pelvis, heading_frame=R, use_limits=use_limits)
    F.write(key=key, frame=frame if frame is not None else bpy.context.scene.frame_current)
    _last["clamped"] = dict(F.clamped)
    return {j: rig.matrix_world @ p for j, p in T.items()}


# ------------------------------------------------------------------------------------------------------------ goals

def goal_name(rig, joint):
    return "%s%s_%s" % (GOAL_PREFIX, rig.name, joint)


def add_goals(rig, joints=CORE, size=0.06):
    """an empty at each joint: move them, then AutoPose (or turn on Live)"""
    bpy.context.view_layer.update()
    m = rig_map(rig)
    cur = joints_now(rig, m, eye_points(rig, m))
    coll = bpy.context.scene.collection
    made = []
    for j in joints:
        name = goal_name(rig, j)
        ob = bpy.data.objects.get(name)
        if ob is None:
            ob = bpy.data.objects.new(name, None)
            coll.objects.link(ob)
        ob.empty_display_type = 'SPHERE' if j != "hips" else 'CUBE'
        ob.empty_display_size = size
        ob.show_in_front = True
        ob.location = rig.matrix_world @ cur[j]
        made.append(name)
    return made


def read_goals(rig):
    out = {}
    for j in CORE + ("elbow_L", "elbow_R", "knee_L", "knee_R", "head", "chest"):
        ob = bpy.data.objects.get(goal_name(rig, j))
        if ob is not None:
            out[j] = ob.matrix_world.translation.copy()
    return out


def remove_goals(rig):
    for ob in [o for o in bpy.data.objects if o.name.startswith(GOAL_PREFIX + rig.name + "_")]:
        bpy.data.objects.remove(ob, do_unlink=True)


_live = {"on": False, "last": {}, "busy": False}


def _live_handler(scene, depsgraph):
    if _live["busy"]:
        return
    s = getattr(scene, "motion_assist", None)
    rig = s.target if s else None
    if rig is None or not s.autopose_live:
        return
    goals = read_goals(rig)
    sig = {j: tuple(round(x, 5) for x in p) for j, p in goals.items()}
    if not goals or sig == _live["last"].get(rig.name):
        return
    _live["last"][rig.name] = sig
    _live["busy"] = True
    try:
        solve(rig, goals, keep=(), pin_pelvis=s.autopose_pin_pelvis, update=False, auto_hips=s.autopose_auto_hips,
              use_limits=s.use_limits)
    except Exception as e:
        print("AutoPose live:", e)
    finally:
        _live["busy"] = False


def register():
    if _live_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_live_handler)


def unregister():
    if _live_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_live_handler)
