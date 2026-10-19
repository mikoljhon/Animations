# Motion Assist: joint limits. Human joints only turn certain ways: knees and elbows are hinges, hips and shoulders are
# ball joints with a limited cone, wrists and ankles bend a little and barely twist. A rotation is measured like a local
# bone rotation (child relative to parent, from the rest pose) and split into swing (where the bone points) and twist
# (roll about its own length); each is clamped to an anatomical range.

import math

from mathutils import Quaternion, Vector

Y = Vector((0.0, 1.0, 0.0))

# child: (parent, kind, tip directions, ranges in degrees)
#   ball:  ranges = (dir1 +, dir1 -, dir2 +, dir2 -, twist min, twist max): how far the bone tip may swing toward each
#          anatomical direction (an ellipse in between) and how far it may roll
#   hinge: ranges = (bend min, bend max, twist min, twist max), bending toward the given direction only
# directions: front (where the character faces), back, up, side (outward for .L/.R bones, the character's left otherwise)
TABLE = {
    "spine": ("pelvis", "ball", ("front", "side"), (45, 30, 30, 30, -25, 25)),
    "chest": ("spine", "ball", ("front", "side"), (40, 25, 25, 25, -30, 30)),
    "neck": ("chest", "ball", ("front", "side"), (50, 45, 35, 35, -50, 50)),
    "head": ("neck", "ball", ("front", "side"), (35, 35, 25, 25, -45, 45)),
    "clavicle.%s": ("chest", "ball", ("front", "up"), (30, 25, 45, 15, -20, 20)),
    "upperarm.%s": ("clavicle.%s", "ball", ("front", "up"), (135, 45, 100, 95, -90, 90)),
    "forearm.%s": ("upperarm.%s", "hinge", ("front",), (-5, 150, -85, 85)),
    "hand.%s": ("forearm.%s", "ball", ("front", "side"), (25, 35, 70, 80, -20, 20)),
    "thigh.%s": ("pelvis", "ball", ("front", "side"), (130, 30, 60, 30, -45, 45)),
    "shin.%s": ("thigh.%s", "hinge", ("back",), (-3, 158, -8, 8)),
    "foot.%s": ("shin.%s", "ball", ("up", "side"), (45, 50, 20, 20, -25, 25)),
}
# measured from a reference pose instead of the rest pose: the shoulder from the arm held out sideways (T-pose), so
# reaching overhead is 90 degrees away rather than nearly 180 (where swing and twist stop being well defined)
# the wrist and the ankle from their neutral (hand in line with the forearm, foot square to the shin): a rig's rest hand
# and foot are rarely neutral (Mender's rest foot points down 16 degrees), and a limit measured from there is off by that
REFS = {"upperarm.%s": "side", "hand.%s": "neutral", "foot.%s": "neutral"}
HINGE_SLACK = 4.0    # degrees a knee or elbow may leave its bend plane (more where the rig's rest pose is bent off-plane)
CONE_POWER = 3.0     # 2 = an ellipse between the four ranges; higher allows more of a diagonal (bend forward and out)
HIP_TWIST = 40.0      # how far the knee may turn away from the hips' forward direction (thigh rotation), degrees
ANKLE_TWIST = 25.0    # how far the foot may turn away from the knee's direction, degrees
_cache = {}


class Limit:
    def __init__(self, bone, parent, kind, ranges, a1=None, a2=None, hinge=None, ref=None, slack=0.0):
        self.bone, self.parent, self.kind, self.ranges = bone, parent, kind, ranges
        self.a1, self.a2, self.hinge, self.ref = a1, a2, hinge, ref
        self.slack = slack        # hinge only: how far (degrees) the bone may leave the bend plane (an elbow's carrying angle)


def body_axes(rig):
    """armature-space forward (where the character faces) and left, from the rest pose"""
    B = rig.data.bones
    fwd = B["toe.L"].tail_local - B["foot.L"].head_local
    fwd.z = 0.0
    left = B["thigh.L"].head_local - B["thigh.R"].head_local
    left.z = 0.0
    return fwd.normalized(), left.normalized()


def _neutral_axis(B, name, par, fwd):
    """armature-space direction a bone points in when its joint is neutral: a hand carries on along the forearm, a foot
    stands square to the shin and points forward"""
    p = B[par].tail_local - B[par].head_local
    p.normalize()
    if name.startswith("foot"):
        n = fwd - p * fwd.dot(p)
        return n.normalized() if n.length > 1e-6 else p.orthogonal().normalized()
    return p


def rig_limits(rig):
    """the Limit for every bone of the table that exists on this rig (cached per rig)"""
    key = (rig.name, len(rig.data.bones))
    if key in _cache:
        return _cache[key]
    B = rig.data.bones
    if not all(n in B for n in ("toe.L", "foot.L", "thigh.L", "thigh.R")):
        _cache[key] = {}
        return _cache[key]
    fwd, left = body_axes(rig)
    out = {}
    for pattern, (parent, kind, dirs, ranges) in TABLE.items():
        for s in (("L", "R") if "%s" in pattern else ("",)):
            name = pattern % s if s else pattern
            par = parent % s if "%s" in parent else parent
            if name not in B or par not in B:
                continue
            side = left if s != "R" else -left
            world = {"front": fwd, "back": -fwd, "up": Vector((0.0, 0.0, 1.0)), "side": side}
            inv = B[name].matrix_local.to_3x3().inverted()

            def local(d):
                v = inv @ world[d]
                v.y = 0.0
                return v.normalized() if v.length > 1e-6 else None

            if kind == "hinge":
                d = local(dirs[0]) or Vector((0.0, 0.0, 1.0))
                h = Y.cross(d).normalized()
                # on an IK chain the solver bends about the plane of the slightly bent rest pose, so use that hinge
                upper = B[name].head_local - B[par].head_local
                lower = B[name].tail_local - B[name].head_local
                n = upper.cross(lower)
                ik = any(c.type == 'IK' for c in rig.pose.bones[name].constraints)
                if ik and n.length > math.sin(math.radians(1.0)) * upper.length * lower.length:
                    n = inv @ n
                    n.y = 0.0
                    if n.length > 1e-6:
                        n.normalize()
                        h = n if n.dot(h) > 0.0 else -n
                # ranges are bends from straight; the rig's rest pose is usually a little bent already (Mender's elbow 18
                # degrees), so shift them to be measured from the rest pose
                straight = B[name].matrix_local.to_3x3().inverted() @ (B[par].tail_local - B[par].head_local)
                shift = math.degrees(hinge_angle(straight.normalized(), h)) if straight.length > 1e-9 else 0.0
                off = math.degrees(math.asin(max(-1.0, min(1.0, straight.normalized().dot(h))))) if straight.length > 1e-9 else 0.0
                out[name] = Limit(name, par, kind, (ranges[0] + shift, ranges[1] + shift) + tuple(ranges[2:]), hinge=h,
                                  slack=max(HINGE_SLACK, abs(off) + 2.0))
            else:
                ref = None
                if pattern in REFS:
                    rest = B[name].matrix_local.to_3x3().to_quaternion()
                    target = _neutral_axis(B, name, par, fwd) if REFS[pattern] == "neutral" else world[REFS[pattern]]
                    q_world = (B[name].matrix_local.to_3x3() @ Y).rotation_difference(target)
                    ref = rest.inverted() @ q_world @ rest
                    inv = (rest @ ref).to_matrix().inverted()
                a1 = local(dirs[0]) or Vector((1.0, 0.0, 0.0))
                a2 = local(dirs[1]) or Y.cross(a1)
                a2 = (a2 - a1 * a2.dot(a1))
                a2 = a2.normalized() if a2.length > 1e-6 else Y.cross(a1).normalized()
                out[name] = Limit(name, par, kind, ranges, a1=a1, a2=a2, ref=ref)
    _cache[key] = out
    return out


def split(q):
    """q = swing @ twist, twist about the bone's own Y axis"""
    q = q.normalized()
    tw = Quaternion((q.w, 0.0, q.y, 0.0))
    tw = tw.normalized() if tw.magnitude > 1e-9 else Quaternion()
    return q @ tw.inverted(), tw


def twist_angle(tw):
    a = 2.0 * math.atan2(tw.y, tw.w)
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def hinge_angle(direction, hinge):
    """signed bend about the hinge axis that best points the bone's Y along `direction` (both in its local frame)"""
    p = direction - hinge * direction.dot(hinge)
    if p.length < 1e-9:
        return 0.0
    p.normalize()
    return math.atan2(Y.cross(p).dot(hinge), Y.dot(p))


def clamp_rotation(q, lim):
    """the closest allowed local rotation to q (a Quaternion in the bone's rest frame)"""
    if lim.ref is not None:
        return lim.ref @ _clamp_plain(lim.ref.inverted() @ q, lim)
    return _clamp_plain(q, lim)


def _clamp_plain(q, lim):
    sw, tw = split(q)
    r = [math.radians(x) for x in lim.ranges]
    t = _clamp(twist_angle(tw), r[-2], r[-1])
    y2 = sw @ Y
    if lim.kind == "hinge":
        phi = _clamp(hinge_angle(y2, lim.hinge), r[0], r[1])
        sl = math.radians(lim.slack)
        psi = _clamp(math.asin(_clamp(y2.dot(lim.hinge), -1.0, 1.0)), -sl, sl)
        sw2 = Y.rotation_difference((Quaternion(lim.hinge, phi) @ Y) * math.cos(psi) + lim.hinge * math.sin(psi))
    else:
        theta = math.acos(_clamp(Y.dot(y2), -1.0, 1.0))
        d = y2 - Y * y2.dot(Y)
        sw2 = sw
        if d.length > 1e-9 and theta > 1e-6:
            d.normalize()
            a, b = d.dot(lim.a1), d.dot(lim.a2)
            A = max(r[0] if a >= 0 else r[1], 1e-3)
            Bv = max(r[2] if b >= 0 else r[3], 1e-3)
            tmax = (abs(a / A) ** CONE_POWER + abs(b / Bv) ** CONE_POWER) ** (-1.0 / CONE_POWER)
            if theta > tmax:
                sw2 = Quaternion(Y.cross(d).normalized(), tmax)
    return sw2 @ Quaternion(Y, t)


def excess(q, lim):
    """how far (degrees) a local rotation is outside its limits"""
    c = clamp_rotation(q, lim)
    return math.degrees(q.normalized().rotation_difference(c).angle)


def swing_only(q, lim):
    """q without its roll about the bone's own axis (in the joint's reference frame, if it has one)"""
    qr = lim.ref.inverted() @ q if lim.ref is not None else q
    sw = split(qr)[0]
    return lim.ref @ sw if lim.ref is not None else sw


def _signed_angle(a, b, axis):
    return math.atan2(a.cross(b).dot(axis), a.dot(b))


def knee_direction(axis, ref, wanted=None, weight=1.0, max_turn=HIP_TWIST):
    """direction the knee should point (perpendicular to the hip-ankle axis): the wanted bend kept within max_turn degrees
    of ref (the body's forward); a wanted direction that points backward is ignored"""
    ref = ref - axis * ref.dot(axis)
    ref = ref.normalized() if ref.length > 1e-6 else axis.orthogonal().normalized()
    if wanted is None or weight <= 0.0:
        return ref
    w = wanted - axis * wanted.dot(axis)
    if w.length < 1e-6:
        return ref
    w.normalize()
    beta = _signed_angle(ref, w, axis)
    if abs(beta) > math.radians(100.0):
        return ref
    beta = _clamp(beta, -math.radians(max_turn), math.radians(max_turn)) * min(max(weight, 0.0), 1.0)
    return Quaternion(axis, beta) @ ref


def foot_direction(shin_axis, knee_dir, wanted, max_turn=ANKLE_TWIST):
    """the foot's forward direction with its turn about the shin kept within max_turn degrees of the knee's direction"""
    k = knee_dir - shin_axis * knee_dir.dot(shin_axis)
    w = wanted - shin_axis * wanted.dot(shin_axis)
    if k.length < 1e-6 or w.length < 1e-6:
        return wanted
    gamma = _signed_angle(k.normalized(), w.normalized(), shin_axis)
    g2 = _clamp(gamma, -math.radians(max_turn), math.radians(max_turn))
    return Quaternion(shin_axis, g2 - gamma) @ wanted
