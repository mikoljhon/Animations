# limb_solver.py : exact two-bone limb solve, no Blender IK, no pole-angle guesswork.
#   For an arm (upperarm -> forearm) or a leg (thigh -> shin):
#     * the end (wrist / ankle) goes exactly onto its target (if the limb can reach; otherwise straight toward it)
#     * the middle joint (elbow / knee) points EXACTLY the way the real captured elbow / knee pointed (around the root-end line)
#     * the lower bone only turns about its anatomical hinge axis and only the natural way (flexion), inside the hinge range
#       -> an elbow or knee can never bend backwards, whatever the data says
#   The roll of the upper bone is whatever makes that hinge lie across the bend (that is how a real arm/leg is built).
#   exec'd by rig_guard.py (needs bpy, math, Matrix, Quaternion, Vector, _limits)
import math
from mathutils import Matrix, Quaternion, Vector

LIMBS = {'L': {}, 'R': {}}
LIMB_PREV = {}
# how far (deg) an elbow / knee may point away from its natural direction (anatomy.py uses the same numbers on the capture data)
POINT_LIMIT = {'elbow': 110.0, 'knee': 100.0}
LIMB_CLAMPED = {}        # rig name -> how many limb solves had to be swung back to a possible elbow / knee direction


def _perp(v, u):
    return v - u * v.dot(u)


def natural_point(kind, u, fwd, up):
    """the natural direction the elbow / knee points for an upper-bone direction u, in a body frame (fwd, up):
    elbow: backward when the arm hangs, down when the arm is raised forward; knee: forward when the thigh hangs, up when it is raised."""
    s = -1.0 if kind == 'elbow' else 1.0
    w = 2.0 * max(0.0, u.dot(fwd))
    r = _perp(fwd * s, u) + _perp(up * s, u) * w
    if r.length < 1e-6:
        r = _perp(up * s, u)
    if r.length < 1e-6:
        r = u.orthogonal()
    return r.normalized()


def point_dir(root, mid, end):
    """where a middle joint points (away from the bend, perpendicular to the upper bone), and the bend angle (deg)"""
    u = (mid - root).normalized()
    l = end - mid
    p = -_perp(l, u)
    return u, (p.normalized() if p.length > 1e-9 else u.orthogonal()), math.degrees(u.angle(l)) if l.length > 1e-9 else 0.0


def _frame(a, b):
    """orthonormal frame (as matrix columns) from a primary direction a and a secondary direction b"""
    a = a.normalized()
    b2 = b - a * b.dot(a)
    if b2.length < 1e-8:
        b2 = a.orthogonal()
    b2.normalize()
    c = a.cross(b2)
    return Matrix((a, b2, c)).transposed()


def _bend_dir_local(rel_rest, hinge, phi):
    """lower bone direction expressed in the upper bone's frame, for hinge angle phi"""
    return rel_rest @ (Quaternion(hinge, phi) @ Vector((0.0, 1.0, 0.0)))


def _phi_for_bend(rel_rest, hinge, beta, lo, hi):
    """hinge angle phi (inside [lo, hi]) whose bend away from the upper bone's own axis equals beta (radians); flexion branch"""
    Y = Vector((0.0, 1.0, 0.0))
    f = lambda p: Y.angle(_bend_dir_local(rel_rest, hinge, p))
    # the straight angle: where the bend is smallest
    best = min(((f(lo + (hi - lo) * i / 200.0), lo + (hi - lo) * i / 200.0) for i in range(201)))
    p_straight = best[1]
    a, b = p_straight, hi
    if f(b) < beta:
        return b                      # cannot bend that far: the hinge stops at its limit
    if beta <= best[0]:
        return p_straight
    for _ in range(40):
        m = 0.5 * (a + b)
        if f(m) < beta:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def solve_limb(rig, upper, lower, root, target, mid_hint, lim_lower, lengths_scale=1.0, natural=None):
    """armature-space: root = joint position of the upper bone (shoulder / hip), target = where the end must go, mid_hint = a point the
    middle joint should lie toward (the real elbow / knee). Returns (R_upper, R_lower) armature-space 3x3 rotations and the end it reaches.
    natural = (kind 'elbow'|'knee', body forward, body up): the middle joint may never point the anatomically impossible way (more than
    POINT_LIMIT from natural_point); if the hint asks for that (broken data, a bad blend), the bend plane is swung around the root-end line
    to the nearest possible direction."""
    B = rig.data.bones
    a = B[upper].length
    b = B[lower].length
    v = target - root
    d = v.length
    lo_d, hi_d = abs(a - b) + 1e-4, (a + b) * 0.9995
    d_c = min(max(d, lo_d), hi_d)
    ax = v.normalized() if d > 1e-6 else Vector((0.0, 0.0, -1.0))
    # the plane of the bend: the real middle joint's direction around the root-end line
    m = mid_hint - root
    perp = m - ax * m.dot(ax)
    pm = perp.length
    key = (rig.name, upper)
    prev = LIMB_PREV.get(key)
    if prev is not None:
        pp = prev - ax * prev.dot(ax)
        if pp.length > 1e-4:
            pp.normalize()
            w = max(0.0, min(1.0, pm / (0.12 * (a + b))))      # nearly straight: the real direction is meaningless, keep the last one
            w = w * w * (3 - 2 * w)
            perp = pp * (1 - w) + (perp / pm if pm > 1e-6 else pp) * w
            pm = perp.length
    if pm < 1e-5:
        perp = ax.orthogonal()
    perp.normalize()
    # law of cosines: the angle at the root between the root-end line and the upper bone
    cos_al = max(-1.0, min(1.0, (a * a + d_c * d_c - b * b) / (2 * a * d_c)))
    al = math.acos(cos_al)
    T = root + ax * d_c
    cos_g = max(-1.0, min(1.0, (a * a + b * b - d_c * d_c) / (2 * a * b)))
    bend_deg = 180.0 - math.degrees(math.acos(cos_g))          # how bent the limb is (0 = straight)
    # the hinge angle depends only on how far the end is, not on which way the bend plane faces: solve it once
    rest_u = B[upper].matrix_local.to_3x3()
    rest_l = B[lower].matrix_local.to_3x3()
    rel_rest = rest_u.inverted() @ rest_l          # lower bone frame in the upper bone frame, at rest
    lo, hi = math.radians(lim_lower.ranges[0]), math.radians(lim_lower.ranges[1])
    h = lim_lower.hinge
    phi = _phi_for_bend(rel_rest, h, math.radians(bend_deg), lo, hi)
    dl = _bend_dir_local(rel_rest, h, phi)         # lower direction in the upper frame
    A = _frame(Vector((0.0, 1.0, 0.0)), dl)

    def pose_for(pp):
        E_ = root + ax * (a * math.cos(al)) + pp * (a * math.sin(al))
        u_, l_ = (E_ - root).normalized(), (T - E_).normalized()
        return E_, u_, l_, _frame(u_, l_) @ A.transposed()
    if natural is not None and bend_deg > 20.0:
        kind, fwd, up = natural[:3]
        excess = natural[3] if len(natural) > 3 else None        # upper-joint range check (shoulder / hip cone), deg outside
        lim = POINT_LIMIT[kind]

        def cost(pp):
            E_, u_, l_, R_ = pose_for(pp)
            _u, p_, _b = point_dir(root, E_, T)
            dv = math.degrees(p_.angle(natural_point(kind, u_, fwd, up)))
            ex = excess(R_) if excess is not None else 0.0
            return max(0.0, dv - lim) + max(0.0, ex - 3.0), dv, ex
        if cost(perp)[0] > 0.0:
            # swing the bend plane around the root-end line to the NEAREST direction that is anatomically possible
            best = None
            for k in range(1, 121):
                for sgn in (1.0, -1.0):
                    pp = Quaternion(ax, sgn * math.radians(3.0 * k)) @ perp
                    if cost(pp)[0] <= 0.0:
                        best = pp
                        break
                if best is not None:
                    break
            if best is None:
                best = min((Quaternion(ax, math.radians(3.0 * k)) @ perp for k in range(120)), key=lambda pp: cost(pp)[0])
            perp = best.normalized()
            LIMB_CLAMPED[rig.name] = LIMB_CLAMPED.get(rig.name, 0) + 1
    LIMB_PREV[key] = perp.copy()
    E, u_dir, l_dir, R_u = pose_for(perp)
    R_l = R_u @ rel_rest @ Quaternion(h, phi).to_matrix()
    reach = E + (R_l @ Vector((0.0, 1.0, 0.0))) * b
    return R_u, R_l, phi, reach


def write_rotation(rig, name, R_world_arm, R_parent_arm):
    """set a pose bone's local rotation so that its armature-space rotation is R_world_arm, given the parent's armature-space rotation"""
    B, PB = rig.data.bones, rig.pose.bones
    b = B[name]
    if b.parent:
        rest_rel = b.parent.matrix_local.to_3x3().inverted() @ b.matrix_local.to_3x3()
        Qb = (R_parent_arm @ rest_rel).inverted() @ R_world_arm
    else:
        Qb = b.matrix_local.to_3x3().inverted() @ R_world_arm
    pb = PB[name]
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = Qb.to_quaternion()
    pb.location = (0.0, 0.0, 0.0)
