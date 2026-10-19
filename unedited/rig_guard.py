# rig_guard.py (v2) : the character is not allowed to move in a way that would break it, AND it follows the real motion exactly.
#   Per frame:
#     1. the torso (pelvis, spine, chest, neck, head) is posed from the params (kd_apply; all IK constraints stay muted)
#     2. arms and legs are solved EXACTLY with limb_solver: hand/foot on target, elbow/knee pointing the way the real capture's did,
#        the elbow/knee only bending the natural way inside its hinge range (it cannot bend backwards)
#     3. every joint is projected onto its anatomical range (Motion Assist limits; ankle measured from the rig's own rest foot), the roll unwrapped
#        so it cannot snap, and every joint is speed-limited to what a human joint can do in one frame
#     4. the result is keyed as plain FK rotations
#   exec'd in Blender after kd_ctrl.py (needs bpy, math, importlib, kd_apply)
#   v1 (IK + elbow swivel search + clamps) is kept as rig_guard_v1_backup.py: its swivel search steered elbows AWAY from the real ones.
import bpy, math, importlib
from mathutils import Matrix, Quaternion, Vector

_limits = importlib.import_module("bl_ext.user_default.motion_assist.limits")
exec(open(r"C:\ai\unedited\limb_solver.py", encoding="utf-8").read())
GUARD = {}
DEBUG = {}
MAX_TURN = {'spine': 30.0, 'chest': 30.0, 'neck': 40.0, 'head': 45.0, 'clavicle': 35.0, 'upperarm': 55.0, 'forearm': 65.0, 'hand': 60.0,
            'thigh': 55.0, 'shin': 70.0, 'foot': 60.0, 'toe': 70.0}   # degrees per frame at 24 fps: no joint may snap faster than this
# shoulders and hips: the exact solver already gives them the real person's orientation; the generic human table's roll reference does not fit
# these rigs (it 'corrected' natural poses into wrong ones), so for them only the swing cone is enforced and the roll follows the real motion
SWING_ONLY = ('upperarm', 'thigh')


def _depth(b):
    d = 0
    while b.parent:
        d += 1
        b = b.parent
    return d


def guard_setup(rig):
    B = rig.data.bones
    names = [b.name for b in sorted(B, key=_depth) if not b.name.startswith("ik_") and b.name != 'root']
    cons = []
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type in ('IK', 'COPY_ROTATION'):
                cons.append(c)
                c.mute = True                         # the exact solver replaces them, for good
    lims = _limits.rig_limits(rig)
    for n_, lim_ in lims.items():
        if n_.startswith('foot'):
            # Relic's (and the KillDroid's) foot rests pitched down (a heel-up foot): measure the ankle from the rig's own rest foot
            lim_.ref = None
            lim_.ranges = (35, 45, 25, 25, -30, 30)
    GUARD[rig.name] = dict(names=names, cons=cons, lims=lims, twist={}, prev_rel={},
                           stats=dict(frames=0, clamped=0, worst=0.0, by_bone={}, limited=0))
    return GUARD[rig.name]


def set_meshes_hidden(hide):
    for o in bpy.data.objects:
        # the KillDroid's own parts stay evaluated: its arms are kept out of its hull against them (kd_clear_arm)
        if o.type == 'MESH' and not (o.parent is not None and o.parent.name == "KillDroid_Rig"):
            o.hide_viewport = hide


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp_cont(q, lim, prev_twist, swing_only=False):
    """Motion Assist's clamp, but the roll is unwrapped against the previous frame (no snapping to the opposite limit); with swing_only the roll
    is left alone. Returns (clamped quaternion, roll used)"""
    Y = _limits.Y
    qq = lim.ref.inverted() @ q if lim.ref is not None else q
    sw, tw = _limits.split(qq)
    t = _limits.twist_angle(tw)
    if prev_twist is not None:
        t = prev_twist + _wrap(t - prev_twist)
    r = [math.radians(x) for x in lim.ranges]
    tc = t if swing_only else min(max(t, r[-2]), r[-1])
    y2 = sw @ Y
    cl = _limits._clamp
    if lim.kind == "hinge":
        phi = cl(_limits.hinge_angle(y2, lim.hinge), r[0], r[1])
        sl = math.radians(lim.slack)
        psi = cl(math.asin(cl(y2.dot(lim.hinge), -1.0, 1.0)), -sl, sl)
        sw2 = Y.rotation_difference((Quaternion(lim.hinge, phi) @ Y) * math.cos(psi) + lim.hinge * math.sin(psi))
    else:
        theta = math.acos(cl(Y.dot(y2), -1.0, 1.0))
        d = y2 - Y * y2.dot(Y)
        sw2 = sw
        if d.length > 1e-9 and theta > 1e-6:
            d.normalize()
            a, b = d.dot(lim.a1), d.dot(lim.a2)
            A = max(r[0] if a >= 0 else r[1], 1e-3)
            Bv = max(r[2] if b >= 0 else r[3], 1e-3)
            tmax = (abs(a / A) ** _limits.CONE_POWER + abs(b / Bv) ** _limits.CONE_POWER) ** (-1.0 / _limits.CONE_POWER)
            if theta > tmax:
                sw2 = Quaternion(Y.cross(d).normalized(), tmax)
    out = sw2 @ Quaternion(Y, tc)
    if lim.ref is not None:
        out = lim.ref @ out
    return out, tc


exec(open(r"C:\ai\unedited\body_clear.py", encoding="utf-8").read())


def relic_body_frame(rig):
    """Relic's torso frame (world): origin pelvis head, unit axis to the neck, its length, forward, left; and the helmet centre + radius"""
    PB, B = rig.pose.bones, rig.data.bones
    M = rig.matrix_world
    o = M @ PB['pelvis'].head
    ax = (M @ PB['neck'].head) - o
    alen = ax.length
    up = ax / max(alen, 1e-9)
    lf = (M @ PB['upperarm.L'].head) - (M @ PB['upperarm.R'].head)
    lf = (lf - up * lf.dot(up)).normalized()
    fw = lf.cross(up).normalized()
    bd = relic_body()
    if 'head_rest' not in bd:
        P0, P1 = B['pelvis'].head_local, B['neck'].head_local
        u0 = (P1 - P0).normalized()
        l0 = B['upperarm.L'].head_local - B['upperarm.R'].head_local
        l0 = (l0 - u0 * l0.dot(u0)).normalized()
        f0 = l0.cross(u0).normalized()
        h = bd['head']
        bd['head_rest'] = P0 + u0 * (h['center_t'] * bd['alen']) + f0 * h['center_f'] + l0 * h['center_l']
    hc = M @ ((PB['head'].matrix @ B['head'].matrix_local.inverted()) @ bd['head_rest'])
    sc_ = M.to_scale().x                       # the profile was measured in the rig's own units
    return o, up, alen, fw, lf, hc, bd['head']['radius'] * sc_, sc_


def fist_clearance(rig, F, frame=None):
    """(torso clearance or None, helmet clearance) of a world point F outside Relic's armour"""
    o, up, alen, fw, lf, hc, hr, sc_ = frame or relic_body_frame(rig)
    d = F - o
    c, _rs, _dr = torso_clear(d.dot(up) / alen, d.dot(fw), d.dot(lf), relic_body()['table'], sc_)
    return c, (F - hc).length - hr


def keep_fist_clear(rig, Ww, hdir, hcl):
    """move the wrist target Ww (world) so the fist stays at least as far outside Relic's armour as the fighter's fist was outside theirs
    (clamped to MIN_CLEAR..MAX_CLEAR); only ever pushes outward"""
    o, up, alen, fw, lf, hc, hr, sc_ = relic_body_frame(rig)
    ct_h, ch_h = hcl
    F = Ww + hdir * FIST_OFF
    req = MIN_CLEAR if ct_h is None else min(max(ct_h, MIN_CLEAR), MAX_CLEAR)
    d = F - o
    t, f, l = d.dot(up) / alen, d.dot(fw), d.dot(lf)
    c, rs, dr = torso_clear(t, f, l, relic_body()['table'], sc_)
    if c is not None and c < req:
        nr = rs + req
        F = F + fw * (dr[0] * nr - f) + lf * (dr[1] * nr - l)
    req_h = min(max(ch_h, MIN_CLEAR), MAX_CLEAR)
    dh = F - hc
    if dh.length < hr + req_h:
        F = hc + (dh.normalized() if dh.length > 1e-6 else fw) * (hr + req_h)
    return F - hdir * FIST_OFF


# ------------------------------------------------------------------------------------------------ the KillDroid's arms and its own hull
# Its cab, engine hump and turret are rigid parts on the torso bones, its forearms as thick as a man's waist, its crusher half a metre long:
# a human arm path that hugs a slim body (a guard, a hanging arm, a curl) cuts straight into it. After each arm solve, the forearm / hand /
# jaw parts are tested polygon against polygon with the hull as posed (the merged parts have no usable inside / outside, so no signed
# distances); if they meet, the wrist target is moved along ONE outward direction (from the hull polygons in contact toward the arm's own
# centre; radial from the spine axis if that points inward) by the least distance that clears it (doubling, then bisection to ~1.5 cm, so
# the push changes smoothly from frame to frame) and the arm is solved again. Upper-arm parts are not tested: they meet the cab at the
# shoulder drums by design.
KD_HULL_BONES = ("pelvis", "spine", "chest", "neck", "head")
KD_SPARE = 0.02              # m: pushed this much past touching, so the joint clamp that follows cannot turn the arm back in
_KD_GEO = {}


def _bone_rest_geometry(rig, objs):
    """vertices (numpy, the rig's REST armature space), polygons and the bone of each vertex, for bone-parented parts"""
    import numpy as np
    V, Pl, owner = [], [], []
    for o in objs:
        b = rig.data.bones[o.parent_bone]
        M = b.matrix_local @ Matrix.Translation((0.0, b.length, 0.0)) @ o.matrix_parent_inverse @ o.matrix_basis
        n0 = len(V)
        V += [(M @ v.co)[:] for v in o.data.vertices]
        Pl += [tuple(i + n0 for i in p.vertices) for p in o.data.polygons]
        owner += [o.parent_bone] * len(o.data.vertices)
    return np.array(V), Pl, owner


def kd_geometry(rig):
    if rig.name not in _KD_GEO:
        import numpy as np
        parts = [o for o in rig.children if o.type == 'MESH' and o.parent_type == 'BONE']
        # the hull: every part on a torso bone, rams included, read as evaluated (object space here, placed per frame by kd_hull_now)
        hull = []
        for o in parts:
            if o.parent_bone in KD_HULL_BONES:
                hull.append((o, np.array([v.co[:] for v in o.data.vertices]), [tuple(p.vertices) for p in o.data.polygons]))
        hp, n0 = [], 0
        for o, v_, p_ in hull:
            hp += [tuple(i + n0 for i in p) for p in p_]
            n0 += len(v_)
        arms = {}
        for sd in 'LR':
            # the arm: its rigid parts at rest (the rams on it aim themselves, so they are left to the QA gate)
            av, ap, _o = _bone_rest_geometry(rig, [o for o in parts if not o.constraints and (o.parent_bone in ('forearm.' + sd, 'hand.' + sd) or
                                                                                         (sd == 'R' and o.parent_bone.startswith('claw')))])
            arms[sd] = (av, ap)
        _KD_GEO[rig.name] = (hull, hp, arms)
    return _KD_GEO[rig.name]


def _xf(M, V):
    """a 4x4 Matrix applied to an (n, 3) numpy array"""
    import numpy as np
    A = np.array(M)
    return V @ A[:3, :3].T + A[:3, 3]


def kd_hull_now(rig):
    """the hull as posed now (armature space; the torso was posed and evaluated before the limbs): (BVH, vertices, polygons)"""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    hull, hp, _arms = kd_geometry(rig)
    Mi = rig.matrix_world.inverted()
    V = np.concatenate([_xf(Mi @ o.matrix_world, v_) for o, v_, _p in hull])
    return BVHTree.FromPolygons(V.tolist(), hp), V, hp


def kd_arm_now(rig, sd, S, R_u, R_l):
    """this arm's forearm / hand / jaw parts for the solve R_u, R_l from the shoulder S (armature space): (BVH, vertices)"""
    from mathutils.bvhtree import BVHTree
    B = rig.data.bones
    av, ap = kd_geometry(rig)[2][sd]
    E = S + (R_u @ Vector((0.0, 1.0, 0.0))) * B['upperarm.' + sd].length
    V = _xf(Matrix.Translation(E) @ R_l.to_4x4() @ B['forearm.' + sd].matrix_local.inverted(), av)    # the whole forearm chain at rest
    return BVHTree.FromPolygons(V.tolist(), ap), V


def kd_clear_arm(rig, hull_, sd, S, W, solve):
    """(R_u, R_l, wrist target, push length): the arm solved for W, or for W moved the least distance that takes it out of the hull"""
    tree_h, Vh, Ph = hull_
    R_u, R_l = solve(W)
    t_a, Va = kd_arm_now(rig, sd, S, R_u, R_l)
    pairs = tree_h.overlap(t_a)
    if not pairs:
        return R_u, R_l, W, 0.0
    PB = rig.pose.bones
    ch = Vector(sum((Vector(Vh[Ph[i][0]]) for i, _j in pairs), Vector()) / len(pairs))
    ca = Vector(Va.mean(axis=0))
    o_, u_ = PB['pelvis'].head, (PB['neck'].head - PB['pelvis'].head).normalized()
    rad = ca - o_
    rad = rad - u_ * rad.dot(u_)
    d = ca - ch
    if d.length < 1e-4 or (rad.length > 1e-4 and d.normalized().dot(rad.normalized()) < -0.2):
        d = rad
    d = d.normalized()
    clear = lambda t: not tree_h.overlap(kd_arm_now(rig, sd, S, *solve(W + d * t))[0])
    lo, hi = 0.0, 0.08
    while hi < 1.3 and not clear(hi):
        lo, hi = hi, hi * 2.0
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        if clear(mid):
            hi = mid
        else:
            lo = mid
    hi += KD_SPARE
    R_u, R_l = solve(W + d * hi)                      # (also leaves the solver's bend-plane memory on the chosen solve)
    return R_u, R_l, W + d * hi, hi


def body_frames(rig, world=False):
    """(forward, up) of the chest (for the arms) and of the pelvis (for the legs), from the posed joints; armature space unless world"""
    PB = rig.pose.bones
    M = rig.matrix_world if world else Matrix.Identity(4)
    H = lambda n: M @ PB[n].head

    def fr(l, u):
        l = l.normalized()
        u = (u - l * u.dot(l)).normalized()
        return l.cross(u).normalized(), u
    arms = fr(H('upperarm.L') - H('upperarm.R'), H('neck') - H('pelvis'))
    legs = fr(H('thigh.L') - H('thigh.R'), H('spine') - H('pelvis'))
    return arms, legs


def _solve_limbs(rig, P):
    """after the torso is posed: arms and legs by the exact solver (armature space). An elbow or knee can never point the
    anatomically impossible way (limb_solver.natural_point), whatever the data asks for."""
    PB = rig.pose.bones
    g = GUARD[rig.name]
    lims = g['lims']
    Mi = rig.matrix_world.inverted()
    (af, au), (lf_, lu) = body_frames(rig)
    B = rig.data.bones

    def range_excess(bone, R_parent):
        """deg by which an armature-space rotation of a shoulder / hip bone would sit outside its swing cone (as clamp_pass sees it)"""
        lim = lims.get(bone)
        if lim is None:
            return lambda R_: 0.0
        rr = B[bone].parent.matrix_local.to_3x3().inverted() @ B[bone].matrix_local.to_3x3()
        base = (R_parent @ rr).inverted()

        def ex(R_):
            q = (base @ R_).to_quaternion()
            qc, _t = clamp_cont(q, lim, None, swing_only=True)
            e = math.degrees(q.rotation_difference(qc).angle)
            return min(e, 360.0 - e)
        return ex
    # stance targets of the feet, in world space, before anything moves
    A_stance = {sd: rig.matrix_world @ PB['ik_foot.' + sd].head for sd in 'LR'}
    # keep the LEG SHAPE: Relic's stance is wider than the fighter's, so a planted leg would straighten; lower (or raise) the pelvis until each
    # planted leg's hip-ankle distance is the fighter's (scaled), i.e. the knees bend as much as the fighter's
    drops = []
    for sd in 'LR':
        w = P.get('aground_' + sd, 0.0) if 'arel_' + sd in P else 0.0
        if w > 0.05:
            Hw = rig.matrix_world @ PB['thigh.' + sd].head
            dt = P['arel_' + sd].length
            v, h2 = Hw.z - A_stance[sd].z, (Hw - A_stance[sd]).xy.length_squared
            if dt * dt > h2:
                drops.append((v - math.sqrt(dt * dt - h2), w))
    if drops:
        dz = max(-0.15, min(0.15, sum(d * w for d, w in drops) / sum(w for d, w in drops)))
        if abs(dz) > 1e-4:
            rest_rot = B['pelvis'].matrix_local.to_3x3()
            PB['pelvis'].location = PB['pelvis'].location + rest_rot.inverted() @ (rig.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, -dz)))
            bpy.context.view_layer.update()
            (af, au), (lf_, lu) = body_frames(rig)

    def arm(sd, hull_=None):
        # arm: true shoulder, wrist target, real elbow direction
        S = PB['upperarm.' + sd].head.copy()
        if 'wrel_' + sd in P:
            Sw = rig.matrix_world @ S
            Ww = Sw + P['wrel_' + sd]
            if 'hcl_' + sd in P and rig.name == "Relic_AI_Rig":
                W0 = Ww
                Ww = keep_fist_clear(rig, Ww, P['hdir_' + sd], P['hcl_' + sd])     # never inside his (bulkier) body
                P['_push_' + sd] = (Ww - W0).length                                  # QA: this fist was moved on purpose
            W = Mi @ Ww
            Eh = Mi @ (Sw + P['erel_' + sd])
        else:
            W = PB['ik_hand.' + sd].head.copy()
            Eh = PB['ik_elbow.' + sd].head.copy()
        R_c = PB['clavicle.' + sd].matrix.to_3x3().normalized()
        R_u, R_l, phi, reach = solve_limb(rig, 'upperarm.' + sd, 'forearm.' + sd, S, W, Eh, lims['forearm.' + sd],
                                          natural=('elbow', af, au, range_excess('upperarm.' + sd, R_c)))
        if hull_ is not None:
            # the KillDroid: its forearm / hand / jaws never inside its own hull (cab, hump, turret)
            solve_ = lambda W_, sd=sd, S=S, Eh=Eh, R_c=R_c: solve_limb(rig, 'upperarm.' + sd, 'forearm.' + sd, S, W_, Eh, lims['forearm.' + sd],
                                                                       natural=('elbow', af, au, range_excess('upperarm.' + sd, R_c)))[:2]
            R_u, R_l, W, moved = kd_clear_arm(rig, hull_, sd, S, W, solve_)
            if 'wrel_' + sd in P:
                P['_push_' + sd] = moved                                             # QA: this hand was moved on purpose
        write_rotation(rig, 'upperarm.' + sd, R_u, PB['clavicle.' + sd].matrix.to_3x3().normalized())
        write_rotation(rig, 'forearm.' + sd, R_l, R_u)
        PB['hand.' + sd].rotation_mode = 'QUATERNION'
        PB['hand.' + sd].rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        PB['hand.' + sd].location = (0.0, 0.0, 0.0)

    def leg(sd):
        # leg: hip joint, ankle target, real knee direction; the foot takes the orientation its control was given
        Hj = PB['thigh.' + sd].head.copy()
        A = Mi @ A_stance[sd]
        if 'arel_' + sd in P:
            # raised foot: from the rig's own hip joint, so the knee bends exactly as much as the fighter's; planted foot: the stance target
            A_air = Mi @ ((rig.matrix_world @ Hj) + P['arel_' + sd])
            A = A_air.lerp(A, P.get('aground_' + sd, 1.0))
        if 'krel_' + sd in P:
            Kh = Mi @ ((rig.matrix_world @ Hj) + P['krel_' + sd])
        else:
            Kh = PB['ik_knee.' + sd].head.copy()
        R_p = PB['pelvis'].matrix.to_3x3().normalized()
        R_t, R_s, phi_k, reach_k = solve_limb(rig, 'thigh.' + sd, 'shin.' + sd, Hj, A, Kh, lims['shin.' + sd],
                                              natural=('knee', lf_, lu, range_excess('thigh.' + sd, R_p)))
        write_rotation(rig, 'thigh.' + sd, R_t, PB['pelvis'].matrix.to_3x3().normalized())
        write_rotation(rig, 'shin.' + sd, R_s, R_t)
        R_f = PB['ik_foot.' + sd].matrix.to_3x3().normalized()
        write_rotation(rig, 'foot.' + sd, R_f, R_s)
    if rig.name == "KillDroid_Rig":
        # its legs first: the rams on its hips aim themselves at the thighs, so the hull is measured as it really stands this frame
        for sd in 'LR':
            leg(sd)
        bpy.context.view_layer.update()
        hull_ = kd_hull_now(rig)
        for sd in 'LR':
            arm(sd, hull_)
    else:
        for sd in 'LR':
            arm(sd)
            leg(sd)


def apply_safe(rig, f, P, key=True, tol=1.0):
    """pose the rig for params P at frame f: torso from P, limbs exact, every joint inside its range and speed, keyed as FK. Returns the worst pull-back."""
    g = GUARD.get(rig.name) or guard_setup(rig)
    kd_apply(rig, f, P, key=False)                     # torso + control bones (constraints muted)
    bpy.context.view_layer.update()
    _solve_limbs(rig, P)
    bpy.context.view_layer.update()
    return clamp_pass(rig, f, key=key, tol=tol)


def clamp_pass(rig, f, key=True, tol=1.0, rate=True):
    """project the CURRENT pose (already evaluated) onto the joint ranges and speed caps and key it"""
    g = GUARD[rig.name]
    B, PB = rig.data.bones, rig.pose.bones
    Reval = {n: PB[n].matrix.to_3x3().normalized() for n in g['names']}
    Rroot = PB['root'].matrix.to_3x3().normalized()
    Rnew = {}
    worst = 0.0
    clamped = False
    for n in g['names']:
        par = B[n].parent
        pn = par.name if par else None
        Rp_eval = Reval[pn] if pn in Reval else Rroot
        Rp_new = Rnew[pn] if pn in Rnew else Rp_eval
        rel_pose = Rp_eval.inverted() @ Reval[n]
        lim = g['lims'].get(n)
        if lim is not None and lim.parent == pn:
            rel_rest = B[pn].matrix_local.to_3x3().inverted() @ B[n].matrix_local.to_3x3()
            q = (rel_rest.inverted() @ rel_pose).to_quaternion()
            qc, tc_ = clamp_cont(q, lim, g['twist'].get(n), swing_only=n.split('.')[0] in SWING_ONLY)
            g['twist'][n] = tc_
            e = math.degrees(q.normalized().rotation_difference(qc).angle)
            e = min(e, 360.0 - e)
            if e > 40 and DEBUG.get('on'):
                sw_d, tw_d = _limits.split(lim.ref.inverted() @ q if lim.ref is not None else q)
                y2 = sw_d @ _limits.Y
                print('GDBG frame', f, n, 'excess %.0f' % e, 'swing %.0f' % math.degrees(math.acos(max(-1, min(1, _limits.Y.dot(y2))))),
                      'twist %.0f' % math.degrees(_limits.twist_angle(tw_d)), 'dir a1 %.2f a2 %.2f' % (y2.dot(lim.a1) if lim.a1 is not None else 0, y2.dot(lim.a2) if lim.a2 is not None else 0), 'ranges', lim.ranges)
            if e > tol:
                clamped = True
                worst = max(worst, e)
                st = g['stats']['by_bone'].setdefault(n, [0, 0.0])
                st[0] += 1
                st[1] = max(st[1], e)
            rel_pose = rel_rest @ qc.to_matrix()
        mt = MAX_TURN.get(n.split('.')[0])
        prev = g['prev_rel'].get(n)
        if rate and mt is not None and prev is not None:
            qn, qp = rel_pose.to_quaternion(), prev.to_quaternion()
            if qn.dot(qp) < 0.0:
                qn = -qn
            ang = math.degrees(qp.rotation_difference(qn).angle)
            ang = min(ang, 360.0 - ang)
            if ang > mt:
                rel_pose = qp.slerp(qn, mt / ang).to_matrix()
                g['stats']['limited'] += 1
        g['prev_rel'][n] = rel_pose.copy()
        Rb_new = Rp_new @ rel_pose
        Rnew[n] = Rb_new
        rest_rel = (B[pn].matrix_local.to_3x3().inverted() if pn else Matrix.Identity(3)) @ B[n].matrix_local.to_3x3()
        Qb = (Rp_new @ rest_rel).inverted() @ Rb_new
        pb = PB[n]
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = Qb.to_quaternion()
        if n != 'pelvis':
            pb.location = (0.0, 0.0, 0.0)
    g['stats']['frames'] += 1
    if clamped:
        g['stats']['clamped'] += 1
    g['stats']['worst'] = max(g['stats']['worst'], worst)
    if key:
        for n in g['names']:
            PB[n].keyframe_insert('rotation_quaternion', frame=f)
        PB['pelvis'].keyframe_insert('location', frame=f)
        for n in ('ik_foot.L', 'ik_foot.R', 'ik_hand.L', 'ik_hand.R', 'ik_elbow.L', 'ik_elbow.R', 'ik_knee.L', 'ik_knee.R'):
            if n in PB:
                PB[n].keyframe_insert('location', frame=f)
    return worst


def guard_report(rig):
    st = GUARD[rig.name]['stats']
    s = "GUARD frames %d, frames where a joint had to be pulled back %d, worst pull-back %.0f deg, joint turns slowed to a human speed %d times" % (
        st['frames'], st['clamped'], st['worst'], st['limited'])
    for b, (n, w) in sorted(st['by_bone'].items(), key=lambda kv: -kv[1][1]):
        s += "\n  %-12s pulled back on %3d frames, worst %3.0f deg" % (b, n, w)
    return s
