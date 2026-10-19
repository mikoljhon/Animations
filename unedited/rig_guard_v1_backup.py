# rig_guard.py : the character is not allowed to move in a way that would break it.
#   After a pose is solved (IK + controls), EVERY joint is projected onto its anatomical range (Motion Assist `limits`: elbows and knees are
#   one-way hinges, hips/shoulders are limited cones with limited twist, ankles and wrists bend a little, spine/neck/head have ranges) and the
#   result is written as plain FK rotations; the IK / copy-rotation constraints are muted so nothing can push a joint past its range afterwards.
#   exec'd in Blender after kd_ctrl.py (needs bpy, math, importlib, Matrix, Quaternion, kd_apply)
import bpy, math, importlib
from mathutils import Matrix, Quaternion

_limits = importlib.import_module("bl_ext.user_default.motion_assist.limits")
GUARD = {}
DEBUG = {}
BASE_RANGES = {}
MAX_TURN = {'spine': 30.0, 'chest': 30.0, 'neck': 40.0, 'head': 45.0, 'clavicle': 35.0, 'upperarm': 55.0, 'forearm': 65.0, 'hand': 60.0,
            'thigh': 55.0, 'shin': 70.0, 'foot': 60.0, 'toe': 70.0}   # degrees per frame at 24 fps: no joint may snap faster than this


def _depth(b):
    d = 0
    while b.parent:
        d += 1
        b = b.parent
    return d


def guard_setup(rig):
    """call once per rig: the processing order and the constraints to toggle; hides the meshes while baking (the armature alone is cheap to evaluate)"""
    B = rig.data.bones
    names = [b.name for b in sorted(B, key=_depth) if not b.name.startswith("ik_") and b.name != 'root']
    cons = []
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type in ('IK', 'COPY_ROTATION'):
                cons.append(c)
    limbs = {}
    for sd in 'LR':
        for nm, chain in (('arm', ('upperarm.' + sd, 'forearm.' + sd, 'hand.' + sd)), ('leg', ('thigh.' + sd, 'shin.' + sd, 'foot.' + sd))):
            ikc = [c for c in rig.pose.bones[chain[1]].constraints if c.type == 'IK']
            if ikc:
                limbs[nm + sd] = dict(con=ikc[0], base=ikc[0].pole_angle, chain=chain, last=0.0)
    lims = _limits.rig_limits(rig)
    for n_, lim_ in lims.items():
        if n_.startswith('thigh'):
            lim_.ranges = (135, 30, 70, 30, -70, 60)         # hip: flexion 135, extension 30, abduction 70, adduction 30, rotation -70..60 (high kicks)
        elif n_.startswith('foot'):
            # Relic's foot rests pitched down about 49 degrees (a digitigrade, heel-up foot), so a human 'square to the shin' neutral is wrong for
            # him: measure the ankle from HIS rest foot instead (a flat stance foot = rest) and give it a human-sized range around that
            lim_.ref = None
            lim_.ranges = (35, 45, 25, 25, -30, 30)
    GUARD[rig.name] = dict(names=names, cons=cons, lims=lims, limbs=limbs, twist={}, prev_rel={}, stats=dict(frames=0, clamped=0, worst=0.0, by_bone={}, swivel=0))
    return GUARD[rig.name]


def _mute(g, state):
    for c in g['cons']:
        c.mute = state


def set_meshes_hidden(hide):
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.hide_viewport = hide


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp_cont(q, lim, prev_twist):
    """like Motion Assist's clamp_rotation, but the roll about the bone's own axis is UNWRAPPED against the previous frame, so a joint whose roll drifts
    past 180 degrees is held at its limit instead of snapping to the opposite limit (that snap showed up as an arm spinning half a turn in one frame).
    Returns (clamped quaternion, unwrapped twist used)"""
    Y = _limits.Y
    qq = lim.ref.inverted() @ q if lim.ref is not None else q
    sw, tw = _limits.split(qq)
    t = _limits.twist_angle(tw)
    if prev_twist is not None:
        t = prev_twist + _wrap(t - prev_twist)
    r = [math.radians(x) for x in lim.ranges]
    tc = min(max(t, r[-2]), r[-1])
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


def _excess(rig, g, bones):
    """total excess (degrees) of the given bones, from the evaluated pose, relative to their evaluated parents"""
    B, PB = rig.data.bones, rig.pose.bones
    tot = 0.0
    for n in bones:
        lim = g['lims'].get(n)
        if lim is None:
            continue
        pn = lim.parent
        rel_pose = PB[pn].matrix.to_3x3().normalized().inverted() @ PB[n].matrix.to_3x3().normalized()
        rel_rest = B[pn].matrix_local.to_3x3().inverted() @ B[n].matrix_local.to_3x3()
        q = (rel_rest.inverted() @ rel_pose).to_quaternion()
        e = math.degrees(q.normalized().rotation_difference(_limits.clamp_rotation(q, lim)).angle)
        tot += min(e, 360.0 - e)
    return tot


SWIVEL = [0.0] + [d * s for d in range(15, 181, 15) for s in (1, -1)]


def _best_swivel(rig, g, key, tol):
    """turn the knee / elbow around the line from the root of the limb to its end until every joint of the limb is inside its range (the hand or foot
    stays exactly on its target; only the bend direction changes). The bend direction is kept CONTINUOUS: it may only turn a little from frame to
    frame (a bigger turn is allowed only when a smaller one cannot keep the limb inside its range)."""
    L = g['limbs'][key]
    con, base, chain = L['con'], L['base'], L['chain']
    last = L['last']

    def cost(delta):
        con.pole_angle = base + math.radians(delta)
        bpy.context.view_layer.update()
        return _excess(rig, g, chain)

    ex0 = cost(last)
    if ex0 <= tol:
        return last
    tiers = [(8, 16, 24, 32, 40), (60, 80, 100), (130, 160, 180)]
    if not L.get('init'):
        tiers = [tuple(range(15, 181, 15))]               # first frame: look all the way round once
    best = (ex0, last)
    for tier in tiers:
        for d in tier:
            for sg in (1, -1):
                c = cost(last + sg * d)
                if c + 1.2 * d < best[0] + 1.2 * abs(best[1] - last) - 1e-6:
                    best = (c, last + sg * d)
        if best[0] <= tol:
            break
    if DEBUG.get('on') and abs(best[1] - last) > 45:
        print('SWJUMP', key, 'from %.0f to %.0f' % (last, best[1]), 'excess before %.0f after %.0f' % (ex0, best[0]))
    L['init'] = True
    con.pole_angle = base + math.radians(best[1])
    bpy.context.view_layer.update()
    L['last'] = best[1]
    g['stats']['swivel'] += 1
    return best[1]


def apply_safe(rig, f, P, key=True, tol=1.0):
    """solve the pose for params P, clamp every joint into its anatomical range, bake as FK rotations at frame f. Returns the worst clamp (degrees)."""
    g = GUARD.get(rig.name) or guard_setup(rig)
    _mute(g, False)
    kd_apply(rig, f, P, key=False)
    bpy.context.view_layer.update()
    if 'wrel_L' in P:                                  # re-aim the hands from the shoulders the rig really has (they ride on the leaning, twisting chest)
        P = dict(P)
        for sd in 'LR':
            shw = rig.matrix_world @ rig.pose.bones['upperarm.' + sd].head
            P['w' + sd] = shw + P['wrel_' + sd]
            P['elbow_' + sd] = shw + P['erel_' + sd]
        kd_apply(rig, f, P, key=False)
        bpy.context.view_layer.update()
    B, PB = rig.data.bones, rig.pose.bones
    for key in g['limbs']:
        if _excess(rig, g, g['limbs'][key]['chain']) > tol * 3:
            _best_swivel(rig, g, key, tol * 3)
        else:
            g['limbs'][key]['last'] = g['limbs'][key]['last']
    Reval = {n: PB[n].matrix.to_3x3().normalized() for n in g['names']}
    Rroot = PB['root'].matrix.to_3x3().normalized()
    Rnew = {}
    worst = 0.0
    clamped_bones = []
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
            if n.startswith(('thigh', 'upperarm')):
                # when a hip or shoulder is swung far, the reported roll about its own axis grows without the joint doing anything unnatural
                # (a high kick turns the hip out well past 45 degrees): let the twist range grow with the swing, up to a hard cap
                sw_, _tw = _limits.split(q)
                th_ = math.degrees(math.acos(max(-1.0, min(1.0, _limits.Y.dot(sw_ @ _limits.Y)))))
                base = BASE_RANGES.setdefault(n, tuple(lim.ranges))
                grow = max(0.0, th_ - 50.0)
                cap = 135.0 if n.startswith('thigh') else 120.0
                lim.ranges = tuple(base[:4]) + (max(base[4], -min(cap, abs(base[4]) + grow)), min(cap, base[5] + grow))
            qc, tc_ = clamp_cont(q, lim, g['twist'].get(n))
            if DEBUG.get('watch') and n == DEBUG['watch'][0] and f in DEBUG['watch'][1]:
                sw_d, tw_d = _limits.split(q)
                print('WATCH', f, n, 'q', tuple(round(x, 2) for x in q), 'twist raw %.0f' % math.degrees(_limits.twist_angle(tw_d)), 'twist used %.0f' % math.degrees(tc_), 'swing %.0f' % math.degrees(math.acos(max(-1, min(1, _limits.Y.dot(sw_d @ _limits.Y))))), 'swivel', g['limbs'].get('armL', {}).get('last'), 'legR swivel', g['limbs'].get('legR', {}).get('last'), 'ranges', lim.ranges)
            g['twist'][n] = tc_
            e = math.degrees(q.normalized().rotation_difference(qc).angle)
            e = min(e, 360.0 - e)
            if e > 60 and DEBUG.get('on'):
                sw, tw = _limits.split(q)
                y2 = sw @ _limits.Y
                print('GDBG frame', f, n, 'excess %.0f' % e, 'swing %.0f deg' % math.degrees(math.acos(max(-1, min(1, _limits.Y.dot(y2))))), 'twist %.0f deg' % math.degrees(_limits.twist_angle(tw)), 'swing dir (a1 %.2f, a2 %.2f)' % (y2.dot(lim.a1) if lim.a1 is not None else 0, y2.dot(lim.a2) if lim.a2 is not None else 0), 'ranges', lim.ranges)
            if e > tol:
                clamped_bones.append((n, e))
                worst = max(worst, e)
                st = g['stats']['by_bone'].setdefault(n, [0, 0.0])
                st[0] += 1
                st[1] = max(st[1], e)
            rel_pose = rel_rest @ qc.to_matrix()
        # rate limit: a joint cannot turn more than a human joint can in one frame; a bigger jump is spread over the next frames
        mt = MAX_TURN.get(n.split('.')[0])
        prev = g['prev_rel'].get(n)
        if mt is not None and prev is not None:
            qn, qp = rel_pose.to_quaternion(), prev.to_quaternion()
            if qn.dot(qp) < 0.0:
                qn = -qn
            ang = math.degrees(qp.rotation_difference(qn).angle)
            ang = min(ang, 360.0 - ang)
            if ang > mt:
                rel_pose = qp.slerp(qn, mt / ang).to_matrix()
                g['stats']['limited'] = g['stats'].get('limited', 0) + 1
        g['prev_rel'][n] = rel_pose.copy()
        Rb_new = Rp_new @ rel_pose
        Rnew[n] = Rb_new
        # basis rotation: Rb = Rp @ (Rrest_p^-1 Rrest_b) @ Qbasis
        rest_rel = (B[pn].matrix_local.to_3x3().inverted() if pn else Matrix.Identity(3)) @ B[n].matrix_local.to_3x3()
        Qb = (Rp_new @ rest_rel).inverted() @ Rb_new
        pb = PB[n]
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = Qb.to_quaternion()
        if n != 'pelvis':
            pb.location = (0.0, 0.0, 0.0)
    _mute(g, True)
    g['stats']['frames'] += 1
    if clamped_bones:
        g['stats']['clamped'] += 1
    g['stats']['worst'] = max(g['stats']['worst'], worst)
    if key:
        for n in g['names']:
            PB[n].keyframe_insert('rotation_quaternion', frame=f)
        PB['pelvis'].keyframe_insert('location', frame=f)
        for n in ('fist1.L', 'fist2.L', 'fist1.R', 'fist2.R'):
            if n in PB and n not in g['names']:
                PB[n].keyframe_insert('rotation_quaternion', frame=f)
        for n in ('ik_foot.L', 'ik_foot.R', 'ik_hand.L', 'ik_hand.R'):
            if n in PB:
                PB[n].keyframe_insert('location', frame=f)
    return worst


def guard_report(rig):
    st = GUARD[rig.name]['stats']
    s = "GUARD frames %d, frames where a joint had to be pulled back %d, worst pull-back %.0f deg, joint turns slowed to a human speed %d times" % (st['frames'], st['clamped'], st['worst'], st.get('limited', 0))
    for b, (n, w) in sorted(st['by_bone'].items(), key=lambda kv: -kv[1][1]):
        s += "\n  %-12s pulled back on %3d frames, worst %3.0f deg" % (b, n, w)
    return s
