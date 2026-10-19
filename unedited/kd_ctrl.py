# kd_ctrl.py : direct, continuous control of the KillDroid rig (no solver).  exec'd by fight_anim.py
#   - torso (pelvis/spine/chest/neck/head) is posed with plain rotation algebra from lean / side lean / twist / head look
#   - legs: the rig's own IK (ik_foot.* + ik_knee.*), arms: new IK bones (ik_hand.*, ik_elbow.*) built here
#   params per frame P: hips (world), yaw, lean (fwd, side), tw, look (world point or None), wL wR aL aR (world), fyaw (L, R)
import bpy, math
from mathutils import Vector, Matrix, Quaternion


def kd_add_arm_ik(rig):
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    if 'ik_hand.L' not in eb:
        root = eb['root']
        for s in 'LR':
            fa, ha = eb['forearm.' + s], eb['hand.' + s]
            h = eb.new('ik_hand.' + s)
            h.head, h.tail = fa.tail.copy(), ha.tail.copy()
            h.parent = root
            e = eb.new('ik_elbow.' + s)
            e.head = fa.head + Vector((0.0, 1.0, 0.0))
            e.tail = e.head + Vector((0.0, 0.0, 0.1))
            e.parent = root
    bpy.ops.object.mode_set(mode='OBJECT')
    for s in 'LR':
        pb = rig.pose.bones['forearm.' + s]
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        ik = pb.constraints.new('IK')
        ik.target = rig
        ik.subtarget = 'ik_hand.' + s
        ik.pole_target = rig
        ik.pole_subtarget = 'ik_elbow.' + s
        ik.pole_angle = math.radians(-90)
        ik.chain_count = 2


def _dims(rig):
    """rest shoulder offset and hip half-width from the bones, plus an arm-length scale (KillDroid = 1.0)"""
    bn = rig.data.bones
    pel = bn['pelvis'].head_local
    sh = bn['upperarm.L'].head_local - pel
    hip = bn['thigh.L'].head_local - pel
    arm = bn['upperarm.L'].length + bn['forearm.L'].length
    return sh, abs(hip.x), arm / 1.2


def _basis(rest_b, D):
    return rest_b.inverted() @ D


def _fit_chest(rest, R0, Rp, H, target):
    """spine fit: the chest rotation (armature space) = the real thorax frame R0 tilted about its own left axis (the shoulder line) by the angle
    that puts the neck on the real hips -> neck line from the pelvis head H; the spine takes half of the way from the pelvis (golden section)"""
    tgt = target.normalized()
    Dp = Matrix.Translation(H) @ (Rp @ rest['pelvis'].to_3x3()).to_4x4()
    inh_s = Dp @ rest['pelvis'].inverted() @ rest['spine']
    qp = Rp.to_quaternion()

    def err(dl):
        Rc = R0 @ Matrix.Rotation(dl, 3, 'X')
        Rs = qp.slerp(Rc.to_quaternion(), 0.5).to_matrix()
        Ds = Matrix.Translation(inh_s.to_translation()) @ (Rs @ rest['spine'].to_3x3()).to_4x4()
        inh_c = Ds @ rest['spine'].inverted() @ rest['chest']
        Dc = Matrix.Translation(inh_c.to_translation()) @ (Rc @ rest['chest'].to_3x3()).to_4x4()
        neck = (Dc @ rest['chest'].inverted() @ rest['neck']).to_translation()
        return (neck - H).angle(tgt, math.pi)
    a, b = math.radians(-35.0), math.radians(70.0)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c_, d_ = b - g * (b - a), a + g * (b - a)
    fc, fd = err(c_), err(d_)
    for _ in range(30):
        if fc < fd:
            b, d_, fd = d_, c_, fc
            c_ = b - g * (b - a)
            fc = err(c_)
        else:
            a, c_, fc = c_, d_, fd
            d_ = a + g * (b - a)
            fd = err(d_)
    return R0 @ Matrix.Rotation(0.5 * (a + b), 3, 'X')


def kd_apply(rig, f, P, key=True):
    """pose the rig for params P at frame f and key it"""
    Mi = rig.matrix_world.inverted()
    L = lambda p: Mi @ Vector(p)
    pb = rig.pose.bones
    bn = rig.data.bones
    rest = {n: bn[n].matrix_local.copy() for n in ('root', 'pelvis', 'spine', 'chest', 'neck', 'head', 'ik_foot.L', 'ik_foot.R', 'ik_knee.L', 'ik_knee.R',
                                                     'ik_hand.L', 'ik_hand.R', 'ik_elbow.L', 'ik_elbow.R')}
    Rz = lambda a: Matrix.Rotation(a, 3, 'Z')
    Rx = lambda a: Matrix.Rotation(a, 3, 'X')
    Ry = lambda a: Matrix.Rotation(a, 3, 'Y')
    sh_off, hip_x, ascale = _dims(rig)
    yaw = P['yaw']
    th, ph = P['lean']
    ta = P.get('tw', 0.0)
    Rb = Rz(yaw)

    def torso(c):
        return Rb @ (Rz(ta * c) @ Ry(ph * c) @ Rx(th * c))

    H = L(P['hips'])
    # pelvis
    if 'plean' in P:                                  # the real pelvis tilt (a high kick tilts the pelvis back 30+ degrees)
        Rp = Rb @ (Ry(P['plean'][1]) @ Rx(P['plean'][0]))
    else:
        Rp = torso(0.15)
    D = Matrix.Translation(H) @ (Rp @ rest['pelvis'].to_3x3()).to_4x4()
    bs = _basis(rest['pelvis'], D)
    pb['pelvis'].location, pb['pelvis'].rotation_quaternion = bs.to_translation(), bs.to_quaternion()
    Pp = D
    prev_name = 'pelvis'
    # spine fit (P['spine_fit'], the real thorax frame from mocap_keys): 0.85 of the lean on the chest falls 15-20 deg short in a deep bend
    # (a pick-up, a kneel) and a side lean applied after a deep forward lean tips the shoulder line; instead the chest copies the real shoulder
    # line and is tilted about it until the pelvis -> neck line is the real one
    Rchest = None
    if P.get('spine_fit') and 'plean' in P and 'thorax' in P:
        Rchest = _fit_chest(rest, Rb @ P['thorax'], Rp, H, Rb @ P['trunk'])
    for name, c in (('spine', 0.45), ('chest', 0.85), ('neck', 0.85)):
        inh = Pp @ rest[prev_name].inverted() @ rest[name]
        Rc = torso(c) if Rchest is None else Rchest
        if 'plean' in P and name == 'spine':             # the spine sits between the real pelvis and the chest
            Rc = Rp.to_quaternion().slerp((torso(0.85) if Rchest is None else Rchest).to_quaternion(), 0.5).to_matrix()
        D = Matrix.Translation(inh.to_translation()) @ (Rc @ rest[name].to_3x3()).to_4x4()
        bs = inh.inverted() @ D
        pb[name].location, pb[name].rotation_quaternion = (0.0, 0.0, 0.0), bs.to_quaternion()
        Pp, prev_name = D, name
    # head: looks toward the target, limited, blended with the torso direction
    inh = Pp @ rest['neck'].inverted() @ rest['head']
    Rh = torso(0.85) if Rchest is None else Rchest.copy()
    look = P.get('look')
    if look is not None:
        d = (L(look) - inh.to_translation())
        d_local = Rh.inverted() @ d
        # in the design frame the face looks along -Y (rest); yaw about Z, pitch about X
        yw = math.atan2(-d_local.x, -d_local.y)
        pt = math.atan2(d_local.z, math.hypot(d_local.x, d_local.y))
        yw = max(-0.9, min(0.9, yw)) * 0.8
        pt = max(-0.45, min(0.45, pt)) * 0.7
        Rh = Rh @ Rz(yw) @ Rx(pt)
    D = Matrix.Translation(inh.to_translation()) @ (Rh @ rest['head'].to_3x3()).to_4x4()
    bs = inh.inverted() @ D
    pb['head'].location, pb['head'].rotation_quaternion = (0.0, 0.0, 0.0), bs.to_quaternion()
    # clavicles and arms stay in the rest relation to the chest (the arm IK moves the arm bones)
    fwd = Rb @ Vector((0.0, -1.0, 0.0))
    left = Rb @ Vector((1.0, 0.0, 0.0))
    # leg targets
    leg_len = rig.data.bones['thigh.L'].length + rig.data.bones['shin.L'].length
    for s, a_key, fy in (('L', 'aL', P.get('fyaw', (0.0, 0.0))[0]), ('R', 'aR', P.get('fyaw', (0.0, 0.0))[1])):
        A = L(P[a_key])
        hip = H + left * (hip_x if s == 'L' else -hip_x)
        dv = A - hip
        if dv.length > leg_len * 0.985:                 # the foot cannot be farther than the leg is long: keep it reachable (no stretch, no flip)
            A = hip + dv.normalized() * leg_len * 0.985
        fd = P.get('fdelta', {}).get(s)
        if fd is not None:                               # full 3D foot orientation from the motion capture (relative to the flat foot at the start)
            Rw_inv = rig.matrix_world.to_3x3().inverted()
            Rfoot = Rw_inv @ fd @ rig.matrix_world.to_3x3() @ Rz(P['fyaw0'][s]) @ rest['ik_foot.' + s].to_3x3()
            Rfoot = Rfoot
        else:
            Rfoot = Rz(yaw + fy) @ Rx(P.get('fpitch', {}).get(s, 0.0)) @ rest['ik_foot.' + s].to_3x3()
        D = Matrix.Translation(A) @ Rfoot.to_4x4()
        bs = _basis(rest['ik_foot.' + s], D)
        pb['ik_foot.' + s].location, pb['ik_foot.' + s].rotation_quaternion = bs.to_translation(), bs.to_quaternion()
        ko = P.get('knee_' + s)
        if ko is not None:
            kp = L(ko)                                   # real knee position from the motion capture
        else:
            ax = (A - hip)
            axn = ax.normalized() if ax.length > 1e-6 else Vector((0, 0, -1))
            k_up = max(0.0, min(1.0, (A.z - (hip.z - 0.6 * leg_len)) / (0.8 * leg_len))) * 1.2   # a raised foot: the knee points forward and up
            ffd = Rz(yaw + fy) @ Vector((0.0, -1.0, 0.0))          # the knee points the way the foot points (knee over the toes)
            pref = ffd + Vector((0.0, 0.0, k_up))
            n = pref - axn * pref.dot(axn)
            if n.length < 0.35:
                n = n + (Vector((0, 0, 1)) - axn * axn.z)
            kp = (hip + A) * 0.5 + n.normalized() * (0.9 * ascale)
        D = Matrix.Translation(kp)
        bs = _basis(rest['ik_knee.' + s], D)
        pb['ik_knee.' + s].location, pb['ik_knee.' + s].rotation_quaternion = bs.to_translation(), Quaternion((1, 0, 0, 0))
    # arm targets
    Rt = torso(0.85)
    for s, w_key, sg in (('L', 'wL', 1.0), ('R', 'wR', -1.0)):
        W = L(P[w_key])
        D = Matrix.Translation(W) @ rest['ik_hand.' + s].to_3x3().to_4x4()
        bs = _basis(rest['ik_hand.' + s], D)
        pb['ik_hand.' + s].location, pb['ik_hand.' + s].rotation_quaternion = bs.to_translation(), Quaternion((1, 0, 0, 0))
        shoulder = H + Rt @ Vector((sh_off.x * sg, sh_off.y, sh_off.z))
        arm_len = rig.data.bones['upperarm.L'].length + rig.data.bones['forearm.L'].length
        dw = W - shoulder
        if dw.length > arm_len * 0.985:
            W = shoulder + dw.normalized() * arm_len * 0.985
            D = Matrix.Translation(W) @ rest['ik_hand.' + s].to_3x3().to_4x4()
            bs = _basis(rest['ik_hand.' + s], D)
            pb['ik_hand.' + s].location = bs.to_translation()
        mid = (shoulder + W) * 0.5
        eo = P.get('elbow_' + s)
        ep = L(eo) if eo is not None else mid + Rt @ Vector((0.30 * sg, 0.55, -0.55)) * ascale
        D = Matrix.Translation(ep)
        bs = _basis(rest['ik_elbow.' + s], D)
        pb['ik_elbow.' + s].location, pb['ik_elbow.' + s].rotation_quaternion = bs.to_translation(), Quaternion((1, 0, 0, 0))
    if P.get('fist') is not None and 'set_fists' in globals():
        set_fists(rig, P['fist'], key=key, frame=f)
    if key:
        for n in ('pelvis', 'spine', 'chest', 'neck', 'head', 'ik_foot.L', 'ik_foot.R', 'ik_knee.L', 'ik_knee.R', 'ik_hand.L', 'ik_hand.R', 'ik_elbow.L', 'ik_elbow.R'):
            p = pb[n]
            p.rotation_mode = 'QUATERNION'
            p.keyframe_insert('rotation_quaternion', frame=f)
            if n in ('pelvis', 'ik_foot.L', 'ik_foot.R', 'ik_knee.L', 'ik_knee.R', 'ik_hand.L', 'ik_hand.R', 'ik_elbow.L', 'ik_elbow.R'):
                p.keyframe_insert('location', frame=f)


def kd_calibrate(rig, plist):
    """pick the IK pole angles that turn each knee / elbow toward its pole target over several bent poses (sum of angles off the limb line)"""
    res = {}
    W = lambda n, tail=False: rig.matrix_world @ (rig.pose.bones[n].tail if tail else rig.pose.bones[n].head)
    limbs = (('shin.L', 'thigh.L', 'ik_knee.L'), ('shin.R', 'thigh.R', 'ik_knee.R'),
             ('forearm.L', 'upperarm.L', 'ik_elbow.L'), ('forearm.R', 'upperarm.R', 'ik_elbow.R'))
    tot = {l[0]: {} for l in limbs}
    for P in plist:
        kd_apply(rig, 1, P, key=False)
        for bone, upper, pole in limbs:
            con = [c for c in rig.pose.bones[bone].constraints if c.type == 'IK'][0]
            for deg in range(-180, 180, 15):
                con.pole_angle = math.radians(deg)
                bpy.context.view_layer.update()
                a, j, t, pl = W(upper), W(bone), W(bone, True), W(pole)
                ax = (t - a).normalized()
                dj = (j - a) - ax * (j - a).dot(ax)
                dp = (pl - a) - ax * (pl - a).dot(ax)
                if dj.length < 0.04 or dp.length < 1e-4:
                    continue
                tot[bone][deg] = tot[bone].get(deg, 0.0) + math.degrees(dj.angle(dp)) * min(1.0, dj.length / 0.2)
    for bone, upper, pole in limbs:
        con = [c for c in rig.pose.bones[bone].constraints if c.type == 'IK'][0]
        deg, v = min(tot[bone].items(), key=lambda kv: kv[1])
        con.pole_angle = math.radians(deg)
        res[bone] = (deg, round(v / len(plist), 1))
    bpy.context.view_layer.update()
    return res


def kd_shoulders(rig, P):
    """world positions of the KillDroid's left and right shoulder for params P (same math as kd_apply)"""
    Mi = rig.matrix_world.inverted()
    Rz = lambda a: Matrix.Rotation(a, 3, 'Z')
    Rx = lambda a: Matrix.Rotation(a, 3, 'X')
    Ry = lambda a: Matrix.Rotation(a, 3, 'Y')
    th, ph = P['lean']
    ta = P.get('tw', 0.0)
    Rt = Rz(P['yaw']) @ (Rz(ta * 0.85) @ Ry(ph * 0.85) @ Rx(th * 0.85))
    H = Vector(P['hips'])
    sh_off = _dims(rig)[0]
    Rw = rig.matrix_world.to_3x3()                   # the rig may be turned in the scene (Relic is, half a circle)
    return [H + Rw @ (Rt @ Vector((sh_off.x * sg, sh_off.y, sh_off.z))) for sg in (1.0, -1.0)]
