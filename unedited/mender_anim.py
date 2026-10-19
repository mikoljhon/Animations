# mender_anim.py : the opening animatic for the Cave scene (9 s, 24 fps): Mender swings the pick, the last strike breaks the rubble open in a
# burst of dust, Relic lights up (ring first, then the eyes) and Mender scoots back with the glove hand up.
# Exec after mender_cave3.py (same call), window scene = Cave. build_opening() builds the keyframes, cameras, markers and dust; it returns a report.
# Shots: S1 wide f1-48, S2 dig f49-96, S3 reaction two-shot f97-168, S4 Relic's face f169-216.

FPS = 24
SWING = {}
BASE_LEAN = (0.30, 0.66, 1.00, 0.60)

POSES = {   # alpha = pick lifted about the shoulder line (deg), lean = y components of pelvis/spine/chest/neck, hip_h = hips above the ankles
    'STRIKE': dict(alpha=0.0, lean=(0.32, 0.72, 1.08, 0.64), hip_h=0.32),
    'REBOUND': dict(alpha=11.0, lean=(0.28, 0.62, 0.95, 0.58), hip_h=0.345),
    'RAISED': dict(alpha=33.0, lean=(0.20, 0.48, 0.74, 0.46), hip_h=0.37),
    'RAISED_HIGH': dict(alpha=42.0, lean=(0.14, 0.38, 0.60, 0.38), hip_h=0.39),
    'STRIKE_BIG': dict(alpha=-3.0, lean=(0.34, 0.78, 1.18, 0.68), hip_h=0.30),
    'PAUSE': dict(alpha=6.0, lean=(0.26, 0.55, 0.85, 0.52), hip_h=0.34),
}


def chest_world(rig):
    bpy.context.view_layer.update()
    return rig.matrix_world @ rig.pose.bones['chest'].matrix


def swing_base(rig):
    """call right after place_mender() (pick free in the world): remember the pick and the shoulder line relative to the chest"""
    pick = bpy.data.objects[CP + "Pick"]
    Mc = chest_world(rig)
    Mi = Mc.inverted()
    SWING['Rel'] = Mi @ pick.matrix_world
    SWING['SL'] = Mi @ shoulder_world(rig, 'L')
    SWING['SR'] = Mi @ shoulder_world(rig, 'R')
    SWING['REL'] = hand_matrix(rig, 'R').inverted() @ pick.matrix_world       # the pick relative to the mechanical hand that holds it
    head = Vector((0.0, 0.0, 0.8))
    z = {}
    for sgn in (1.0, -1.0):
        SRw, SLw = Mc @ SWING['SR'], Mc @ SWING['SL']
        axis = (SLw - SRw).normalized()
        Rm = Matrix.Translation(SRw) @ Matrix.Rotation(math.radians(10.0 * sgn), 4, axis) @ Matrix.Translation(-SRw)
        z[sgn] = ((Rm @ Mc @ SWING['Rel']) @ head).z
    SWING['sgn'] = 1.0 if z[1.0] > z[-1.0] else -1.0
    return {"sgn": SWING['sgn']}


def hand_matrix(rig, s):
    """world matrix of hand s: origin at the wrist, axes = the posed hand frame (x across, y back of the hand, z along the fingers)"""
    pb = rig.pose.bones['hand.' + s]
    bpy.context.view_layer.update()
    R = hand_world_frame(rig, s)
    M = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            M[r][c] = float(R[r, c])
    M.translation = rig.matrix_world @ pb.head
    return M


def grip_both(rig, B, Rp, s_left, s_right, fwd, left, right):
    """both hands on the shaft. The mechanical (right) hand is solved first from the analytic pick; the pick is then taken as that hand really
    holds it (SWING['REL'], a constant grip), and the glove hand grips THAT shaft, so the two hands and the baked pick never disagree."""
    up = Vector((0, 0, 1))
    zw = Rp[:, 2]
    sl = shoulder_world(rig, 'L')
    sr = shoulder_world(rig, 'R')
    res = {}

    def solve(s, P, axis, sh, outv, g_local):
        hint = np.array(P - np.array(sh))
        zh, yh, wrist = grip_hand(rig, s, P, axis, outv, g_local=g_local, forearm_hint_w=hint)
        pole = (left if s == 'L' else right) * 0.7 + Vector((0, 0, -0.8)) - fwd * 0.2
        miss = reach_arm(rig, s, Vector(wrist), pole)
        orient_hand_w(rig, s, zh, yh)
        res['miss_' + s] = round(float(miss), 3)

    solve('R', B + zw * s_right, zw, sr, right * 0.6 + up * 0.8, (0.0, -0.030, 0.046))
    if 'REL' in SWING:
        Mp = hand_matrix(rig, 'R') @ SWING['REL']
        B2 = np.array(Mp.translation)
        z2 = np.array(Mp.to_3x3())[:, 2]
        SWING['LAST_PICK'] = Mp
    else:
        B2, z2 = B, zw
    solve('L', B2 + z2 * s_left, z2, sl, left * 0.6 + up * 0.8, (0.0, -0.030, 0.052))
    curl_hand(rig, 'L', {'index': [58, 74, 52], 'middle': [64, 78, 54], 'ring': [68, 80, 54], 'pinky': [70, 80, 52], 'thumb': [24, 30, 18]})
    curl_hand(rig, 'R', {'mi': [48, 62], 'mm': [52, 66], 'mr': [56, 70], 'mt': [10, 16]})
    bpy.context.view_layer.update()
    return res


def swing_pose(rig, alpha=0.0, lean=BASE_LEAN, hip_h=0.34, look=GLINT, s_right=0.28, s_left=0.44):
    """the dig crouch with the pick lifted by `alpha` degrees about the shoulder line (both hands stay on the shaft at the same distance from the
    shoulders, so the arms always reach)"""
    r = crouch(rig, DIG["C"], W=look, hip_h=hip_h, lean=lean, keep_object=True)
    Mc = chest_world(rig)
    SLw, SRw = Mc @ SWING['SL'], Mc @ SWING['SR']
    axis = (SLw - SRw).normalized()
    Rm = Matrix.Translation(SRw) @ Matrix.Rotation(math.radians(alpha * SWING['sgn']), 4, axis) @ Matrix.Translation(-SRw)
    Mp = Rm @ Mc @ SWING['Rel']
    B = np.array(Mp.translation)
    Rp = np.array(Mp.to_3x3())
    free_prop(bpy.data.objects[CP + "Pick"], B, Rp)
    return grip_both(rig, B, Rp, s_left, s_right, r["fwd"], r["left"], r["right"])


def record_pose(rig):
    bpy.context.view_layer.update()
    rec = {}
    for pb in rig.pose.bones:
        rec[pb.name] = (pb.location.copy(), pb.rotation_quaternion.copy(), pb.scale.copy())
    return rec


def apply_pose(rig, rec):
    for pb in rig.pose.bones:
        l, q, s = rec[pb.name]
        pb.location = l
        pb.rotation_quaternion = q
        pb.scale = s
    bpy.context.view_layer.update()


def blend_pose(a, b, t):
    out = {}
    for n in a:
        la, qa, sa = a[n]
        lb, qb, sb = b[n]
        qb2 = qb.copy()
        if qa.dot(qb2) < 0:
            qb2.negate()
        out[n] = (la.lerp(lb, t), qa.slerp(qb2, t), sa.lerp(sb, t))
    return out


def attach_posed(ob, rig, bone, W):
    """parent ob to a bone keeping the world matrix W at the rig's CURRENT pose (attach() needs the rest pose; this does not)"""
    ob.parent = rig
    ob.parent_type = 'BONE'
    ob.parent_bone = bone
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    Pm = ob.matrix_world.copy()
    ob.matrix_parent_inverse = Pm.inverted() @ W
    bpy.context.view_layer.update()


def clear_anim(idb):
    if idb is not None and idb.animation_data is not None:
        idb.animation_data_clear()


def make_cam(sc, name, loc, tgt, lens, dof=None):
    cd = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, cd)
        sc.collection.objects.link(cam)
    clear_anim(cam)
    clear_anim(cd)
    cd.lens = lens
    cd.clip_start = 0.05
    cd.clip_end = 200.0
    cd.dof.use_dof = dof is not None
    if dof is not None:
        cd.dof.aperture_fstop = dof
    cam.location = Vector(loc)
    tn = name + "_Target"
    t = bpy.data.objects.get(tn)
    if t is None:
        t = bpy.data.objects.new(tn, None)
        t.empty_display_type = 'PLAIN_AXES'
        t.empty_display_size = 0.12
        sc.collection.objects.link(t)
    clear_anim(t)
    t.location = Vector(tgt)
    for c in list(cam.constraints):
        cam.constraints.remove(c)
    tc = cam.constraints.new('TRACK_TO')
    tc.target = t
    tc.track_axis = 'TRACK_NEGATIVE_Z'
    tc.up_axis = 'UP_Y'
    return cam, t


def key_path(obj, path, frame, value):
    setattr(obj, path, value)
    obj.keyframe_insert(data_path=path, frame=frame)


def iter_fcurves(idb):
    """every f-curve of an ID's action, in the slotted-action API of Blender 5 or the legacy one"""
    ad = getattr(idb, "animation_data", None)
    if not ad or not ad.action:
        return []
    act = ad.action
    try:
        return list(act.fcurves)
    except Exception:
        pass
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                out.extend(cb.fcurves)
    return out


def key_prop(ob, prop, frame, value, interp=None):
    ob[prop] = value
    ob.keyframe_insert(data_path='["%s"]' % prop, frame=frame)
    if interp:
        for fc in iter_fcurves(ob):
            if fc.data_path == '["%s"]' % prop:
                for kp in fc.keyframe_points:
                    if int(round(kp.co[0])) == frame:
                        kp.interpolation = interp


def dust_node():
    m = bpy.data.materials["Cave_Dust"]
    for n in m.node_tree.nodes:
        if n.type == 'MATH' and n.operation == 'MULTIPLY' and not n.inputs[1].is_linked:
            return m.node_tree, n
    raise RuntimeError("dust multiply node not found")
