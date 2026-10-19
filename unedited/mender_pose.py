# mender_pose.py : posing helpers for Mender (aim bones, orient the hand, curl fingers, move IK feet, props in hands) and the first poses.
# Exec after mender_build.py and mender_scene.py.


def get_rig():
    return bpy.data.objects[RIGN]


def reset_pose(rig=None):
    rig = rig or get_rig()
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def aim(rig, name, d, twist=0.0):
    """turn bone `name` (head fixed) so its Y axis points along d (rig space), then twist about that axis (degrees)"""
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    m = pb.matrix.copy()
    y = m.col[1].to_3d().normalized()
    dv = Vector(d).normalized()
    R = y.rotation_difference(dv).to_matrix().to_4x4()
    new = Matrix.Translation(m.translation) @ R @ m.to_3x3().to_4x4()
    if twist:
        new = Matrix.Translation(new.translation) @ Matrix.Rotation(math.radians(twist), 4, dv) @ new.to_3x3().to_4x4()
    pb.matrix = new
    bpy.context.view_layer.update()


def move_ik(rig, name, delta):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    m = pb.matrix.copy()
    m.translation = m.translation + Vector(delta)
    pb.matrix = m
    bpy.context.view_layer.update()


def bone_world(rig, name, t=0.5):
    pb = rig.pose.bones[name]
    return rig.matrix_world @ (pb.head * (1 - t) + pb.tail * t)


def _rest_frame_world(s):
    """the hand frame (columns x, y, z) of the rest pose in scaled rig space, orthonormalised"""
    wr, R = hand_frame(s)
    scv = np.asarray(SC, float)
    z = unit(R[:, 2] * scv)
    y = unit(R[:, 1] * scv)
    y = unit(y - z * (y @ z))
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def orient_hand(rig, s, Rt):
    """give the hand frame (z along the fingers, y = back of the hand, x across) the orientation Rt (columns x, y, z in rig space)"""
    Rw = _rest_frame_world(s)
    D = np.asarray(Rt) @ Rw.T
    pb = rig.pose.bones['hand.' + s]
    bpy.context.view_layer.update()
    m = pb.matrix.copy()
    new3 = Matrix(D.tolist()) @ pb.bone.matrix_local.to_3x3()
    pb.matrix = Matrix.Translation(m.translation) @ new3.to_4x4()
    bpy.context.view_layer.update()


def curl_chain(rig, bones, angles_deg, a_rest):
    """rotate each bone of a finger chain about its flexion axis (a_rest = hand x axis in the rest pose); positive = toward the palm"""
    for b, ang in zip(bones, angles_deg):
        pb = rig.pose.bones[b]
        bpy.context.view_layer.update()
        delta = pb.matrix @ pb.bone.matrix_local.inverted()
        ax = (delta.to_3x3() @ Vector(a_rest)).normalized()
        m = pb.matrix.copy()
        h = m.translation.copy()
        pb.matrix = Matrix.Translation(h) @ Matrix.Rotation(math.radians(ang), 4, ax) @ Matrix.Translation(-h) @ m
        bpy.context.view_layer.update()


def curl_hand(rig, s, spec):
    """spec: {'index': [a1, a2, a3], ..., 'thumb': [...], or for the mechanical hand 'mi': [a1, a2], ...} degrees added to the rest curl"""
    a_rest = _rest_frame_world(s)[:, 0]
    for nm, angs in spec.items():
        bones = ['%s%d.%s' % (nm, k + 1, s) for k in range(len(angs))]
        curl_chain(rig, bones, angs, a_rest)


def hold_scanner(rig, scanner, face_target):
    """walkie-talkie grip: the hand (thumb up) closes on the scanner's grip, the screen turns toward face_target"""
    pb = rig.pose.bones['hand.L']
    bpy.context.view_layer.update()
    Rt = _rest_frame_world('L')
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    Rp = np.array(delta.to_3x3()) @ Rt                      # posed hand frame
    head = np.array(rig.matrix_world @ pb.head)
    P = head + Rp @ (np.array([0.0, -0.028, 0.060]) * 0.93)
    z_s = -Rp[:, 0]
    to_face = unit(np.array(Vector(face_target)) - P)
    Rs = basis(z_s, -to_face)
    g_local = np.array([0.0, 0.032, -0.070]) * np.asarray(SC, float)
    origin = P - Rs @ g_local
    attach(scanner, rig, 'hand.L', frame4(origin, Rs))


def hero_pose(rig=None):
    """Mender scanning: weight on one hip, head turned toward the scanner in the left hand (walkie-talkie grip), mechanical hand relaxed"""
    rig = rig or get_rig()
    reset_pose(rig)
    move_ik(rig, 'ik_foot.L', (0.035, -0.070, 0.0))
    move_ik(rig, 'ik_foot.R', (-0.025, 0.035, 0.0))
    aim(rig, 'pelvis', (-0.07, 0.0, 1.0))
    aim(rig, 'spine', (0.00, 0.035, 1.0), 0)
    aim(rig, 'chest', (0.05, 0.030, 1.0), 8)
    aim(rig, 'neck', (0.02, -0.04, 1.0), 0)
    aim(rig, 'head', (0.06, -0.16, 1.0), 20)
    aim(rig, 'upperarm.L', (0.34, -0.50, -0.80))
    aim(rig, 'forearm.L', (0.16, -0.90, 0.20))
    zt = unit((0.16, -0.90, 0.20))
    xt = unit(-(np.array([0.0, 0.0, 1.0]) - zt * zt[2]))
    yt = np.cross(zt, xt)
    orient_hand(rig, 'L', np.stack([xt, yt, zt], axis=1))
    curl_hand(rig, 'L', {'index': [62, 70, 50], 'middle': [68, 74, 52], 'ring': [72, 76, 50], 'pinky': [76, 74, 46],
                         'thumb': [18, 24, 16]})
    aim(rig, 'upperarm.R', (-0.22, -0.12, -0.97))
    aim(rig, 'forearm.R', (-0.04, -0.42, -0.90))
    aim(rig, 'hand.R', (0.0, -0.32, -0.95))
    curl_hand(rig, 'R', {'mi': [6, 6], 'mm': [8, 8], 'mr': [10, 10], 'mt': [-6, -4]})
    bpy.context.view_layer.update()
    return "hero"
