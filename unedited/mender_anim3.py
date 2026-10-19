# mender_anim3.py : second cut of the opening animatic, closer to the Episode 1 script:
# Mender digs, the last slab slides off in a burst of dust, the chest ring lights and he FREEZES (head on the ring), the eyes light and Relic's
# head lifts, Mender DROPS THE PICK and scrambles backward with both hands up.
# Exec after mender_anim2.py (same call), window scene = Cave. build_opening2() rebuilds everything (about 1 minute).

KEYS2 = [(1, 'RAISED'), (8, 'STRIKE'), (12, 'REBOUND'), (24, 'RAISED'), (31, 'STRIKE'), (35, 'REBOUND'), (49, 'RAISED_HIGH'), (58, 'STRIKE_BIG'),
         (66, 'PAUSE'), (78, 'FREEZE_RING'), (98, 'FREEZE_RING'), (103, 'FREEZE_VISOR'), (109, 'FREEZE_VISOR'), (115, 'STEP'), (125, 'SCRAMBLE2'),
         (150, 'SCRAMBLE3'), (216, 'SCRAMBLE3')]
DROP = globals().get("DROP_5", 106)                      # the frame Mender lets go of the pick
HEAD_LIFT = (104, 130)          # Relic's head lifts between these frames
LIFT = dict(neck=((0.20, -0.24, 1.0), 0.0), head=((0.46, -0.66, 1.0), 36.0))      # cut 3: a clearer lift off the slope toward Mender


def compute_key_poses2(rig):
    cave_state(False)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    place_mender(rig)
    swing_base(rig)
    ring_p = ring_world()[0]
    vis_p = visor_world()[0]
    recs, miss = {}, {}
    for nm, kw in POSES.items():
        miss[nm] = swing_pose(rig, **kw)
        recs[nm] = record_pose(rig)
    fr = dict(alpha=6.0, lean=(0.26, 0.55, 0.85, 0.52), hip_h=0.34)
    miss['FREEZE_RING'] = swing_pose(rig, look=ring_p, **fr)
    recs['FREEZE_RING'] = record_pose(rig)
    miss['FREEZE_VISOR'] = swing_pose(rig, look=vis_p, **fr)
    recs['FREEZE_VISOR'] = record_pose(rig)
    miss['SCRAMBLE2'] = scramble_pose(rig, back=0.34, hip_h=0.48)
    recs['SCRAMBLE2'] = record_pose(rig)
    miss['SCRAMBLE3'] = scramble_pose(rig, back=0.40, hip_h=0.50)
    recs['SCRAMBLE3'] = record_pose(rig)
    apply_pose(rig, blend_pose(recs['FREEZE_VISOR'], recs['SCRAMBLE2'], 0.5))
    for s in ('L', 'R'):
        pb = rig.pose.bones['ik_foot.' + s]
        m = pb.matrix.copy()
        m.translation.z += 0.075
        pb.matrix = m
    bpy.context.view_layer.update()
    recs['STEP'] = record_pose(rig)
    return recs, miss


def key_rig2(rig, recs):
    global KEYS
    old = KEYS
    KEYS = KEYS2
    try:
        return key_rig(rig, recs)
    finally:
        KEYS = old


def to_quat_mode(pb):
    M = pb.matrix_basis.copy()
    pb.rotation_mode = 'QUATERNION'
    pb.matrix_basis = M


def lift_relic_head(rr, amount=1.0):
    """tilt Relic's neck and head toward Mender and forward off the slope; amount 0 = the buried pose, 1 = lifted"""
    for n in ('neck', 'head'):
        d, tw = LIFT[n]
        base = POSE[n]
        dd = (base[0][0] + (d[0] - base[0][0]) * amount, base[0][1] + (d[1] - base[0][1]) * amount, 1.0)
        _aim(rr, n, dd, base[1] + (tw - base[1]) * amount)


def key_relic_head():
    rr = bpy.data.objects[RIG]
    clear_anim(rr)
    cave_pose(True)
    for n in ('neck', 'head'):
        to_quat_mode(rr.pose.bones[n])
    bpy.context.view_layer.update()
    base = {n: rr.pose.bones[n].rotation_quaternion.copy() for n in ('neck', 'head')}
    lift_relic_head(rr, 1.0)
    lifted = {n: rr.pose.bones[n].rotation_quaternion.copy() for n in ('neck', 'head')}
    f0, f1 = HEAD_LIFT
    for n in ('neck', 'head'):
        pb = rr.pose.bones[n]
        q1 = lifted[n].copy()
        if base[n].dot(q1) < 0:
            q1.negate()
        for f, q in ((1, base[n]), (f0, base[n]), (f1, q1), (F_END, q1)):
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=f)
    return "relic head keyed"


def smooth01(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def bake_pick(sc, drop=DROP):
    """sample where the pick is at every frame while it rides the hand, then free it: from `drop` it falls, bounces once and lies on the rubble"""
    pick = bpy.data.objects[CP + "Pick"]
    W = {}
    for f in range(1, F_END + 1):
        sc.frame_set(f)
        W[f] = pick.matrix_world.copy()
    clear_anim(pick)
    pick.parent = None
    pick.rotation_mode = 'QUATERNION'
    Wd = W[drop]
    p0 = Wd.translation.copy()
    q0 = Wd.to_quaternion()
    zdir = Wd.to_3x3() @ Vector((0, 0, 1))
    dir_h = Vector((zdir.x, zdir.y, 0.0)).normalized()
    side = dir_h.cross(Vector((0, 0, 1))).normalized()
    sc.frame_set(drop + 14)                     # dug-out state: the rubble where the pick will land
    bxy = Vector((p0.x, p0.y)) - Vector((dir_h.x, dir_h.y)) * 0.12
    hxy = bxy + Vector((dir_h.x, dir_h.y)) * 0.78
    gb = surface_top(bxy.x, bxy.y)
    gh = surface_top(hxy.x, hxy.y)
    zb = (gb[0].z if gb else p0.z - 0.5) + 0.035
    zh = (gh[0].z if gh else zb) + 0.035
    axis = Vector((hxy.x - bxy.x, hxy.y - bxy.y, zh - zb)).normalized()
    R1 = frame_zx(np.array(axis), np.array(side))
    q1 = Matrix(R1.tolist()).to_quaternion()
    rest = Vector((bxy.x, bxy.y, zb))
    prev = None
    for f in range(1, F_END + 1):
        if f < drop:
            loc, q, _s = W[f].decompose()
        else:
            k = f - drop
            if k <= 7:
                u = k / 7.0
                xy = Vector((p0.x, p0.y)).lerp(Vector((rest.x, rest.y)), smooth01(u))
                z = p0.z + (rest.z - p0.z) * (u * u)
                loc = Vector((xy.x, xy.y, z))
                q = q0.slerp(q1, smooth01(u))
            else:
                bounce = {8: 0.050, 9: 0.062, 10: 0.040, 11: 0.012, 12: 0.018, 13: 0.006}.get(f - drop, 0.0)
                loc = rest + Vector((0, 0, bounce))
                q = q1.copy()
        q = q.copy()
        if prev is not None and prev.dot(q) < 0:
            q.negate()
        prev = q.copy()
        pick.location = loc
        pick.rotation_quaternion = q
        pick.keyframe_insert('location', frame=f)
        pick.keyframe_insert('rotation_quaternion', frame=f)
    for fc in iter_fcurves(pick):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
    sc.frame_set(1)
    return {"rest": tuple(round(x, 3) for x in rest), "drop_at": tuple(round(x, 3) for x in p0)}


def build_opening2():
    win = bpy.context.window_manager.windows[0]
    sc = bpy.data.scenes['Cave']
    win.scene = sc
    rig = bpy.data.objects[CRIG]
    rep = {}
    pick = bpy.data.objects[CP + "Pick"]
    clear_anim(rig)
    clear_anim(bpy.data.objects["Cave_Controls"])
    clear_anim(pick)
    pick.parent = None
    clear_anim(bpy.data.objects[RIG])
    sc.frame_set(1)
    cave_pose(True)                                  # Relic back to the buried pose before anything is computed
    recs, miss = compute_key_poses2(rig)
    rep['misses'] = miss
    # the pick rides the mechanical hand until it is baked
    apply_pose(rig, recs['STRIKE'])
    swing_pose(rig, **POSES['STRIKE'])
    attach_posed(pick, rig, 'hand.R', pick.matrix_world.copy())
    rep['keys'] = key_rig2(rig, recs)
    rep['controls'] = key_controls(sc)
    rep['relic'] = key_relic_head()
    sc.frame_start, sc.frame_end = 1, F_END
    rep['pick'] = bake_pick(sc)
    face = face_cam_positions(frame=169)
    rep['cameras'] = build_cameras(sc, face)
    sc.frame_set(1)
    return rep
