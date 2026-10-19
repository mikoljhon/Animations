# mender_anim6.py : cut 3 orchestration in three steps (each short enough for one call). Exec after mender_anim5.py; window scene = Cave.

def build_step1():
    """clean slate, Relic's head tracks, the pick on the right hand at the base pose, keys for Mender, the controls and Relic"""
    win = bpy.context.window_manager.windows[0]
    sc = bpy.data.scenes['Cave']
    win.scene = sc
    rig = bpy.data.objects[CRIG]
    rr = bpy.data.objects[RIG]
    pick = bpy.data.objects[CP + "Pick"]
    ctl = bpy.data.objects["Cave_Controls"]
    for ob in (rig, ctl, rr, pick):
        clear_anim(ob)
    pick.parent = None
    for g, (bone, names, *_r) in ACC.items():
        for n in names:
            ob = bpy.data.objects[CP + n]
            clear_anim(ob)
            ob.location = (0, 0, 0)
            ob.rotation_mode = 'QUATERNION'
            ob.rotation_quaternion = (1, 0, 0, 0)
    sc.frame_set(1)
    rep = {}
    quats = compute_relic_quats(rr)
    rep['setup'] = setup_solver(rig)
    attach_posed(pick, rig, 'hand.R', pick.matrix_world.copy())
    recs = load_cache()
    rep['rig'] = key_dense(rig, recs)
    rep['controls'] = key_controls(sc)
    rep['relic'] = key_relic_dense(rr, quats)
    sc.frame_start, sc.frame_end = 1, F_TOTAL
    sc.frame_set(1)
    return rep


def build_step2():
    """bake the pick from the hand and let it fall"""
    sc = bpy.data.scenes['Cave']
    bpy.context.window_manager.windows[0].scene = sc
    rep = bake_pick(sc)
    sc.frame_set(1)
    return rep


def build_step3():
    """gear springs, cameras"""
    sc = bpy.data.scenes['Cave']
    bpy.context.window_manager.windows[0].scene = sc
    rig = bpy.data.objects[CRIG]
    rep = {}
    rep['acc_max_deg'] = key_accessories(sc, rig)
    face = face_cam_positions(frame=169)
    rep['cams'] = build_cameras3(sc, face)
    sc.frame_set(1)
    return rep
