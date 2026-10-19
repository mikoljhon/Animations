# mender_anim2.py : builds the opening animatic (see mender_anim.py). Exec after mender_anim.py, window scene = Cave.

KEYS = [(1, 'RAISED'), (8, 'STRIKE'), (12, 'REBOUND'), (24, 'RAISED'), (31, 'STRIKE'), (35, 'REBOUND'), (49, 'RAISED_HIGH'), (58, 'STRIKE_BIG'),
        (66, 'PAUSE'), (78, 'STEP'), (90, 'RECOIL'), (150, 'RECOIL2'), (216, 'RECOIL2')]
F_END = globals().get("F_TOTAL_5", 216)


def compute_key_poses(rig):
    """every pose of the animatic as a recorded set of bone channels (+ the world matrix of the pick for the dig poses)"""
    cave_state(False)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    place_mender(rig)
    swing_base(rig)
    pick = bpy.data.objects[CP + "Pick"]
    recs, picks, miss = {}, {}, {}
    for nm, kw in POSES.items():
        miss[nm] = swing_pose(rig, **kw)
        recs[nm] = record_pose(rig)
        picks[nm] = pick.matrix_world.copy()
    miss['RECOIL'] = recoil_pose(rig, keep_object=True)
    recs['RECOIL'] = record_pose(rig)
    picks['RECOIL'] = pick.matrix_world.copy()
    miss['RECOIL2'] = recoil_pose(rig, back=0.38, hip_h=0.505, keep_object=True)
    recs['RECOIL2'] = record_pose(rig)
    picks['RECOIL2'] = pick.matrix_world.copy()
    # the hop back: halfway between the pause and the recoil, both boots lifted
    apply_pose(rig, blend_pose(recs['PAUSE'], recs['RECOIL'], 0.5))
    for s in ('L', 'R'):
        pb = rig.pose.bones['ik_foot.' + s]
        m = pb.matrix.copy()
        m.translation.z += 0.075
        pb.matrix = m
    bpy.context.view_layer.update()
    recs['STEP'] = record_pose(rig)
    picks['STEP'] = picks['PAUSE']
    return recs, picks, miss


def key_rig(rig, recs):
    clear_anim(rig)
    moving = set()
    for rec in recs.values():
        for n, (l, q, s) in rec.items():
            if l.length > 1e-5 or abs(q.w - 1.0) > 1e-5 or abs(q.x) > 1e-5 or abs(q.y) > 1e-5 or abs(q.z) > 1e-5:
                moving.add(n)
    prev = {}
    count = 0
    for f, nm in KEYS:
        rec = recs[nm]
        for n in sorted(moving):
            l, q, s = rec[n]
            q = q.copy()
            if n in prev and prev[n].dot(q) < 0:
                q.negate()
            prev[n] = q.copy()
            pb = rig.pose.bones[n]
            pb.location = l
            pb.rotation_quaternion = q
            pb.keyframe_insert('location', frame=f)
            pb.keyframe_insert('rotation_quaternion', frame=f)
            count += 2
    return {"bones": len(moving), "keys": count}


def key_controls(sc):
    ctl = bpy.data.objects["Cave_Controls"]
    clear_anim(ctl)
    for f, v in ((1, 0.0), (58, 0.0), (82, 1.0), (F_END, 1.0)):
        key_prop(ctl, 'dug', f, v)
    ring = [(1, 0.0, 'LINEAR'), (73, 0.0, 'CONSTANT'), (74, 0.9, 'CONSTANT'), (76, 0.1, 'CONSTANT'), (78, 0.7, 'CONSTANT'), (79, 0.0, 'CONSTANT'),
            (82, 1.0, 'CONSTANT'), (85, 0.55, 'CONSTANT'), (87, 1.0, 'BEZIER'), (140, 1.0, 'BEZIER'), (156, 1.25, 'BEZIER'), (176, 1.0, 'BEZIER'),
            (F_END, 1.0, 'BEZIER')]
    for f, v, i in ring:
        key_prop(ctl, 'ring', f, v, i)
    eyes = [(1, 0.0, 'LINEAR'), (98, 0.0, 'CONSTANT'), (99, 0.9, 'CONSTANT'), (101, 0.0, 'CONSTANT'), (103, 1.0, 'CONSTANT'), (104, 0.35, 'CONSTANT'),
            (106, 1.0, 'LINEAR'), (190, 1.0, 'LINEAR'), (192, 0.05, 'LINEAR'), (195, 1.0, 'LINEAR'), (F_END, 1.0, 'LINEAR')]
    for f, v, i in eyes:
        key_prop(ctl, 'eyes', f, v, i)
    nt, node = dust_node()
    clear_anim(nt)
    for f, v in ((1, 0.030), (56, 0.030), (60, 0.45), (72, 0.45), (86, 0.20), (100, 0.09), (160, 0.05), (F_END, 0.04)):
        node.inputs[1].default_value = v
        node.inputs[1].keyframe_insert('default_value', frame=f)
    return "ok"


def shake(f, f0, amp=0.035, n=7):
    """a decaying camera jolt starting at frame f0"""
    k = f - f0
    if k < 0 or k >= n:
        return Vector((0, 0, 0))
    a = amp * (1.0 - k / n)
    sgn = 1.0 if k % 2 == 0 else -1.0
    return Vector((0.4 * a * sgn, 0.0, a * sgn))


def lerpv(a, b, t):
    return Vector(a).lerp(Vector(b), t)


def face_cam_positions(d0=1.75, d1=1.30, frame=169):
    """start and end camera positions in front of Relic's visor, searched at `frame` of the finished animation (dug out, Mender in place)"""
    sc = bpy.context.scene
    f0 = sc.frame_current
    sc.frame_set(frame)
    a = visor_cam(dist=d0)
    b = visor_cam(dist=d1)
    sc.frame_set(f0)
    return (Vector(a[0]), Vector(a[1]), Vector(b[0]), Vector(b[1]), a[2], b[2])


def build_cameras(sc, face):
    """four cameras with timeline markers: wide f1-48, dig f49-96, two-shot f97-168, Relic's face f169-216"""
    sc.timeline_markers.clear()
    cams = {}
    # S1 wide, slow push
    c, t = make_cam(sc, "Anim_Cam_Wide", (-2.0, -3.0, 1.5), (0.6, 1.0, 1.0), 34.0)
    c.location = (-2.0, -3.0, 1.5)
    c.keyframe_insert('location', frame=1)
    c.location = (-1.75, -2.65, 1.45)
    c.keyframe_insert('location', frame=48)
    cams['S1'] = (c, 1)
    # S2 dig, push and a jolt on the big strike (f58)
    c, t = make_cam(sc, "Anim_Cam_Dig", (-0.7, -1.3, 1.6), (0.75, 1.0, 1.1), 36.0)
    for f in range(49, 97):
        tt = (f - 49) / 47.0
        c.location = lerpv((-0.7, -1.3, 1.6), (-0.45, -1.0, 1.55), tt) + shake(f, 58)
        c.keyframe_insert('location', frame=f)
    # the aim point follows Mender as he rises from the strike
    for f, p in ((49, (0.75, 1.0, 1.1)), (64, (0.75, 1.0, 1.1)), (92, (0.98, 0.93, 1.38))):
        t.location = p
        t.keyframe_insert('location', frame=f)
    cams['S2'] = (c, 49)
    # S3 reaction two-shot, slow push
    c, t = make_cam(sc, "Anim_Cam_Two", (-1.3, -1.4, 2.0), (0.78, 0.97, 1.4), 32.0)
    c.location = (-1.3, -1.4, 2.0)
    c.keyframe_insert('location', frame=97)
    c.location = (-1.1, -1.15, 1.95)
    c.keyframe_insert('location', frame=168)
    cams['S3'] = (c, 97)
    # S4 Relic's face, slow push-in with a shallow depth of field
    cpos, vtgt, cpos2, vtgt2, info, info2 = face
    c, t = make_cam(sc, "Anim_Cam_Face", tuple(cpos), tuple(vtgt), 55.0, dof=3.2)
    c.location = cpos
    c.keyframe_insert('location', frame=169)
    c.data.dof.focus_distance = (Vector(vtgt) - Vector(cpos)).length
    c.data.keyframe_insert('dof.focus_distance', frame=169)
    c.location = cpos2
    c.keyframe_insert('location', frame=F_END)
    c.data.dof.focus_distance = (Vector(vtgt2) - Vector(cpos2)).length
    c.data.keyframe_insert('dof.focus_distance', frame=F_END)
    cams['S4'] = (c, 169)
    for nm, (cam, f) in cams.items():
        m = sc.timeline_markers.new(nm, frame=f)
        m.camera = cam
    sc.camera = cams['S1'][0]
    return {"visor_cam": [info, info2]}


def build_opening():
    win = bpy.context.window_manager.windows[0]
    sc = bpy.data.scenes['Cave']
    win.scene = sc
    rig = bpy.data.objects[CRIG]
    rep = {}
    # make sure nothing animated is left over from an earlier build, so the poses are computed on a clean slate
    clear_anim(rig)
    clear_anim(bpy.data.objects["Cave_Controls"])
    recs, picks, miss = compute_key_poses(rig)
    rep['misses'] = miss
    # the pick rides the mechanical hand from here on
    pick = bpy.data.objects[CP + "Pick"]
    apply_pose(rig, recs['STRIKE'])
    attach_posed(pick, rig, 'hand.R', picks['STRIKE'])
    rep['keys'] = key_rig(rig, recs)
    rep['controls'] = key_controls(sc)
    sc.frame_start, sc.frame_end = 1, F_END
    face = face_cam_positions()
    rep['cameras'] = build_cameras(sc, face)
    sc.frame_start = 1
    sc.frame_end = F_END
    sc.render.fps = FPS
    sc.frame_set(1)
    return rep


def preview_frames(frames, tag, res=(640, 360), samples=12):
    """render single frames at low quality through the animated scene (the marker cameras pick the shot)"""
    sc = bpy.context.scene
    os.makedirs(os.path.join(OUT_DIR, "renders"), exist_ok=True)
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.eevee.taa_render_samples = samples
    sc.render.image_settings.file_format = 'PNG'
    win = bpy.context.window_manager.windows[0]
    prefs = bpy.context.preferences.view
    old = prefs.render_display_type
    prefs.render_display_type = 'NONE'
    paths = []
    try:
        for f in frames:
            sc.frame_set(f)
            p = os.path.join(OUT_DIR, "renders", "%s_%03d.png" % (tag, f))
            sc.render.filepath = p
            with bpy.context.temp_override(window=win, scene=sc):
                bpy.ops.render.render(write_still=True)
            paths.append(p)
    finally:
        prefs.render_display_type = old
    return paths
