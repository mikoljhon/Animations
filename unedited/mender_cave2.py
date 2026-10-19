# mender_cave2.py : the first cave pose for Mender: crouched on the rubble beside Relic's shoulder, headlamp on the gold glint,
# glove hand brushing the dust off, mechanical hand on the pick that is planted in the rubble.
# Exec after mender_cave.py (same call).

GLINT = Vector((0.437, 1.232, 1.028))       # the shoulder window of Relic in world space (buried state)


def surface_top(x, y, z_top=3.4):
    """world point and normal of the highest surface under (x, y), Relic included (Mender and lights ignored)"""
    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    o = Vector((x, y, z_top))
    d = Vector((0.0, 0.0, -1.0))
    for _ in range(16):
        ok, loc, nrm, idx, ob, mat = sc.ray_cast(deps, o, d)
        if not ok:
            return None
        if ob.name.startswith(("MenderC_", "Mender_")):
            o = loc + d * 0.001
            continue
        return loc, nrm
    return None


def crouch(rig, C, W=GLINT, hip_h=0.40, stance=0.17, lean=(0.28, 0.62, 0.95, 0.55), left_adv=0.10, right_back=0.05, head_frac=1.0, keep_object=False):
    """root at C (world xy), facing W. Feet on the rubble by ray casts, pelvis low over them, spine leaning forward by `lean` (y components of the
    pelvis, spine, chest, neck directions), head toward W. Returns a dict of world positions for the arm solvers.
    keep_object=True leaves the rig object where it is (location and heading) and only poses the bones: needed for animation, where the object
    must not move between key poses."""
    C = Vector((C[0], C[1], 0.0))
    if keep_object:
        psi = rig.rotation_euler.z
        fxy = Vector((math.sin(psi), -math.cos(psi)))
    else:
        fxy = Vector((W.x - C.x, W.y - C.y)).normalized()
        psi = math.atan2(fxy.x, -fxy.y)
        rig.rotation_mode = 'XYZ'
        rig.rotation_euler = (0, 0, psi)
        rig.location = (C.x, C.y, 0.0)
    reset_pose(rig)
    fwd = Vector((fxy.x, fxy.y, 0.0))
    right = fwd.cross(Vector((0, 0, 1))).normalized()
    left = -right

    def stand_foot(center):
        cpt, n = foot_plane((center.x, center.y), (fwd.x, fwd.y))
        fs = (fwd - n * fwd.dot(n)).normalized()
        return cpt - fs * 0.10 + n * 0.099, fs, n, cpt

    aL, fL, nL, pL = stand_foot(C + left * stance + fwd * left_adv)
    aR, fR, nR, pR = stand_foot(C + right * stance - fwd * right_back)
    zmean = 0.5 * (pL.z + pR.z)
    if not keep_object:
        rig.location = (C.x, C.y, zmean)
    bpy.context.view_layer.update()
    place_foot(rig, 'L', aL, fL, nL)
    place_foot(rig, 'R', aR, fR, nR)
    mid = 0.5 * (aL + aR) - fwd * 0.04
    hip = Vector((mid.x, mid.y, 0.5 * (aL.z + aR.z) + hip_h))
    bpy.context.view_layer.update()
    cur = rig.matrix_world @ rig.pose.bones['pelvis'].head
    move_ik(rig, 'pelvis', w2l(rig, hip - cur))
    aim(rig, 'pelvis', (0.0, -lean[0], 1.0))
    aim(rig, 'spine', (0.0, -lean[1], 1.0))
    aim(rig, 'chest', (0.0, -lean[2], 1.0))
    aim(rig, 'neck', (0.0, -lean[3], 1.0))
    head = rig.matrix_world @ rig.pose.bones['head'].head
    look_dir(rig, 'head', W - head, head_frac)
    bpy.context.view_layer.update()
    return {"fwd": fwd, "right": right, "left": left, "feet": (aL, aR), "psi": psi}


def shoulder_world(rig, s):
    bpy.context.view_layer.update()
    return rig.matrix_world @ rig.pose.bones['upperarm.' + s].head


PICK_TIP_L = np.array([0.27, 0.0, 0.855]) * np.asarray(SC, float)      # the spike tip in the pick's own frame (butt at the origin, +Z to the head)


def frame_zx(z, xh):
    """right-handed frame with columns (x, y, z): z along `z`, x as close to xh as possible"""
    z = unit(z)
    x = np.asarray(xh, float) - z * (np.asarray(xh, float) @ z)
    x = unit(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def pick_pose(tip_w, z_w, x_hint=(0.0, 0.0, -1.0)):
    """frame of the pick (columns x spike side, y, z butt -> head) that puts the spike tip on tip_w with the shaft along z_w; returns butt, R"""
    R = frame_zx(z_w, x_hint)
    B = np.asarray(tip_w, float) - R @ PICK_TIP_L
    return B, R


def free_prop(ob, B, R):
    """detach a prop from its bone and give it a plain world matrix"""
    ob.parent = None
    ob.matrix_world = frame4(B, R)
    bpy.context.view_layer.update()


def dig_pose(rig, C, tip_w, elev_deg=-34.0, yaw_deg=14.0, s_right=0.10, s_left=0.44, hip_h=0.36, lean=(0.30, 0.66, 1.00, 0.60),
             stance=0.17, left_adv=0.10, right_back=0.05, look=GLINT, pick_name="MenderC_Pick"):
    """Mender digs: crouched on the rubble, two hands on the pick, the spike on tip_w, the head turned to `look`.
    Returns the misses of both wrists (0 = the arm reached) and the final positions."""
    r = crouch(rig, C, W=look, hip_h=hip_h, stance=stance, lean=lean, left_adv=left_adv, right_back=right_back)
    fwd, right, left = r["fwd"], r["right"], r["left"]
    up = Vector((0, 0, 1))
    el = math.radians(elev_deg)
    yw = math.radians(yaw_deg)
    hdir = (fwd * math.cos(yw) + left * math.sin(yw)).normalized()      # yaw > 0 turns the shaft toward his left
    z_w = np.array(hdir * math.cos(el) + up * math.sin(el))
    B, Rp = pick_pose(np.array(tip_w), z_w, x_hint=(0.0, 0.0, -1.0))
    pick = bpy.data.objects[pick_name]
    free_prop(pick, B, Rp)
    zw = Rp[:, 2]
    PR = B + zw * s_right
    PL = B + zw * s_left
    sl = shoulder_world(rig, 'L')
    sr = shoulder_world(rig, 'R')
    res = {}
    for s, P, sh, outv in (('L', PL, sl, left * 0.6 + up * 0.8), ('R', PR, sr, right * 0.6 + up * 0.8)):
        hint = np.array(P - np.array(sh))
        zh, yh, wrist = grip_hand(rig, s, P, zw, outv, g_local=(0.0, -0.030, 0.052) if s == 'L' else (0.0, -0.030, 0.046), forearm_hint_w=hint)
        pole = (left if s == 'L' else right) * 0.7 + Vector((0, 0, -0.8)) - fwd * 0.2
        miss = reach_arm(rig, s, Vector(wrist), pole)
        orient_hand_w(rig, s, zh, yh)
        res['miss_' + s] = round(float(miss), 3)
        res['reach_' + s] = round(float((Vector(wrist) - sh).length), 3)
    curl_hand(rig, 'L', {'index': [58, 74, 52], 'middle': [64, 78, 54], 'ring': [68, 80, 54], 'pinky': [70, 80, 52], 'thumb': [24, 30, 18]})
    curl_hand(rig, 'R', {'mi': [48, 62], 'mm': [52, 66], 'mr': [56, 70], 'mt': [10, 16]})
    bpy.context.view_layer.update()
    res['pick_butt'] = tuple(round(float(x), 3) for x in B)
    res['tip'] = tuple(round(float(x), 3) for x in tip_w)
    return res


DIG = dict(C=(1.0, 0.85), dist=0.18, elev_deg=-16.0, yaw_deg=0.0, s_right=0.28, s_left=0.44, hip_h=0.34)


def place_mender(rig=None, **kw):
    """the cave dig pose: Mender crouched on the rubble 0.9 m from Relic's shoulder, spike of the pick on the rim of the dish, lamp on the glint.
    Always computed in the BURIED state (the rubble under his boots is the same in both states); the dug state is restored afterwards."""
    rig = rig or bpy.data.objects[CRIG]
    P = dict(DIG)
    P.update(kw)
    ctl = bpy.data.objects["Cave_Controls"]
    was = float(ctl["dug"])
    cave_state(False)
    C = P["C"]
    fxy = Vector((GLINT.x - C[0], GLINT.y - C[1])).normalized()
    txy = Vector((GLINT.x, GLINT.y)) - fxy * P["dist"]
    g = surface_top(txy.x, txy.y)
    T = np.array([txy.x, txy.y, g[0].z - 0.01])
    res = dig_pose(rig, C, T, elev_deg=P["elev_deg"], yaw_deg=P["yaw_deg"], s_right=P["s_right"], s_left=P["s_left"], hip_h=P["hip_h"])
    cave_state(was > 0.5)
    return res


def grip_hand(rig, s, P_w, axis_w, out_w, g_local=(0.0, -0.030, 0.052), forearm_hint_w=None):
    """hand s closes on a shaft through P_w running along axis_w. out_w = direction away from the shaft toward the back of the hand.
    Returns (z_w, y_w) of the hand frame and the wrist target; the sign of the across axis is picked so the wrist bends least."""
    ax = unit(np.array(axis_w))
    ow = np.array(out_w, float)
    ow = unit(ow - ax * (ow @ ax))
    best = None
    for sg in (1.0, -1.0):
        xh = ax * sg
        zh = np.cross(xh, ow)                  # x = y × z  =>  z = x × y
        Rh = np.stack([xh, ow, zh], axis=1)
        wrist = np.array(P_w, float) - Rh @ (np.array(g_local) * 0.95)
        score = 0.0
        if forearm_hint_w is not None:
            score = float(unit(zh) @ unit(np.array(forearm_hint_w)))
        if best is None or score > best[0]:
            best = (score, zh, ow, wrist)
    return best[1], best[2], best[3]
