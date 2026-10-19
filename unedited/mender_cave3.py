# mender_cave3.py : Mender's reaction when Relic lights up: he scoots back, rises a little, the glove hand comes up palm-out and the pick
# swings round to point at the glowing chest, the headlamp snaps up to the visor.
# Exec after mender_cave2.py (same call). Window scene = Cave.

VISOR_FALLBACK = Vector((0.11, 1.38, 1.28))


def scramble_pose(rig=None, back=0.34, hip_h=0.48, lean=(0.00, -0.04, -0.08, 0.00), look=None, keep_object=True):
    """Mender after dropping the pick: scooted back and up, both hands raised palm-out, head on the visor (or on `look`)"""
    rig = rig or bpy.data.objects[CRIG]
    ctl = bpy.data.objects["Cave_Controls"]
    was = float(ctl["dug"])
    cave_state(False)
    if look is None:
        try:
            look, vn = visor_world()
        except Exception:
            look = VISOR_FALLBACK
    C0 = Vector(DIG["C"])
    f0 = Vector((GLINT.x - C0.x, GLINT.y - C0.y)).normalized()
    C = (C0.x - f0.x * back, C0.y - f0.y * back)
    r = crouch(rig, C, W=look, hip_h=hip_h, lean=lean, stance=0.20, left_adv=0.04, right_back=0.10, keep_object=keep_object)
    fwd, right, left = r["fwd"], r["right"], r["left"]
    up = Vector((0, 0, 1))
    res = {}
    for s, sgn, side in (('L', 1.0, left), ('R', -1.0, right)):
        sh = shoulder_world(rig, s)
        wr = sh + fwd * 0.36 + up * (0.10 if s == 'L' else 0.06) + side * 0.06
        miss = reach_arm(rig, s, wr, side * 0.7 + Vector((0, 0, -0.8)) - fwd * 0.3)
        orient_hand_w(rig, s, np.array(up * 0.85 + fwd * 0.35), np.array(-fwd))
        res['miss_' + s] = round(float(miss), 3)
    curl_hand(rig, 'L', {'index': [8, 12, 8], 'middle': [10, 14, 8], 'ring': [14, 16, 10], 'pinky': [18, 20, 12], 'thumb': [4, 6, 4]})
    curl_hand(rig, 'R', {'mi': [10, 14], 'mm': [12, 16], 'mr': [14, 18], 'mt': [0, 2]})
    bpy.context.view_layer.update()
    cave_state(was > 0.5)
    return res


def recoil_pose(rig=None, back=0.30, hip_h=0.47, lean=(0.00, -0.04, -0.08, 0.00), s_right=0.30, aim_at="visor", pick_name="MenderC_Pick",
                keep_object=False):
    """Mender startled. The rubble under his boots is the same in both states, so the buried state is used for the ray casts and the dug state
    is restored afterwards. keep_object=True: the rig object stays where the dig pose put it (for animation)."""
    rig = rig or bpy.data.objects[CRIG]
    ctl = bpy.data.objects["Cave_Controls"]
    was = float(ctl["dug"])
    cave_state(False)
    try:
        vp, vn = visor_world()
    except Exception:
        vp = VISOR_FALLBACK
    ring, ringf = ring_world()
    C0 = Vector(DIG["C"])
    f0 = Vector((GLINT.x - C0.x, GLINT.y - C0.y)).normalized()
    C = (C0.x - f0.x * back, C0.y - f0.y * back)
    r = crouch(rig, C, W=vp, hip_h=hip_h, lean=lean, stance=0.20, left_adv=0.04, right_back=0.10, keep_object=keep_object)
    fwd, right, left = r["fwd"], r["right"], r["left"]
    up = Vector((0, 0, 1))
    res = {}
    # glove hand up, palm out: "whoa"
    sl = shoulder_world(rig, 'L')
    wl = sl + fwd * 0.36 + up * 0.10 + left * 0.06
    miss = reach_arm(rig, 'L', wl, left * 0.7 + Vector((0, 0, -0.8)) - fwd * 0.3)
    orient_hand_w(rig, 'L', np.array(up * 0.85 + fwd * 0.35), np.array(-fwd))
    curl_hand(rig, 'L', {'index': [8, 12, 8], 'middle': [10, 14, 8], 'ring': [14, 16, 10], 'pinky': [18, 20, 12], 'thumb': [4, 6, 4]})
    res['miss_L'] = round(float(miss), 3)
    # pick in the mechanical hand, pointing at Relic
    sr = shoulder_world(rig, 'R')
    P = sr + fwd * 0.34 + right * 0.02 + Vector((0, 0, -0.10))
    tgt = vp if aim_at == "visor" else ring
    z_w = (tgt - P).normalized()
    B, Rp = pick_pose(np.array(P - z_w * s_right), np.array(z_w), x_hint=(0.0, 0.0, -1.0))
    # pick_pose puts the SPIKE TIP on the point it is given, so give it the tip position that matches the butt we want
    Rp = frame_zx(np.array(z_w), (0.0, 0.0, -1.0))
    B = np.array(P - z_w * s_right)
    pick = bpy.data.objects[pick_name]
    free_prop(pick, B, Rp)
    zw = Rp[:, 2]
    hint = np.array(P - sr)
    zh, yh, wrist = grip_hand(rig, 'R', np.array(P), zw, right * 0.6 + up * 0.8, g_local=(0.0, -0.030, 0.046), forearm_hint_w=hint)
    miss = reach_arm(rig, 'R', Vector(wrist), right * 0.7 + Vector((0, 0, -0.8)) - fwd * 0.2)
    orient_hand_w(rig, 'R', zh, yh)
    curl_hand(rig, 'R', {'mi': [48, 62], 'mm': [52, 66], 'mr': [56, 70], 'mt': [10, 16]})
    res['miss_R'] = round(float(miss), 3)
    bpy.context.view_layer.update()
    res['visor'] = tuple(round(float(x), 2) for x in vp)
    cave_state(was > 0.5)
    return res
