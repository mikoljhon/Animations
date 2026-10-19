# mender_anim7.py : cut 5 of the opening animatic. A heavy pick swung through a real arc, slower.
# Exec after mender_anim6.py / mender_fx.py (see ma5_load.py).  This file adds swing_pose2 (the pick can go back behind the head: the
# grip point is tucked toward the shoulders so the elbows bend, and the pick is cocked back about the leading hand) and the new timeline.


def swing_pose2(rig, alpha=0.0, cock=0.0, tuck=0.0, lean=BASE_LEAN, hip_h=0.34, look=GLINT, s_right=0.28, s_left=0.44, drop=0.0):
    """swing_pose plus: cock = extra degrees the pick turns about the right grip (the head falls back behind the shoulders), tuck = fraction
    of the distance between the grip and the shoulder line that the hands move in (bent elbows), drop = metres the hands sink"""
    r = crouch(rig, DIG["C"], W=look, hip_h=hip_h, lean=lean, keep_object=True)
    Mc = chest_world(rig)
    SLw, SRw = Mc @ SWING['SL'], Mc @ SWING['SR']
    axis = (SLw - SRw).normalized()
    sg = SWING['sgn']
    Rm = Matrix.Translation(SRw) @ Matrix.Rotation(math.radians(alpha * sg), 4, axis) @ Matrix.Translation(-SRw)
    Mp = Rm @ Mc @ SWING['Rel']
    if abs(cock) > 1e-6 or tuck > 1e-6 or abs(drop) > 1e-6:
        zw = Vector(Mp.to_3x3().col[2])
        G = Mp.translation + zw * s_right                     # where the mechanical hand holds the shaft
        Rc = Matrix.Translation(G) @ Matrix.Rotation(math.radians(cock * sg), 4, axis) @ Matrix.Translation(-G)
        Mp = Rc @ Mp
        G2 = Mp.translation + Vector(Mp.to_3x3().col[2]) * s_right
        mid = (SLw + SRw) * 0.5
        v = mid - G2
        v = v - axis * v.dot(axis)                            # only toward the shoulder line
        Mp.translation = Mp.translation + v * tuck + Vector((0, 0, -drop))
    B = np.array(Mp.translation)
    Rp = np.array(Mp.to_3x3())
    free_prop(bpy.data.objects[CP + "Pick"], B, Rp)
    return grip_both(rig, B, Rp, s_left, s_right, r["fwd"], r["left"], r["right"])


CACHE = os.path.join(OUT_DIR, "anim_cache5")


# ------------------------------------------------------------------ the new timeline (frames at 24 fps)
SLOW = 1.35                    # everything after the big strike runs this much slower than cut 3
C1, C2, C3 = 42, 91, 146       # contact frames of the three strikes
CONTACTS = [(42.0, 45.0), (91.0, 94.0), (146.0, 149.0)]
HOLD_END = 149.0


def G(f_old):
    """cut 3 frame (>= 58) -> frame in the new timeline"""
    return C3 + (f_old - 58.0) * SLOW


def Wf(g):
    """frame in the new timeline -> cut 3 frame"""
    return 58.0 + (g - C3) / SLOW


def Gi(f_old):
    return int(round(G(f_old))) if f_old >= 58 else int(f_old)


def strike_segs(s, a0, top, L, H, D, hold, wrench, a_next):
    segs, t = [], float(s)
    segs.append((t, a0, t + 4, a0 - 8.0, 'io')); t += 4
    segs.append((t, a0 - 8.0, t + L, top, 'io')); t += L                    # the long lift, back behind the head
    segs.append((t, top, t + H, top + 5.0, 'out')); t += H                  # hang at the top
    segs.append((t, top + 5.0, t + D, -3.0, 'in2')); t += D                 # the downswing: fastest at the stone (quadratic: no 100 degree last frame)
    segs.append((t, -3.0, t + hold, -3.5, 'lin')); t += hold                # the spike is in the rock
    segs.append((t, -3.5, t + wrench, a_next, 'out')); t += wrench          # wrenching it free
    return segs, t


def build_alpha5():
    segs = [(1.0, 18.0, 5.0, 18.0, 'lin')]
    s1, t = strike_segs(5, 18.0, 130.0, 20, 4, 9, 3, 9, 20.0)
    s2, t = strike_segs(54, 20.0, 135.0, 20, 4, 9, 3, 9, 22.0)
    s3, t = strike_segs(103, 22.0, 145.0, 24, 6, 9, 3, 9, 22.0)
    return segs + s1 + s2 + s3[:-1]


ALPHA5 = build_alpha5()


def alpha5(f):
    if f >= HOLD_END:
        return alpha_pick(Wf(f))
    for f0, a0, f1, a1, kind in ALPHA5:
        if f <= f1:
            if f <= f0:
                return a0
            return a0 + (a1 - a0) * ease(kind, (f - f0) / (f1 - f0))
    return ALPHA5[-1][3]


A_TAB = [-4.0, 0.0, 11.0, 33.0, 47.0, 90.0, 135.0, 150.0]
LEAN_TAB = np.array([[0.31, 0.69, 1.04, 0.62], [0.30, 0.66, 1.00, 0.60], [0.26, 0.58, 0.88, 0.54], [0.14, 0.38, 0.58, 0.36],
                     [0.05, 0.22, 0.34, 0.20], [0.03, 0.10, 0.13, 0.08], [-0.04, -0.08, -0.12, -0.03], [-0.06, -0.10, -0.15, -0.04]])
HIP_TAB = [0.325, 0.34, 0.36, 0.40, 0.445, 0.46, 0.47, 0.475]


def lean_of(a):
    a = min(max(a, -4.0), 150.0)
    return np.array([float(np.interp(a, A_TAB, LEAN_TAB[:, k])) for k in range(4)])


def hip_of(a):
    return float(np.interp(min(max(a, -4.0), 150.0), A_TAB, HIP_TAB))


def breath(f):
    t = (f - 1) / 24.0
    gau = lambda x, c, w: math.exp(-((x - c) / w) ** 2)
    heavy = max(gau(Wf(f), 78.0, 22.0), 0.6 * gau(f, C1 + 10, 16.0), 0.75 * gau(f, C2 + 10, 16.0))
    amp = 0.012 + 0.014 * heavy
    freq = 0.30 + 0.20 * heavy
    ph = 2 * math.pi * freq * t
    s = math.sin(ph + 0.4)
    return np.array([0.0, 0.004 * s, amp * s, 0.003 * s]), 0.004 * math.sin(ph + 0.9)


COCK_TOP, TUCK_TOP = 62.0, 0.33


def compute_tracks():
    ring_p = np.array(ring_world()[0])
    vis_p = np.array(visor_world()[0])
    glint = np.array(GLINT)
    fr = np.arange(1, F_TOTAL + 1)

    def look_target(f):
        w = Wf(f)
        if w < 74.5:
            return glint
        if w < 99.5:
            return ring_p
        return vis_p

    T = {}
    def scalar_sod(fn, f0, z, r):
        return run_sod(lambda f: np.array([fn(f)]), f0, z, r)[:, 0]

    T['a_pick'] = np.array([alpha5(float(f)) for f in fr])
    T['a_body'] = scalar_sod(lambda f: alpha5(f + 3.0), 3.4, 0.55, 0.0)
    T['a_hip'] = scalar_sod(lambda f: alpha5(f + 3.5), 3.8, 0.50, 0.0)
    T['a_head'] = scalar_sod(lambda f: alpha5(f - 0.5), 2.5, 0.45, 0.0)
    T['cock'] = scalar_sod(lambda f: COCK_TOP * smoothstep((alpha5(f) - 25.0) / 100.0), 2.6, 0.80, 0.0)     # the pick head trails the hands
    T['tuck'] = np.array([TUCK_TOP * smoothstep((alpha5(float(f)) - 30.0) / 100.0) for f in fr])
    T['freeze'] = np.array([smoothstep((Wf(f) - 60.0) / 10.0) for f in fr])
    T['look'] = run_sod(look_target, 5.5, 0.55, 0.0)
    T['wT'] = scalar_sod(lambda f: 1.0 if Wf(f) >= 106.5 else 0.0, 2.4, 0.45, 0.0)
    T['wAL'] = scalar_sod(lambda f: 1.0 if Wf(f) >= 107.5 else 0.0, 3.2, 0.30, 0.0)
    T['wAR'] = scalar_sod(lambda f: 1.0 if Wf(f) >= 108.5 else 0.0, 3.0, 0.35, 0.0)
    T['uL'] = np.array([smoothstep((Wf(f) - 107.5) / 8.5) for f in fr])
    T['uR'] = np.array([smoothstep((Wf(f) - 110.5) / 10.5) for f in fr])
    T['back'] = np.array([0.34 + 0.06 * smoothstep((Wf(f) - 125.0) / 25.0) for f in fr])
    T['hipB'] = np.array([0.48 + 0.02 * smoothstep((Wf(f) - 125.0) / 25.0) for f in fr])
    return T


def solve_frame(rig, f, T):
    i = f - 1
    br, brh = breath(float(f))
    c = T['freeze'][i]
    lean = lean_of(T['a_body'][i])
    lean = lean + c * (FREEZE_LEAN - lean_of(6.0))
    hip = hip_of(T['a_hip'][i]) + c * (FREEZE_HIP - hip_of(6.0))
    lean = lean + br
    hip = hip + brh
    nod = min(max((10.0 - T['a_head'][i]) / 20.0, 0.0), 1.0)
    look = Vector(T['look'][i]) + Vector((0.0, 0.0, -0.11 * nod))
    lean_t = tuple(float(x) for x in lean)
    alpha = float(T['a_pick'][i])
    cock = float(T['cock'][i]) * smoothstep((alpha + 2.0) / 18.0)
    tuck = float(T['tuck'][i])
    if in_contact(float(f)):
        alpha = touch_alpha(rig, lean_t, float(hip), look)
        cock, tuck = 0.0, 0.0
    miss = swing_pose2(rig, alpha=alpha, cock=cock, tuck=tuck, lean=lean_t, hip_h=float(hip), look=look)
    miss['alpha'] = round(alpha, 2)
    rec = record_pose(rig)
    if f < DROP_F:
        return rec, miss
    feetA = {s: rig.pose.bones['ik_foot.' + s].matrix.copy() for s in 'LR'}
    lookB = Vector(T['look'][i])
    lb = tuple(float(x) + float(b) for x, b in zip((0.00, -0.04, -0.08, 0.0), br))
    scramble_pose(rig, back=float(T['back'][i]), hip_h=float(T['hipB'][i] + brh), lean=lb, look=lookB)
    recB = record_pose(rig)
    feetB = {s: rig.pose.bones['ik_foot.' + s].matrix.copy() for s in 'LR'}
    w = {}
    for n in TORSO:
        w[n] = float(T['wT'][i])
    for n in ARM_L:
        w[n] = float(T['wAL'][i])
    for n in ARM_R:
        w[n] = float(T['wAR'][i])
    rec = blend_groups(rec, recB, w)
    apply_pose(rig, rec)
    for s, u in (('L', float(T['uL'][i])), ('R', float(T['uR'][i]))):
        pa, pb_ = feetA[s], feetB[s]
        m = Matrix.Translation(pa.translation.lerp(pb_.translation, u) + Vector((0, 0, 0.075 * math.sin(math.pi * u)))) @ \
            pa.to_quaternion().slerp(pb_.to_quaternion(), u).to_matrix().to_4x4()
        rig.pose.bones['ik_foot.' + s].matrix = m
    bpy.context.view_layer.update()
    return record_pose(rig), miss


# ------------------------------------------------------------------ controls, Relic's head, puffs, cameras, orchestration
def _mapped(seq):
    """[(old_frame, value, interp)] -> same with the new timeline; frames of cut 3 before the big strike are kept"""
    out, last = [], -1
    for f, v, i in seq:
        g = Gi(f) if f != F_END_OLD else F_TOTAL
        g = max(g, last + 1)
        out.append((g, v, i))
        last = g
    return out


F_END_OLD = 216


def key_controls(sc):
    ctl = bpy.data.objects["Cave_Controls"]
    clear_anim(ctl)
    for f, v in ((1, 0.0), (C3, 0.0), (Gi(82), 1.0), (F_TOTAL, 1.0)):
        key_prop(ctl, 'dug', f, v)
    ring = [(1, 0.0, 'LINEAR'), (73, 0.0, 'CONSTANT'), (74, 0.9, 'CONSTANT'), (76, 0.1, 'CONSTANT'), (78, 0.7, 'CONSTANT'), (79, 0.0, 'CONSTANT'),
            (82, 1.0, 'CONSTANT'), (85, 0.55, 'CONSTANT'), (87, 1.0, 'BEZIER'), (140, 1.0, 'BEZIER'), (156, 1.25, 'BEZIER'), (176, 1.0, 'BEZIER'),
            (216, 1.0, 'BEZIER')]
    for f, v, i in _mapped(ring):
        key_prop(ctl, 'ring', f, v, i)
    eyes = [(1, 0.0, 'LINEAR'), (98, 0.0, 'CONSTANT'), (99, 0.9, 'CONSTANT'), (101, 0.0, 'CONSTANT'), (103, 1.0, 'CONSTANT'), (104, 0.35, 'CONSTANT'),
            (106, 1.0, 'LINEAR'), (190, 1.0, 'LINEAR'), (192, 0.05, 'LINEAR'), (195, 1.0, 'LINEAR'), (216, 1.0, 'LINEAR')]
    for f, v, i in _mapped(eyes):
        key_prop(ctl, 'eyes', f, v, i)
    nt, node = dust_node()
    clear_anim(nt)
    dust = [(1, 0.030), (56, 0.030), (60, 0.17), (72, 0.15), (86, 0.09), (100, 0.06), (160, 0.045), (216, 0.04)]
    last = -1
    for f, v in dust:
        g = Gi(f) if f != 216 else F_TOTAL
        if f == 56:
            g = C3 - 3
        g = max(g, last + 1)
        last = g
        node.inputs[1].default_value = v
        node.inputs[1].keyframe_insert('default_value', frame=g)
    return "ok"


def relic_head_tracks():
    step = lambda f: np.array([1.0 if f >= G(104) else 0.0])
    ah = run_sod(step, 1.9, 0.42, 0.0)[:, 0]
    an = run_sod(step, 1.4, 0.55, 0.0)[:, 0]
    return an, ah


def key_puffs(sc, rig):
    coll = bpy.data.collections[CCOLL]
    T = rim_target()
    ring_p = Vector(ring_world()[0])
    sc.frame_set(1)
    for ob in list(coll.objects):
        if ob.name.startswith(CP + "Puff_"):
            bpy.data.objects.remove(ob, do_unlink=True)
    made = []
    made.append(add_puff(sc, coll, "strike1", T, 0.10, 0.38, C1, 20, 4.8, 1).name)
    made.append(add_puff(sc, coll, "strike2", T + Vector((0.02, 0.0, 0.0)), 0.10, 0.42, C2, 22, 5.4, 2).name)
    made.append(add_puff(sc, coll, "strike3", T + Vector((0.0, 0.0, 0.08)), 0.18, 0.95, C3, 30, 6.5, 3).name)
    made.append(add_puff(sc, coll, "slab", ring_p + Vector((0.0, 0.0, 0.12)), 0.15, 0.80, Gi(62), 54, 4.5, 4).name)
    sc.frame_set(DROP + 7)
    pk = bpy.data.objects[CP + "Pick"]
    made.append(add_puff(sc, coll, "pickdrop", pk.matrix_world.translation + Vector((0.0, 0.0, 0.04)), 0.10, 0.36, DROP + 7, 20, 3.8, 5).name)
    f_scuff = Gi(118)
    sc.frame_set(f_scuff)
    foot = rig.matrix_world @ rig.pose.bones['foot.L'].head
    made.append(add_puff(sc, coll, "scuff", foot + Vector((0.0, 0.0, 0.02)), 0.10, 0.42, f_scuff, 22, 3.6, 6).name)
    sc.frame_set(1)
    return made


def build_cameras3(sc, face):
    """cut 5 cameras: wide (the first two swings), dig (the big swing and the break), two-shot (the reaction), Relic's face"""
    sc.timeline_markers.clear()
    cams = {}
    s2, s3, s4 = C3 - 20, Gi(97), Gi(169)
    # S1 wide: a long eased push, hand-held, a little jolt on each strike
    P0, P1 = Vector((-1.15, -2.25, 1.9)), Vector((-0.7, -1.7, 1.8))
    A = Vector((1.05, 0.82, 1.62))
    c, t = make_cam(sc, "Anim_Cam_Wide", P0, A, 34.0)
    dirv = (A - P0).normalized()
    n1 = s2 - 2
    key_cam(c, t, 1, s2 - 1,
            lambda f: P0.lerp(P1, smoothstep((f - 1) / float(n1))) + hh_noise((f - 1) / 24.0, 3, 0.018) + dirv * (kick(f, C1, 0.014) + kick(f, C2, 0.014)),
            lambda f: A + hh_noise((f - 1) / 24.0, 4, 0.02))
    cams['S1'] = (c, 1)
    # S2 dig: push in on the big swing, the strike hits the camera, then the aim rises with Mender
    P0, P1 = Vector((-1.1, -1.95, 1.75)), Vector((-0.45, -1.0, 1.55))
    A0, A1 = Vector((1.0, 0.88, 1.7)), Vector((0.55, 1.0, 1.3))     # after the strike the aim slides toward the pit so the rocks falling are in the picture
    c, t = make_cam(sc, "Anim_Cam_Dig", P0, A0, 34.0)
    dirv = (A0 - P0).normalized()
    ga = G(64.0)
    key_cam(c, t, s2, s3 - 1,
            lambda f: P0.lerp(P1, smoothstep((f - (C3 + 4)) / float(s3 - C3 - 4))) + hh_noise((f - s2) / 24.0, 5, 0.016) + shake(f, C3, 0.045, 9) + dirv * kick(f, C3, 0.05, 2.0, 14.0),
            lambda f: A0.lerp(A1, smoothstep((f - (C3 + 1)) / 20.0)) + hh_noise((f - s2) / 24.0, 6, 0.018))
    cams['S2'] = (c, s2)
    # S3 two-shot: slow push; a nudge when the eyes light and a jolt at the drop
    P0, P1 = Vector((-1.3, -1.4, 2.0)), Vector((-1.1, -1.15, 1.95))
    A = Vector((0.78, 0.97, 1.4))
    c, t = make_cam(sc, "Anim_Cam_Two", P0, A, 32.0)
    dirv = (A - P0).normalized()
    key_cam(c, t, s3, s4 - 1,
            lambda f: P0.lerp(P1, smoothstep((f - s3) / float(s4 - s3))) + hh_noise((f - s3) / 24.0, 7, 0.016) + dirv * (kick(f, Gi(99), 0.04, 3.0, 16.0)) + shake(f, DROP, 0.02, 7),
            lambda f: A + hh_noise((f - s3) / 24.0, 8, 0.02))
    cams['S3'] = (c, s3)
    # S4 Relic's face: slow push-in, shallow depth of field following the move
    cpos, vtgt, cpos2, vtgt2, info, info2 = face
    c, t = make_cam(sc, "Anim_Cam_Face", cpos, vtgt, 55.0, dof=3.2)
    key_cam(c, t, s4, F_TOTAL,
            lambda f: cpos.lerp(cpos2, smoothstep((f - s4) / float(F_TOTAL - s4))) + hh_noise((f - s4) / 24.0, 9, 0.006),
            lambda f: vtgt.lerp(vtgt2, smoothstep((f - s4) / float(F_TOTAL - s4))) + hh_noise((f - s4) / 24.0, 10, 0.006))
    c.data.dof.focus_distance = (vtgt - cpos).length
    c.data.keyframe_insert('dof.focus_distance', frame=s4)
    c.data.dof.focus_distance = (vtgt2 - cpos2).length
    c.data.keyframe_insert('dof.focus_distance', frame=F_TOTAL)
    cams['S4'] = (c, s4)
    for nm, (cam, f) in cams.items():
        m = sc.timeline_markers.new(nm, frame=f)
        m.camera = cam
    sc.camera = cams['S1'][0]
    return "cameras keyed"


def build_step3(face_frame=None):
    sc = bpy.data.scenes['Cave']
    bpy.context.window_manager.windows[0].scene = sc
    rig = bpy.data.objects[CRIG]
    rep = {}
    rep['acc_max_deg'] = key_accessories(sc, rig)
    face = face_cam_positions(frame=face_frame or Gi(169))
    rep['cams'] = build_cameras3(sc, face)
    sc.frame_set(1)
    return rep
