# fight_anim.py : keys the fight of episode 1 on fight_v01.blend (part 1: drop, standoff walk, beat 1) and bakes the first-person camera.
#   blender.exe -b fight_v01.blend --python fight_anim.py  ->  fight_v02.blend
# Poses come from Motion Assist AutoPose: each key gives hips / neck / wrists / ankles in fighter-local coordinates
#   (side: + = his LEFT, fwd: + = the way he faces, up) around a ground point `base`; Blender interpolates between keys.
import bpy, math, importlib, random
from mathutils import Vector, Matrix, Quaternion

OUT = r"C:\ai\unedited"
MA = "bl_ext.user_default.motion_assist"
ap = importlib.import_module(MA + ".autopose")
sc = bpy.data.scenes["Street"]
rr = bpy.data.objects["Relic_AI_Rig"]
kr = bpy.data.objects["KillDroid_Rig"]
cam = bpy.data.objects["POV"]
FPS = 24
# clear the fight lane of boulders so the camera can see the KillDroid come in
for nm, nx in (("Boulder19", -6.8), ("Boulder22", 6.4), ("Boulder28", -7.4)):
    bpy.data.objects[nm].location.x = nx
sc.render.fps = FPS
G = 9.81

# ------------------------------------------------------------------------------------------------ rest dimensions
bpy.context.view_layer.update()
REST = {}
for rig in (rr, kr):
    m = ap.rig_map(rig)
    cur = ap.joints_now(rig, m, ap.eye_points(rig, m))
    w = {j: rig.matrix_world @ p for j, p in cur.items()}
    REST[rig.name] = w
    print("REST", rig.name, {j: tuple(round(x, 2) for x in p) for j, p in w.items() if j in ("hips", "neck", "wrist_L", "ankle_L", "elbow_L")})

exec(open(OUT + r'\kd_ctrl.py', encoding='utf-8').read())
kd_add_arm_ik(kr)
kd_add_arm_ik(rr)
exec(open(OUT + '/rig_guard.py', encoding='utf-8').read())     # the joint-limit guard: no pose is ever baked without it
XTRA = {}
FWD = {rr.name: Vector((0.0, 1.0, 0.0)), kr.name: Vector((0.0, -1.0, 0.0))}


def to_world(rig, base, yaw, p):
    """fighter-local (side, fwd, up) -> world, around the ground point base, with an extra yaw (radians, + turns toward his left)."""
    f = FWD[rig.name]
    f = Matrix.Rotation(yaw, 3, 'Z') @ f
    left = Matrix.Rotation(math.pi / 2, 3, 'Z') @ f
    return Vector((base[0], base[1], 0.0)) + left * p[0] + f * p[1] + Vector((0, 0, p[2]))


KEYS = {}


def _pchip(xs, ys, x):
    """monotone cubic Hermite (no overshoot), xs ascending"""
    n = len(xs)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    k = max(i for i in range(n - 1) if xs[i] <= x)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = 0.0, 0.0
    t = (x - xs[k]) / h[k]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * ys[k] + h10 * h[k] * m[k] + h01 * ys[k + 1] + h11 * h[k] * m[k + 1]


def bake_rig(rig, f0, f1):
    """solve the pose on EVERY frame from smoothly interpolated goal paths (no bezier between separate solutions)"""
    ks = KEYS[rig.name]
    fr = sorted(ks)
    for f in range(f0, f1 + 1):
        goals = {}
        for j in ks[fr[0]]:
            goals[j] = Vector([_pchip(fr, [ks[q][j][c] for q in fr], f) for c in range(3)])
        ap.solve(rig, goals, keep=ap.CORE, key=True, frame=f)


def _fcurves(ad):
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            cb = strip.channelbag(ad.action_slot) if ad.action_slot else None
            if cb:
                out.extend(cb.fcurves)
    return out


HEAD_UP = {"KillDroid_Rig": 1.25, "Relic_AI_Rig": 0.9}


def rig_params(rig, f, other=None):
    ks = KEYS[rig.name]
    xt = XTRA[rig.name]
    fr = sorted(ks)
    P = {}
    for j, k in (("hips", "hips"), ("wL", "wrist_L"), ("wR", "wrist_R"), ("aL", "ankle_L"), ("aR", "ankle_R")):
        P[j] = Vector([_pchip(fr, [ks[q][k][c] for q in fr], f) for c in range(3)])
    P["yaw"] = _pchip(fr, [xt[q][0] for q in fr], f)
    P["lean"] = (_pchip(fr, [xt[q][1] for q in fr], f), _pchip(fr, [xt[q][2] for q in fr], f))
    P["tw"] = _pchip(fr, [xt[q][3] for q in fr], f)
    if other is not None:
        ko = KEYS[other.name]
        fo = sorted(ko)
        oh = Vector([_pchip(fo, [ko[q]["hips"][c] for q in fo], f) for c in range(3)])
        P["look"] = oh + Vector((0.0, 0.0, HEAD_UP[other.name] - ko[fo[0]]["hips"][2] * 0.0 - 0.1))
    return P


def bake_ctrl(rig, other, f0, f1, calib_frames):
    print("POLES", rig.name, kd_calibrate(rig, [rig_params(rig, f, other) for f in calib_frames]))
    guard_setup(rig)
    set_meshes_hidden(True)
    for f in range(f0, f1 + 1):
        apply_safe(rig, f, rig_params(rig, f, other))
    set_meshes_hidden(False)
    print(rig.name, guard_report(rig))


def smooth_rig(rig, f0, f1, sigma=1.3):
    """gaussian-smooth every pose-bone channel over time (kills solver snaps); quaternions kept on one side"""
    ad = rig.animation_data
    if not ad or not ad.action:
        return
    import numpy as np
    r = int(math.ceil(sigma * 3))
    kern = np.array([math.exp(-0.5 * (i / sigma) ** 2) for i in range(-r, r + 1)])
    kern /= kern.sum()
    chans = {}
    for fc in _fcurves(ad):
        chans.setdefault(fc.data_path, []).append(fc)
    for path, fcs in chans.items():
        if not path.endswith(("rotation_quaternion", "location", "rotation_euler", "scale")):
            continue
        fcs.sort(key=lambda c: c.array_index)
        arr = np.array([[fc.evaluate(f) for fc in fcs] for f in range(f0, f1 + 1)])
        if path.endswith("rotation_quaternion") and len(fcs) == 4:
            for i in range(1, len(arr)):
                if np.dot(arr[i], arr[i - 1]) < 0:
                    arr[i] = -arr[i]
        pad = np.concatenate([np.repeat(arr[:1], r, 0), arr, np.repeat(arr[-1:], r, 0)])
        out = np.stack([np.convolve(pad[:, c], kern, mode="valid") for c in range(arr.shape[1])], axis=1)
        if path.endswith("rotation_quaternion") and len(fcs) == 4:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        for ci, fc in enumerate(fcs):
            for kp in list(fc.keyframe_points):
                pass
            for i, f in enumerate(range(f0, f1 + 1)):
                fc.keyframe_points[f - f0].co[1] = out[i, ci]
                fc.keyframe_points[f - f0].interpolation = 'LINEAR'
            fc.update()


def pose(rig, frame, base, hips, wL, wR, aL, aR, lean=(0.0, 0.0), yaw=0.0):
    """key one pose. hips/wL/wR/aL/aR = (side, fwd, up) local; lean = (forward angle, sideways angle toward his left) of the torso."""
    rest = REST[rig.name]
    L = (rest["neck"] - rest["hips"]).length
    th, ph = lean
    off = (math.sin(ph) * L, math.sin(th) * L, math.cos(th) * math.cos(ph) * L)
    neck = (hips[0] + off[0], hips[1] + off[1], hips[2] + off[2])
    goals = {"hips": to_world(rig, base, yaw, hips), "neck": to_world(rig, base, yaw, neck),
             "wrist_L": to_world(rig, base, yaw, wL), "wrist_R": to_world(rig, base, yaw, wR),
             "ankle_L": to_world(rig, base, yaw, aL), "ankle_R": to_world(rig, base, yaw, aR)}
    KEYS.setdefault(rig.name, {})[frame] = goals
    XTRA.setdefault(rig.name, {})[frame] = (yaw, lean[0], lean[1], 0.0)


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------------ the KillDroid
# rest numbers (design frame): hips z 1.32, ankle z 0.21, shoulders +-0.65, wrists hang at z 0.92 side +-0.85
K_H = REST[kr.name]["hips"].z
K_A = REST[kr.name]["ankle_L"].z
K_AS = abs(REST[kr.name]["ankle_L"].x)          # ankle half-width
K_WS = abs(REST[kr.name]["wrist_L"].x)
K_WZ = REST[kr.name]["wrist_L"].z
print("KD nums", K_H, K_A, K_AS, K_WS, K_WZ)


def kd_stand(f, base, bob=0.0, sway=0.0, wl=None, wr=None, lean=(0.04, 0.0), fl=0.0, fr=0.0, yaw=0.0, hz=None):
    """standing / walking KillDroid; feet at fl, fr forward of the base (and lift z in the 3rd item if given)."""
    hz = K_H if hz is None else hz
    wl = wl or (K_WS + 0.05, 0.05, K_WZ + 0.05)
    wr = wr or (-K_WS - 0.05, 0.05, K_WZ + 0.05)
    fl = fl if isinstance(fl, tuple) else (fl, 0.0)
    fr = fr if isinstance(fr, tuple) else (fr, 0.0)
    pose(kr, f, base, (sway, 0.0, hz + bob), wl, wr, (K_AS, fl[0], K_A + fl[1]), (-K_AS, fr[0], K_A + fr[1]), lean=lean, yaw=yaw)


def kd_walk(f0, f1, y0, y1, period=40, step=4):
    """heavy walk along -Y from y0 to y1, one key every `step` frames. feet stay planted in the world while they are on the ground."""
    n = f1 - f0
    v = (y0 - y1) / n                                # metres per frame, positive = toward -Y
    S = v * period * 0.6                             # stride length
    for f in range(f0, f1 + 1, step):
        yb = y0 - (f - f0) * v
        sw = 0.0
        feet = []
        for k, phase0 in enumerate((0.0, 0.5)):
            phi = (((f - f0) / period) + phase0) % 1.0
            if phi < 0.6:
                fy = S / 2 - S * (phi / 0.6)
                lz = 0.0
            else:
                u = (phi - 0.6) / 0.4
                fy = -S / 2 + S * smooth(u)
                lz = 0.20 * math.sin(math.pi * u)
            feet.append((fy, lz))
        # hips dip when the feet pass each other
        dip = -0.02 * abs(math.cos(2 * math.pi * ((f - f0) / period)))
        sway = 0.05 * math.sin(2 * math.pi * ((f - f0) / period))
        # arms swing a little opposite to the feet; the claw arm hangs heavier
        swing = math.sin(2 * math.pi * ((f - f0) / period))
        wl = (K_WS + 0.06, 0.05 + 0.25 * swing, K_WZ + 0.10)
        wr = (-K_WS - 0.06, 0.05 - 0.15 * swing, K_WZ + 0.10)
        kd_stand(f, (0.0, yb), bob=dip, sway=sway, wl=wl, wr=wr, fl=feet[0], fr=feet[1], hz=K_H - 0.07)


# ------------------------------------------------------------------------------------------------ Relic
R_H = REST[rr.name]["hips"].z
R_A = REST[rr.name]["ankle_L"].z
R_AS = abs(REST[rr.name]["ankle_L"].x)
print("Relic nums", R_H, R_A, R_AS)


def relic_guard(f, base, drop=0.12, lean=(0.10, 0.0), wl=None, wr=None, al=None, ar=None, yaw=0.0, hips_side=0.0, hips_fwd=0.0):
    """fight stance: knees bent, fists up in front of the chest. Feet: left foot a little back and out, right foot forward (orthodox-ish)."""
    wl = wl or (0.16, 0.50, 1.42)
    wr = wr or (-0.18, 0.60, 1.38)
    al = al or (R_AS + 0.14, -0.20, R_A)
    ar = ar or (-R_AS - 0.02, 0.28, R_A)
    pose(rr, f, base, (hips_side, hips_fwd, R_H - drop), wl, wr, al, ar, lean=lean, yaw=yaw)


# ================================================================================================ the timeline (24 fps)
F_LAND = 48
base_R = (0.0, 2.0)
# --- Relic stands at the crack lip, starts relaxed, takes the stance as the KillDroid lands
relic_guard(1, base_R, drop=0.0, lean=(0.0, 0.0), wl=(0.47, 0.03, 1.00), wr=(-0.47, 0.03, 1.00), al=(R_AS, 0.0, R_A), ar=(-R_AS, 0.0, R_A))
relic_guard(30, base_R, drop=0.0, lean=(0.02, 0.0), wl=(0.45, 0.10, 0.95), wr=(-0.45, 0.10, 0.95), al=(R_AS, 0.0, R_A), ar=(-R_AS, 0.0, R_A))
relic_guard(F_LAND + 8, base_R, drop=0.18, lean=(0.10, 0.0), wl=(0.30, 0.20, 1.10), wr=(-0.30, 0.20, 1.10), al=(R_AS + 0.10, -0.10, R_A), ar=(-R_AS - 0.02, 0.18, R_A))
relic_guard(F_LAND + 40, base_R)

# --- KillDroid drops from the bridge deck (hips start z 14+K_H at y 26, lands at y 12 at F_LAND)
exec(open(OUT + r'\msflow_retarget.py', encoding='utf-8').read())
GEN = "C:/ai/msflow/generations/walk1"
K_LEG, K_ARM = 1.30, 2.0                       # KillDroid leg / arm length over the human skeleton's
CLIP_A = load_clip(GEN + r"\sample_00_rep00.npy")
CLIP_B = load_clip(GEN + r"\sample_01_rep00.npy")
TRA, TRB = clip_travel(CLIP_A, K_LEG), clip_travel(CLIP_B, K_LEG)
print("CLIP TRAVEL", TRA, TRB, len(CLIP_A), len(CLIP_B))
Y_HOLD = 5.2
y_deck, z_deck = 26.0, 14.0
y_land = Y_HOLD + TRA + TRB + 0.4
T_FALL = (F_LAND - 6)
for f in range(1, F_LAND - 5, 6):
    u = (f - 1) / float(T_FALL - 1)                  # 0 at the edge, 1 at touchdown
    y = y_deck + (y_land - y_deck) * u
    hzf = K_H + (z_deck + K_H - K_H) * (1.0 - u * u)  # free fall: height ~ u squared
    dz = hzf - K_H                                       # everything falls together
    kd_stand(f, (0.0, y), hz=hzf, lean=(0.10 + 0.25 * u, 0.0),
             wl=(K_WS - 0.10, 0.25, dz + K_WZ + 0.55 + 0.2 * u), wr=(-K_WS + 0.10, 0.20, dz + K_WZ + 0.55 + 0.2 * u),
             fl=(0.10, dz + 0.55 - 0.40 * u), fr=(-0.10, dz + 0.45 - 0.35 * u))
# the landing: deep crouch, left (organic) fist on the ground, claw arm held back
kd_stand(F_LAND, (0.0, y_land), hz=K_H - 0.58, lean=(0.45, 0.0),
         wl=(K_WS + 0.05, 0.55, 0.20), wr=(-K_WS - 0.15, -0.35, K_WZ + 0.30),
         fl=(0.25, 0.0), fr=(-0.25, 0.45))
kd_stand(F_LAND + 10, (0.0, y_land), hz=K_H - 0.62, lean=(0.50, 0.0),
         wl=(K_WS + 0.05, 0.58, 0.18), wr=(-K_WS - 0.15, -0.30, K_WZ + 0.30),
         fl=(0.25, 0.0), fr=(-0.25, 0.45))
# rise slowly to full height (1.5 s)
kd_stand(F_LAND + 30, (0.0, y_land), hz=K_H - 0.25, lean=(0.20, 0.0),
         wl=(K_WS + 0.05, 0.40, 0.60), wr=(-K_WS - 0.05, 0.10, K_WZ + 0.15), fl=(0.20, 0.0), fr=(-0.20, 0.30))
kd_stand(F_LAND + 52, (0.0, y_land), hz=K_H - 0.02, lean=(0.04, 0.0), fl=(0.18, 0.0), fr=(-0.18, 0.0))

# --- the walk: 12 m -> 5.2 m over 168 frames (7 s)
F_W0 = F_LAND + 64
def put_clip(f0, clip, base0):
    keys = clip_to_keys_kd(kr, clip, f0, base0, K_LEG, K_ARM, ground_z=K_A)
    for fr, P in keys:
        KEYS.setdefault(kr.name, {})[fr] = {"hips": P["hips"], "neck": P["hips"], "wrist_L": P["wL"], "wrist_R": P["wR"], "ankle_L": P["aL"], "ankle_R": P["aR"]}
        XTRA.setdefault(kr.name, {})[fr] = (P["yaw"], P["lean"][0], P["lean"][1], P["tw"])
    return keys[-1][0], keys[-1][1]["hips"]
fA_end, hA_end = put_clip(F_W0, CLIP_A, (0.0, y_land))
fB0 = fA_end + 30                                # he stops, looks, and goes on
fB_end, hB_end = put_clip(fB0, CLIP_B, (0.0, hA_end.y - 0.4))
F_W1 = fB_end
print("WALK", F_W0, fA_end, fB0, fB_end, "ends at y", hB_end.y)
# stops, plants, opens the claw, looks at him
F_HOLD = F_W1 + 36
kd_stand(F_HOLD, (0.0, hB_end.y), hz=K_H - 0.12, lean=(0.10, 0.0), wl=(K_WS + 0.10, 0.20, K_WZ + 0.25), wr=(-K_WS - 0.10, 0.35, K_WZ + 0.40),
         fl=(0.22, 0.0), fr=(-0.22, 0.12))

# --- beat 1: the claw slash from high on his right, Relic steps out of it to his left
F_WIND = F_HOLD + 24                     # wind-up (1 s): he drops his weight, claw rises behind his head
F_STRIKE = F_WIND + 18                   # the slash lands at the old spot
F_REC = F_STRIKE + 40
kd_stand(F_WIND, (0.0, hB_end.y), hz=K_H - 0.32, lean=(-0.12, 0.10), wl=(K_WS + 0.20, 0.35, K_WZ + 0.35), wr=(-0.60, -0.25, 2.95),
         fl=(0.32, 0.05), fr=(-0.32, -0.10))
kd_stand(F_STRIKE - 7, (0.0, 4.3), hz=K_H - 0.45, lean=(0.20, 0.0), wl=(K_WS, 0.55, K_WZ + 0.35), wr=(-0.35, 0.55, 2.45),
         fl=(0.34, 0.45), fr=(-0.30, -0.20), yaw=-0.08)
kd_stand(F_STRIKE - 4, (0.0, 3.95), hz=K_H - 0.43, lean=(0.28, 0.0), wl=(K_WS - 0.02, 0.45, K_WZ + 0.2), wr=(-0.20, 0.70, 2.05),
         fl=(0.32, 0.55), fr=(-0.32, -0.20), yaw=-0.12)
kd_stand(F_STRIKE - 2, (0.0, 3.70), hz=K_H - 0.44, lean=(0.34, 0.0), wl=(K_WS - 0.05, 0.40, K_WZ + 0.15), wr=(-0.05, 0.95, 1.50),
         fl=(0.31, 0.62), fr=(-0.32, -0.20), yaw=-0.14)
kd_stand(F_STRIKE, (0.0, 3.45), hz=K_H - 0.46, lean=(0.38, 0.0), wl=(K_WS - 0.10, 0.30, K_WZ + 0.10), wr=(0.10, 1.15, 0.95),
         fl=(0.30, 0.70), fr=(-0.32, -0.20), yaw=-0.16)
kd_stand(F_REC, (0.0, 3.45), hz=K_H - 0.44, lean=(0.36, 0.0), wl=(K_WS - 0.10, 0.30, K_WZ + 0.10), wr=(0.10, 1.17, 0.92),
         fl=(0.30, 0.70), fr=(-0.32, -0.20), yaw=-0.16)

# Relic: sees the wind-up and steps out (his left) just before the strike
relic_guard(F_HOLD, base_R)
relic_guard(F_WIND + 8, base_R, drop=0.20, lean=(0.12, 0.0), hips_side=0.0)
relic_guard(F_STRIKE - 1, (-0.45, 1.95), drop=0.30, lean=(0.05, 0.28), wl=(0.30, 0.30, 1.12), wr=(-0.25, 0.42, 1.20),
            al=(R_AS + 0.30, 0.00, R_A), ar=(-R_AS + 0.25, 0.10, R_A), hips_side=0.0)
relic_guard(F_STRIKE + 8, (-0.85, 1.95), drop=0.24, lean=(0.10, 0.20), wl=(0.30, 0.40, 1.25), wr=(-0.20, 0.55, 1.28),
            al=(R_AS + 0.10, -0.05, R_A), ar=(-R_AS + 0.10, 0.18, R_A))
relic_guard(F_REC, (-0.85, 1.95), drop=0.20, lean=(0.10, 0.15))

F_END = F_REC
sc.frame_start, sc.frame_end = 1, F_END
print("KEYED UNTIL", F_END)

bake_ctrl(kr, rr, 1, F_END, (56, F_WIND, F_STRIKE - 5, F_HOLD, F_W0 + 100))
bake_ctrl(rr, kr, 1, F_END, (F_LAND + 8, F_WIND + 8, F_STRIKE - 1, F_STRIKE + 8, F_REC))
sc.frame_set(1)

# ================================================================================================ the first-person camera
R = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_4x4()
P = Matrix.Translation(Vector((0.0, -0.274, 1.862)))
pb = rr.pose.bones["head"]
cam.animation_data_clear()
cam.rotation_mode = 'QUATERNION'
rng = random.Random(7)
SHAKES = [(F_LAND, 0.9, 20), (F_STRIKE, 0.5, 14)]        # (frame, amount, decay frames): landing thud and the claw hitting the street
look_q = None
spring_q = None
for f in range(1, F_END + 1):
    sc.frame_set(f)
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    M = rr.matrix_world @ delta @ P @ R
    loc, q_head, _s = M.decompose()
    # look toward the KillDroid's head, mixing with the body's own head direction
    head_t = kr.matrix_world @ kr.pose.bones["head"].tail
    d = (head_t - loc)
    q_look = d.to_track_quat('-Z', 'Y')
    q_target = q_head.slerp(q_look, 0.75)
    if spring_q is None:
        spring_q = q_target.copy()
    spring_q = spring_q.slerp(q_target, 0.30)
    q = spring_q.copy()
    # camera shake (rotation) and a small walking bob in his breathing
    sh = 0.0
    for fs, amt, dec in SHAKES:
        if f >= fs:
            sh += amt * math.exp(-(f - fs) / dec)
    if sh > 0.001:
        n1, n2, n3 = (math.sin(f * 2.7) + 0.5 * math.sin(f * 5.1), math.sin(f * 3.3 + 1.0) + 0.5 * math.sin(f * 6.2), math.sin(f * 2.1 + 2.0))
        q = q @ Quaternion((0, 0, 1), 0.020 * sh * n1) @ Quaternion((1, 0, 0), 0.020 * sh * n2) @ Quaternion((0, 1, 0), 0.015 * sh * n3)
        loc = loc + Vector((0.01 * n2, 0.0, 0.015 * sh * n1)) * sh
    cam.location, cam.rotation_quaternion = loc, q
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
sc.camera = cam
sc.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\fight_v02.blend")
print("FIGHT ANIM SAVED")
