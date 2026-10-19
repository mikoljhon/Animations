# mender_anim4.py : cut 3 of the opening animatic. Dense, parameter-driven animation instead of sparse key poses:
#  - the pick angle alpha(t) is authored with real spacing: slow-in on the wind-up, a small anticipation lift, a cubic ease-IN strike
#    (fast right up to the contact), a 1-2 frame hold on the contact, then an ease-OUT rebound;
#  - the body follows alpha through second-order (spring-damper) filters with a lead, so it overshoots and settles after the pick stops
#    (overlapping action / follow-through); hips, chest and head each get their own frequency and damping;
#  - breathing and a head nod ride on top, the head tracks a look point that is itself filtered;
#  - every frame is solved with the arm IK so both hands stay exactly on the shaft;
#  - after the drop the pose is blended from "holding the pick" to "hands up" per body group, with overshoot and a stepped hop.
# Exec after mender_anim3.py (same call), window scene = Cave. Poses are solved WITHOUT animation on the rig (animated channels would be overwritten by
# every depsgraph update), cached to C:\ai\unedited\anim_cache, and keyed in one go by key_dense().

import json

F_TOTAL = globals().get("F_TOTAL_5", 216)
CACHE = os.path.join(OUT_DIR, "anim_cache")
FREEZE_LEAN = np.array([0.26, 0.55, 0.85, 0.52])
FREEZE_HIP = 0.34
DROP_F = globals().get("DROP_5", 106)


# ------------------------------------------------------------------ math helpers
def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def ease(kind, u):
    u = min(max(u, 0.0), 1.0)
    if kind == 'in':
        return u ** 3
    if kind == 'in2':
        return u ** 2
    if kind == 'out':
        return 1.0 - (1.0 - u) ** 3
    if kind == 'io':
        return smoothstep(u)
    return u


class SOD:
    """second-order dynamics: y + k1 y' + k2 y'' = x + k3 x'  (f = natural frequency in Hz, z = damping, r = initial response).
    z < 1 overshoots and rings, z = 1 is critically damped, r < 0 anticipates (moves the other way first)."""

    def __init__(self, f, z, r, x0):
        self.k1 = z / (math.pi * f)
        self.k2 = 1.0 / ((2.0 * math.pi * f) ** 2)
        self.k3 = r * z / (2.0 * math.pi * f)
        self.xp = np.array(x0, float)
        self.y = np.array(x0, float)
        self.yd = np.zeros_like(self.y)

    def step(self, T, x):
        x = np.array(x, float)
        xd = (x - self.xp) / T
        self.xp = x
        k2s = max(self.k2, T * T / 2.0 + T * self.k1 / 2.0, T * self.k1)
        self.y = self.y + T * self.yd
        self.yd = self.yd + T * (x + self.k3 * xd - self.y - self.k1 * self.yd) / k2s
        return self.y


def run_sod(fn, f0, z, r, frames=F_TOTAL, sub=8, start=1):
    """filter the signal fn(frame) through a SOD and return its value at each integer frame start..start+frames-1 (numpy array)"""
    x0 = np.array(fn(float(start)), float)
    s = SOD(f0, z, r, x0)
    T = (1.0 / 24.0) / sub
    out = []
    for i in range(frames):
        f = start + i
        if i > 0:
            for k in range(sub):
                s.step(T, fn(f - 1 + (k + 1) / sub))
        out.append(np.array(s.y, float).copy())
    return np.array(out)


def interp_ext(x, xs, ys):
    """linear interpolation with linear extrapolation beyond the ends"""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if x <= xs[0]:
        return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if x >= xs[-1]:
        return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(x, xs, ys))


def qpow(q, t):
    """the quaternion q raised to the power t (t outside 0..1 extrapolates)"""
    q = q.normalized()
    if q.w < 0:
        q = Quaternion((-q.w, -q.x, -q.y, -q.z))
    axis, ang = q.to_axis_angle()
    return Quaternion(axis, ang * t)


# ------------------------------------------------------------------ the authored pick angle (degrees, positive = head up)
ALPHA_SEGS = [
    (1.0, 33.0, 4.0, 36.0, 'io'),          # a little lift: anticipation
    (4.0, 36.0, 8.0, -3.0, 'in'),          # the strike: slow start, fastest right at the contact (f8)
    (8.0, -3.0, 9.5, -3.5, 'lin'),         # the contact: held and pressed
    (9.5, -3.5, 14.0, 12.0, 'out'),        # rebound off the stone: fast then slowing
    (14.0, 12.0, 22.0, 34.0, 'io'),        # the lift
    (22.0, 34.0, 25.0, 37.0, 'io'),        # anticipation again
    (25.0, 37.0, 31.0, -3.0, 'in'),        # strike 2 (contact f31)
    (31.0, -3.0, 32.5, -3.5, 'lin'),
    (32.5, -3.5, 37.0, 12.0, 'out'),
    (37.0, 12.0, 45.0, 44.0, 'io'),        # the big wind-up
    (45.0, 44.0, 51.0, 47.0, 'io'),        # up on his toes, a long beat at the top
    (51.0, 47.0, 58.0, -4.0, 'in'),        # the big strike (contact f58)
    (58.0, -4.0, 60.0, -4.5, 'lin'),
    (60.0, -4.5, 68.0, 6.0, 'out'),        # the pick sags back, he stops
    (68.0, 6.0, 72.0, 6.0, 'lin'),
    (72.0, 6.0, 75.0, 10.5, 'out'),        # flinch as the ring flickers on (f74)
    (75.0, 10.5, 82.0, 6.0, 'io'),
    (82.0, 6.0, 98.0, 6.0, 'lin'),         # frozen
    (98.0, 6.0, 101.0, 12.0, 'out'),       # flinch as the eyes light (f99)
    (101.0, 12.0, 106.0, 16.0, 'io'),      # hands pull up and back just before he lets go (f106)
    (106.0, 16.0, 400.0, 16.0, 'lin'),
]


def alpha_pick(f):
    for f0, a0, f1, a1, kind in ALPHA_SEGS:
        if f <= f1:
            if f <= f0:
                return a0
            return a0 + (a1 - a0) * ease(kind, (f - f0) / (f1 - f0))
    return ALPHA_SEGS[-1][3]


# body posture as a function of the pick angle (from the keyed poses of cut 2), extrapolated linearly so an overshoot keeps going
A_TAB = [-4.0, 0.0, 11.0, 33.0, 42.0, 47.0]
# alpha 0 is the base dig pose, whose spike tip sits exactly on the rim target; -4 leans a hair deeper (the body presses into the strike)
LEAN_TAB = np.array([[0.31, 0.69, 1.04, 0.62], [0.30, 0.66, 1.00, 0.60], [0.26, 0.58, 0.88, 0.54], [0.14, 0.38, 0.58, 0.36],
                     [0.08, 0.28, 0.42, 0.26], [0.05, 0.22, 0.34, 0.20]])
HIP_TAB = [0.325, 0.34, 0.36, 0.40, 0.43, 0.445]       # he rises out of the crouch for the wind-up and drops into the strike


def lean_of(a):
    a = max(a, -4.0)                       # no leaning past the hardest strike: the spike is on the stone, the overshoot is on the way back up
    return np.array([interp_ext(a, A_TAB, LEAN_TAB[:, k]) for k in range(4)])


def hip_of(a):
    return interp_ext(max(a, -4.0), A_TAB, HIP_TAB)


CONTACTS = [(8.0, 9.5), (31.0, 32.5), (58.0, 60.0)]      # frames where the spike is on the stone: alpha is solved so it touches exactly
TOUCH_SINK = 0.012                                       # the tip sits this far into the rubble


def in_contact(f):
    return any(a <= f <= b for a, b in CONTACTS)


def tip_height(rig, alpha, Mc, SRw, SLw):
    """height of the spike tip above the rubble under it for a pick angle alpha, with the torso where it is now"""
    axis = (SLw - SRw).normalized()
    Rm = Matrix.Translation(SRw) @ Matrix.Rotation(math.radians(alpha * SWING['sgn']), 4, axis) @ Matrix.Translation(-SRw)
    Mp = Rm @ Mc @ SWING['Rel']
    tip = Mp @ Vector(tuple(PICK_TIP_L))
    # signed distance of the tip to the plane through the rim target (tangent to the rubble there): the arc crosses it once, so the search is
    # monotonic (the real rubble is bumpy and the arc passes over Relic's shoulder pile, which fooled a height-above-ground test)
    return (tip - SWING['T']).dot(SWING['Tn'])


def touch_alpha(rig, lean, hip, look):
    """pose the torso (no arms yet) and find the pick angle at which the spike tip rests TOUCH_SINK below the rubble surface"""
    crouch(rig, DIG["C"], W=look, hip_h=hip, lean=lean, keep_object=True)
    Mc = chest_world(rig)
    SLw, SRw = Mc @ SWING['SL'], Mc @ SWING['SR']
    axis = (SLw - SRw).normalized()
    best = (1e9, 0.0)
    for k in range(-50, 51):                       # the arc of the tip about the shoulder line: where does it pass closest to the rim target?
        a = 0.5 * k
        Rm = Matrix.Translation(SRw) @ Matrix.Rotation(math.radians(a * SWING['sgn']), 4, axis) @ Matrix.Translation(-SRw)
        tip = (Rm @ Mc @ SWING['Rel']) @ Vector(tuple(PICK_TIP_L))
        d = (tip - SWING['T']).length
        if d < best[0]:
            best = (d, a)
    SWING['touch_d'] = best[0]
    return best[1]


def breath(f):
    """lean and hip additions from breathing: steady, then heavier for a few seconds after the exertion"""
    t = (f - 1) / 24.0
    heavy = math.exp(-((f - 78.0) / 22.0) ** 2)
    amp = 0.012 + 0.014 * heavy
    freq = 0.30 + 0.20 * heavy
    ph = 2 * math.pi * freq * t
    s = math.sin(ph + 0.4)
    return np.array([0.0, 0.004 * s, amp * s, 0.003 * s]), 0.004 * math.sin(ph + 0.9)


def compute_tracks():
    ring_p = np.array(ring_world()[0])
    vis_p = np.array(visor_world()[0])
    glint = np.array(GLINT)
    fr = np.arange(1, F_TOTAL + 1)

    def look_target(f):
        if f < 74.5:
            return glint
        if f < 99.5:
            return ring_p
        return vis_p

    T = {}
    def scalar_sod(fn, f0, z, r):
        return run_sod(lambda f: np.array([fn(f)]), f0, z, r)[:, 0]

    T['a_pick'] = np.array([alpha_pick(float(f)) for f in fr])
    T['a_body'] = scalar_sod(lambda f: alpha_pick(f + 3.0), 3.4, 0.55, 0.0)      # the body leads the pick, then overshoots about 13% on the way back up
    T['a_hip'] = scalar_sod(lambda f: alpha_pick(f + 3.5), 3.8, 0.50, 0.0)       # the hips lead more and ring faster
    T['a_head'] = scalar_sod(lambda f: alpha_pick(f - 0.5), 2.5, 0.45, 0.0)      # the head lags: it nods after the contact
    T['freeze'] = np.array([smoothstep((f - 60.0) / 10.0) for f in fr])
    T['look'] = run_sod(look_target, 5.5, 0.55, 0.0)
    # the scramble: weights per body group, stepped with overshoot
    T['wT'] = scalar_sod(lambda f: 1.0 if f >= 106.5 else 0.0, 2.4, 0.45, 0.0)
    T['wAL'] = scalar_sod(lambda f: 1.0 if f >= 107.5 else 0.0, 3.2, 0.30, 0.0)
    T['wAR'] = scalar_sod(lambda f: 1.0 if f >= 108.5 else 0.0, 3.0, 0.35, 0.0)
    T['uL'] = np.array([smoothstep((f - 107.5) / 8.5) for f in fr])
    T['uR'] = np.array([smoothstep((f - 110.5) / 10.5) for f in fr])
    T['back'] = np.array([0.34 + 0.06 * smoothstep((f - 125.0) / 25.0) for f in fr])
    T['hipB'] = np.array([0.48 + 0.02 * smoothstep((f - 125.0) / 25.0) for f in fr])
    return T


# ------------------------------------------------------------------ one frame
ARM_L = ['clavicle.L', 'upperarm.L', 'forearm.L', 'hand.L'] + ['%s%d.L' % (n, k) for n in ('index', 'middle', 'ring', 'pinky', 'thumb') for k in (1, 2, 3)]
ARM_R = ['clavicle.R', 'upperarm.R', 'forearm.R', 'hand.R'] + ['%s%d.R' % (n, k) for n in ('mi', 'mm', 'mr', 'mt') for k in (1, 2)]
TORSO = ['pelvis', 'spine', 'chest', 'neck', 'head']


def blend_groups(recA, recB, w):
    """pose-space blend with a weight per bone (quaternions by power, so a weight above 1 overshoots)"""
    out = {}
    for n, (la, qa, sa) in recA.items():
        lb, qb, sb = recB[n]
        t = w.get(n, 0.0)
        if t == 0.0:
            out[n] = (la.copy(), qa.copy(), sa.copy())
            continue
        d = qa.inverted() @ qb
        q = (qa @ qpow(d, t)).normalized()
        out[n] = (la.lerp(lb, t), q, sa.lerp(sb, t))
    return out


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
    if in_contact(float(f)):
        alpha = touch_alpha(rig, lean_t, float(hip), look)       # the spike is on the stone: land it exactly, whatever the body is doing
    miss = swing_pose(rig, alpha=alpha, lean=lean_t, hip_h=float(hip), look=look)
    miss['alpha'] = round(alpha, 2)
    rec = record_pose(rig)
    if f < DROP_F:
        return rec, miss
    # after the drop: blend toward the hands-up pose, group by group
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


def ser(rec):
    return {n: [list(l), list(q), list(s)] for n, (l, q, s) in rec.items()}


def des(d):
    return {n: (Vector(v[0]), Quaternion(v[1]), Vector(v[2])) for n, v in d.items()}


def setup_solver(rig):
    """put the rig and the pick in the base dig pose and remember the grip relations (no animation on the rig while solving)"""
    cave_state(False)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    place_mender(rig)
    r = swing_base(rig)
    # the rim target of the spike (see place_mender) and the slope normal there
    C = DIG["C"]
    fxy = Vector((GLINT.x - C[0], GLINT.y - C[1])).normalized()
    txy = Vector((GLINT.x, GLINT.y)) - fxy * DIG["dist"]
    g = surface_top(txy.x, txy.y)
    SWING['T'] = Vector((txy.x, txy.y, g[0].z - 0.01))
    SWING['Tn'] = Vector(g[1]).normalized()
    return r


def solve_range(a, b, T=None):
    rig = bpy.data.objects[CRIG]
    os.makedirs(CACHE, exist_ok=True)
    if T is None:
        T = compute_tracks()
    setup_solver(rig)
    worst = 0.0
    for f in range(a, b + 1):
        rec, miss = solve_frame(rig, f, T)
        worst = max(worst, abs(miss.get('miss_L', 0.0)), abs(miss.get('miss_R', 0.0)))
        with open(os.path.join(CACHE, "pose_%04d.json" % f), "w") as fh:
            json.dump(ser(rec), fh)
    return {"frames": (a, b), "worst_miss": round(worst, 3)}
