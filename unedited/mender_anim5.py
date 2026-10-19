# mender_anim5.py : cut 3, part 2: key the cached dense poses, Relic's head lift with follow-through, secondary motion on the gear (spring sims),
# eased and hand-held cameras. Exec after mender_anim4.py (same call); window scene = Cave.

# ------------------------------------------------------------------ keys from the cache
def load_cache(frames=F_TOTAL):
    recs = {}
    for f in range(1, frames + 1):
        with open(os.path.join(CACHE, "pose_%04d.json" % f)) as fh:
            recs[f] = des(json.load(fh))
    return recs


def key_dense(rig, recs):
    clear_anim(rig)
    moving = set()
    for rec in recs.values():
        for n, (l, q, s) in rec.items():
            if l.length > 1e-5 or abs(q.w - 1.0) > 1e-5 or abs(q.x) > 1e-5 or abs(q.y) > 1e-5 or abs(q.z) > 1e-5:
                moving.add(n)
    prev = {}
    n_keys = 0
    for f in sorted(recs):
        for n in sorted(moving):
            l, q, s = recs[f][n]
            q = q.copy()
            if n in prev and prev[n].dot(q) < 0:
                q.negate()
            prev[n] = q.copy()
            pb = rig.pose.bones[n]
            pb.location = l
            pb.rotation_quaternion = q
            pb.keyframe_insert('location', frame=f)
            pb.keyframe_insert('rotation_quaternion', frame=f)
            n_keys += 2
    for fc in iter_fcurves(rig):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
    return {"bones": len(moving), "keys": n_keys}


# ------------------------------------------------------------------ Relic's head lift with follow-through
def lift_relic(rr, a_neck, a_head):
    for n, amt in (('neck', a_neck), ('head', a_head)):
        d, tw = LIFT[n]
        base = POSE[n]
        dd = (base[0][0] + (d[0] - base[0][0]) * amt, base[0][1] + (d[1] - base[0][1]) * amt, 1.0)
        _aim(rr, n, dd, base[1] + (tw - base[1]) * amt)


def relic_head_tracks():
    """the lift is a step input at f104 through two spring filters: the head overshoots about 25% and settles, the neck is slower and softer"""
    step = lambda f: np.array([1.0 if f >= 104.0 else 0.0])
    ah = run_sod(step, 1.9, 0.42, 0.0)[:, 0]
    an = run_sod(step, 1.4, 0.55, 0.0)[:, 0]
    return an, ah


def compute_relic_quats(rr):
    cave_pose(True)
    for n in ('neck', 'head'):
        to_quat_mode(rr.pose.bones[n])
    an, ah = relic_head_tracks()
    out = {'neck': [], 'head': []}
    base = {n: rr.pose.bones[n].rotation_quaternion.copy() for n in ('neck', 'head')}
    for i in range(F_TOTAL):
        # _aim is incremental (its twist adds to the roll the bone already has), so every frame starts again from the buried pose
        for n in ('neck', 'head'):
            rr.pose.bones[n].rotation_quaternion = base[n]
        bpy.context.view_layer.update()
        lift_relic(rr, float(an[i]), float(ah[i]))
        for n in ('neck', 'head'):
            out[n].append(rr.pose.bones[n].rotation_quaternion.copy())
    cave_pose(True)
    for n in ('neck', 'head'):
        to_quat_mode(rr.pose.bones[n])
    return out


def key_relic_dense(rr, quats):
    clear_anim(rr)
    for n in ('neck', 'head'):
        pb = rr.pose.bones[n]
        prev = None
        for i, q in enumerate(quats[n]):
            q = q.copy()
            if prev is not None and prev.dot(q) < 0:
                q.negate()
            prev = q.copy()
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=i + 1)
    for fc in iter_fcurves(rr):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
    return "relic keyed"


# ------------------------------------------------------------------ secondary motion: spring-damper pendulums for the gear
ACC = {   # group name -> (bone, objects, natural frequency Hz, damping, gravity scale, pivot at 'low' or 'high' end, inertia gain)
    'antenna': ('chest', ['Antenna', 'AntennaFlag'], 2.4, 0.12, 0.0, 'low', 0.10),
    'scarf': ('chest', ['ScarfTail'], 1.2, 0.30, 0.15, 'high', 0.40),      # gravity only a little free: the cloth lies against the back
    'cable': ('chest', ['PackCable'], 0.9, 0.22, 0.15, 'high', 0.40),
    'wrench': ('pelvis', ['WrenchShaft', 'WrenchHead'], 0.9, 0.35, 1.0, 'high', 0.30),
}


TARGET_DEG = {'antenna': 14.0, 'scarf': 30.0, 'cable': 26.0, 'wrench': 20.0}      # the biggest swing each piece of gear should make in the whole clip


def mesh_points(ob):
    co = np.empty(len(ob.data.vertices) * 3, dtype=np.float32)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(float)


def parent_matrix(rig, bone):
    pb = rig.pose.bones[bone]
    B = pb.matrix.copy()
    B.translation = pb.tail
    return rig.matrix_world @ B


def sim_pendulum(P, Rm, rhat, L, f0, z, gs, gain, sub=8, amp_clamp=0.75):
    """P (N,3) world position of the pivot per frame, Rm (N,3,3) object->world rotation; returns the rotation vectors theta (N,3) in the OBJECT frame"""
    N = len(P)
    dt = 1.0 / 24.0
    acc = np.zeros_like(P)
    acc[1:-1] = (P[2:] - 2 * P[1:-1] + P[:-2]) / (dt * dt)
    acc[0], acc[-1] = acc[1], acc[-2]
    acc = np.clip(acc, -45.0, 45.0)
    for k in range(3):                                   # a light 3-tap smooth: one frame of acceleration is a coarse sample of a sharp hit
        acc[1:-1, k] = 0.25 * acc[:-2, k] + 0.5 * acc[1:-1, k] + 0.25 * acc[2:, k]
    w0 = 2.0 * math.pi * f0
    h = dt / sub
    th = np.zeros(3)
    om = np.zeros(3)
    out = np.zeros((N, 3))
    g = np.array([0.0, 0.0, -9.81]) * gs
    for i in range(N):
        out[i] = th
        R = Rm[i]
        Rinv = R.T
        j = min(i + 1, N - 1)
        for s in range(sub):
            a_w = acc[i] + (acc[j] - acc[i]) * (s / sub)
            force = Rinv @ (g - a_w * gain)
            ang = np.linalg.norm(th)
            if ang > 1e-9:
                K = th / ang
                rh = rhat * math.cos(ang) + np.cross(K, rhat) * math.sin(ang) + K * (K @ rhat) * (1 - math.cos(ang))
            else:
                rh = rhat
            alpha = np.cross(rh, force) / L - (w0 ** 2) * th - 2.0 * z * w0 * om
            om = om + alpha * h
            th = th + om * h
            n = np.linalg.norm(th)
            if n > amp_clamp:
                th = th * (amp_clamp / n)
                om = om * 0.5
    return out


def key_accessories(sc, rig, strength=1.0):
    """spring sims for the gear that hangs or sticks out; keys the objects' own location/rotation (their parent bones are already animated)"""
    pos = {g: [] for g in ACC}
    rot = {g: [] for g in ACC}
    for f in range(1, F_TOTAL + 1):
        sc.frame_set(f)
        for g, (bone, names, *_r) in ACC.items():
            ob = bpy.data.objects[CP + names[0]]
            W0 = parent_matrix(rig, bone) @ ob.matrix_parent_inverse
            pos[g].append(W0)
    report = {}
    for g, (bone, names, f0, z, gs, pivot, gain) in ACC.items():
        objs = [bpy.data.objects[CP + n] for n in names]
        pts = np.concatenate([mesh_points(o) for o in objs])
        zi = pts[:, 2]
        end = pts[zi <= np.percentile(zi, 8)] if pivot == 'low' else pts[zi >= np.percentile(zi, 92)]
        piv = end.mean(axis=0)
        com = pts.mean(axis=0)
        r = com - piv
        L = max(float(np.linalg.norm(r)), 0.05)
        rhat = r / L
        mats = pos[g]
        P = np.array([m @ Vector(tuple(piv)) for m in mats])
        Rm = np.array([np.array(m.to_3x3()) for m in mats])
        target = math.radians(TARGET_DEG[g]) * strength
        gn = gain
        for _it in range(4):                       # scale the inertial gain until the biggest swing is about the target angle
            th = sim_pendulum(P, Rm, rhat, L, f0, z, gs, gn)
            m = float(np.linalg.norm(th, axis=1).max())
            if abs(m - target) < 0.03 * target:
                break
            gn = gn * (target / max(m, 1e-3)) if m < 0.7 else gn * 0.5
        report[g] = (round(float(np.degrees(np.linalg.norm(th, axis=1).max())), 1), round(gn, 3))
        for ob in objs:
            clear_anim(ob)
            ob.rotation_mode = 'QUATERNION'
            prevq = None
            for i in range(F_TOTAL):
                ang = float(np.linalg.norm(th[i]))
                if ang < 1e-8:
                    Rot = Matrix.Identity(4)
                else:
                    Rot = Matrix.Rotation(ang, 4, Vector(tuple(th[i] / ang)))
                pv = Vector(tuple(piv))
                B = Matrix.Translation(pv) @ Rot @ Matrix.Translation(-pv)
                loc, q, _s = B.decompose()
                if g == 'antenna' and ob.name.endswith('AntennaFlag'):
                    t = i / 24.0
                    fl = 0.45 * math.sin(2 * math.pi * 4.3 * t) + 0.25 * math.sin(2 * math.pi * 7.7 * t + 1.1)
                    ax = Vector(tuple(rhat))
                    Bf = B @ (Matrix.Translation(pv) @ Matrix.Rotation(fl, 4, ax) @ Matrix.Translation(-pv))
                    loc, q, _s = Bf.decompose()
                if prevq is not None and prevq.dot(q) < 0:
                    q.negate()
                prevq = q.copy()
                ob.location = loc
                ob.rotation_quaternion = q
                ob.keyframe_insert('location', frame=i + 1)
                ob.keyframe_insert('rotation_quaternion', frame=i + 1)
            for fc in iter_fcurves(ob):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'
    return report


# ------------------------------------------------------------------ cameras: eased moves, a little hand-held drift, reactions on the beats
def hh_noise(t, seed, amp, base=0.45):
    rng_ = np.random.default_rng(seed)
    ph = rng_.uniform(0, 2 * math.pi, 9).reshape(3, 3)
    out = []
    for k in range(3):
        v = 0.0
        for j, fq in enumerate((1.0, 2.3, 4.1)):
            v += (0.6 ** j) * math.sin(2 * math.pi * base * fq * t + ph[k, j])
        out.append(v * amp)
    return Vector(out)


def kick(f, f0, amp, rise=2.0, fall=10.0):
    """a push that comes in fast and eases back: 0 before f0"""
    if f < f0:
        return 0.0
    if f < f0 + rise:
        return amp * smoothstep((f - f0) / rise)
    return amp * (1.0 - smoothstep((f - f0 - rise) / fall))


def key_cam(cam, tgt, f0, f1, loc_fn, tgt_fn):
    clear_anim(cam)
    clear_anim(tgt)
    for f in range(f0, f1 + 1):
        cam.location = loc_fn(f)
        cam.keyframe_insert('location', frame=f)
        tgt.location = tgt_fn(f)
        tgt.keyframe_insert('location', frame=f)


def build_cameras3(sc, face):
    sc.timeline_markers.clear()
    cams = {}
    # S1 wide: an eased push, hand-held, a tiny jolt on each strike
    P0, P1 = Vector((-2.0, -3.0, 1.5)), Vector((-1.75, -2.65, 1.45))
    A = Vector((0.6, 1.0, 1.0))
    c, t = make_cam(sc, "Anim_Cam_Wide", P0, A, 34.0)
    dirv = (A - P0).normalized()
    key_cam(c, t, 1, 48,
            lambda f: P0.lerp(P1, smoothstep((f - 1) / 47.0)) + hh_noise((f - 1) / 24.0, 3, 0.018) + dirv * (kick(f, 8, 0.012) + kick(f, 31, 0.012)),
            lambda f: A + hh_noise((f - 1) / 24.0, 4, 0.02))
    cams['S1'] = (c, 1)
    # S2 dig: push, the big strike at f58 hits the camera, then the aim rises with Mender
    P0, P1 = Vector((-0.7, -1.3, 1.6)), Vector((-0.45, -1.0, 1.55))
    A0, A1 = Vector((0.75, 1.0, 1.1)), Vector((0.98, 0.93, 1.38))
    c, t = make_cam(sc, "Anim_Cam_Dig", P0, A0, 36.0)
    dirv = (A0 - P0).normalized()
    key_cam(c, t, 49, 96,
            lambda f: P0.lerp(P1, smoothstep((f - 49) / 47.0)) + hh_noise((f - 49) / 24.0, 5, 0.016) + shake(f, 58, 0.045, 9) + dirv * kick(f, 58, 0.05, 2.0, 12.0),
            lambda f: A0.lerp(A1, smoothstep((f - 64.0) / 28.0)) + hh_noise((f - 49) / 24.0, 6, 0.018))
    cams['S2'] = (c, 49)
    # S3 two-shot: slow push; a nudge when the eyes light (f99) and a jolt at the drop (f106)
    P0, P1 = Vector((-1.3, -1.4, 2.0)), Vector((-1.1, -1.15, 1.95))
    A = Vector((0.78, 0.97, 1.4))
    c, t = make_cam(sc, "Anim_Cam_Two", P0, A, 32.0)
    dirv = (A - P0).normalized()
    key_cam(c, t, 97, 168,
            lambda f: P0.lerp(P1, smoothstep((f - 97) / 71.0)) + hh_noise((f - 97) / 24.0, 7, 0.016) + dirv * (kick(f, 99, 0.04, 3.0, 14.0)) + shake(f, 106, 0.02, 7),
            lambda f: A + hh_noise((f - 97) / 24.0, 8, 0.02))
    cams['S3'] = (c, 97)
    # S4 Relic's face: slow push-in, shallow depth of field following the move
    cpos, vtgt, cpos2, vtgt2, info, info2 = face
    c, t = make_cam(sc, "Anim_Cam_Face", cpos, vtgt, 55.0, dof=3.2)
    key_cam(c, t, 169, F_TOTAL,
            lambda f: cpos.lerp(cpos2, smoothstep((f - 169) / 47.0)) + hh_noise((f - 169) / 24.0, 9, 0.006),
            lambda f: vtgt.lerp(vtgt2, smoothstep((f - 169) / 47.0)) + hh_noise((f - 169) / 24.0, 10, 0.006))
    c.data.dof.focus_distance = (vtgt - cpos).length
    c.data.keyframe_insert('dof.focus_distance', frame=169)
    c.data.dof.focus_distance = (vtgt2 - cpos2).length
    c.data.keyframe_insert('dof.focus_distance', frame=F_TOTAL)
    cams['S4'] = (c, 169)
    for nm, (cam, f) in cams.items():
        m = sc.timeline_markers.new(nm, frame=f)
        m.camera = cam
    sc.camera = cams['S1'][0]
    return "cameras keyed"
