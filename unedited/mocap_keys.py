# mocap_keys.py : play a real motion-capture segment (cut from the CMU library by mocap_ref/cut_segments.py) on one of our rigs.
#   The captured joints are fitted onto the rig's proportions: hips/ankles scale with the leg length, fists with the arm length (measured from the
#   rig's own shoulders), knees and elbows become the IK pole targets (so the knees and elbows bend the way the real person's did), torso lean /
#   side lean / shoulder twist / foot yaw / heel lift come straight from the data.   exec'd in Blender (needs math, numpy, Vector, kd_shoulders)
import math
import numpy as np
from mathutils import Vector, Matrix

_J = ['hips', 'spine', 'chest', 'neck', 'head', 'eye_L', 'eye_R', 'clav_L', 'shoulder_L', 'elbow_L', 'wrist_L', 'knuckle_L', 'thumb_L', 'clav_R',
      'shoulder_R', 'elbow_R', 'wrist_R', 'knuckle_R', 'thumb_R', 'hip_L', 'knee_L', 'ankle_L', 'toe_L', 'hip_R', 'knee_R', 'ankle_R', 'toe_R']
JI = {n: i for i, n in enumerate(_J)}


exec(open(r"C:\ai\unedited\body_clear.py", encoding="utf-8").read())


def human_fist_clear(Lt, p):
    """clearance (metres, the fighter's own scale) of point p outside the fighter's torso and head. Lt: this frame's joints in metres,
    hips-relative (x = left, y = back, z = up)."""
    ax = Lt[JI['neck']] - Lt[JI['hips']]
    alen = float(np.linalg.norm(ax)); up = ax / max(alen, 1e-9)
    lf = Lt[JI['shoulder_L']] - Lt[JI['shoulder_R']]
    lf = lf - up * float(lf @ up); lf = lf / max(float(np.linalg.norm(lf)), 1e-9)
    fw = np.cross(lf, up)
    d = p - Lt[JI['hips']]
    ct, _rs, _dir = torso_clear(float(d @ up) / alen, float(d @ fw), float(d @ lf), HUMAN, alen)
    hc = Lt[JI['head']] + up * (0.12 * alen) + fw * (0.03 * alen)
    chd = float(np.linalg.norm(p - hc)) - HUMAN_HEAD_R * alen
    return ct, chd


def load_seg(path):
    d = np.load(path, allow_pickle=True)
    return dict(local=d['local'].astype(float), root=d['root'].astype(float), heading=d['heading'].astype(float), leg=float(d['leg']), fps=float(d['fps']),
                desc=str(d['desc']), clip=str(d['clip']))


def _resample(a, t):
    """linear sample of array a (frames first) at fractional frame positions t"""
    t = np.clip(t, 0, len(a) - 1)
    i0 = np.minimum(t.astype(int), len(a) - 2)
    w = (t - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[i0] * (1 - w) + a[i0 + 1] * w


def mocap_frames(rig, seg, f0, base, yawT, fwd_T, left_T, fist=(1.0, 1.0), slow=1.0, ankle_ground=None, ank0=None, hips_lift=0.0):
    """returns {frame: params} for kd_apply.  fwd_T / left_T = world unit vectors (x, y) of the fighter's T-frame (toward the opponent / his left)."""
    bn = rig.data.bones
    Lg = bn['thigh.L'].length + bn['shin.L'].length
    Ag = bn['upperarm.L'].length + bn['forearm.L'].length
    ank0 = bn['foot.L'].head_local.z if ank0 is None else ank0
    HIP_X = abs(bn['thigh.L'].head_local.x - bn['pelvis'].head_local.x)
    LEG = seg['leg']
    L = seg['local'] * LEG                           # metres, hips-relative, in the hips' heading frame (x = left, y = -forward, z = up)
    R = seg['root']; Hd = seg['heading']
    # measured limb lengths of the captured person
    def blen(a, b):
        return float(np.linalg.norm(L[:, JI[a]] - L[:, JI[b]], axis=1).mean())
    Ld = 0.5 * (blen('hip_L', 'knee_L') + blen('knee_L', 'ankle_L') + blen('hip_R', 'knee_R') + blen('knee_R', 'ankle_R'))
    Ad = 0.5 * (blen('shoulder_L', 'elbow_L') + blen('elbow_L', 'wrist_L') + blen('shoulder_R', 'elbow_R') + blen('elbow_R', 'wrist_R'))
    kl, ka = Lg / Ld, Ag / Ad
    # ground height (the lowest ankle) and start heading
    ank_w = np.stack([R[:, 2] + L[:, JI['ankle_L'], 2], R[:, 2] + L[:, JI['ankle_R'], 2]], axis=1)
    g = ank_w.min() if ankle_ground is None else ankle_ground
    h0 = Hd[0] / (np.linalg.norm(Hd[0]) + 1e-9)
    l0 = np.array([-h0[1], h0[0]])
    n_in = len(L)
    n_out = int((n_in - 1) / seg['fps'] * 24.0 * slow) + 1
    tsrc = np.arange(n_out) / 24.0 * seg['fps'] / slow
    Ls, Rs, Hs = _resample(L, tsrc), _resample(R, tsrc), _resample(Hd, tsrc)
    fT = np.array([fwd_T[0], fwd_T[1]]); lT = np.array([left_T[0], left_T[1]])
    out = {}
    psi_prev = 0.0
    _F0, _FY0 = {}, {}
    # foot baselines (flat at the start)
    def foot_ang(Lt, s):
        a, t = Lt[:, JI['ankle_' + s]], Lt[:, JI['toe_' + s]]
        hor = np.hypot(t[:, 0] - a[:, 0], t[:, 1] - a[:, 1]) + 1e-6
        return np.arctan2(a[:, 2] - t[:, 2], hor)
    fa = {s: foot_ang(Ls, s) for s in 'LR'}
    base_ang = {s: float(np.median(fa[s][:3])) for s in 'LR'}
    for i in range(n_out):
        h = Hs[i] / (np.linalg.norm(Hs[i]) + 1e-9)
        psi = math.atan2(float(h @ l0), float(h @ h0))
        psi = psi_prev + (math.atan2(math.sin(psi - psi_prev), math.cos(psi - psi_prev)))
        psi_prev = psi
        c, s_ = math.cos(psi), math.sin(psi)
        d = Rs[i, :2] - Rs[0, :2]
        df, dl = float(d @ h0), float(d @ l0)

        def vec(p, k, c=c, s_=s_):                   # bind THIS frame's heading now (arms are evaluated after the loop)
            """local (x, y, z) offset (metres) -> world vector (T-frame mapped), scaled by k"""
            x, y, z = p
            of = -y * c - x * s_
            ol = x * c + (-y) * s_
            return Vector((fT[0] * of + lT[0] * ol, fT[1] * of + lT[1] * ol, z)) * k
        hips_z = ank0 + (Rs[i, 2] - g) * kl + hips_lift
        hips = Vector((base[0] + (fT[0] * df + lT[0] * dl) * kl, base[1] + (fT[1] * df + lT[1] * dl) * kl, hips_z))
        P = {'hips': hips, 'yaw': yawT + psi}
        Lt = Ls[i]
        # torso
        neck = Lt[JI['neck']]
        th = math.atan2(-neck[1], neck[2])
        ph = math.atan2(neck[0], neck[2])
        sv = Lt[JI['shoulder_L']] - Lt[JI['shoulder_R']]
        hv = Lt[JI['hip_L']] - Lt[JI['hip_R']]
        tw = math.atan2(sv[1], sv[0]) - math.atan2(hv[1], hv[0])
        tw = math.atan2(math.sin(tw), math.cos(tw))
        P['lean'] = (th, ph)
        spv = Lt[JI['spine']] - Lt[JI['hips']]
        hl, hr = Lt[JI['hip_L']], Lt[JI['hip_R']]
        P['plean'] = (math.atan2(-spv[1], spv[2]), math.atan2(-(hl[2] - hr[2]), max(1e-6, hl[0] - hr[0])))   # real pelvis tilt: forward, toward his left
        P['tw'] = tw
        # the real thorax frame for the spine fit (kd_apply): up = the hips -> neck line, left = the shoulder line; columns left, back, up
        upv = Vector(neck).normalized()
        lfv = Vector(sv)
        lfv = (lfv - upv * lfv.dot(upv)).normalized()
        P['thorax'] = Matrix((lfv, upv.cross(lfv), upv)).transposed()
        P['trunk'] = Vector(neck)
        # ankles, knees (poles)
        # the rig's hips are wider than the captured person's: move each foot out by the difference (along the pelvis's own left axis) so the
        # legs keep the real person's angle instead of slanting inward (foot lock below keeps planted feet planted)
        half_w = 0.5 * abs(float(Lt[JI['hip_L'], 0] - Lt[JI['hip_R'], 0])) * kl
        dw = max(0.0, HIP_X - half_w)
        lv = vec(np.array([1.0, 0.0, 0.0]), 1.0)
        lv.z = 0.0
        lv = lv.normalized() if lv.length > 1e-6 else lv
        for sd in 'LR':
            a = Lt[JI['ankle_' + sd]]
            av = vec(a, kl)
            sgn = 1.0 if sd == 'L' else -1.0
            P['a' + sd] = Vector((hips.x + av.x + lv.x * dw * sgn, hips.y + av.y + lv.y * dw * sgn, ank0 + (Rs[i, 2] + a[2] - g) * kl))
            kn = Lt[JI['knee_' + sd]]
            kv = vec(kn, kl)
            P['knee_' + sd] = Vector((hips.x + kv.x, hips.y + kv.y, ank0 + (Rs[i, 2] + kn[2] - g) * kl))
            P['krel_' + sd] = vec(kn - Lt[JI['hip_' + sd]], kl)     # knee offset from the REAL hip joint (the rig's hips are wider)
            # a RAISED foot is placed from the rig's own hip joint (ankle offset from the real hip, scaled): the knee bends exactly as much as
            # the fighter's. A foot on the ground keeps the widened stance + foot lock above. aground = 1 on the ground .. 0 from 15 cm up.
            P['arel_' + sd] = vec(a - Lt[JI['hip_' + sd]], kl)
            hgt = (Rs[i, 2] + a[2] - g) * kl
            wg = min(1.0, max(0.0, 1.0 - hgt / 0.15))
            P['aground_' + sd] = wg * wg * (3 - 2 * wg)
        # foot yaw (relative to the pelvis heading) and heel lift
        fy, fp = [], {}
        for sd in 'LR':
            a, t = Lt[JI['ankle_' + sd]], Lt[JI['toe_' + sd]]
            fy.append(math.atan2(t[0] - a[0], -(t[1] - a[1])))
            fp[sd] = float(np.clip(fa[sd][i] - base_ang[sd], -0.9, 1.0))
        P['fyaw'] = tuple(fy)
        P['fpitch'] = fp
        # full 3D foot orientation relative to the first (flat) frame
        fd, fy0 = {}, {}
        for sd in 'LR':
            a, t_, k_ = Lt[JI['ankle_' + sd]], Lt[JI['toe_' + sd]], Lt[JI['knee_' + sd]]
            dvec = vec(t_ - a, 1.0).normalized()
            uvec = vec(k_ - a, 1.0)
            uvec = (uvec - dvec * uvec.dot(dvec)).normalized()
            M = Matrix((dvec.cross(uvec), dvec, uvec)).transposed()           # columns: x = y cross z, y = toe direction, z = up
            if sd not in _F0:
                _F0[sd] = M.copy()
                hd = Vector((dvec.x, dvec.y))
                fy0[sd] = yawT + math.atan2(hd.x * left_T[0] + hd.y * left_T[1], hd.x * fwd_T[0] + hd.y * fwd_T[1])
                _FY0[sd] = fy0[sd]
            fd[sd] = M @ _F0[sd].inverted()
        P['fdelta'] = fd
        P['fyaw0'] = dict(_FY0)
        out[f0 + i] = (P, Lt, kl, ka, vec)
    # foot lock: while a foot is down and (nearly) still, pin it where it landed so it does not skate; short blend at each end
    frs = sorted(out)
    for sd in 'LR':
        key = 'a' + sd
        pos = [out[f][0][key].copy() for f in frs]
        spd = [0.0] + [(pos[i].xy - pos[i - 1].xy).length for i in range(1, len(pos))]
        low = [(pos[i].z < ank0 + 0.04 * kl) for i in range(len(pos))]
        still = [low[i] and (spd[i] < 0.010 * kl or (i + 1 < len(pos) and spd[i + 1] < 0.010 * kl and spd[i] < 0.02 * kl)) for i in range(len(pos))]
        i = 0
        while i < len(pos):
            if still[i]:
                j = i
                while j + 1 < len(pos) and still[j + 1]:
                    j += 1
                if j - i >= 3:
                    anchor = pos[i + (j - i) // 2].xy.copy()
                    for k in range(i, j + 1):
                        w = min(1.0, (k - i + 1) / 3.0, (j - k + 1) / 3.0)
                        P_ = out[frs[k]][0]
                        P_[key] = Vector((pos[k].x * (1 - w) + anchor.x * w, pos[k].y * (1 - w) + anchor.y * w, pos[k].z))
                i = j + 1
            else:
                i += 1
    # arms need the shoulders of the rig for the given torso pose
    res = {}
    for fr, (P, Lt, kl_, ka_, vec) in out.items():
        shL, shR = kd_shoulders(rig, P)
        for sd, sh in (('L', shL), ('R', shR)):
            s_loc = Lt[JI['shoulder_' + sd]]
            w = vec(Lt[JI['wrist_' + sd]] - s_loc, ka_)
            e = vec(Lt[JI['elbow_' + sd]] - s_loc, ka_)
            P['w' + sd] = Vector((sh.x + w.x, sh.y + w.y, sh.z + w.z))
            P['elbow_' + sd] = Vector((sh.x + e.x, sh.y + e.y, sh.z + e.z))
            P['wrel_' + sd] = Vector((w.x, w.y, w.z))              # offsets from the TRUE shoulder (the guard re-aims the hand with these once the torso is posed)
            P['erel_' + sd] = Vector((e.x, e.y, e.z))
            # how far the fighter's fist was OUTSIDE their own (slim) body, scaled to Relic: the solver keeps Relic's fist at least that far
            # outside HIS armour (body_clear.py)
            hp = 0.5 * (Lt[JI['wrist_' + sd]] + Lt[JI['knuckle_' + sd]])
            ct, chd = human_fist_clear(Lt, hp)
            P['hcl_' + sd] = (None if ct is None else ct * ka_, chd * ka_)
            hd = vec(Lt[JI['knuckle_' + sd]] - Lt[JI['wrist_' + sd]], 1.0)
            P['hdir_' + sd] = hd.normalized() if hd.length > 1e-6 else Vector((0.0, 0.0, -1.0))
        P['fist'] = fist
        res[fr] = P
    _stabilize(res, rig, kl, ka)
    return res


def blend_params(P0, P1, t):
    """linear blend of two param dicts (t 0..1)"""
    t = max(0.0, min(1.0, t))
    t = t * t * (3 - 2 * t)
    P = {}
    for k in P0:
        a, b = P0[k], P1.get(k, P0[k])
        if isinstance(a, Vector):
            P[k] = a * (1 - t) + b * t
        elif isinstance(a, (tuple, list)):
            P[k] = tuple(x * (1 - t) + y * t for x, y in zip(a, b))
        elif isinstance(a, dict):
            P[k] = {}
            for kk in a:
                if isinstance(a[kk], Matrix):               # rotations: blend as quaternions, not as raw matrices
                    P[k][kk] = a[kk].to_quaternion().slerp(b[kk].to_quaternion(), t).to_matrix()
                else:
                    P[k][kk] = a[kk] * (1 - t) + b[kk] * t
        elif isinstance(a, (int, float)):
            P[k] = a * (1 - t) + b * t
        else:
            P[k] = a
    return P


def mocap_world_joints(seg, f0, base, fwd_T, left_T, k=1.0, slow=1.0, ank_h=0.08):
    """the raw captured skeleton in world coordinates (uniformly scaled by k), for a reference figure: {frame: {joint: Vector}}"""
    LEG = seg['leg']
    L = seg['local'] * LEG
    R = seg['root']; Hd = seg['heading']
    ank_w = np.stack([R[:, 2] + L[:, JI['ankle_L'], 2], R[:, 2] + L[:, JI['ankle_R'], 2]], axis=1)
    g = ank_w.min()
    h0 = Hd[0] / (np.linalg.norm(Hd[0]) + 1e-9)
    l0 = np.array([-h0[1], h0[0]])
    n_out = int((len(L) - 1) / seg['fps'] * 24.0 * slow) + 1
    tsrc = np.arange(n_out) / 24.0 * seg['fps'] / slow
    Ls, Rs, Hs = _resample(L, tsrc), _resample(R, tsrc), _resample(Hd, tsrc)
    fT = np.array(fwd_T); lT = np.array(left_T)
    out = {}
    psi_prev = 0.0
    for i in range(n_out):
        h = Hs[i] / (np.linalg.norm(Hs[i]) + 1e-9)
        psi = math.atan2(float(h @ l0), float(h @ h0))
        psi = psi_prev + math.atan2(math.sin(psi - psi_prev), math.cos(psi - psi_prev))
        psi_prev = psi
        c, s_ = math.cos(psi), math.sin(psi)
        d = Rs[i, :2] - Rs[0, :2]
        df, dl = float(d @ h0), float(d @ l0)
        hx = base[0] + (fT[0] * df + lT[0] * dl) * k
        hy = base[1] + (fT[1] * df + lT[1] * dl) * k
        joints = {}
        for n, j in JI.items():
            x, y, z = Ls[i, j]
            of = -y * c - x * s_
            ol = x * c + (-y) * s_
            joints[n] = Vector((hx + (fT[0] * of + lT[0] * ol) * k, hy + (fT[1] * of + lT[1] * ol) * k, ank_h + (Rs[i, 2] + z - g) * k))
        out[f0 + i] = joints
    return out


def _unit_perp(v, ax):
    p = v - ax * v.dot(ax)
    return p


def _stab_pole(res, frs, root_fn, end_key, pole_key, min_off, max_turn_deg=None):
    """follow the REAL bend direction exactly; only while the limb is nearly straight (the real knee/elbow almost on the root-end line, so its
    direction is meaningless) lean on the previous frame's direction, blending smoothly back to the real one as the limb bends.
    (The old version also capped the turn per frame; that made the elbow lag onto the wrong side for many frames.)"""
    prev = None
    for fr in frs:
        P = res[fr]
        root = root_fn(P)
        end = P[end_key]
        axv = end - root
        if axv.length < 1e-6:
            continue
        ax = axv.normalized()
        v = P[pole_key] - root
        perp = _unit_perp(v, ax)
        m = perp.length
        d = perp / m if m > 1e-6 else None
        if prev is not None:
            pp = _unit_perp(prev, ax)
            pp = pp.normalized() if pp.length > 1e-4 else None
            if pp is not None:
                w = max(0.0, min(1.0, m / min_off))
                w = w * w * (3 - 2 * w)
                d = pp if d is None else (pp * (1 - w) + d * w)
                d = d.normalized() if d.length > 1e-6 else pp
        if d is None:
            d = _unit_perp(Vector((0, 0, -1)) if abs(ax.z) < 0.9 else Vector((0, -1, 0)), ax).normalized()
        prev = d
        P[pole_key] = root + ax * v.dot(ax) + d * max(m, min_off * 0.5)


def _fix_rot_outliers(res, frs, key, sd, max_deg=70.0):
    """a captured foot whose direction suddenly flips (bad capture frames): replace outlier orientations by interpolation between good neighbours"""
    mats = [res[f][key][sd] for f in frs]
    n = len(mats)
    bad = [False] * n
    last = 0
    for i in range(1, n):
        ang = math.degrees(mats[i].to_quaternion().rotation_difference(mats[last].to_quaternion()).angle)
        if ang > 180:
            ang = 360 - ang
        if ang > max_deg:
            bad[i] = True
        else:
            last = i
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            a = mats[i - 1] if i > 0 else None
            b = mats[j] if j < n else None
            for k in range(i, j):
                if a is not None and b is not None:
                    t = (k - i + 1) / (j - i + 1)
                    mats[k] = a.to_quaternion().slerp(b.to_quaternion(), t).to_matrix()
                else:
                    mats[k] = (a or b).copy()
            i = j
        else:
            i += 1
    for f, m in zip(frs, mats):
        res[f][key][sd] = m


def _stabilize(res, rig, kl, ka):
    frs = sorted(res)
    bn = rig.data.bones
    Lg = bn['thigh.L'].length + bn['shin.L'].length
    Ag = bn['upperarm.L'].length + bn['forearm.L'].length
    for sd in 'LR':
        _fix_rot_outliers(res, frs, 'fdelta', sd)
        _stab_pole(res, frs, lambda P: P['hips'], 'a' + sd, 'knee_' + sd, 0.12 * Lg)
        sg = 1.0 if sd == 'L' else -1.0
        _stab_pole(res, frs, lambda P, sd=sd: kd_shoulders(rig, P)[0 if sd == 'L' else 1], 'w' + sd, 'elbow_' + sd, 0.12 * Ag)
