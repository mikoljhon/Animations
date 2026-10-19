# msflow_retarget.py : turn MSFlow joint clips (HumanML3D 22 joints, 20 fps, y up) into per-frame fight_anim keys for a character.
#   exec'd by fight_anim.py (needs numpy, Vector, math, KEYS, KDX, kr, kd_shoulders in scope)
import numpy as np, math
from mathutils import Vector

# HumanML3D joint indices
PEL, LHIP, RHIP, SP1, LKN, RKN, SP2, LAN, RAN, SP3, LFT, RFT, NECK, LCOL, RCOL, HEAD, LSH, RSH, LEL, REL, LWR, RWR = range(22)


def load_clip(path, trim=True, pad=6):
    J = np.load(path).astype(float)
    if trim:
        sp = np.linalg.norm(np.diff(J[:, PEL, [0, 2]], axis=0), axis=1) * 20.0
        mv = np.where(sp > 0.12)[0]
        if len(mv):
            a, b = max(0, mv[0] - pad), min(len(J) - 1, mv[-1] + pad)
            J = J[a:b + 1]
    return J


def resample(J, fps_in=20.0, fps_out=24.0):
    T = len(J)
    n = int((T - 1) / fps_in * fps_out) + 1
    t = np.arange(n) * fps_in / fps_out
    i0 = np.minimum(t.astype(int), T - 2)
    w = (t - i0)[:, None, None]
    return J[i0] * (1 - w) + J[i0 + 1] * w


def _heading(J):
    left = J[:, LHIP] - J[:, RHIP]
    left[:, 1] = 0
    left /= np.linalg.norm(left, axis=1, keepdims=True) + 1e-9
    # forward = left x up (up = +Y): (lx,0,lz) x (0,1,0) = (-lz, 0, lx)
    f = np.stack([-left[:, 2], np.zeros(len(J)), left[:, 0]], axis=1)
    return left, f


def clip_to_keys_kd(rig, J, f0, base0, k_leg, k_arm, hips_z=None, fwd_sign=-1.0, torso_k=1.0, ground_z=0.21):
    """add one KEYS/KDX entry per frame (24 fps) starting at frame f0. The KillDroid faces world -Y (fwd_sign -1)."""
    J = resample(J)
    left, f = _heading(J)
    psi = np.arctan2(f[:, 0], f[:, 2])
    psi = np.unwrap(psi) - psi[0]
    pel = J[:, PEL]
    ymin = J[:, [LAN, RAN], 1].min()
    n = len(J)
    out = []
    for i in range(n):
        fr = f0 + i
        # world mapping: gen (x, z) -> KD world (x, -z)  (left=+X, fwd=-Y)
        def W(p):
            return np.array([p[0], fwd_sign * p[2]])
        d = W(pel[i] - pel[0]) * k_leg
        bx, by = base0[0] + d[0], base0[1] + d[1]
        hz = (pel[i, 1] * k_leg) if hips_z is None else hips_z + (pel[i, 1] - pel[:, 1].mean()) * k_leg
        hips = Vector((bx, by, hz))
        anks = []
        for a in (LAN, RAN):
            v = W(J[i, a] - pel[i]) * k_leg
            anks.append(Vector((bx + v[0], by + v[1], ground_z + (J[i, a, 1] - ymin) * k_leg)))
        # torso lean and twist in the pelvis heading frame
        fh, lh = f[i], left[i]
        sp = J[i, NECK] - pel[i]
        sf, sl, su = float(np.dot(sp, fh)), float(np.dot(sp, lh)), float(sp[1])
        th = math.atan2(sf, su) * torso_k
        ph = math.atan2(sl, su) * torso_k
        sh = J[i, LSH] - J[i, RSH]
        sh[1] = 0
        shn = sh / (np.linalg.norm(sh) + 1e-9)
        tw = math.atan2(float(-np.dot(shn, fh)), float(np.dot(shn, lh))) * torso_k
        yaw = float(psi[i])
        P = {"hips": hips, "yaw": yaw, "lean": (th, ph), "tw": tw}
        shL, shR = kd_shoulders(rig, P)
        for key, wi, shi, shw in (("wL", LWR, LSH, shL), ("wR", RWR, RSH, shR)):
            rel = (J[i, wi] - J[i, shi]) * k_arm
            rf, rl, ru = float(np.dot(rel, fh)), float(np.dot(rel, lh)), float(rel[1])
            # KD heading vectors in world: fwd = Rz(yaw) @ (0,-1,0) = (sin, -cos); left = Rz(yaw) @ (1,0,0) = (cos, sin)
            fw = np.array([math.sin(yaw), -math.cos(yaw)])
            lw = np.array([math.cos(yaw), math.sin(yaw)])
            v = fw * rf + lw * rl
            P[key] = Vector((shw.x + v[0], shw.y + v[1], shw.z + ru))
        P["aL"], P["aR"] = anks
        out.append((fr, P))
    return out


def clip_travel(J, k_leg):
    """how far (metres, along the clip's start heading) the pelvis travels over the clip after scaling"""
    J = resample(J)
    d = J[-1, PEL] - J[0, PEL]
    return float(math.hypot(d[0], d[2])) * k_leg
