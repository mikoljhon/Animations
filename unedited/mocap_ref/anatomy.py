# anatomy.py : is a captured pose physically possible for a human? (works on joint positions only)
#   The elbow "point" (the bony tip, opposite to where the forearm swings) can only face a limited arc for a given upper-arm direction:
#   backward when the arm hangs, down when the arm is raised forward, never forward when the arm hangs. Same for the knee (forward when
#   the thigh hangs, up when the thigh is raised forward). A pose whose elbow or knee points far outside that arc is broken capture data
#   (the forearm flipped 180 deg about the upper arm, e.g. the cgspeed BVH conversion of CMU subjects 143/144 punches).
#   Frames are in the seq30 local frame: x = the person's left, y = back, z = up.
import numpy as np

FWD = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
ELBOW_MAX = 110.0       # deg away from the natural elbow-point direction before a frame counts as impossible
KNEE_MAX = 100.0


def _perp(v, u):
    return v - u * (v * u).sum(-1, keepdims=True)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def point_dir(a, b, c):
    """direction the middle joint points (away from the bend), perpendicular to the upper bone; and the bend angle"""
    u = _unit(b - a)
    f = c - b
    p = -_perp(f, u)
    bend = np.degrees(np.arccos(np.clip((u * _unit(f)).sum(-1), -1, 1)))
    return u, _unit(p), bend


def elbow_ref(u, fwd=FWD, up=UP):
    """natural elbow-point direction for upper-arm direction u: backward for a hanging arm, down for an arm raised forward"""
    w = 2.0 * np.clip((u * fwd).sum(-1, keepdims=True), 0, None)
    return _unit(_perp(-fwd, u) + w * _perp(-up, u) + 1e-6 * _perp(-up, u))


def knee_ref(u, fwd=FWD, up=UP):
    """natural knee-point direction for thigh direction u: forward for a hanging thigh, up for a thigh raised forward"""
    w = 2.0 * np.clip((u * fwd).sum(-1, keepdims=True), 0, None)
    return _unit(_perp(fwd, u) + w * _perp(up, u) + 1e-6 * _perp(up, u))


def angle(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def check(L, J, min_bend=35.0):
    """L: (F, joints, 3) local frames. Returns per-limb arrays of the deviation (deg) on frames where the joint is bent > min_bend (else nan)."""
    out = {}
    for sd in "LR":
        u, p, bend = point_dir(L[:, J["shoulder_" + sd]], L[:, J["elbow_" + sd]], L[:, J["wrist_" + sd]])
        dev = angle(p, elbow_ref(u))
        out["elbow_" + sd] = np.where(bend > min_bend, dev, np.nan)
        u, p, bend = point_dir(L[:, J["hip_" + sd]], L[:, J["knee_" + sd]], L[:, J["ankle_" + sd]])
        dev = angle(p, knee_ref(u))
        out["knee_" + sd] = np.where(bend > min_bend, dev, np.nan)
    return out


def repair_flipped_arms(L, J, limit=ELBOW_MAX, majority=0.5, min_bend=35.0):
    """The broken BVH takes have ONE arm flipped for the WHOLE take (the forearm turned 180 deg about the upper arm, so the elbow bends
    backward). If most of an arm's bent frames are impossible, flip that forearm on every frame (a straight arm is barely changed, so
    the motion stays continuous). Returns the repaired copy and {side: (bad frames before, bent frames, bad after)} for the arms flipped."""
    L = L.copy()
    rep = {}
    for sd in "LR":
        r = check(L, J, min_bend)["elbow_" + sd]
        n, bad = int(np.sum(~np.isnan(r))), int(np.nansum(r > limit))
        if n == 0 or bad < majority * n:
            continue
        s, e = L[:, J["shoulder_" + sd]], L[:, J["elbow_" + sd]]
        u = _unit(e - s)
        for jn in ("wrist_", "knuckle_", "thumb_"):
            v = L[:, J[jn + sd]] - e
            L[:, J[jn + sd]] = e + 2.0 * (v * u).sum(-1, keepdims=True) * u - v
        r2 = check(L, J, min_bend)["elbow_" + sd]
        rep[sd] = (bad, n, int(np.nansum(r2 > limit)))
    return L, rep


def repair_elbows(L, J, limit=ELBOW_MAX):
    """flip the forearm (and hand joints) 180 deg about the upper-arm axis on frames where the elbow points the impossible way.
    Returns the repaired copy and the number of frames fixed per arm."""
    L = L.copy()
    fixed = {}
    for sd in "LR":
        s, e = L[:, J["shoulder_" + sd]], L[:, J["elbow_" + sd]]
        u, p, bend = point_dir(s, e, L[:, J["wrist_" + sd]])
        bad = (bend > 20.0) & (angle(p, elbow_ref(u)) > limit)
        for jn in ("wrist_", "knuckle_", "thumb_"):
            v = L[:, J[jn + sd]] - e
            flipped = e + 2.0 * (v * u).sum(-1, keepdims=True) * u - v
            L[bad, J[jn + sd]] = flipped[bad]
        fixed[sd] = int(bad.sum())
    return L, fixed
