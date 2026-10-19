# body_clear.py : keep Relic's fists as far OUTSIDE his body as the real fighter's fists were outside theirs.
#   The capture comes from a slim human (CMU, retargeted onto one small skeleton: shoulders 23 cm apart); Relic is armoured (shoulders 72 cm,
#   chest armour 22 cm in front of the spine), so a guard that sits just in front of the human's chin lands inside Relic's chest.
#   Both bodies are modelled the same way: along the hips->neck axis (t = 0..1), how far the body reaches forward / back / left / right,
#   plus a head sphere. Clearance = how far a point is outside that shape (negative = inside).
#   (The KillDroid's hull is boxy and its forearms are as thick as a man's waist: it is kept clear against its real geometry instead,
#   kd_clear_arm in rig_guard.py.)
#   Pure python (no numpy / mathutils): exec'd by mocap_keys.py (human side) and rig_guard.py (Relic side). Points are given in the
#   body's own frame: t (fraction of the hips->neck length), f (forward), l (left), in the units of the profile.
import json, math

# the human: fractions of their hips->neck length (average adult proportions; the CMU joints sit near the spine)
#           t      front  back   left   right
HUMAN = [(-0.25, 0.147, 0.184, 0.294, 0.294), (0.0, 0.165, 0.184, 0.312, 0.312), (0.3, 0.184, 0.129, 0.239, 0.239),
         (0.6, 0.276, 0.129, 0.276, 0.276), (0.85, 0.257, 0.147, 0.294, 0.294), (1.0, 0.147, 0.110, 0.147, 0.147),
         (1.12, 0.08, 0.08, 0.10, 0.10)]
HUMAN_HEAD_R = 0.20          # head radius, fraction of hips->neck; centre = the head joint + 0.12 up + 0.03 forward (same fractions)
T_RANGE = (-0.3, 1.12)
MIN_CLEAR = 0.03             # m: a fist never comes closer to Relic's armour than this
MAX_CLEAR = 0.12             # m: beyond this the fighter's hand was simply away from the body; nothing to keep
FIST_OFF = 0.09              # m: Relic's fist centre beyond the wrist, along the hand
_RELIC = {}


def profile_at(table, t):
    ts = [r[0] for r in table]
    t = min(max(t, ts[0]), ts[-1])
    for i in range(len(ts) - 1):
        if ts[i] <= t <= ts[i + 1]:
            w = (t - ts[i]) / (ts[i + 1] - ts[i])
            return tuple(table[i][k] * (1 - w) + table[i + 1][k] * w for k in (1, 2, 3, 4))
    return tuple(table[-1][1:])


def torso_clear(t, f, l, table, scale=1.0):
    """clearance of point (t, f, l) outside the torso cross-section at height t (profile scaled by scale), along the point's own direction
    from the axis; returns (clearance, surface radius along that direction, unit (f, l)) or (None, ..) when not beside the torso"""
    if not (T_RANGE[0] <= t <= T_RANGE[1]):
        return None, None, None
    fr, bk, lf, rt = [x * scale for x in profile_at(table, t)]
    ef, el = (fr if f >= 0 else bk), (lf if l >= 0 else rt)
    r = math.hypot(f, l)
    if r < 1e-9:
        return -min(ef, el), min(ef, el), (1.0, 0.0)
    rho = math.sqrt((f / max(ef, 1e-4)) ** 2 + (l / max(el, 1e-4)) ** 2)
    rs = r / max(rho, 1e-9)
    return r - rs, rs, (f / r, l / r)


def relic_body():
    if not _RELIC:
        d = json.load(open(r"C:\ai\unedited\anim_cache5\relic_torso.json"))
        _RELIC['table'] = [(r['t'], r['front'], r['back'], r['left'], r['right']) for r in d['rows'] if -0.3 <= r['t'] <= 1.1]
        _RELIC['head'] = d['head']
        _RELIC['alen'] = d['axis_len']
    return _RELIC
