# fight_moves.py : a small library of trained-fighter moves (jab, cross, hook, teep, roundhouse, slips, ducks, angle step, parry, back step)
# built from the research in fight_reference.md.  Authored in HUMAN units (hip height 0.93 m, leg 0.85, arm 0.58) and scaled to each body.
#   exec'd after kd_ctrl.py and msflow_retarget.py inside fight_anim.py (needs math, Vector, Matrix, kd_shoulders, to_world, FWD)
#
# frame of reference ("T-frame"): origin = the fighter's ground point `base`, forward = toward the opponent, side + = his LEFT.
# orthodox stance: left foot and shoulder lead.  Everything the pro does is a chain: rear leg loads -> hips turn -> shoulders turn -> arm extends last.
import math
from mathutils import Vector, Matrix

H_HUMAN = dict(hip=0.93, leg=0.85, ankle=0.08, arm=0.58)

STANCE = dict(
    hs=0.0, hf=0.0, hz=0.86,                 # hips: side, fwd, height (human units)
    byaw=-0.60, tw=0.0, th=0.10, ph=0.0,     # body yaw off the line to the opponent (bladed to his right), twist, forward lean, side lean
    Lfs=0.14, Lff=0.34, Lfz=0.0, Lfy=0.15,   # lead (left) foot: side, fwd, lift, yaw off the line
    Rfs=-0.16, Rff=-0.26, Rfz=0.0, Rfy=-0.75,    # rear (right) foot
    Lpt=0.0, Rpt=0.18,                        # heel-up pitch of each foot (rad): the rear heel is always a little up
    # (foot s/f values are the positions of the BALL of the foot; the ankle sits behind it and the foot pivots on the ball)
    # fists relative to their own shoulder, in the chest frame (side, fwd, up), human units
    LgS=-0.04, LgF=0.46, LgU=0.14, RgS=0.10, RgF=0.36, RgU=0.24,
    LFc=1.0, RFc=1.0,                        # fist closed 0..1
    Le=0.0, Re=0.0, LKe=0.0, RKe=0.0,        # punch extension 0..1 toward the target (L / R fist), kick extension 0..1 (L / R foot)
)


class Body:
    """scale factors of one rig from its rest bones"""

    def __init__(self, rig):
        bn = rig.data.bones
        self.rig = rig
        self.leg = bn['thigh.L'].length + bn['shin.L'].length
        self.arm = bn['upperarm.L'].length + bn['forearm.L'].length
        self.hip0 = bn['pelvis'].head_local.z
        self.ank0 = bn['foot.L'].head_local.z
        fh = bn['foot.L'].head_local
        ft = bn['foot.L'].tail_local
        self.foot_h = math.hypot(ft.x - fh.x, ft.y - fh.y)             # ankle -> ball, horizontal (metres)
        self.foot_len = (ft - fh).length
        self.kl = self.leg / H_HUMAN['leg']
        self.ka = self.arm / H_HUMAN['arm']


def _pchip(xs, ys, x):
    n = len(xs)
    if n == 1 or x <= xs[0]:
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
    t = (x - xs[k]) / h[k]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * ys[k] + h10 * h[k] * m[k] + h01 * ys[k + 1] + h11 * h[k + 1 - 1] * m[k + 1]


def ch(spec, name, t):
    """value of channel `name` at time t (frames) in a move spec; constant stance value if the move does not touch it"""
    keys = spec['ch'].get(name)
    if not keys:
        return STANCE[name]
    ks = [(0.0, STANCE[name])] + [(a, b) for a, b in keys if a > 0.0] if keys[0][0] > 0.0 else list(keys)
    return _pchip([k[0] for k in ks], [k[1] for k in ks], t)


# ------------------------------------------------------------------------------------------------------------ the moves
# time in frames at 24 fps.  Each returns {'dur': frames, 'ch': {...}, 'delta': (side, fwd) base shift in human units, 'name': ...}

def cross(power=1.0):
    """rear (right) hand.  load on the rear leg, rear foot pivots, hips turn, shoulders turn further, the fist leaves last, snaps back"""
    return dict(name='cross', dur=20, delta=(0, 0), hit=9, ch=dict(
        hz=[(3, 0.83), (6, 0.85), (10, 0.855), (18, 0.86)],
        hf=[(3, -0.04), (7, 0.12 * power), (10, 0.12 * power), (18, 0.0)],
        byaw=[(3, -0.74), (7, -0.20), (9, -0.08), (11, -0.10), (19, -0.60)],
        tw=[(3, -0.10), (7, 0.12), (9, 0.30), (11, 0.28), (19, 0.0)],
        th=[(3, 0.05), (8, 0.22), (11, 0.20), (19, 0.10)],
        ph=[(3, -0.06), (9, 0.05), (19, 0.0)],
        Rfy=[(3, -0.80), (8, -0.15), (11, -0.15), (19, -0.75)],
        Rpt=[(3, 0.20), (7, 0.55), (11, 0.50), (19, 0.18)],
        Lfy=[(8, 0.30), (11, 0.30), (19, 0.15)],
        Re=[(5, 0.0), (7, 0.55), (9, 1.0), (10, 1.03), (12, 0.45), (15, 0.0)],
        LgF=[(7, 0.46), (9, 0.36), (14, 0.46)], LgU=[(9, 0.20), (14, 0.14)],
    ))


def jab():
    """lead (left) hand: little hip turn, a small step in with the lead foot, fast out and back"""
    return dict(name='jab', dur=14, delta=(0, 0), hit=5, ch=dict(
        hf=[(2, 0.02), (5, 0.07), (8, 0.05), (13, 0.0)],
        byaw=[(2, -0.58), (5, -0.42), (8, -0.45), (14, -0.60)],
        tw=[(5, 0.10), (8, 0.08), (14, 0.0)],
        th=[(5, 0.14), (14, 0.10)],
        Lff=[(2, 0.34), (5, 0.40), (8, 0.38), (13, 0.34)],
        Lpt=[(2, 0.0), (5, 0.12), (13, 0.0)],
        Le=[(1, 0.0), (3, 0.6), (5, 1.0), (6, 1.02), (8, 0.5), (11, 0.0)],
    ))


def hook(power=1.0):
    """lead (left) hook: lead foot pivots, hips and trunk whip around, elbow stays high, short horizontal arc"""
    return dict(name='hook', dur=22, delta=(0, 0), hit=9, ch=dict(
        hz=[(3, 0.83), (8, 0.85), (19, 0.86)],
        hf=[(3, -0.02), (8, 0.06), (19, 0.0)],
        byaw=[(3, -0.72), (7, -0.15), (9, 0.05 * power), (11, 0.0), (21, -0.60)],
        tw=[(3, -0.18), (8, 0.08), (9, 0.18), (12, 0.14), (21, 0.0)],
        th=[(3, 0.05), (9, 0.16), (21, 0.10)],
        ph=[(3, 0.10), (9, -0.08), (21, 0.0)],
        Lfy=[(3, 0.10), (8, 0.95), (12, 0.80), (21, 0.15)],
        Lpt=[(3, 0.0), (8, 0.50), (12, 0.45), (21, 0.0)],
        Le=[(4, 0.0), (6, 0.5), (9, 1.0), (10, 1.0), (12, 0.4), (16, 0.0)],
        LgS=[(4, -0.04), (9, 0.30), (12, 0.20), (16, -0.04)],     # the fist travels out wide and in: sweeping arc
    ))


def teep():
    """push kick with the rear leg: rear knee to a 90 degree chamber, hips thrust forward, small pivot, tall torso, fast retract"""
    return dict(name='teep', dur=24, delta=(0, 0), hit=11, ch=dict(
        hz=[(4, 0.85), (8, 0.88), (11, 0.88), (20, 0.86)],
        hf=[(4, 0.02), (11, 0.14), (14, 0.14), (22, 0.0)],
        byaw=[(5, -0.60), (11, -0.30), (14, -0.30), (23, -0.60)],
        th=[(6, 0.05), (11, -0.02), (14, 0.0), (23, 0.10)],
        Lfy=[(5, 0.15), (11, 0.60), (16, 0.60), (23, 0.15)],     # standing foot pivots
        RKe=[(2, 0.0), (6, 0.35), (11, 1.0), (13, 1.0), (16, 0.35), (20, 0.0)],
        Rfz=[(2, 0.0), (6, 0.45), (9, 0.55), (13, 0.55), (17, 0.30), (21, 0.0)],
        Rpt=[(2, 0.18), (6, 0.5), (9, 0.0), (11, -0.55), (14, -0.45), (19, 0.3), (23, 0.18)],
        Lpt=[(5, 0.0), (11, 0.22), (16, 0.22), (23, 0.0)],
        Rff=[(2, -0.26), (6, -0.05), (11, 0.30), (13, 0.30), (17, 0.0), (22, -0.26)],
        Rfs=[(6, -0.12), (11, -0.04), (22, -0.16)],
        RgF=[(8, 0.40), (22, 0.36)], RgU=[(8, 0.26), (22, 0.24)],
    ))


def roundhouse():
    """rear-leg roundhouse: chamber (knee up and out), standing foot pivots about 80 degrees, hips whip around, shin lands as the body weight arrives, recoil"""
    return dict(name='roundhouse', dur=30, delta=(0, 0), hit=13, ch=dict(
        hz=[(4, 0.84), (9, 0.90), (13, 0.90), (24, 0.86)],
        hf=[(4, -0.02), (10, 0.12), (14, 0.12), (27, 0.0)],
        byaw=[(4, -0.78), (9, -0.10), (13, 0.35), (16, 0.35), (28, -0.60)],
        tw=[(4, -0.15), (13, 0.30), (16, 0.25), (28, 0.0)],
        th=[(4, 0.04), (9, 0.0), (13, -0.14), (16, -0.10), (28, 0.10)],
        ph=[(9, -0.06), (13, -0.28), (16, -0.20), (28, 0.0)],      # leans away from the kicking side
        Lfy=[(4, 0.15), (9, 1.00), (14, 1.45), (17, 1.40), (28, 0.15)],
        RKe=[(3, 0.0), (7, 0.30), (13, 1.0), (15, 1.0), (19, 0.3), (24, 0.0)],
        Rfz=[(3, 0.0), (8, 0.55), (13, 0.85), (15, 0.85), (20, 0.40), (26, 0.0)],
        Rpt=[(3, 0.18), (8, 0.5), (13, 0.20), (15, 0.20), (22, 0.4), (28, 0.18)],
        Lpt=[(4, 0.0), (9, 0.5), (14, 0.6), (17, 0.5), (28, 0.0)],
        Rff=[(3, -0.26), (8, -0.02), (13, 0.30), (15, 0.28), (21, 0.0), (27, -0.26)],
        Rfs=[(8, -0.30), (13, -0.15), (27, -0.16)],
        RgU=[(8, 0.35), (13, 0.40), (24, 0.24)], RgF=[(8, 0.22), (13, 0.12), (24, 0.36)],  # the kicking-side arm swings back for balance
        LgU=[(8, 0.22), (13, 0.30), (24, 0.14)], LgF=[(8, 0.34), (13, 0.28), (24, 0.46)],  # the other hand covers the head
    ))


def slip(side=+1):
    """slip a straight punch: knees bend a little, the head moves a few inches off the line (side +1 = to his left), hands stay up, then back"""
    s = side
    return dict(name='slip', dur=16, delta=(0, 0), ch=dict(
        hz=[(3, 0.82), (7, 0.80), (13, 0.85), (16, 0.86)],
        hs=[(3, 0.04 * s), (7, 0.08 * s), (13, 0.02 * s), (16, 0.0)],
        ph=[(3, 0.12 * s), (7, 0.30 * s), (13, 0.10 * s), (16, 0.0)],
        byaw=[(7, -0.60 + 0.12 * s), (16, -0.60)],
        tw=[(7, 0.10 * s), (16, 0.0)],
        th=[(7, 0.14), (16, 0.10)],
    ))


def duck():
    """duck under a hook or wide swing: bend the KNEES, head drops and travels in a U, hands stay up"""
    return dict(name='duck', dur=20, delta=(0, 0), ch=dict(
        hz=[(4, 0.70), (9, 0.62), (14, 0.72), (20, 0.86)],
        hf=[(4, 0.0), (9, -0.04), (20, 0.0)],
        th=[(4, 0.22), (9, 0.26), (14, 0.18), (20, 0.10)],
        ph=[(9, -0.06), (14, 0.08), (20, 0.0)],
        Lff=[(9, 0.38), (20, 0.34)], Rff=[(9, -0.28), (20, -0.26)],
        LgU=[(9, 0.20), (20, 0.14)], RgU=[(9, 0.28), (20, 0.24)],
    ))


def angle_step(side=+1, dist=0.55, fwd=0.2):
    """step off the line: the foot on the side of travel goes first, the other follows, weight low, head off line.  side +1 = to his left"""
    s = side
    L_first = (s > 0)
    lead_s, rear_s = (STANCE['Lfs'], STANCE['Rfs'])
    d = dist
    ch = dict(
        hz=[(3, 0.82), (7, 0.80), (11, 0.83), (14, 0.86)],
        hs=[(4, 0.06 * s), (8, d * 0.65 * s), (12, d * s)],
        hf=[(8, fwd * 0.6), (12, fwd)],
        ph=[(5, 0.16 * s), (9, 0.10 * s), (14, 0.0)],
        th=[(8, 0.14), (14, 0.10)],
    )
    # the foot on the travel side leaves first, the other catches up
    first, second = ('L', 'R') if L_first else ('R', 'L')
    f_s0 = {'L': STANCE['Lfs'], 'R': STANCE['Rfs']}
    f_f0 = {'L': STANCE['Lff'], 'R': STANCE['Rff']}
    ch[first + 'fs'] = [(2, f_s0[first]), (7, f_s0[first] + d * s), (12, f_s0[first] + d * s)]
    ch[first + 'ff'] = [(2, f_f0[first]), (7, f_f0[first] + fwd), (12, f_f0[first] + fwd)]
    ch[first + 'fz'] = [(2, 0.0), (5, 0.10), (8, 0.0)]
    ch[second + 'fs'] = [(6, f_s0[second]), (11, f_s0[second] + d * s), (14, f_s0[second] + d * s)]
    ch[second + 'ff'] = [(6, f_f0[second]), (11, f_f0[second] + fwd), (14, f_f0[second] + fwd)]
    ch[second + 'fz'] = [(6, 0.0), (9, 0.09), (12, 0.0)]
    return dict(name='angle_step', dur=15, delta=(d * s, fwd), ch=ch)


def back_step(dist=0.7):
    """step back out of range: rear foot first, then lead, weight stays centered, hands up"""
    ch = dict(
        hz=[(3, 0.83), (8, 0.85), (13, 0.86)],
        hf=[(4, -0.12), (12, -dist)],
        th=[(5, 0.06), (12, 0.10)],
        Rff=[(2, -0.26), (6, -0.26 - dist), (12, -0.26 - dist)],
        Rfz=[(2, 0.0), (4, 0.10), (7, 0.0)],
        Lff=[(6, 0.34), (10, 0.34 - dist), (13, 0.34 - dist)],
        Lfz=[(6, 0.0), (8, 0.09), (11, 0.0)],
    )
    return dict(name='back_step', dur=14, delta=(0, -dist), ch=ch)


def parry():
    """parry a straight punch with the lead glove: small sweep across, head stays, then counter-ready"""
    return dict(name='parry', dur=12, delta=(0, 0), ch=dict(
        LgS=[(2, -0.04), (4, -0.16), (7, -0.02), (11, -0.04)],
        LgF=[(2, 0.46), (4, 0.52), (7, 0.46)],
        LgU=[(4, 0.16), (7, 0.14)],
        tw=[(4, 0.06), (11, 0.0)],
    ))


def idle(dur=24, amp=1.0):
    """fighter's bounce: small weight shifts, knees pumping, hands floating"""
    return dict(name='idle', dur=dur, delta=(0, 0), ch=dict(
        hz=[(6, 0.86 - 0.02 * amp), (12, 0.86 + 0.0), (18, 0.86 - 0.02 * amp), (24, 0.86)],
        hs=[(6, 0.015 * amp), (18, -0.015 * amp), (24, 0.0)],
        byaw=[(12, -0.60 + 0.03 * amp), (24, -0.60)],
        LgU=[(12, 0.15), (24, 0.14)], RgU=[(12, 0.25), (24, 0.24)],
    ))


MOVES = dict(cross=cross, jab=jab, hook=hook, teep=teep, roundhouse=roundhouse, slip=slip, duck=duck, angle_step=angle_step,
             back_step=back_step, parry=parry, idle=idle)


# ------------------------------------------------------------------------------------------------------------ compile to per-frame params
def torso_rot(P, c=0.85):
    Rz = lambda a: Matrix.Rotation(a, 3, 'Z')
    Rx = lambda a: Matrix.Rotation(a, 3, 'X')
    Ry = lambda a: Matrix.Rotation(a, 3, 'Y')
    th, ph = P['lean']
    return Rz(P['yaw']) @ (Rz(P.get('tw', 0.0) * c) @ Ry(ph * c) @ Rx(th * c))


def frame_params(body, base, yawT, spec, t, target=None):
    """params dict for kd_apply at time t of move `spec`, fighter at ground point `base`, line to the opponent at yaw yawT.
    target: world point the fists / feet aim at when extended"""
    rig = body.rig
    v = lambda n: ch(spec, n, t)
    kl, ka = body.kl, body.ka
    # human height above the ankle * leg ratio + the rig's own ankle height
    hz = body.ank0 + (v('hz') - H_HUMAN['ankle']) * kl
    loc = lambda s, f, z: to_world(rig, base, yawT, (s * kl, f * kl, z))
    P = {}
    P['hips'] = loc(v('hs'), v('hf'), hz)
    P['yaw'] = yawT + v('byaw')
    P['lean'] = (v('th'), v('ph'))
    P['tw'] = v('tw')
    # feet: the s/f channels are the BALL of the foot; the foot yaws about the ball and the heel lifts with the pitch channel
    P['fyaw'] = (v('Lfy') - v('byaw'), v('Rfy') - v('byaw'))
    P['fist'] = (v('LFc'), v('RFc'))
    P['fpitch'] = {}
    for key, side_ch, pt_ch, fy_ch, lift_ch, nm in (('aL', ('Lfs', 'Lff'), 'Lpt', 'Lfy', 'Lfz', 'L'), ('aR', ('Rfs', 'Rff'), 'Rpt', 'Rfy', 'Rfz', 'R')):
        ball_s, ball_f = v(side_ch[0]), v(side_ch[1])
        pt = v(pt_ch)
        psi = v(fy_ch)
        ux, uy = math.sin(psi), math.cos(psi)                       # foot direction in (side, fwd)
        back = body.foot_h * math.cos(pt) / kl                       # ball -> ankle, horizontal, in human units
        a_s, a_f = ball_s - back * ux, ball_f - back * uy
        z = body.ank0 + v(lift_ch) * kl + body.foot_len * math.sin(max(pt, 0.0)) * 1.0
        P[key] = loc(a_s, a_f, z)
        P['fpitch'][nm] = pt
    # fists: relative to the shoulder in the chest frame; extension blends toward the target
    Rt = torso_rot(P)
    shL, shR = kd_shoulders(rig, P)
    for key, sh, g, e in (('wL', shL, ('LgS', 'LgF', 'LgU'), v('Le')), ('wR', shR, ('RgS', 'RgF', 'RgU'), v('Re'))):
        s_, f_, u_ = (v(g[0]) * ka, v(g[1]) * ka, v(g[2]) * ka)
        guard = sh + rig.matrix_world.to_3x3() @ (Rt @ Vector((s_, -f_, u_)))
        if target is not None and e > 0.0:
            d = Vector(target) - sh
            reach = body.arm * 0.97
            far = sh + d.normalized() * min(reach, d.length)
            P[key] = guard * (1.0 - e) + far * e
        else:
            P[key] = guard
    # kicks: the kicking foot blends toward the target (stops a bit short, the shin/foot lands)
    for key, k in (('aL', 'LKe'), ('aR', 'RKe')):
        e = v(k)
        if target is not None and e > 0.0:
            P[key] = P[key] * (1.0 - e) + (Vector(target) - (Vector(target) - P[key]).normalized() * 0.15 * kl) * e
    return P


def compile_moves(body, base0, yawT, sequence, target=None, f0=1):
    """sequence: list of (start_frame, spec).  Returns {frame: params}.  Frames between moves hold the guard; each move re-bases by its delta."""
    out = {}
    base = Vector((base0[0], base0[1], 0.0))
    rigsec = sorted(sequence, key=lambda x: x[0])
    for start, spec in rigsec:
        for i in range(int(spec['dur']) + 1):
            fr = start + i
            out[fr] = frame_params(body, base, yawT, spec, float(i), target)
        d = spec.get('delta', (0, 0))
        left = to_world(body.rig, (0, 0), yawT, (1.0, 0.0, 0.0))
        fwd = to_world(body.rig, (0, 0), yawT, (0.0, 1.0, 0.0))
        base = base + left * d[0] * body.kl + fwd * d[1] * body.kl
    return out
