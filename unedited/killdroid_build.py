# killdroid_build.py : the KillDroid of UNEDITED (StarCapital's hunter): 2.5 m cyborg, bare red muscle, black armor, a three-clawed steel right arm,
# a thick organic left arm, a camera rig with two red eyes. Built with the primitives of mender_build.py (numpy meshes, a skinned body, rigid armor on bones).
# Design frame: feet at z=0, facing -Y, +X = his left. Run in Blender (background is fine):
#   blender.exe -b --python killdroid_build.py        (saves C:\ai\unedited\killdroid_v01.blend and renders a design sheet)
import bpy, bmesh, math, os, types
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

_g = globals()
exec(compile(open(r"C:\ai\unedited\mender_build.py", encoding="utf-8").read(), "mender_build.py", "exec"), _g)
exec(compile(open(r"C:\ai\unedited\mender_scene.py", encoding="utf-8").read(), "mender_scene.py", "exec"), _g)

OUT_DIR = r"C:\ai\unedited"
RIGN = "KillDroid_Rig"
MC = "KillDroid"
KD_SCALE = 1.12            # the whole KillDroid is built at 2.5 m and scaled up to about 2.8 m
SC = np.array([KD_SCALE, KD_SCALE, KD_SCALE])


# ------------------------------------------------------------------ skeleton
def kd_joint_table():
    J = {}
    J['hip_c'] = np.array([0, 0, 1.18])
    J['pelvis_t'] = np.array([0, 0, 1.30])
    J['spine_t'] = np.array([0, 0, 1.56])
    J['chest_t'] = np.array([0, 0, 1.90])
    J['neck_t'] = np.array([0, 0, 2.06])
    J['head_t'] = np.array([0, 0, 2.46])
    for s, sx in (('L', 1.0), ('R', -1.0)):
        J['clav_' + s] = np.array([0.12 * sx, 0, 1.94])
        J['sh_' + s] = np.array([0.58 * sx, 0, 1.86])
        J['el_' + s] = np.array([0.74 * sx, 0.03, 1.33])
        J['wr_' + s] = np.array([0.76 * sx, -0.05, 0.82])
        J['hd_' + s] = np.array([0.76 * sx, -0.06, 0.60])
        J['hip_' + s] = np.array([0.20 * sx, 0, 1.18])
        J['kn_' + s] = np.array([0.25 * sx, -0.03, 0.66])
        J['an_' + s] = np.array([0.26 * sx, 0.03, 0.19])
        J['ball_' + s] = np.array([0.26 * sx, -0.22, 0.08])
        J['toe_' + s] = np.array([0.26 * sx, -0.44, 0.07])
    return J


J = kd_joint_table()
# the three claw fingers of the right hand: (palm end, middle joint, tip), curling in toward the axis under the palm
CLAW = {'claw1': [np.array([-0.76, -0.12, 0.62]), np.array([-0.76, -0.17, 0.43]), np.array([-0.76, -0.11, 0.29])],
        'claw2': [np.array([-0.82, 0.03, 0.62]), np.array([-0.87, 0.07, 0.43]), np.array([-0.81, 0.04, 0.29])],
        'claw3': [np.array([-0.70, 0.03, 0.62]), np.array([-0.65, 0.07, 0.43]), np.array([-0.71, 0.04, 0.29])]}


def hand_dir(s):
    return unit(J['hd_' + s] - J['wr_' + s])


def hand_end(s):
    return J['hd_' + s]


def bone_defs():
    B = [('root', (0, 0, 0), (0, -0.3, 0), None),
         ('pelvis', J['hip_c'], J['pelvis_t'], 'root'),
         ('spine', J['pelvis_t'], J['spine_t'], 'pelvis'),
         ('chest', J['spine_t'], J['chest_t'], 'spine'),
         ('neck', J['chest_t'], J['neck_t'], 'chest'),
         ('head', J['neck_t'], J['head_t'], 'neck')]
    for s, sx in (('L', 1.0), ('R', -1.0)):
        B += [('clavicle.' + s, J['clav_' + s], J['sh_' + s], 'chest'),
              ('upperarm.' + s, J['sh_' + s], J['el_' + s], 'clavicle.' + s),
              ('forearm.' + s, J['el_' + s], J['wr_' + s], 'upperarm.' + s),
              ('hand.' + s, J['wr_' + s], J['hd_' + s], 'forearm.' + s)]
        if s == 'R':
            for nm, jt in CLAW.items():
                B.append((nm + 'a.R', jt[0], jt[1], 'hand.R'))
                B.append((nm + 'b.R', jt[1], jt[2], nm + 'a.R'))
        B += [('thigh.' + s, J['hip_' + s], J['kn_' + s], 'pelvis'),
              ('shin.' + s, J['kn_' + s], J['an_' + s], 'thigh.' + s),
              ('foot.' + s, J['an_' + s], J['ball_' + s], 'shin.' + s),
              ('toe.' + s, J['ball_' + s], J['toe_' + s], 'foot.' + s),
              ('ik_foot.' + s, J['an_' + s], J['ball_' + s], 'root'),
              ('ik_knee.' + s, np.array([J['kn_' + s][0], -1.2, J['kn_' + s][2]]), np.array([J['kn_' + s][0], -1.2, J['kn_' + s][2] + 0.1]), 'root')]
    return B


# ------------------------------------------------------------------ the body surface helpers
TZ = np.array([1.06, 1.18, 1.30, 1.45, 1.60, 1.75, 1.88, 1.98, 2.05, 2.11])
TRX = np.array([0.30, 0.31, 0.30, 0.29, 0.38, 0.52, 0.60, 0.45, 0.18, 0.15])
TRY = np.array([0.20, 0.21, 0.21, 0.22, 0.26, 0.30, 0.31, 0.26, 0.16, 0.15])
TCY = np.array([0.0, 0.0, 0.0, -0.005, -0.02, -0.04, -0.035, -0.01, 0.0, 0.0])


def torso_front(x, z):
    """y of the torso's front surface at (x, z) (negative = in front)"""
    rx, ry, cy = np.interp(z, TZ, TRX), np.interp(z, TZ, TRY), np.interp(z, TZ, TCY)
    return cy - ry * math.sqrt(max(1.0 - (x / rx) ** 2, 0.0))


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dome(C, rx, ry, h, axis, sides=26):
    """a closed dome shell: base radius rx (x) and ry (y), height h along `axis`"""
    prof = [(0.0, 0.0), (rx, 0.0), (rx * 0.99, h * 0.18), (rx * 0.93, h * 0.42), (rx * 0.78, h * 0.68), (rx * 0.50, h * 0.88), (rx * 0.22, h * 0.98), (0.0, h)]
    V, p = lathe(prof, sides)
    V = V * np.array([1.0, ry / rx, 1.0])
    return place(V, C, basis(axis, (0, -1, 0))), p


def band(P0, P1, width, thick, r=0.015):
    """a rounded plate from P0 to P1 (length along its local x), `width` along local z, `thick` along local y"""
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    d = P1 - P0
    L = float(np.linalg.norm(d))
    x = d / L
    z = np.cross(x, [0.0, 1.0, 0.0])
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 0.0, 1.0])
    z = unit(z)
    y = np.cross(z, x)
    return rbox((P0 + P1) / 2, (L, thick, width), r, np.stack([x, y, z], 1))


def ring(C, R_, r_, axis=(0, 0, 1)):
    return torus(C, R_, r_, axis=axis, nu=28, nv=8)


def build_skin():
    S = Suit()
    chains = {'torso': ([J['hip_c'], J['pelvis_t'], J['spine_t'], J['chest_t'], J['neck_t'], J['head_t']], ['pelvis', 'spine', 'chest', 'neck', 'head'], 0.09)}
    for s in ('L', 'R'):
        chains['leg_' + s] = ([J['hip_' + s], J['kn_' + s], J['an_' + s]], ['thigh.' + s, 'shin.' + s], 0.10)
        chains['arm_' + s] = ([J['clav_' + s], J['sh_' + s], J['el_' + s], J['wr_' + s]], ['clavicle.' + s, 'upperarm.' + s, 'forearm.' + s], 0.09)
    CH.update(chains)

    def host(V, polys, mat, region):
        j, b, bl = chains[region]
        S.host(V, polys, mat, j, b, bl)

    # torso: a V-shaped trunk
    Vt, Pt = tube(np.stack([np.zeros_like(TZ), TCY, TZ], 1), TRX, TRY, sides=30, caps=(True, True), up=(1.0, 0.0, 0.0))
    host(Vt, Pt, 0, 'torso')
    # pecs, abs, traps, lats, neck
    for sx in (-1.0, 1.0):
        zc = 1.80
        host(*ellipsoid((0.215 * sx, torso_front(0.215, zc) + 0.035, zc), (0.215, 0.105, 0.150), 22, 12, R=rot_y(-0.10 * sx)), 0, 'torso')
        host(*ellipsoid((0.46 * sx, 0.07, 1.74), (0.15, 0.15, 0.31), 18, 10, R=rot_y(0.12 * sx)), 0, 'torso')          # lats
        host(*ellipsoid((0.22 * sx, 0.0, 2.00), (0.21, 0.17, 0.10), 18, 10, R=rot_y(-0.5 * sx)), 0, 'torso')           # traps
        for k, zr in enumerate((1.33, 1.425, 1.52, 1.615)):
            x = 0.083 * sx
            y = torso_front(abs(x), zr) + 0.014
            host(*ellipsoid((x, y, zr), (0.080, 0.050, 0.046), 14, 8), 0, 'torso')
    host(*ellipsoid((0.0, torso_front(0.0, 1.20) + 0.01, 1.20), (0.20, 0.07, 0.10), 16, 8), 0, 'torso')              # lower abs / pelvis front
    host(*tube(np.array([[0, 0.0, 2.00], [0, 0.01, 2.14], [0, 0.015, 2.24]]), [0.19, 0.165, 0.155], sides=22, caps=(False, False)), 0, 'torso')  # neck
    # head: skull, jaw, brow, nose
    host(*ellipsoid((0.0, 0.015, 2.33), (0.150, 0.170, 0.185), 22, 14), 0, 'torso')
    host(*ellipsoid((0.0, -0.065, 2.20), (0.138, 0.135, 0.105), 18, 10), 0, 'torso')
    host(*ellipsoid((0.0, -0.125, 2.345), (0.145, 0.055, 0.040), 16, 8), 0, 'torso')
    host(*ellipsoid((0.0, -0.150, 2.285), (0.022, 0.028, 0.040), 10, 8), 0, 'torso')
    # legs: thigh and calf muscle down to the ankle
    for s, sx in (('L', 1.0), ('R', -1.0)):
        P = np.array([[0.20, 0.0, 1.14], [0.22, -0.01, 0.92], [0.25, -0.03, 0.66], [0.26, -0.00, 0.42], [0.26, 0.03, 0.19]]) * np.array([sx, 1, 1])
        r = np.array([0.265, 0.255, 0.200, 0.190, 0.125])
        host(*tube(P, r, r * 1.04, sides=22, caps=(True, True), up=(1.0, 0.0, 0.0)), 0, 'leg_' + s)
        host(*ellipsoid((0.26 * sx, 0.07, 0.46), (0.15, 0.16, 0.25), 14, 8), 0, 'leg_' + s)                          # calf
    # arms
    sh, el, wr = J['sh_L'], J['el_L'], J['wr_L']
    path = [sh] + [lerp(sh, el, u) for u in (0.25, 0.5, 0.8)] + [el] + [lerp(el, wr, u) for u in (0.25, 0.5, 0.8)] + [wr]
    rad = [0.17, 0.205, 0.215, 0.175, 0.145, 0.165, 0.160, 0.130, 0.108]
    host(*tube(np.array(path), np.array(rad), np.array(rad) * 1.03, sides=22, caps=(True, True), up=(0.0, 1.0, 0.0)), 0, 'arm_L')
    host(*ellipsoid(sh + np.array([0.0, 0.0, 0.03]), (0.215, 0.205, 0.225), 18, 12), 0, 'arm_L')                        # the bare deltoid
    sh, el = J['sh_R'], J['el_R']
    path = [sh] + [lerp(sh, el, u) for u in (0.25, 0.5, 0.8)] + [el]
    rad = [0.16, 0.185, 0.19, 0.16, 0.14]
    host(*tube(np.array(path), np.array(rad), np.array(rad) * 1.03, sides=22, caps=(True, True), up=(0.0, 1.0, 0.0)), 0, 'arm_R')
    return S, chains


def skin_to_obj(S, coll, rig, name="KillDroid_Body"):
    return make_skinned(name, S, coll, rig)


# ------------------------------------------------------------------ rigid armor and gear: (name, V, polys, material key, bone)
def armor_parts():
    parts = []

    def add(name, vp, mk, bone):
        parts.append((name, vp[0], vp[1], mk, bone))

    # --- shoulders: the big layered right pauldron, a smaller left plate
    for s, sx, big in (('R', -1.0, True), ('L', 1.0, False)):
        ax = np.array([0.46 * sx, 0.0, 0.89])
        layers = ((0.0, 0.40, 0.30, 0.24), (-0.10, 0.37, 0.28, 0.20), (-0.20, 0.33, 0.25, 0.17)) if big else ((0.0, 0.30, 0.26, 0.15),)
        for k, (dz, rx, ry, h) in enumerate(layers):
            C = J['sh_' + s] + np.array([0.04 * sx, 0.0, 0.10 + dz])
            add('Pauldron%s%d' % (s, k), dome(C, rx, ry, h, ax), 'armor' if k != 1 else 'armorRust', 'clavicle.' + s)
    add('CollarGuardR', rbox((-0.27, 0.02, 2.06), (0.34, 0.20, 0.07), 0.025, rot_y(0.35)), 'armor', 'chest')
    add('CollarGuardL', rbox((0.27, 0.02, 2.06), (0.30, 0.18, 0.06), 0.022, rot_y(-0.35)), 'armor', 'chest')
    # --- chest harness: a diagonal band from the right shoulder to the left hip, a sternum plate, a buckle
    n = 8
    P = []
    for i in range(n + 1):
        t = i / n
        x, z = lerp((-0.30, 1.98), (0.16, 1.43), t)
        P.append(np.array([x, torso_front(abs(x), z) - 0.012, z]))
    for i in range(n):
        add('Harness%d' % i, band(P[i], P[i + 1], 0.13, 0.045, 0.015), 'armor', 'chest')
    add('SternumPlate', rbox((0.0, torso_front(0.0, 1.68) - 0.02, 1.68), (0.17, 0.07, 0.22), 0.025), 'armor', 'chest')
    add('ChestPlateR', rbox((-0.36, torso_front(0.36, 1.90) - 0.025, 1.90), (0.30, 0.07, 0.20), 0.025, rot_y(0.55)), 'armorRust', 'chest')
    add('Buckle', rbox((0.14, torso_front(0.14, 1.52) - 0.04, 1.50), (0.10, 0.05, 0.09), 0.015), 'steel', 'chest')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('BackPlate' + s, rbox((0.28 * sx, 0.24, 1.78), (0.30, 0.07, 0.36), 0.03, rot_y(-0.2 * sx)), 'armor', 'chest')
    # --- belt and hip armor
    add('Belt', ring((0, 0, 1.31), 0.325, 0.045), 'armor', 'pelvis')
    add('BeltBuckle', rbox((0, -0.335, 1.31), (0.20, 0.06, 0.15), 0.02), 'steel', 'pelvis')
    add('Codpiece', rbox((0, -0.215, 1.10), (0.22, 0.07, 0.24), 0.03), 'armor', 'pelvis')
    for s, sx in (('L', 1.0), ('R', -1.0)):
        add('HipPlate' + s, rbox((0.35 * sx, 0.0, 1.20), (0.12, 0.30, 0.28), 0.03, rot_y(0.25 * sx)), 'armor', 'pelvis')
        add('FaldF' + s, rbox((0.20 * sx, -0.20, 1.07), (0.15, 0.06, 0.26), 0.025, rot_x(0.12)), 'armorRust', 'pelvis')
    # --- legs: thigh plates, knee armor, greaves, boots
    for s, sx in (('L', 1.0), ('R', -1.0)):
        hip, kn, an = J['hip_' + s], J['kn_' + s], J['an_' + s]
        c = lerp(hip, kn, 0.45) + np.array([0.0, -0.20, 0.0])
        add('ThighPlate' + s, rbox(c + np.array([0.0, -0.05, 0.0]), (0.33, 0.09, 0.46), 0.035), 'armor', 'thigh.' + s)
        add('ThighSide' + s, rbox(lerp(hip, kn, 0.5) + np.array([0.25 * sx, -0.02, 0.0]), (0.09, 0.30, 0.44), 0.03), 'armorRust', 'thigh.' + s)
        add('ThighBand' + s, ring(lerp(hip, kn, 0.80), 0.26, 0.026), 'armor', 'thigh.' + s)
        kc = kn + np.array([0.0, -0.24, 0.0])
        add('KneeCap' + s, rbox(kc, (0.36, 0.14, 0.26), 0.06), 'armor', 'shin.' + s)
        for dx in (-0.07, 0.07):
            V, p = cyl(kc + np.array([dx, -0.04, 0.09]), kc + np.array([dx * 1.2, -0.12, 0.13]), 0.030, 0.0, sides=10)
            add('KneeSpike%s%d' % (s, int(dx * 100)), (V, p), 'steel', 'shin.' + s)
        path = np.array([kn + np.array([0.0, -0.03, -0.08]), lerp(kn, an, 0.4) + np.array([0.0, -0.02, 0.0]), lerp(kn, an, 0.75) + np.array([0.0, 0.0, 0.0]), an + np.array([0.0, 0.0, 0.08])])
        add('Greave' + s, tube(path, [0.235, 0.215, 0.185, 0.160], [0.24, 0.225, 0.195, 0.17], sides=22, caps=(False, False), up=(1.0, 0.0, 0.0)), 'armor', 'shin.' + s)
        for u in (0.25, 0.5, 0.75):
            add('GreaveRib%s%d' % (s, int(u * 100)), ring(lerp(kn, an, u) + np.array([0.0, -0.01, 0.0]), 0.24 - 0.06 * u, 0.022), 'armorRust', 'shin.' + s)
        add('ShinFront' + s, rbox(lerp(kn, an, 0.5) + np.array([0.0, -0.22, 0.0]), (0.20, 0.08, 0.40), 0.035), 'armor', 'shin.' + s)
        # boot
        an_, ball, toe = J['an_' + s], J['ball_' + s], J['toe_' + s]
        add('BootHeel' + s, rbox(an_ + np.array([0.0, 0.08, -0.07]), (0.34, 0.32, 0.30), 0.06), 'armor', 'foot.' + s)
        add('AnkleCuff' + s, ring(an_ + np.array([0.0, 0.0, 0.06]), 0.20, 0.03), 'armorRust', 'foot.' + s)
        add('BootTop' + s, rbox(an_ + np.array([0.0, -0.12, -0.08]), (0.34, 0.34, 0.24), 0.06), 'armor', 'foot.' + s)
        add('BootToe' + s, rbox(ball + np.array([0.0, -0.12, 0.0]), (0.36, 0.40, 0.16), 0.06), 'armorRust', 'toe.' + s)
        add('BootSole' + s, rbox(lerp(an_, toe, 0.5) + np.array([0.0, 0.0, -0.095]), (0.38, 0.84, 0.07), 0.025), 'rubber', 'foot.' + s)
        for k in range(4):
            add('BootRib%s%d' % (s, k), rbox(an_ + np.array([0.0, -0.06 - 0.075 * k, 0.045]), (0.355, 0.03, 0.05), 0.012), 'steel', 'foot.' + s)
    # --- head gear: helmet cap, visor band, the red eyes, the side camera
    add('Helmet', dome(np.array([0.0, 0.025, 2.355]), 0.185, 0.205, 0.17, (0.0, 0.18, 0.98)), 'armor', 'head')
    add('HelmetBack', ellipsoid((0.0, 0.10, 2.30), (0.17, 0.12, 0.17), 18, 10), 'armor', 'head')
    add('Visor', rbox((0.0, -0.150, 2.335), (0.345, 0.07, 0.085), 0.022), 'armor', 'head')
    add('Respirator', rbox((0.0, -0.150, 2.160), (0.215, 0.06, 0.095), 0.025), 'armor', 'head')
    for k in range(4):
        add('RespVent%d' % k, rbox((-0.054 + 0.036 * k, -0.186, 2.160), (0.012, 0.012, 0.06), 0.004), 'steel', 'head')
    add('TopSensor', cyl((0.0, 0.02, 2.52), (0.0, 0.02, 2.555), 0.040, 0.030, sides=12), 'steel', 'head')
    add('BrowPlate', rbox((0.0, -0.165, 2.395), (0.28, 0.05, 0.035), 0.012), 'armorRust', 'head')
    for sx in (-1.0, 1.0):
        add('EyeLens%d' % (sx > 0), ellipsoid((0.075 * sx, -0.183, 2.335), (0.022, 0.014, 0.022), 14, 8), 'eye', 'head')
        add('EyeRing%d' % (sx > 0), ring((0.075 * sx, -0.176, 2.335), 0.034, 0.008, axis=(0, 1, 0)), 'armor', 'head')
    add('SideCamBody', cyl((0.20, -0.02, 2.37), (0.20, -0.17, 2.37), 0.040, 0.040, sides=14), 'steel', 'head')
    add('SideCamLens', ellipsoid((0.20, -0.18, 2.37), (0.028, 0.018, 0.028), 12, 8), 'eye', 'head')
    add('SideCamMount', rbox((0.18, 0.0, 2.37), (0.05, 0.08, 0.06), 0.015), 'armor', 'head')
    add('NeckGuard', tube(np.array([[0, 0.02, 2.04], [0, 0.03, 2.14]]), [0.225, 0.19], [0.21, 0.18], sides=22, caps=(False, False)), 'armor', 'neck')
    # --- the right arm: armored upper arm, joint, long forearm, palm and the three claws
    sh, el, wr = J['sh_R'], J['el_R'], J['wr_R']
    for u in (0.35, 0.60):
        add('SleeveR%d' % int(u * 100), ring(lerp(sh, el, u), 0.205 - 0.02 * u, 0.028, axis=unit(el - sh)), 'armor', 'upperarm.R')
    add('ElbowR', ellipsoid(el, (0.17, 0.16, 0.17), 16, 10), 'armorRust', 'upperarm.R')
    fa = [lerp(el, wr, u) for u in (0.0, 0.25, 0.55, 0.85, 1.0)]
    add('ForearmShell', tube(np.array(fa), [0.185, 0.200, 0.180, 0.150, 0.135], [0.185, 0.205, 0.185, 0.155, 0.135], sides=20, caps=(False, False), up=(0, 1, 0)), 'armor', 'forearm.R')
    for u in (0.2, 0.4, 0.6, 0.8):
        add('ForearmRib%d' % int(u * 100), ring(lerp(el, wr, u), 0.165 - 0.045 * u, 0.017, axis=unit(wr - el)), 'steel', 'forearm.R')
    add('ForearmPiston', cyl(el + np.array([0.12, -0.07, -0.05]), wr + np.array([0.10, -0.06, 0.08]), 0.026, 0.026, sides=10), 'steel', 'forearm.R')
    add('Palm', rbox(wr + np.array([0.0, 0.0, -0.10]), (0.30, 0.26, 0.22), 0.05), 'steel', 'hand.R')
    for nm, jt in CLAW.items():
        add(nm + 'KnuckleA', ellipsoid(jt[0], (0.058, 0.058, 0.058), 12, 8), 'steel', nm + 'a.R')
        add(nm + 'SegA', cyl(jt[0], jt[1], 0.045, 0.036, sides=10), 'steel', nm + 'a.R')
        add(nm + 'KnuckleB', ellipsoid(jt[1], (0.044, 0.044, 0.044), 10, 8), 'armor', nm + 'b.R')
        add(nm + 'SegB', cyl(jt[1], jt[2], 0.036, 0.020, sides=10), 'steel', nm + 'b.R')
        tip = jt[2] + unit(jt[2] - jt[1]) * 0.09
        add(nm + 'Tip', cyl(jt[2], tip, 0.022, 0.0, sides=8), 'steel', nm + 'b.R')
    # hoses along the claw arm
    add('HoseR', tube(np.array([sh + np.array([0.0, 0.10, -0.10]), lerp(sh, el, 0.5) + np.array([0.02, 0.17, 0.0]), el + np.array([0.0, 0.13, 0.0]),
                                lerp(el, wr, 0.5) + np.array([-0.03, 0.15, 0.0])]), 0.018, sides=8, caps=(True, True)), 'rubber', 'upperarm.R')
    # --- the left arm: big organic fist with a bracer and a shoulder strap
    sh, el, wr = J['sh_L'], J['el_L'], J['wr_L']
    add('BracerL', tube(np.array([lerp(el, wr, 0.55), lerp(el, wr, 0.9)]), [0.17, 0.125], [0.17, 0.125], sides=20, caps=(False, False), up=(0, 1, 0)), 'armor', 'forearm.L')
    add('BracerRibL', ring(lerp(el, wr, 0.7), 0.155, 0.018, axis=unit(wr - el)), 'steel', 'forearm.L')
    fist = wr + np.array([0.0, -0.01, -0.13])
    add('FistPalm', rbox(fist, (0.25, 0.20, 0.22), 0.06), 'skin', 'hand.L')
    for k, xo in enumerate((-0.085, -0.028, 0.028, 0.085)):
        add('Finger%d' % k, rbox(fist + np.array([xo, -0.075, -0.10]), (0.055, 0.075, 0.12), 0.022), 'skin', 'hand.L')
    add('Thumb', rbox(fist + np.array([-0.13, -0.05, -0.02]), (0.06, 0.09, 0.14), 0.025, rot_z(0.3)), 'skin', 'hand.L')
    return parts


# ------------------------------------------------------------------ materials
def kd_materials(body_ob):
    M = {}
    M['skin'] = pbr('KD_Skin', (0.050, 0.011, 0.009), (0.135, 0.030, 0.022), rough=0.52, nscale=7, bump=0.45, bscale=130, dirt=0.9,
                    dirt_col=(0.020, 0.004, 0.004), ob=body_ob)
    M['armor'] = pbr('KD_Armor', (0.010, 0.010, 0.012), (0.040, 0.040, 0.046), metal=0.75, rough=0.40, nscale=6, bump=0.32, bscale=60, dirt=0.9,
                     dirt_col=(0.02, 0.012, 0.01), wear_col=(0.30, 0.07, 0.04), wear=0.38)
    M['armorRust'] = pbr('KD_ArmorRust', (0.075, 0.020, 0.013), (0.20, 0.055, 0.030), metal=0.7, rough=0.5, nscale=6, bump=0.4, bscale=50, dirt=1.0,
                         dirt_col=(0.02, 0.01, 0.008), wear_col=(0.10, 0.10, 0.11), wear=0.4)
    M['steel'] = pbr('KD_Steel', (0.12, 0.12, 0.13), (0.30, 0.30, 0.32), metal=1.0, rough=0.34, nscale=8, bump=0.25, dirt=0.7, wear_col=(0.28, 0.08, 0.04), wear=0.5)
    M['rubber'] = pbr('KD_Rubber', (0.008, 0.008, 0.010), (0.025, 0.025, 0.028), rough=0.7, nscale=7, bump=0.3, dirt=0.4)
    M['eye'] = glow_mat('KD_Eye', (1.0, 0.0, 0.0), 5.0)
    return M


def build_killdroid():
    for n in list(bpy.data.collections.keys()):
        if n == MC:
            wipe(n)
    coll = _coll(MC)
    rig = make_rig(coll)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'XYZ'
    setup_ik(rig)
    S, chains = build_skin()
    body = skin_to_obj(S, coll, rig)
    M = kd_materials(body)
    body.data.materials.append(M['skin'])
    parts = armor_parts()
    objs = []
    for name, Vp, Pp, mk, bone in parts:
        ob = make_obj("KillDroid_" + name, Vp, Pp, coll)
        ob.data.materials.append(M[mk])
        ob.parent = rig
        ob.parent_type = 'BONE'
        ob.parent_bone = bone
        objs.append(ob)
    bpy.context.view_layer.update()
    for ob in objs:
        ob.matrix_parent_inverse = ob.matrix_world.inverted()
    return {"body_verts": len(body.data.vertices), "rigid_parts": len(objs), "bones": len(rig.data.bones)}


# ------------------------------------------------------------------ studio and design sheet
def make_kd_studio():
    sc = bpy.data.scenes.get("KD_Studio") or bpy.data.scenes.new("KD_Studio")
    sc.render.engine = 'BLENDER_EEVEE'
    sc.eevee.taa_render_samples = 40
    sc.view_settings.view_transform = 'AgX'
    sc.render.resolution_x, sc.render.resolution_y = 640, 900
    w = bpy.data.worlds.get("KD_World") or bpy.data.worlds.new("KD_World")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.09, 0.095, 0.105, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    sc.world = w
    mc = bpy.data.collections.get(MC)
    if mc and mc.name not in [c.name for c in sc.collection.children]:
        sc.collection.children.link(mc)
    for nm, loc, e, col, sz in (("KD_Key", (-3.0, -4.0, 4.0), 900.0, (1.0, 0.92, 0.85), 3.0), ("KD_Rim", (3.5, 3.0, 3.2), 700.0, (0.7, 0.8, 1.0), 3.0),
                               ("KD_Fill", (3.5, -3.5, 1.5), 200.0, (1.0, 0.7, 0.6), 4.0)):
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.energy, ld.size, ld.color = e, sz, col
        lo = bpy.data.objects.new(nm, ld)
        sc.collection.objects.link(lo)
        lo.location = loc
        look_at(lo, (0.0, 0.0, 1.2))
    fl = bpy.data.objects.new("KD_Floor", bpy.data.meshes.new("KD_Floor"))
    fl.data.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
    sc.collection.objects.link(fl)
    m = bpy.data.materials.get("KD_FloorMat") or bpy.data.materials.new("KD_FloorMat")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.05, 0.05, 0.055, 1.0)
    fl.data.materials.append(m)
    cd = bpy.data.cameras.new("KD_Cam")
    cd.lens = 50
    cam = bpy.data.objects.new("KD_Cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return sc, cam


def design_sheet(path):
    sc, cam = make_kd_studio()
    views = [((-2.4, -6.2, 1.55), (0, 0, 1.28), 62), ((-6.0, -2.0, 1.45), (0, 0, 1.28), 62), ((5.0, 4.6, 1.5), (0, 0, 1.28), 62),
             ((0.2, -2.1, 2.1), (0, 0, 2.25), 70)]
    files = []
    for i, (loc, tgt, lens) in enumerate(views):
        cam.location = tuple(np.array(loc) * KD_SCALE)
        cam.data.lens = lens
        look_at(cam, tuple(np.array(tgt) * KD_SCALE))
        sc.render.filepath = path.replace(".png", "_%d.png" % i)
        bpy.ops.render.render(write_still=True, scene=sc.name)
        files.append(sc.render.filepath)
    return files


if __name__ == "__main__" or True:
    info = build_killdroid()
    print("BUILT", info)
    files = design_sheet(r"C:\ai\unedited\renders\killdroid_v01.png")
    print("RENDERED", files)
    bpy.ops.wm.save_as_mainfile(filepath=r"C:\ai\unedited\killdroid_v01.blend")
    print("SAVED")
