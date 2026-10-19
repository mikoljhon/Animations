# swivel_check.py : which way does each elbow point (around the shoulder-wrist line) compared with (a) its IK pole target and (b) the real capture?
import bpy, math, sys
import numpy as np
from mathutils import Vector
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
a = sys.argv.index("--") + 1
f0, f1 = int(sys.argv[a]), int(sys.argv[a + 1])
def perp(v, ax):
    p = v - ax * v.dot(ax)
    return p
def sangle(u, v, ax):
    return math.degrees(math.atan2(u.cross(v).dot(ax), u.dot(v)))
rows = {s: [] for s in "LR"}
rowsK = {s: [] for s in "LR"}
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    for s in "LR":
        W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
        sh, el, wr = W("upperarm." + s), W("forearm." + s), W("forearm." + s, True)
        pole = W("ik_elbow." + s)
        E = lambda n: bpy.data.objects["ref_" + n].matrix_world.translation
        rsh, rel, rwr = E("shoulder_" + s), E("elbow_" + s), E("wrist_" + s)
        ax = (wr - sh).normalized(); rax = (rwr - rsh).normalized()
        pe, pp = perp(el - sh, ax), perp(pole - sh, ax)
        pr = perp(rel - rsh, rax)
        if pe.length < 0.02 or pr.length < 0.02:
            continue
        # compare the real elbow direction expressed around Relic's own axis (the axes agree to within a few degrees)
        pr2 = perp(pr, ax)
        rows[s].append((f, sangle(pp, pe, ax), sangle(pr2, pe, ax), pe.length, pr.length))
        hp, kn, an = W("thigh." + s), W("shin." + s), W("shin." + s, True)
        kp = W("ik_knee." + s)
        rhp, rkn, ran = E("hip_" + s), E("knee_" + s), E("ankle_" + s)
        axk = (an - hp).normalized(); raxk = (ran - rhp).normalized()
        qe, qp, qr = perp(kn - hp, axk), perp(kp - hp, axk), perp(perp(rkn - rhp, raxk), axk)
        if qe.length > 0.02 and qr.length > 0.02:
            rowsK[s].append((f, sangle(qp, qe, axk), sangle(qr, qe, axk)))
for s in "LR":
    r = np.array(rows[s])
    if len(r) == 0:
        continue
    e_pole = np.abs(r[:, 1]); e_ref = np.abs(r[:, 2])
    print("SWIVEL %s elbow vs its pole: mean %.0f max %.0f deg | elbow vs the REAL elbow: mean %.0f max %.0f deg, frames off by more than 60 deg: %d of %d" % (
        s, e_pole.mean(), e_pole.max(), e_ref.mean(), e_ref.max(), int((e_ref > 60).sum()), len(r)))
    worst = r[np.argsort(-e_ref)[:6]]
    print("   worst frames (frame, vs pole, vs real):", [(int(x[0]), int(x[1]), int(x[2])) for x in worst])

for s in "LR":
    r = np.array(rowsK[s])
    if len(r) == 0:
        continue
    e_ref = np.abs(r[:, 2])
    print("KNEE %s knee vs its pole: mean %.0f max %.0f | knee vs the REAL knee: mean %.0f max %.0f deg, frames off > 45: %d of %d" % (s, np.abs(r[:, 1]).mean(), np.abs(r[:, 1]).max(), e_ref.mean(), e_ref.max(), int((e_ref > 45).sum()), len(r)))
