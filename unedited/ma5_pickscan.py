import bpy, numpy as np, math
sc = bpy.data.scenes["Cave"]; pick = bpy.data.objects["MenderC_Pick"]; rig = bpy.data.objects["MenderC_Rig"]
Q = []; P = []
for f in range(1, 361):
    sc.frame_set(f)
    W = pick.matrix_world
    Q.append(W.to_quaternion()); P.append(np.array(W.translation))
P = np.array(P)
ang = [0.0] + [math.degrees(Q[i].rotation_difference(Q[i-1]).angle) for i in range(1, 360)]
sp = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
acc_ang = np.r_[0, 0, np.abs(np.diff(ang))[1:]]
print("ANGSTEP", [(f + 1, round(a, 1)) for f, a in enumerate(ang) if a > 25 or (f > 0 and f < 215 and abs(a - ang[f-1]) > 18)][:40])
print("POSSTEP_cm", [(f + 1, round(s * 100, 1)) for f, s in enumerate(sp) if s > 0.22][:40])
print("ANG_FIRST60", [round(a, 1) for a in ang[28:60]])
# also: is the pick offset from the hand constant?
Mh = []
for f in range(1, 212, 1):
    sc.frame_set(f)
    h = rig.matrix_world @ rig.pose.bones["hand.R"].matrix
    Mh.append(h.inverted() @ pick.matrix_world)
d = [ (Mh[i].translation - Mh[0].translation).length for i in range(len(Mh))]
r = [ math.degrees(Mh[i].to_quaternion().rotation_difference(Mh[0].to_quaternion()).angle) for i in range(len(Mh))]
print("PICK_REL_HAND_MAX", round(max(d) * 100, 1), "cm", round(max(r), 1), "deg")
print("PICK_REL_BAD", [(i + 1, round(d[i] * 100, 1), round(r[i], 1)) for i in range(len(d)) if d[i] > 0.02 or r[i] > 6][:30])
