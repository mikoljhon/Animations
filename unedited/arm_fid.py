import bpy, math, sys
import numpy as np
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
a = sys.argv.index("--") + 1
f0, f1 = int(sys.argv[a]), int(sys.argv[a + 1])
res = {s: [] for s in "LR"}
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    for s in "LR":
        W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
        sh, el, wr = W("upperarm." + s), W("forearm." + s), W("forearm." + s, True)
        E = lambda n: bpy.data.objects["ref_" + n].matrix_world.translation
        rsh, rel, rwr = E("shoulder_" + s), E("elbow_" + s), E("wrist_" + s)
        d1, d2 = (wr - sh).normalized(), (rwr - rsh).normalized()
        e1, e2 = (el - sh).normalized(), (rel - rsh).normalized()
        flex = lambda a_, b_, c_: 180 - math.degrees((a_ - b_).angle(c_ - b_))
        res[s].append((math.degrees(d1.angle(d2)), math.degrees(e1.angle(e2)), flex(sh, el, wr) - flex(rsh, rel, rwr)))
for s in "LR":
    r = np.array(res[s])
    print("ARMFID", s, "hand direction error mean %.0f max %.0f deg | upper-arm direction error mean %.0f max %.0f deg | elbow bend error mean %.0f max %.0f deg" % (r[:, 0].mean(), r[:, 0].max(), r[:, 1].mean(), r[:, 1].max(), np.abs(r[:, 2]).mean(), np.abs(r[:, 2]).max()))
