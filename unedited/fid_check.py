import bpy, sys, math
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
off = None
rows = []
for f in range(6, 44, 4):
    sc.frame_set(f)
    out = [f]
    for s in "LR":
        ra = rig.matrix_world @ rig.pose.bones["foot." + s].head
        rk = rig.matrix_world @ rig.pose.bones["shin." + s].head
        ma = bpy.data.objects["ref_ankle_" + s].matrix_world.translation
        mk = bpy.data.objects["ref_knee_" + s].matrix_world.translation
        mh = bpy.data.objects["ref_hips"].matrix_world.translation
        # compare relative to each figure's own hips, ignoring the 1.6 m offset
        rh = rig.matrix_world @ rig.pose.bones["pelvis"].head
        da = (ra - rh) - (ma - mh)
        dk = (rk - rh) - (mk - mh)
        out += [s, "ankle err %.2f m" % da.length, "knee err %.2f m" % dk.length, "ankle rel hips R(%.2f,%.2f,%.2f) ref(%.2f,%.2f,%.2f)" % tuple(list(ra - rh) + list(ma - mh))]
    print("FID", *out)
