import bpy, math
sc = bpy.data.scenes["Street"]; rig = bpy.data.objects["Relic_AI_Rig"]
for f in (20, 30, 40, 100, 160):
    sc.frame_set(f)
    for s in "L":
        W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
        sh, wr, tg = W("upperarm." + s), W("forearm." + s, True), W("ik_hand." + s)
        E = lambda n: bpy.data.objects["ref_" + n].matrix_world.translation
        rsh, rwr, rel = E("shoulder_" + s), E("wrist_" + s), E("elbow_" + s)
        v1 = (wr - sh); v2 = (rwr - rsh); v3 = (tg - sh)
        print("AD", f, s, "Relic wrist-shoulder (%.2f,%.2f,%.2f) |%.2f|" % (*v1, v1.length), "target-shoulder (%.2f,%.2f,%.2f)" % tuple(v3), "ref wrist-shoulder (%.2f,%.2f,%.2f) |%.2f|" % (*v2, v2.length), "angle %.0f" % math.degrees(v1.angle(v2)), "target-vs-ref %.0f" % math.degrees(v3.angle(v2)))
