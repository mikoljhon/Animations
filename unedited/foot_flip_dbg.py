import bpy, math
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
for f in range(180, 187):
    sc.frame_set(f)
    out = []
    for s in "R":
        pb = rig.pose.bones["foot." + s]
        h = rig.matrix_world @ pb.head; t = rig.matrix_world @ pb.tail
        d = (t - h).normalized()
        ik = rig.pose.bones["ik_foot." + s]
        qd = (rig.matrix_world @ ik.matrix).to_quaternion()
        out.append("foot dir (%.2f,%.2f,%.2f) ankle (%.2f,%.2f,%.2f) ik quat (%.2f,%.2f,%.2f,%.2f)" % (*d, *h, *qd))
    print("FF", f, *out)
