import bpy, importlib, math
ap = importlib.import_module("bl_ext.user_default.motion_assist.autopose")
limits = importlib.import_module("bl_ext.user_default.motion_assist.limits")
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
lims = limits.rig_limits(rig)
for b in ("foot.L", "thigh.L", "upperarm.L", "shin.L"):
    l = lims[b]
    print("LIMDEF", b, "parent", l.parent, "ranges", l.ranges, "hinge", l.hinge, "ref", l.ref, "slack", getattr(l,"slack",None))
B, P = rig.data.bones, rig.pose.bones
for f in (1, 100, 204):
    sc.frame_set(f)
    for b in ("foot.L", "thigh.L", "upperarm.L"):
        lim = lims[b]
        rel_pose = P[lim.parent].matrix.to_3x3().normalized().inverted() @ P[b].matrix.to_3x3().normalized()
        rel_rest = B[lim.parent].matrix_local.to_3x3().inverted() @ B[b].matrix_local.to_3x3()
        q = (rel_rest.inverted() @ rel_pose).to_quaternion()
        print("REL", f, b, "excess", round(limits.excess(q, lim),1), "euler deg", [round(math.degrees(x)) for x in q.to_euler()])
