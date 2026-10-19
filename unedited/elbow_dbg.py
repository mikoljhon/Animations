import bpy, math, importlib
from mathutils import Vector
limits = importlib.import_module("bl_ext.user_default.motion_assist.limits")
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
lims = limits.rig_limits(rig)
for b in ("forearm.L", "forearm.R", "shin.L", "shin.R"):
    l = lims[b]; print("HINGE", b, "ranges", l.ranges, "hinge", tuple(round(x,2) for x in l.hinge), "parent", l.parent)
B, P = rig.data.bones, rig.pose.bones
worst = {}
for f in range(1, sc.frame_end + 1):
    sc.frame_set(f)
    for b in ("forearm.L", "forearm.R", "shin.L", "shin.R"):
        lim = lims[b]
        rel_pose = P[lim.parent].matrix.to_3x3().normalized().inverted() @ P[b].matrix.to_3x3().normalized()
        rel_rest = B[lim.parent].matrix_local.to_3x3().inverted() @ B[b].matrix_local.to_3x3()
        q = (rel_rest.inverted() @ rel_pose).to_quaternion()
        # bend angle about the hinge axis: signed angle of the tip direction
        tip = q @ limits.Y
        phi = limits.hinge_angle(tip, lim.hinge)
        deg = math.degrees(phi)
        w = worst.setdefault(b, [999, -999, 0, f])
        w[0] = min(w[0], deg); w[1] = max(w[1], deg)
        if deg < lim.ranges[0] - 3: w[2] += 1
print("ELBOW/KNEE bend angle (deg, relative to the rest bend): min / max / frames below the allowed minimum")
for b, w in worst.items():
    print("BEND", b, "min %.0f max %.0f below-min frames %d  allowed range (%.0f, %.0f)" % (w[0], w[1], w[2], lims[b].ranges[0], lims[b].ranges[1]))
