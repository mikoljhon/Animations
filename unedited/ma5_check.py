import bpy, json, importlib, math
from mathutils import Vector
MA = "bl_ext.user_default.motion_assist"
grip = importlib.import_module(MA + ".grip")
sc = bpy.data.scenes["Cave"]; rig = bpy.data.objects["MenderC_Rig"]; pick = bpy.data.objects["MenderC_Pick"]
out = {"L": [], "R": []}
for f in range(1, 213, 3):
    sc.frame_set(f)
    c, d, hl = grip.prop_axis(pick)
    W = pick.matrix_world
    c, d = W @ c, (W.to_3x3() @ d).normalized()
    for s in "LR":
        h = rig.matrix_world @ rig.pose.bones["hand." + s].head
        v = h - c
        dist = (v - d * v.dot(d)).length
        out[s].append((round(dist * 100, 1), f))
print("MAXDIST", {s: max(v) for s, v in out.items()})
print("OVER4", {s: [x for x in v if x[0] > 4.0][:12] for s, v in out.items()})
