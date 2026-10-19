import bpy, numpy as np, math
sc = bpy.data.scenes["Cave"]; pick = bpy.data.objects["MenderC_Pick"]; rig = bpy.data.objects["MenderC_Rig"]
import importlib
grip = importlib.import_module("bl_ext.user_default.motion_assist.grip")
c, d, hl = grip.prop_axis(pick)
rows = []
for f in list(range(28, 50)) + list(range(80, 96)) + list(range(135, 152)):
    sc.frame_set(f)
    Wi = pick.matrix_world.inverted()
    out = [f]
    for s in "LR":
        w = Wi @ (rig.matrix_world @ rig.pose.bones["hand." + s].head)
        v = np.array(w) - np.array(c)
        out += [round(float(v @ np.array(d)) * 100, 1), round(float(np.linalg.norm(v - np.array(d) * (v @ np.array(d)))) * 100, 1)]
    rows.append(out)
print("SLIDE (frame, L_along_cm, L_perp_cm, R_along_cm, R_perp_cm)")
for r in rows:
    print("ROW", r)
