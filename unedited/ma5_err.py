import bpy, numpy as np, math, importlib
from mathutils import Matrix
MA = "bl_ext.user_default.motion_assist"
grip = importlib.import_module(MA + ".grip")
sc = bpy.data.scenes["Cave"]; rig = bpy.data.objects["MenderC_Rig"]
rec, grips, holds = grip.load(r"C:\ai\unedited\anim_cache5\grips_cut5b.json")
bad = {"L": [], "R": []}; mx = {"L": [0, 0], "R": [0, 0]}
for f in sorted(holds):
    if f > 211: break
    sc.frame_set(f)
    for s in "LR":
        if s not in holds[f]: continue
        W, H = holds[f][s]
        pb = rig.pose.bones["hand." + s]
        dp = (pb.matrix.translation - W).length * 100
        dr = math.degrees(pb.matrix.to_3x3().normalized().to_quaternion().rotation_difference(H.to_quaternion()).angle)
        dr = min(dr, 360 - dr)
        mx[s] = [max(mx[s][0], dp), max(mx[s][1], dr)]
        if dp > 1.5 or dr > 12: bad[s].append((f, round(dp, 1), round(dr, 0)))
print("MAXERR", {s: [round(v[0], 2), round(v[1], 1)] for s, v in mx.items()})
for s in "LR": print("BAD", s, bad[s][:50])
