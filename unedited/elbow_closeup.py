import bpy, sys, math
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 700, 700
sc.eevee.taa_render_samples = 8
for o in bpy.data.objects:
    if "Crust" in o.name or o.name.startswith("ref"): o.hide_render = True
rig = bpy.data.objects["Relic_AI_Rig"]
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 50; cam.data.clip_end = 100
sc.camera = cam
# pick frames where the left elbow is bent 70-110 degrees
cands = []
for f in range(sc.frame_start, sc.frame_end + 1, 3):
    sc.frame_set(f)
    W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
    for sd in "LR":
        sh, el, wr = W("upperarm." + sd), W("forearm." + sd), W("forearm." + sd, True)
        bend = 180 - math.degrees((sh - el).angle(wr - el))
        if 75 < bend < 115:
            cands.append((abs(bend - 95), f, sd))
cands.sort()
picked = []
for c in cands:
    if all(abs(c[1] - p[1]) > 40 for p in picked):
        picked.append(c)
    if len(picked) == 3:
        break
print("PICKED", [(p[1], p[2]) for p in picked])
for _, f, sd in picked:
    sc.frame_set(f)
    W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
    el = W("forearm." + sd); sh = W("upperarm." + sd); wr = W("forearm." + sd, True)
    n = (sh - el).cross(wr - el).normalized()           # normal of the arm's bend plane: look straight at the bend
    for k, (nm, dirv) in enumerate((("plane", n), ("outside", (el - (sh + wr) * 0.5).normalized()))):
        cam.location = el + dirv * 2.2
        cam.rotation_euler = (el - cam.location).to_track_quat('-Z', 'Z').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\elb_%d_%s_%s.png" % (f, sd, nm)
        bpy.ops.render.render(write_still=True)
