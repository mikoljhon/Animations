# hip_views.py : close-ups of Relic's hips on the frames where each thigh is turned furthest from the pelvis (where a leg could pull away
#   from the body).   blender.exe -b pipe_<tag>.blend --python hip_views.py -- <tag> [n_frames]   -> renders\hip_<tag>_<frame>_<view>.png
import bpy, sys, math, os
from mathutils import Vector
a = sys.argv.index("--") + 1
tag = sys.argv[a]
nf = int(sys.argv[a + 1]) if len(sys.argv) > a + 1 else 4
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 520, 520
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
pb = rig.pose.bones
for o in bpy.data.objects:
    if "Crust" in o.name or o.name.startswith("ref"):
        o.hide_render = True
rows = []
for f in range(sc.frame_start, sc.frame_end + 1):
    sc.frame_set(f)
    for sd in "LR":
        q = pb['thigh.' + sd].rotation_quaternion
        rows.append((math.degrees(q.angle), f, sd))
rows.sort(reverse=True)
picked = []
for r in rows:
    if all(abs(r[1] - p[1]) > 12 for p in picked):
        picked.append(r)
    if len(picked) == nf:
        break
cam = bpy.data.objects.new("HV", bpy.data.cameras.new("HV")); sc.collection.objects.link(cam)
cam.data.lens = 50; cam.data.clip_end = 100
sc.camera = cam
os.makedirs(r"C:\ai\unedited\renders\hips_%s" % tag, exist_ok=True)
for ang, f, sd in picked:
    sc.frame_set(f)
    c = rig.matrix_world @ pb['thigh.' + sd].head
    fw = rig.matrix_world.to_3x3() @ Vector((0, -1, 0)); fw.z = 0; fw.normalize()
    lf = Vector((-fw.y, fw.x, 0.0))
    out = lf if sd == 'L' else -lf
    for nm, d in (("front", fw * 1.6 + out * 0.5), ("side", out * 1.6 + fw * 0.2), ("back", -fw * 1.6 + out * 0.5)):
        cam.location = c + d + Vector((0, 0, 0.1))
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\hips_%s\%04d_%s_%s.png" % (tag, f, sd, nm)
        bpy.ops.render.render(write_still=True)
    print("HIP", f, sd, "thigh turned %.0f deg" % ang)
print("HIP DONE")
