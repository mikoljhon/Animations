import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
who = sys.argv[sys.argv.index("--") + 1]
kind = sys.argv[sys.argv.index("--") + 2] if len(sys.argv) > sys.argv.index("--") + 2 else "relic"
sc.render.resolution_x, sc.render.resolution_y = 640, 640
sc.eevee.taa_render_samples = 6
cam1 = sc.camera
for _o in bpy.data.objects:
    if "Crust" in _o.name: _o.hide_render = True
cam2 = bpy.data.objects.new("DBG2", bpy.data.cameras.new("DBG2")); sc.collection.objects.link(cam2)
cam2.data.lens = 32; cam2.data.clip_end = 1500.0
tgt = Vector((0.0, 1.8, 1.2))
cam2.location = (-3.4, 5.6, 1.7) if kind == "relic" else (-4.5, 8.5, 2.2)
cam2.rotation_euler = (tgt - cam2.location).to_track_quat('-Z', 'Y').to_euler()
import os
for d in ("A", "B"):
    os.makedirs(r"C:\ai\unedited\renders\fmv_%s_%s" % (who, d), exist_ok=True)
for f in range(1, sc.frame_end + 1):
    sc.frame_set(f)
    sc.camera = cam1
    sc.render.filepath = r"C:\ai\unedited\renders\fmv_%s_A\f_%04d.png" % (who, f)
    bpy.ops.render.render(write_still=True)
    sc.camera = cam2
    sc.render.filepath = r"C:\ai\unedited\renders\fmv_%s_B\f_%04d.png" % (who, f)
    bpy.ops.render.render(write_still=True)
print("VID DONE")
