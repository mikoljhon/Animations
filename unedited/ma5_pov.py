# First-person test: a camera at Relic's eyes that follows his head bone. Background Blender:
#   blender.exe -b relic_v15_opening_cut6_pass.blend --python ma5_pov.py -- TAG f1,f2,...   (renders 640x360 stills)
import math
import sys

import bpy
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
tag, frames = a[0], [int(x) for x in a[1].split(",")]
y_off = float(a[2]) if len(a) > 2 else -0.274            # head-local y of the camera (the glass is at -0.262)
lens = float(a[3]) if len(a) > 3 else 22.0

sc = bpy.data.scenes["Cave"]
rig = bpy.data.objects["Relic_AI_Rig"]
pb = rig.pose.bones["head"]
Mw = rig.matrix_world
R = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_4x4()
P = Matrix.Translation(Vector((0.0, y_off, 1.862)))

cd = bpy.data.cameras.new("POV")
cd.lens = lens
cd.clip_start = 0.005
cd.clip_end = 60.0
cam = bpy.data.objects.new("POV", cd)
sc.collection.objects.link(cam)
sc.timeline_markers.clear()
sc.camera = cam
sc.render.resolution_x, sc.render.resolution_y = 640, 360
sc.eevee.taa_render_samples = 12
for f in frames:
    sc.frame_set(f)
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    cam.matrix_world = Mw @ delta @ P @ R
    sc.render.filepath = r"C:\ai\unedited\renders\%s_%03d.png" % (tag, f)
    bpy.ops.render.render(write_still=True, scene="Cave")
print("done")
