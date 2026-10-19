import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 900, 600
sc.eevee.taa_render_samples = 6
for o in bpy.data.objects:
    if "Crust" in o.name: o.hide_render = True
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 40; cam.data.clip_end = 100
sc.camera = cam
a = sys.argv.index("--") + 1
tag = sys.argv[a]; frames = [int(x) for x in sys.argv[a+1:]]
for f in frames:
    sc.frame_set(f)
    cam.location = (4.6, 1.2, 1.3)
    cam.rotation_euler = (Vector((0.0, 1.2, 1.2)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = r"C:\ai\unedited\renders\ap_%s_%04d.png" % (tag, f)
    bpy.ops.render.render(write_still=True)
