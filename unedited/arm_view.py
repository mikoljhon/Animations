import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 700, 700
sc.eevee.taa_render_samples = 8
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 70; cam.data.clip_end = 100
sc.camera = cam
a = sys.argv.index("--") + 1
tag = sys.argv[a]; frames = [int(x) for x in sys.argv[a+1:]]
for f in frames:
    sc.frame_set(f)
    for nm, tgt, loc in (("rel", (0.0, 2.3, 1.5), (-2.2, 5.0, 1.8)), ("ref", (0.0, 0.7, 1.5), (-2.2, 3.4, 1.8))):
        cam.location = loc
        cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\arm_%s_%s_%04d.png" % (tag, nm, f)
        bpy.ops.render.render(write_still=True)
