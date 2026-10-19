import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 800, 800
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 60; cam.data.clip_end = 100
frames = [int(x) for x in sys.argv[sys.argv.index("--") + 1:]]
views = {"front": (-0.6, 6.2, 1.4), "side": (4.2, 2.1, 1.4)}
for f in frames:
    sc.frame_set(f)
    for nm, loc in views.items():
        cam.location = loc
        cam.rotation_euler = (Vector((0.0, 2.0, 1.2)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.camera = cam
        sc.render.filepath = r"C:\ai\unedited\renders\cl_%s_%04d.png" % (nm, f)
        bpy.ops.render.render(write_still=True)
