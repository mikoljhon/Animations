import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 800, 800
sc.eevee.taa_render_samples = 8
names = [o.name for o in bpy.data.objects if o.name.startswith("Relic_AI")]
print("OBJS", names)
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 85; cam.data.clip_end = 100
frames = [int(x) for x in sys.argv[sys.argv.index("--") + 1:]]
crust = [o for o in bpy.data.objects if "Crust" in o.name]
views = {"upper": ((-2.0, 5.6, 1.8), (0.0, 2.4, 1.45)), "feet": ((-2.5, 5.0, 0.6), (0.0, 2.1, 0.3))}
for f in frames:
    sc.frame_set(f)
    for hide in (True,):
        for c in crust: c.hide_render = hide
        for nm, (loc, tgt) in views.items():
            cam.location = loc
            cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat('-Z', 'Y').to_euler()
            sc.camera = cam
            sc.render.filepath = r"C:\ai\unedited\renders\cl2_%s_%04d.png" % (nm, f)
            bpy.ops.render.render(write_still=True)
