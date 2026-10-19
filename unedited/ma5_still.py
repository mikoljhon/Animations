# usage: blender -b FILE --python ma5_still.py -- TAG f1,f2,...   (renders at 640x360, 10 samples, the scene's own camera)
import bpy, sys
a = sys.argv[sys.argv.index("--") + 1:]
tag, frames = a[0], [int(x) for x in a[1].split(",")]
sc = bpy.data.scenes["Cave"]
sc.render.resolution_x, sc.render.resolution_y = 640, 360
sc.eevee.taa_render_samples = 10
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = r"C:\ai\unedited\renders\%s_%03d.png" % (tag, f)
    bpy.ops.render.render(write_still=True, scene="Cave")
