import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 900, 600
sc.eevee.taa_render_samples = 8
a = sys.argv.index("--") + 1
tag = sys.argv[a]; frames = [int(x) for x in sys.argv[a+1:]]
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = r"C:\ai\unedited\renders\cmp_%s_%04d.png" % (tag, f)
    bpy.ops.render.render(write_still=True)
