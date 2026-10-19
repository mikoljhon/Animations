import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 520, 520
sc.eevee.taa_render_samples = 6
kr = bpy.data.objects["KillDroid_Rig"]
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG")); sc.collection.objects.link(dbg)
dbg.data.lens = 55; dbg.data.clip_end = 1500.0
args = sys.argv[sys.argv.index("--") + 1:]
tag = args[0]
for f in [int(x) for x in args[1:]]:
    sc.frame_set(f)
    p = kr.matrix_world @ kr.pose.bones["pelvis"].head
    dbg.location = (p.x + 7.5, p.y, 1.5)
    dbg.rotation_euler = (Vector((p.x, p.y, 1.4)) - dbg.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = dbg
    sc.render.filepath = r"C:\ai\unedited\renders\kv_%s_%04d.png" % (tag, f)
    bpy.ops.render.render(write_still=True)
