import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 640, 360
sc.eevee.taa_render_samples = 8
cam = bpy.data.objects["POV"]
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG")); sc.collection.objects.link(dbg)
dbg.data.lens = 30; dbg.data.clip_end = 1500.0
frames = [int(x) for x in sys.argv[sys.argv.index("--") + 1:]]
for f in frames:
    sc.frame_set(f)
    sc.camera = cam
    sc.render.filepath = r"C:\ai\unedited\renders\fp_%04d.png" % f
    bpy.ops.render.render(write_still=True)
    ry = bpy.data.objects["Relic_AI_Rig"].pose.bones["pelvis"]
    kz = bpy.data.objects["KillDroid_Rig"]
    tgt = (kz.matrix_world @ kz.pose.bones["pelvis"].head) * 0.5 + (bpy.data.objects["Relic_AI_Rig"].matrix_world @ ry.head) * 0.5
    dbg.location = (tgt.x + 8.0, tgt.y + 0.5, 1.8 + tgt.z * 0.3)
    dbg.rotation_euler = (tgt - dbg.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = dbg
    sc.render.filepath = r"C:\ai\unedited\renders\sd_%04d.png" % f
    bpy.ops.render.render(write_still=True)
print("PREV DONE")
