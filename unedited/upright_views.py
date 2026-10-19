import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 640, 760
sc.eevee.taa_render_samples = 8
for o in bpy.data.objects:
    if "Crust" in o.name or o.name.startswith("ref"): o.hide_render = True
rig = bpy.data.objects["Relic_AI_Rig"]
cam = bpy.data.objects.new("UV", bpy.data.cameras.new("UV")); sc.collection.objects.link(cam)
cam.data.lens = 45; cam.data.clip_end = 200
sc.camera = cam
a = sys.argv.index("--") + 1
tag = sys.argv[a]; frames = [int(x) for x in sys.argv[a + 1:]]
for f in frames:
    sc.frame_set(f)
    c = rig.matrix_world @ rig.pose.bones["chest"].head
    fw = (rig.matrix_world.to_3x3() @ Vector((0, -1, 0))).normalized()     # where Relic faces
    lf = Vector((-fw.y, fw.x, 0.0))
    for nm, d in (("front", fw * 3.0 + lf * 1.2), ("side", lf * -3.2 + fw * 0.3)):
        cam.location = c + d + Vector((0, 0, 0.15))
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\up_%s_%04d_%s.png" % (tag, f, nm)
        bpy.ops.render.render(write_still=True)
