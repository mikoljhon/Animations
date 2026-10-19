import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 600, 700
sc.eevee.taa_render_samples = 6
for o in bpy.data.objects:
    if "Crust" in o.name: o.hide_render = True
    if o.name.startswith("ref") or o.name.startswith("refj") or o.name.startswith("refbone"): o.hide_render = True
rig = bpy.data.objects["Relic_AI_Rig"]
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 60; cam.data.clip_end = 100
sc.camera = cam
a = sys.argv.index("--") + 1
tag = sys.argv[a]; frames = [int(x) for x in sys.argv[a+1:]]
for f in frames:
    sc.frame_set(f)
    p = rig.matrix_world @ rig.pose.bones["chest"].head
    cam.location = (p.x + 3.6, p.y, p.z + 0.1)
    cam.rotation_euler = (Vector((p.x, p.y, p.z - 0.05)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = r"C:\ai\unedited\renders\as_%s_%04d.png" % (tag, f)
    bpy.ops.render.render(write_still=True)
