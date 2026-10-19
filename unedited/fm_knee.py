import bpy, sys
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 800, 800
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 75; cam.data.clip_end = 100
sc.camera = cam
a = sys.argv.index("--") + 1
tag = sys.argv[a]; frames = [int(x) for x in sys.argv[a+1:]]
rig.show_in_front = True
rig.data.display_type = 'STICK'
for f in frames:
    sc.frame_set(f)
    p = rig.matrix_world @ rig.pose.bones["pelvis"].head
    cam.location = (p.x + 4.2, p.y + 0.3, 0.9)
    cam.rotation_euler = (Vector((p.x, p.y + 0.3, 0.8)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    for show in (False, True):
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.name.startswith("Relic_AI") and "Crust" not in o.name:
                o.hide_render = False
        sc.render.filepath = r"C:\ai\unedited\renders\knee_%s_%04d.png" % (tag, f)
        bpy.ops.render.render(write_still=True)
        break
