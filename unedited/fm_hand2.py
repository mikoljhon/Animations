import bpy, math
from mathutils import Vector, Quaternion
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 600, 600
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
for o in bpy.data.objects:
    if "Crust" in o.name: o.hide_render = True
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 90; cam.data.clip_end = 100
sc.camera = cam
pb = rig.pose.bones["hand.L"]; pb.rotation_mode = 'QUATERNION'
i = 0
for ax in ("X", "Z"):
    for ang in (-0.9, 0.9):
        pb.rotation_quaternion = Quaternion((1 if ax == "X" else 0, 0, 1 if ax == "Z" else 0)[0:3] if False else {"X": (1,0,0), "Z": (0,0,1)}[ax], ang)
        bpy.context.view_layer.update()
        cam.location = (-3.6, 2.4, 1.0)
        cam.rotation_euler = (Vector((-0.47, 1.6, 0.9)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\hand2_%s_%s.png" % (ax, "neg" if ang < 0 else "pos")
        bpy.ops.render.render(write_still=True)
