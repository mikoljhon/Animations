import bpy
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 700, 700
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
for o in bpy.data.objects:
    if "Crust" in o.name: o.hide_render = True
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 80; cam.data.clip_end = 100
sc.camera = cam
# Relic faces +Y at (0,1.5); left hand near (-0.47, 1.6, 1.0)
for nm, loc in (("front", (-0.5, 5.0, 1.05)), ("side", (-4.0, 1.6, 1.05)), ("below", (-0.5, 3.5, 0.2))):
    cam.location = loc
    cam.rotation_euler = (Vector((-0.47, 1.6, 0.95)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = r"C:\ai\unedited\renders\hand_%s.png" % nm
    bpy.ops.render.render(write_still=True)
for n in ("hand.L","forearm.L"):
    pb = rig.pose.bones[n]; b = rig.data.bones[n]
    print("HB", n, [tuple(round(x,2) for x in (b.matrix_local.to_3x3()[:][i])) for i in range(3)])
