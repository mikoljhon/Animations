# bend Relic's elbow by hand about the rig's hinge axis: +90 (what the rig calls flexion) and -40 (the other way), and look from the side
import bpy, math, importlib, sys
from mathutils import Vector, Quaternion
limits = importlib.import_module("bl_ext.user_default.motion_assist.limits")
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 600, 800
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
rig.animation_data_clear()
for c in [c for pb in rig.pose.bones for c in pb.constraints]:
    c.mute = True
for pb in rig.pose.bones:
    pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
for o in bpy.data.objects:
    if "Crust" in o.name: o.hide_render = True
lims = limits.rig_limits(rig)
h = lims["forearm.L"].hinge
print("HINGE forearm.L", tuple(round(x, 3) for x in h), "range", lims["forearm.L"].ranges)
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 50; cam.data.clip_end = 100
sc.camera = cam
tag = sys.argv[sys.argv.index("--") + 1]
for ang in (90, -40):
    rig.pose.bones["forearm.L"].rotation_quaternion = Quaternion(h, math.radians(ang))
    bpy.context.view_layer.update()
    el = rig.matrix_world @ rig.pose.bones["forearm.L"].head
    # Relic faces +Y in the scene, his left arm is on -X: look from his left side (-X) at the elbow
    for nm, loc in (("side", el + Vector((-2.4, 0.0, 0.0))), ("front", el + Vector((-0.6, 2.4, 0.0)))):
        cam.location = loc
        cam.rotation_euler = (el - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\canon_%s_%s_%+d.png" % (tag, nm, ang)
        bpy.ops.render.render(write_still=True)
    wr = rig.matrix_world @ rig.pose.bones["forearm.L"].tail
    print("ANG", ang, "wrist relative to elbow (world): x %.2f y %.2f z %.2f   (+y = in front of Relic)" % tuple(wr - el))
