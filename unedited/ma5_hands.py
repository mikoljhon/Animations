import bpy, math, sys
from mathutils import Vector
sc = bpy.data.scenes["Cave"]; rig = bpy.data.objects["MenderC_Rig"]
cam_d = bpy.data.cameras.new("tmpcam"); cam = bpy.data.objects.new("tmpcam", cam_d); sc.collection.objects.link(cam)
cam_d.lens = 55; cam_d.clip_start = 0.02
sc.render.resolution_x, sc.render.resolution_y = 480, 360
sc.eevee.taa_render_samples = 10
sc.timeline_markers.clear()
sc.camera = cam
fxy = Vector((-0.8275, 0.5615, 0.0)); side = Vector((-0.5615, -0.8275, 0.0))
for f in (35, 37, 39, 40, 41, 42, 43, 45):
    sc.frame_set(f)
    h = rig.matrix_world @ rig.pose.bones["hand.R"].head
    cam.location = h + side * 2.0 + Vector((0, 0, 0.2))
    cam.rotation_euler = (h - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = r"C:\ai\unedited\renders\c6h_%03d.png" % f
    bpy.ops.render.render(write_still=True, scene="Cave")
