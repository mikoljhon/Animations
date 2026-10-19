import bpy, sys, json
from mathutils import Vector
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 700, 700
sc.eevee.taa_render_samples = 8
spans = json.load(open(r"C:\ai\unedited\anim_cache5\fm_spans_relic.json"))
for o in bpy.data.objects:
    if "Crust" in o.name: o.hide_render = True
cam = bpy.data.objects.new("CL", bpy.data.cameras.new("CL")); sc.collection.objects.link(cam)
cam.data.lens = 70; cam.data.clip_end = 100
sc.camera = cam
c0, c1 = spans["cross"][0], spans["cross"][0] + spans["cross"][2]
jb = spans["jab"][0] + spans["jab"][2]
hk = spans["hook"][0] + spans["hook"][2]
shots = [("guard", 3), ("cross", c1), ("jab", jb), ("hook", hk)]
for nm, f in shots:
    sc.frame_set(f)
    cam.location = (-2.6, 5.4, 1.7)
    cam.rotation_euler = (Vector((0.0, 2.3, 1.45)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = r"C:\ai\unedited\renders\cl3_%s.png" % nm
    bpy.ops.render.render(write_still=True)
