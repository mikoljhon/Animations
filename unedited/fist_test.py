# fist_test.py : close-up renders of both of Relic's hands with different fist curls, to check the fist.
#   blender.exe -b fight_v06.blend --python fist_test.py -- c1,c2 c1,c2 ...      (radians for fist1 / fist2)
#   -> renders\fist_<i>_<L|R>_<view>.png
import bpy, sys, math
from mathutils import Vector, Quaternion
OUT = r"C:\ai\unedited"
exec(open(OUT + r"\relic_fists.py", encoding="utf-8").read())
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 420, 420
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
rig.animation_data_clear()
for o in bpy.data.objects:
    if "Crust" in o.name or o.name.startswith("ref"):
        o.hide_render = True
add_fists(rig)
pb = rig.pose.bones
for b in pb:
    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = (1, 0, 0, 0)
bpy.context.view_layer.update()
cam = bpy.data.objects.new("FT", bpy.data.cameras.new("FT")); sc.collection.objects.link(cam)
cam.data.lens = 85; cam.data.clip_end = 100
sc.camera = cam
a = sys.argv.index("--") + 1
curls = [tuple(float(x) for x in s.split(",")) for s in sys.argv[a:]]
fw = rig.matrix_world.to_3x3() @ Vector((0, -1, 0)); fw.z = 0; fw.normalize()
up = Vector((0, 0, 1))
for i, (c1, c2) in enumerate(curls):
    set_fists(rig, (1.0, 1.0), c1=c1, c2=c2)
    bpy.context.view_layer.update()
    for hs in ('R', 'L'):
        hm = rig.matrix_world @ pb['hand.' + hs].matrix
        c = hm @ Vector((0, 0.09, 0))                  # middle of the hand in world
        out = Vector((fw.y, -fw.x, 0.0)) * (1.0 if hs == 'R' else -1.0)    # the outside of this hand
        for nm, d in (("side", out * 0.62 + fw * 0.05), ("front", fw * 0.62 + out * 0.28), ("back", -fw * 0.62 + out * 0.28), ("below", out * 0.4 - up * 0.45 + fw * 0.15)):
            cam.location = c + d
            cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
            sc.render.filepath = OUT + r"\renders\fist_%d_%s_%s.png" % (i, hs, nm)
            bpy.ops.render.render(write_still=True)
    print("FIST", i, c1, c2)
print("FIST DONE")
