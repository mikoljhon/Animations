# quick test of AutoPose on Relic and the KillDroid in the fight scene.  blender.exe -b fight_v01.blend --python fa_test.py
import bpy, math, importlib
from mathutils import Vector, Matrix

MA = "bl_ext.user_default.motion_assist"
ap = importlib.import_module(MA + ".autopose")
sc = bpy.data.scenes["Street"]
rr = bpy.data.objects["Relic_AI_Rig"]
kr = bpy.data.objects["KillDroid_Rig"]
cam = bpy.data.objects["POV"]
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.eevee.taa_render_samples = 16

rr.location = (0.0, 3.0, 0.0)
kr.location = (0.0, 4.9, 0.0)
bpy.context.view_layer.update()
print("RIG HEADS", {n: tuple(round(x, 2) for x in (rr.matrix_world @ rr.pose.bones[n].head)) for n in ("pelvis", "head", "hand.L", "hand.R", "foot.L")})
print("KD HEADS", {n: tuple(round(x, 2) for x in (kr.matrix_world @ kr.pose.bones[n].head)) for n in ("pelvis", "head", "hand.L", "hand.R", "foot.L")})

# Relic: fists up in front of him (his left is -X because he faces +Y)
ap.solve(rr, {"wrist_L": Vector((-0.30, 3.60, 1.35)), "wrist_R": Vector((0.30, 3.62, 1.40))}, keep=("hips", "neck", "ankle_L", "ankle_R"))
# the KillDroid: claw arm (his right, -X side) thrust forward at head height, fist (his left) cocked back
ap.solve(kr, {"wrist_R": Vector((-0.55, 3.55, 2.05)), "wrist_L": Vector((0.65, 5.2, 1.45))}, keep=("hips", "neck", "ankle_L", "ankle_R"))
bpy.context.view_layer.update()


def shoot(tag, loc=None, tgt=None, lens=None, camera=None):
    c = camera or cam
    if loc is not None:
        c = bpy.data.objects.get("DBG") or bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG"))
        if c.name not in sc.collection.objects:
            sc.collection.objects.link(c)
        c.location = loc
        c.data.lens = lens or 30
        c.data.clip_end = 1500.0
        c.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = c
    sc.render.filepath = r"C:\ai\unedited\renders\fa_test_%s.png" % tag
    bpy.ops.render.render(write_still=True)


# re-bake the head camera from the current pose
pb = rr.pose.bones["head"]
R = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_4x4()
P = Matrix.Translation(Vector((0.0, -0.274, 1.862)))
delta = pb.matrix @ pb.bone.matrix_local.inverted()
M = rr.matrix_world @ delta @ P @ R
loc, q, _s = M.decompose()
cam.location, cam.rotation_quaternion = loc, q
cam.animation_data_clear()
shoot("fp")
shoot("side", loc=(-7.0, 4.0, 2.4), tgt=(0.0, 4.0, 1.7), lens=32)
shoot("front", loc=(5.0, 1.0, 2.2), tgt=(0.0, 4.0, 1.7), lens=32)
print("DONE")
