# pose_test.py : MODEL STRESS TEST. Puts Relic in 8 extreme poses and renders each from 4 sides with all scenery hidden, so a rip or a
#   see-through hole shows up against the bright sky. Run it after ANY change to Relic's mesh, weights or rig.
#   blender.exe -b <file.blend> --python pose_test.py -- <tag>    -> renders\posetest_<tag>.png (8 rows x 4 views)
#   Poses are plain joint rotations in armature space (Relic faces -Y there, his left is +X, up is +Z).
import bpy, sys, math, os
from mathutils import Vector, Matrix, Quaternion
OUT = r"C:\ai\unedited"
exec(open(OUT + r"\limb_solver.py", encoding="utf-8").read())          # write_rotation
tag = sys.argv[sys.argv.index("--") + 1]
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 360, 460
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
rig.animation_data_clear()
keep = {o.name for o in bpy.data.objects if o == rig or o.parent == rig or
        (o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers))}
for o in bpy.data.objects:
    if o.type in ('MESH', 'CURVE', 'FONT') and o.name not in keep:
        o.hide_render = True
    if "Crust" in o.name or o.name.startswith("ref"):
        o.hide_render = True
PB, B = rig.pose.bones, rig.data.bones
for pb_ in PB:
    for c_ in pb_.constraints:
        c_.mute = True                       # plain FK, like the fight pipeline (the base file still has leg IK on)
X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))


def reset():
    for b in PB:
        b.rotation_mode = 'QUATERNION'; b.rotation_quaternion = (1, 0, 0, 0); b.location = (0, 0, 0)
    bpy.context.view_layer.update()


def turn(name, axis, deg):
    """rotate bone `name` (and everything below it) by deg about an armature-space axis through its head"""
    bpy.context.view_layer.update()
    R = Matrix.Rotation(math.radians(deg), 3, axis) @ PB[name].matrix.to_3x3().normalized()
    par = B[name].parent
    Rp = PB[par.name].matrix.to_3x3().normalized() if par else Matrix.Identity(3)
    write_rotation(rig, name, R, Rp)
    bpy.context.view_layer.update()


POSES = [   # (name, moves, the joint to look at up close)
    ("arms out to the sides", [("upperarm.L", Y, -80), ("upperarm.R", Y, 80)], "upperarm.L"),
    ("arms straight overhead", [("upperarm.L", X, -160), ("upperarm.R", X, -160)], "upperarm.R"),
    ("guard: elbows fully bent", [("upperarm.L", X, -50), ("upperarm.R", X, -50), ("forearm.L", X, -120), ("forearm.R", X, -120)], "forearm.L"),
    ("punch reach + other arm back", [("upperarm.R", X, -90), ("upperarm.L", X, 45), ("forearm.L", X, -60), ("chest", Z, -25)], "upperarm.R"),
    ("high front kick", [("thigh.R", X, -100)], "thigh.R"),
    ("side kick", [("thigh.L", Y, -70), ("shin.L", X, 40)], "thigh.L"),
    ("deep crouch", [("thigh.L", X, -100), ("thigh.R", X, -100), ("shin.L", X, 110), ("shin.R", X, 110), ("spine", X, -25)], "pelvis"),
    ("twist + lean + head turn", [("spine", Z, 25), ("chest", Z, 20), ("spine", X, -20), ("neck", Z, 30), ("head", Y, 15)], "spine"),
]
cam = bpy.data.objects.new("PT", bpy.data.cameras.new("PT")); sc.collection.objects.link(cam)
cam.data.lens = 60; cam.data.clip_end = 200; sc.camera = cam
odir = OUT + r"\renders\posetest_%s" % tag
os.makedirs(odir, exist_ok=True)
fw = rig.matrix_world.to_3x3() @ Vector((0, -1, 0)); fw.z = 0; fw.normalize()
lf = Vector((-fw.y, fw.x, 0.0))
for i, (nm, moves, focus) in enumerate(POSES):
    reset()
    for bn, ax, deg in moves:
        turn(bn, ax, deg)
    c = rig.matrix_world @ PB['spine'].head
    for vn, d in (("front", fw), ("left", lf), ("back", -fw), ("right", -lf)):
        cam.location = c + d * 4.6 + Vector((0, 0, 0.45))
        cam.rotation_euler = (c + Vector((0, 0, 0.1)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = odir + r"\%d_%s.png" % (i, vn)
        bpy.ops.render.render(write_still=True)
    # close-ups of the joint this pose stresses, from four sides
    fc = rig.matrix_world @ PB[focus].head
    for vn, d in (("cfront", fw + lf * 0.3), ("cleft", lf + fw * 0.3), ("cback", -fw + lf * 0.3), ("cright", -lf + fw * 0.3)):
        cam.location = fc + d.normalized() * 1.5 + Vector((0, 0, 0.15))
        cam.rotation_euler = (fc - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = odir + r"\%d_%s.png" % (i, vn)
        bpy.ops.render.render(write_still=True)
    print("POSE", i, nm)
reset()
print("POSETEST DONE", tag)
