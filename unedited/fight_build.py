# fight_build.py : assembles the fight scene of episode 1: the street (street_v01.blend), the KillDroid (killdroid_v02.blend, the "Breaker") and Relic
# (from the cave file), with a first-person camera on Relic's head. Run:
#   blender.exe -b street_v01.blend --python fight_build.py -- fight_vNN.blend        full build from the street -> fight_vNN.blend + test renders
#   blender.exe -b fight_v07.blend --python fight_build.py -- swap fight_v08.blend    the current base with its KillDroid replaced by KD (Relic, the
#                                                                                    street and everything else untouched)
# Fight bases are versions: an existing file is never overwritten (fight_v07.blend and earlier stay as they are).
import bpy, math, os, sys
import numpy as np
from mathutils import Vector, Matrix, Euler

OUT = r"C:\ai\unedited"
CAVE = OUT + r"\relic_v15_opening_cut6_pass.blend"
KD = OUT + r"\killdroid_v02.blend"          # v02 "Breaker" (2026-10-14); same collection / rig / body names as v01, 28 bones (jaws claw1a.R, claw2a.R)
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sc = bpy.data.scenes["Street"]
bpy.context.window.scene = sc if bpy.context.window else None


def append_collections(path, names):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = [n for n in names if n in src.collections]
    got = []
    for c in dst.collections:
        if c is not None:
            sc.collection.children.link(c)
            got.append(c)
    return got


def save_new(path):
    """save a NEW fight base; refuse to overwrite an existing one"""
    if os.path.exists(path):
        print("REFUSED: %s already exists, pick the next version name" % path)
        sys.exit(2)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print("SAVED", path)


def rest_pose(rig_):
    """a clean standing rest pose: no animation, every pose bone at rest"""
    if rig_.animation_data:
        rig_.animation_data_clear()
    for pb_ in rig_.pose.bones:
        pb_.rotation_mode = 'QUATERNION'
        pb_.location = (0.0, 0.0, 0.0)
        pb_.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb_.scale = (1.0, 1.0, 1.0)


def lamp_check(rig_):
    """the KillDroid's lamps follow rig["lamp_power"]: evaluated spot energy at power 1 and 0"""
    if "lamp_power" not in rig_.keys():
        return None
    lamps = [o for o in bpy.data.objects if o.type == 'LIGHT' and o.name.startswith("KillDroid_Lamp")]
    res = []
    for p in (1.0, 0.0):
        rig_["lamp_power"] = p
        rig_.update_tag()
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        res.append(round(sum(o.evaluated_get(dg).data.energy for o in lamps), 1))
    rig_["lamp_power"] = 1.0
    rig_.update_tag()
    bpy.context.view_layer.update()
    return dict(lamps=len(lamps), energy_at_1=res[0], energy_at_0=res[1])


_dst = ARGS[1] if ARGS[:1] == ["swap"] else (ARGS[0] if ARGS else "fight_v01.blend")
if os.path.exists(OUT + "\\" + _dst):
    print("REFUSED: %s already exists, pick the next version name" % _dst)
    sys.exit(2)

if ARGS[:1] == ["swap"]:
    # replace the old KillDroid (v01: 119 meshes + rig, materials KD_*) with KD; names must come in without ".001" suffixes
    old = bpy.data.collections.get("KillDroid")
    datas, mats = [], set()
    if old is not None:
        for o in list(old.all_objects):
            if o.data is not None:
                datas.append(o.data)
            if o.type == 'MESH':
                mats.update(m for m in o.data.materials if m is not None)
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    for dblk in datas:
        if dblk.users == 0:
            if isinstance(dblk, bpy.types.Mesh):
                bpy.data.meshes.remove(dblk)
            elif isinstance(dblk, bpy.types.Armature):
                bpy.data.armatures.remove(dblk)
            elif isinstance(dblk, bpy.types.Light):
                bpy.data.lights.remove(dblk)
    for m in mats:
        if m.users == 0:
            bpy.data.materials.remove(m)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    append_collections(KD, ["KillDroid"])
    kr = bpy.data.objects["KillDroid_Rig"]
    kr.location = (0.0, 22.0, 0.0)          # under the bridge, facing -Y toward Relic (as in the full build)
    kr.rotation_euler = (0.0, 0.0, 0.0)
    rest_pose(kr)
    bpy.context.view_layer.update()
    clash = sorted(d.name for d in list(bpy.data.objects) + list(bpy.data.materials) + list(bpy.data.armatures) + list(bpy.data.meshes)
                   if d.name.startswith(("KillDroid", "KD_")) and d.name[-4:-3] == "." and d.name[-3:].isdigit())
    print("SWAP KillDroid from", KD, ":", len(bpy.data.collections["KillDroid"].all_objects), "objects,", len(kr.data.bones), "bones,",
          "renamed on append:", clash or "none", "| lamps:", lamp_check(kr))
    # a still of the new KillDroid in the street (third person, debug camera)
    dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG"))
    sc.collection.objects.link(dbg)
    dbg.data.lens = 30
    dbg.data.clip_end = 1500.0
    dbg.location = (-6.5, 14.0, 2.4)
    dbg.rotation_euler = (Vector((0.0, 22.0, 1.5)) - dbg.location).to_track_quat('-Z', 'Y').to_euler()
    cam0 = sc.camera
    sc.camera = dbg
    stem = os.path.splitext(os.path.basename(ARGS[1]))[0]
    sc.render.filepath = OUT + r"\renders\%s_kd.png" % stem
    bpy.ops.render.render(write_still=True)
    sc.camera = cam0
    bpy.data.objects.remove(dbg, do_unlink=True)
    save_new(OUT + "\\" + ARGS[1])
    sys.exit(0)

NEW = ARGS[0] if ARGS else "fight_v01.blend"
STEM = os.path.splitext(NEW)[0]
relic = append_collections(CAVE, ["Relic_AI"])
kdc = append_collections(KD, ["KillDroid"])
rr = bpy.data.objects["Relic_AI_Rig"]
kr = bpy.data.objects["KillDroid_Rig"]
# Relic: upright, at the crack lip, facing +Y (the design frame faces -Y, so turn him half a circle)
rr.rotation_euler = (0.0, 0.0, math.pi)
rr.location = (0.0, 1.5, 0.0)
# the KillDroid: under the bridge, facing -Y toward Relic
kr.location = (0.0, 22.0, 0.0)
# the crust (mud) and ring stay as in the cave; the eyes object is a glyph surface, leave it
for o in bpy.data.objects:
    if o.name.startswith("Relic_AI") and o.type == 'MESH':
        o.hide_render = False

# a clean standing rest pose: drop the cave animation and the lying pose
for rig_ in (rr, kr):
    rest_pose(rig_)
bpy.context.view_layer.update()


# ---------------------------------------------------------------- the first-person camera: baked from Relic's head bone
def build_pov(frame_range=(1, 1)):
    cd = bpy.data.cameras.get("POV") or bpy.data.cameras.new("POV")
    cd.lens = 22.0
    cd.clip_start = 0.005
    cd.clip_end = 1500.0
    cam = bpy.data.objects.get("POV") or bpy.data.objects.new("POV", cd)
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    cam.rotation_mode = 'QUATERNION'
    pb = rr.pose.bones["head"]
    R = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_4x4()
    P = Matrix.Translation(Vector((0.0, -0.274, 1.862)))
    for f in range(frame_range[0], frame_range[1] + 1):
        sc.frame_set(f)
        delta = pb.matrix @ pb.bone.matrix_local.inverted()
        M = rr.matrix_world @ delta @ P @ R
        loc, q, _s = M.decompose()
        cam.location, cam.rotation_quaternion = loc, q
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
    sc.camera = cam
    return cam


cam = build_pov((1, 1))
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.eevee.taa_render_samples = 24

# test renders: the first-person view, and a debug third-person view of both
sc.render.filepath = OUT + r"\renders\%s_fp.png" % STEM
bpy.ops.render.render(write_still=True)
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG"))
sc.collection.objects.link(dbg)
dbg.data.lens = 30
dbg.data.clip_end = 1500.0
dbg.location = (-9.0, -4.0, 3.2)
dbg.rotation_euler = (Vector((0.0, 12.0, 2.5)) - dbg.location).to_track_quat('-Z', 'Y').to_euler()
sc.camera = dbg
sc.render.filepath = OUT + r"\renders\%s_dbg.png" % STEM
bpy.ops.render.render(write_still=True)
dbg.location = (-4.0, 12.0, 1.8)
dbg.data.lens = 28
dbg.rotation_euler = (Vector((0.0, 1.5, 1.2)) - dbg.location).to_track_quat('-Z', 'Y').to_euler()
sc.render.filepath = OUT + r"\renders\%s_relic.png" % STEM
bpy.ops.render.render(write_still=True)
sc.camera = cam
bpy.data.objects.remove(dbg, do_unlink=True)
save_new(OUT + "\\" + NEW)
print("FIGHT BUILT", len(bpy.data.objects), "objects")
