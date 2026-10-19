# pipe_test.py : run clips through the fight pipeline on one character, print the QA gate, save a blend with a reference mannequin + cameras.
#   blender.exe -b fight_v03.blend --python pipe_test.py -- relic|kd TAG clip1 clip2 ...
#   PIPE_NOSAVE=1 in the environment: QA only, no .blend written (qa_clips.sh uses this; the disk is nearly full)
import bpy, math, sys, json, bmesh, os
from mathutils import Vector, Matrix
OUT = r"C:\ai\unedited"
exec(open(OUT + r"\fight_pipeline.py", encoding="utf-8").read())
sc = bpy.data.scenes["Street"]
a = sys.argv.index("--") + 1
who, tag, clips = sys.argv[a], sys.argv[a + 1], sys.argv[a + 2:]
rig = bpy.data.objects["Relic_AI_Rig" if who == "relic" else "KillDroid_Rig"]
other = bpy.data.objects["KillDroid_Rig" if who == "relic" else "Relic_AI_Rig"]
other.location = (30.0, 30.0, 0.0)
rig.animation_data_clear()
pipe = FightPipeline(rig, other)
DEBUG['on'] = True
f = 4
head_t = Vector((0.0, 3.25, 1.7)) if who == "relic" else Vector((0.0, 3.9, 2.45))
spans = {}
for nm in clips:
    end = pipe.add_clip(nm, f, base=(0.0, 2.0), yaw=0.0, look=head_t)
    spans[nm] = (f, end - f, (end - f) // 2)
    f = end + 10
pipe.bake()
rep = pipe.qa()
print(pipe.qa_text(rep))
json.dump(rep, open(OUT + r"\anim_cache5\qa_%s.json" % tag, "w"), default=str, indent=1)
json.dump({k: list(v) for k, v in spans.items()}, open(OUT + r"\anim_cache5\spans_%s.json" % tag, "w"))
if os.environ.get("PIPE_NOSAVE"):
    print("PIPE QA ONLY", tag, "QA", "PASS" if rep['pass'] else "FAIL")
    sys.exit(0)

# reference mannequin (the raw real capture, placed 1.6 m to the side) and cameras for the comparison reel
def _mat(name, col):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = col
    return m
MATS = {"L": _mat("refL", (0.15, 0.35, 0.95, 1)), "R": _mat("refR", (0.95, 0.2, 0.15, 1)), "C": _mat("refC", (0.9, 0.9, 0.9, 1))}
BONES = [("hips", "spine", "C"), ("spine", "chest", "C"), ("chest", "neck", "C"), ("neck", "head", "C"), ("neck", "shoulder_L", "L"), ("shoulder_L", "elbow_L", "L"),
         ("elbow_L", "wrist_L", "L"), ("neck", "shoulder_R", "R"), ("shoulder_R", "elbow_R", "R"), ("elbow_R", "wrist_R", "R"), ("hips", "hip_L", "L"),
         ("hip_L", "knee_L", "L"), ("knee_L", "ankle_L", "L"), ("ankle_L", "toe_L", "L"), ("hips", "hip_R", "R"), ("hip_R", "knee_R", "R"),
         ("knee_R", "ankle_R", "R"), ("ankle_R", "toe_R", "R")]
refc = bpy.data.collections.new("RefFigure"); sc.collection.children.link(refc)
used = sorted({b for pr in BONES for b in pr[:2]})
empt = {}
off = Vector((0.0, -1.6, 0.0))
for jn in used:
    e = bpy.data.objects.new("ref_" + jn, None); refc.objects.link(e); empt[jn] = e
for fr in sorted(pipe.real):
    for jn in used:
        empt[jn].location = pipe.real[fr][jn] + off
        empt[jn].keyframe_insert("location", frame=fr)
cyl = bpy.data.meshes.new("refcyl"); bm = bmesh.new()
bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.03, radius2=0.03, depth=1.0)
bmesh.ops.translate(bm, vec=(0, 0, 0.5), verts=bm.verts)
bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(-math.pi / 2, 3, 'X'), verts=bm.verts)
bm.to_mesh(cyl); bm.free()
for a_, b_, side in BONES:
    o = bpy.data.objects.new("refbone_%s_%s" % (a_, b_), cyl); refc.objects.link(o)
    o.data.materials.append(MATS[side])
    c1 = o.constraints.new('COPY_LOCATION'); c1.target = empt[a_]
    c2 = o.constraints.new('STRETCH_TO'); c2.target = empt[b_]; c2.rest_length = 1.0; c2.volume = 'NO_VOLUME'
for jn in used:
    sp_ = bpy.data.objects.new("refj_" + jn, bpy.data.meshes.new("refj"))
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.05 if jn == "head" else 0.04); bm.to_mesh(sp_.data); bm.free()
    refc.objects.link(sp_); sp_.data.materials.append(MATS["C"])
    cc = sp_.constraints.new('COPY_LOCATION'); cc.target = empt[jn]
for o in bpy.data.objects:
    if "Crust" in o.name:
        o.hide_render = True
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG")); sc.collection.objects.link(dbg)
dbg.data.lens = 38 if who == "relic" else 30; dbg.data.clip_end = 1500.0
mid = Vector((0.0, 1.8, 1.3 if who == "relic" else 1.6))
dbg.location = (6.5 if who == "relic" else 8.0, 1.8, mid.z)
dbg.rotation_euler = (mid - dbg.location).to_track_quat('-Z', 'Y').to_euler()
sc.camera = dbg
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\pipe_%s.blend" % tag)
print("PIPE SAVED", tag, "QA", "PASS" if rep['pass'] else "FAIL")
