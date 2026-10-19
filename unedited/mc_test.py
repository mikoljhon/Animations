# mc_test.py : play real mocap segments on Relic or the KillDroid.  blender.exe -b fight_v01.blend --python mc_test.py -- relic|kd seg1 seg2 ...
import bpy, math, sys, json
from mathutils import Vector, Matrix
OUT = r"C:\ai\unedited"
sc = bpy.data.scenes["Street"]
a = sys.argv.index("--") + 1
who = sys.argv[a]; segs = sys.argv[a + 1:]
NOGUARD = 'noguard' in segs
segs = [x for x in segs if x != 'noguard']
rig = bpy.data.objects["Relic_AI_Rig" if who == "relic" else "KillDroid_Rig"]
other = bpy.data.objects["KillDroid_Rig" if who == "relic" else "Relic_AI_Rig"]
other.location = (30.0, 30.0, 0.0)
rig.animation_data_clear()
FWD = {"Relic_AI_Rig": Vector((0.0, 1.0, 0.0)), "KillDroid_Rig": Vector((0.0, -1.0, 0.0))}
exec(open(OUT + r"\kd_ctrl.py", encoding="utf-8").read())
exec(open(OUT + "/relic_fists.py", encoding="utf-8").read())
if who == "relic":
    add_fists(rig)
kd_add_arm_ik(rig)
exec(open(OUT + "/mocap_keys.py", encoding="utf-8").read())
exec(open(OUT + "/rig_guard.py", encoding="utf-8").read())
fwd = FWD[rig.name]; left = Matrix.Rotation(math.pi / 2, 3, 'Z') @ fwd
base0 = (0.0, 2.0)
frames = {}; spans = {}
REF = {}
KREF = (rig.data.bones['thigh.L'].length + rig.data.bones['shin.L'].length) / 0.80
f = 4
last = None
for nm in segs:
    seg = load_seg(OUT + r"\mocap_ref\segs\%s.npz" % nm)
    fr = mocap_frames(rig, seg, f, base0, 0.0, (fwd.x, fwd.y), (left.x, left.y), fist=(1.0, 1.0) if who == "relic" else None)
    if last is not None:
        first = fr[f]
        for k in range(1, 9):
            frames[f - 9 + k] = blend_params(last, first, k / 9.0)
    frames.update(fr)
    REF.update(mocap_world_joints(seg, f, (base0[0], base0[1] - 1.6), (fwd.x, fwd.y), (left.x, left.y), k=KREF))
    n = max(fr) - f
    spans[nm] = (f, n, n // 2)
    last = fr[max(fr)]
    f = max(fr) + 10
F_END = f
head_t = Vector((0.0, 3.25, 1.7)) if who == "relic" else Vector((0.0, 3.9, 2.45))
keys_sorted = sorted(frames)
filled = {}
for fr in range(1, F_END + 1):
    if fr in frames:
        filled[fr] = frames[fr]
    else:
        near = min(keys_sorted, key=lambda k: abs(k - fr))
        filled[fr] = frames[near]
frames = filled
for _P in frames.values():
    pass
_stabilize(frames, rig, 1.0, 1.0)             # one pass over the whole sequence: bend directions stay continuous across the blends
plist = [frames[k] for k in sorted(frames)[::max(1, len(frames) // 8)]]
for P in plist: P["look"] = head_t
print("POLES", kd_calibrate(rig, plist))
guard_setup(rig)
DEBUG['on'] = True
DEBUG['watch'] = ('foot.R', set(range(4, 12)))

set_meshes_hidden(True)
for fr in range(1, F_END + 1):
    P = dict(frames[fr]); P["look"] = head_t
    if NOGUARD:
        kd_apply(rig, fr, P); w_ = 0
    else:
        w_ = apply_safe(rig, fr, P)
    if w_ > 25:
        print("BIGCLAMP frame", fr, [k for k, (a_, n_, h_) in spans.items() if a_ - 1 <= fr <= a_ + n_ + 1] or ["gap"], round(w_))
set_meshes_hidden(False)
print(guard_report(rig))
sc.frame_start, sc.frame_end = 1, F_END
# the reference figure: the raw capture drawn as a plain mannequin beside Relic (joints = spheres, bones = stretched cylinders)
import bmesh
def _mat(name, col):
    m = bpy.data.materials.new(name); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = col
    return m
MATS = {"L": _mat("refL", (0.15, 0.35, 0.95, 1)), "R": _mat("refR", (0.95, 0.2, 0.15, 1)), "C": _mat("refC", (0.9, 0.9, 0.9, 1))}
BONES = [("hips","spine","C"),("spine","chest","C"),("chest","neck","C"),("neck","head","C"),("neck","shoulder_L","L"),("shoulder_L","elbow_L","L"),("elbow_L","wrist_L","L"),
         ("neck","shoulder_R","R"),("shoulder_R","elbow_R","R"),("elbow_R","wrist_R","R"),("hips","hip_L","L"),("hip_L","knee_L","L"),("knee_L","ankle_L","L"),("ankle_L","toe_L","L"),
         ("hips","hip_R","R"),("hip_R","knee_R","R"),("knee_R","ankle_R","R"),("ankle_R","toe_R","R")]
refc = bpy.data.collections.new("RefFigure"); sc.collection.children.link(refc)
empt = {}
used = sorted({b for pr in BONES for b in pr[:2]})
for jn in used:
    e = bpy.data.objects.new("ref_" + jn, None); refc.objects.link(e); empt[jn] = e
    e.empty_display_size = 0.04
for fr in sorted(REF):
    for jn in used:
        empt[jn].location = REF[fr][jn]
        empt[jn].keyframe_insert("location", frame=fr)
cyl_me = bpy.data.meshes.new("refcyl"); bm = bmesh.new()
bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.03, radius2=0.03, depth=1.0)
bmesh.ops.translate(bm, vec=(0, 0, 0.5), verts=bm.verts)
bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(-math.pi / 2, 3, 'X'), verts=bm.verts)   # along +Y
bm.to_mesh(cyl_me); bm.free()
for a_, b_, side in BONES:
    o = bpy.data.objects.new("refbone_%s_%s" % (a_, b_), cyl_me); refc.objects.link(o)
    o.data.materials.append(MATS[side])
    c1 = o.constraints.new('COPY_LOCATION'); c1.target = empt[a_]
    c2 = o.constraints.new('STRETCH_TO'); c2.target = empt[b_]; c2.rest_length = 1.0; c2.volume = 'NO_VOLUME'
    o.scale = (1, 1, 1)
for jn in used:
    sp_ = bpy.data.objects.new("refj_" + jn, bpy.data.meshes.new("refj"))
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.05 if jn in ("head",) else 0.04); bm.to_mesh(sp_.data); bm.free()
    refc.objects.link(sp_); sp_.data.materials.append(MATS["C"])
    cc = sp_.constraints.new('COPY_LOCATION'); cc.target = empt[jn]
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG")); sc.collection.objects.link(dbg)
dbg.data.lens = 38 if who == "relic" else 30; dbg.data.clip_end = 1500.0
sc.camera = dbg
mid = Vector((0.0, 1.8, 1.3 if who == "relic" else 1.6))
dbg.location = (6.5 if who == "relic" else 8.0, 1.8, 1.3 if who == "relic" else 1.6)
dbg.rotation_euler = (mid - dbg.location).to_track_quat('-Z', 'Y').to_euler()
for _o in bpy.data.objects:
    if "Crust" in _o.name: _o.hide_render = True
TAG = who + "_" + segs[0] + ("_ng" if NOGUARD else "")
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\mc_test_%s.blend" % TAG)
json.dump({k: list(v) for k, v in spans.items()}, open(OUT + r"\anim_cache5\mc_spans_%s.json" % TAG, "w"))
print("MC BAKED", F_END, spans)
