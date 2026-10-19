# fm_test.py : bake every move of fight_moves.py on Relic (or the KillDroid) and render side-view contact sheets.
#   blender.exe -b fight_v01.blend --python fm_test.py -- [relic|kd]
import bpy, math, sys, importlib
from mathutils import Vector, Matrix

OUT = r"C:\ai\unedited"
sc = bpy.data.scenes["Street"]
who = (sys.argv[sys.argv.index("--") + 1:] or ["relic"])[0]
rig = bpy.data.objects["Relic_AI_Rig" if who == "relic" else "KillDroid_Rig"]
other = bpy.data.objects["KillDroid_Rig" if who == "relic" else "Relic_AI_Rig"]
other.location = (30.0, 30.0, 0.0)                    # park the other fighter out of view
rig.animation_data_clear()
FWD = {"Relic_AI_Rig": Vector((0.0, 1.0, 0.0)), "KillDroid_Rig": Vector((0.0, -1.0, 0.0))}
exec(open(OUT + r"\kd_ctrl.py", encoding="utf-8").read())
exec(open(OUT + r"\msflow_retarget.py", encoding="utf-8").read())


def to_world(rg, base, yaw, p):
    f = Matrix.Rotation(yaw, 3, 'Z') @ FWD[rg.name]
    left = Matrix.Rotation(math.pi / 2, 3, 'Z') @ f
    return Vector((base[0], base[1], 0.0)) + left * p[0] + f * p[1] + Vector((0, 0, p[2]))


exec(open(OUT + "/relic_fists.py", encoding="utf-8").read())
if who == "relic":
    print("FISTS", add_fists(rig))
exec(open(OUT + r"\fight_moves.py", encoding="utf-8").read())
kd_add_arm_ik(rig)
body = Body(rig)
print("BODY", who, round(body.kl, 2), round(body.ka, 2), round(body.hip0, 2), round(body.ank0, 2))
base0 = (0.0, 2.0)
yawT = 0.0
dist = 1.0 * body.kl + 0.6 * body.ka                  # where the opponent's face is: just inside the reach of a cross
head_t = (0.0, base0[1] + (1.05 if who == "kd" else 0.9) * body.kl * 1.2, body.hip0 * 1.62)
if who == "relic":
    head_t = (0.0, 2.0 + 1.25, 1.7)
else:
    head_t = (0.0, 2.0 + 1.9, 2.45)
names = ["cross", "jab", "hook", "teep", "roundhouse", "slip", "duck", "angle_step", "back_step", "parry", "idle"]
seq = []
f = 4
spans = {}
for nm in names:
    spec = MOVES[nm]()
    seq.append((f, spec))
    spans[nm] = (f, spec["dur"], spec.get("hit", spec["dur"] // 2))
    f += spec["dur"] + 8
F_END = f
# the fighter always returns to the stance between moves: pad compile with idle holds
frames = compile_moves(body, base0, yawT, seq, target=head_t)
first = min(frames)
# before the first move and in gaps hold the stance: fill by repeating nearest frame params
allf = sorted(frames)
for fr in range(1, F_END + 1):
    if fr not in frames:
        prev = max([x for x in allf if x < fr], default=None)
        nxt = min([x for x in allf if x > fr], default=None)
        src = prev if prev is not None else nxt
        frames[fr] = frames[src]
# note: after a move that shifts the base the next move starts at the new base automatically (compile_moves re-bases)
Pl = [frames[x] for x in (spans["cross"][0] + 8, spans["roundhouse"][0] + 13, spans["duck"][0] + 9, spans["angle_step"][0] + 9)]
for P in Pl:
    P["look"] = Vector(head_t)
print("POLES", kd_calibrate(rig, Pl))
for fr in range(1, F_END + 1):
    P = dict(frames[fr])
    P["look"] = Vector(head_t)
    kd_apply(rig, fr, P)
sc.frame_start, sc.frame_end = 1, F_END
sc.render.resolution_x, sc.render.resolution_y = 520, 520
sc.eevee.taa_render_samples = 6
# a marker sphere where the face is
tg = bpy.data.objects.new("TargetDot", bpy.data.meshes.new("TargetDot"))
import bmesh
bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.1); bm.to_mesh(tg.data); bm.free()
sc.collection.objects.link(tg); tg.location = head_t
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG")); sc.collection.objects.link(dbg)
dbg.data.lens = 50 if who == "relic" else 38
dbg.data.clip_end = 1500.0
sc.camera = dbg
mid = Vector((0.0, 2.6, 1.3 if who == "relic" else 1.6))
dbg.location = (6.0 if who == "relic" else 8.0, 2.6, 1.3 if who == "relic" else 1.6)
dbg.rotation_euler = (mid - dbg.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\fm_test_%s.blend" % who)
import json
json.dump({k: list(v) for k, v in spans.items()}, open(OUT + r"\anim_cache5\fm_spans_%s.json" % who, "w"))
print("FM TEST BAKED", F_END)
