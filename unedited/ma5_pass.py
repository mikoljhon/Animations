# Motion Assist pass on the cut 5 opening (heavy pick swing, 360 frames). Run in background Blender:
#   blender.exe -b relic_v13_opening_cut5.blend --python ma5_pass.py
# Order (see project notes): grips recorded on the ORIGINAL, follow-through on the free parts, then the hand targets, then Fix Joints,
# then the leg follow-through and a legs re-fix. Writes relic_v14_opening_cut5_pass.blend and anim_cache5\ma5_pass_report.json.
import json
import math
import sys
import time

exec(open(r"C:\ai\unedited\ma_cave_pass.py", encoding="utf-8").read())

T0 = time.time()
LOG = open(r"C:\ai\unedited\anim_cache5\ma5_pass_log.txt", "w")


def log(*a):
    LOG.write("%7.1f  %s\n" % (time.time() - T0, " ".join(str(x) for x in a)))
    LOG.flush()


F1, DROP = 360, 211
autopose_mod = mod('autopose')
autopose_mod.ELBOW_TURN = 32.0
sc, rig, s = setup()
grip, autopose, overlap, analysis, bodym = mod("grip"), mod("autopose"), mod("overlap"), mod("analysis"), mod("body")
ad = rig.animation_data
act = ad.action
keep = act.copy()
keep.name = "MenderC_cut5_original"
keep.use_fake_user = True
log("action", act.name, "backed up as", keep.name)
pick = bpy.data.objects["MenderC_Pick"]
props = {"L": pick, "R": pick}

rec = grip.record(sc, rig, 1, F1, props)
log("recorded", {k: sum(1 for v in r["frames"].values() if v["on"]) for k, r in rec.items()})
holds = grip.targets(sc, rig, rec, props, grip.plain(rec))
leans = autopose.reach_lean(sc, rig, 1, F1, holds, max_lean=math.radians(22.0))
log("first leans", len(leans))
grips = grip.solve_grip(sc, rig, rec, props, leans=leans)
log("solve_grip done")
for k in range(2):
    grips = grip.refine(sc, rig, rec, props, grips, leans=leans)
    grips = grip.refine(sc, rig, rec, props, grips, leans=leans, span=12.0, n=5, frame_step=1)
    log("refine round", k)
    holds = grip.targets(sc, rig, rec, props, grips)
    new = autopose.reach_lean(sc, rig, 1, F1, holds, max_lean=math.radians(22.0))
    change = max((abs(new.get(f, 0.0) - leans.get(f, 0.0)) for f in set(new) | set(leans)), default=0.0)
    leans = new
    if change < math.radians(1.5):
        break
grip.save(r"C:\ai\unedited\anim_cache5\grips_cut5.json", rec, grips, holds)

overlap.bake_overlap(sc, rig, ["clavicle.L", "clavicle.R"], 1, F1, freq=6.0, damp=0.7, resp=0.0, strength=0.3, use_limits=True)
overlap.bake_overlap(sc, rig, ["forearm.L", "forearm.R"], DROP + 2, F1, freq=6.0, damp=0.6, resp=0.0, strength=0.5, use_limits=True)
overlap.bake_overlap(sc, rig, ["hand.L", "hand.R"], DROP + 2, F1, freq=7.0, damp=0.55, resp=0.0, strength=0.55, use_limits=True)
log("overlap baked")
holds = grip.targets(sc, rig, rec, props, grips)
leans = autopose.reach_lean(sc, rig, 1, F1, holds, max_lean=math.radians(22.0))
grip.save(r"C:\ai\unedited\anim_cache5\grips_cut5b.json", rec, grips, holds)
json.dump({str(f): v for f, v in leans.items()}, open(r"C:\ai\unedited\anim_cache5\leans_cut5b.json", "w"))
log("targets", len(holds), "leans", len(leans), "max lean deg", round(math.degrees(max(leans.values(), default=0.0)), 1))
autopose.fix_animation(sc, rig, 1, F1, holds=holds, leans=leans)
log("fix_animation done")
sc.frame_set(1)
data = analysis.analyze(sc, rig, s)
legs = [ch for ch in data["info"]["chains"] if "foot" in ch["target"].lower()]
body = bodym.get_body(rig)
probe_bones = np.array(body.names)[body.pbone]
free = {ch["target"]: ~data["touch"][:, np.isin(probe_bones, ch["below"])].any(axis=1) for ch in legs}
lf = overlap.bake_leg_follow(sc, rig, legs, data["f0"], data["f1"], free, strength=1.0)
log("leg follow", lf)
autopose.fix_animation(sc, rig, DROP, DROP + 30, parts=("legs",))
bad = autopose.check_animation(sc, rig, 1, F1)
remaining = {b: {"frames": n, "worst_deg": round(w, 1)} for b, (n, w, f) in bad.items()}
log("remaining", remaining)

# hands against the targets, and the pick path unchanged
errs = {"L": [], "R": []}
for f in sorted(holds):
    if f % 2:
        continue
    sc.frame_set(f)
    for sd in "LR":
        if sd in holds[f]:
            w = rig.matrix_world @ rig.pose.bones["hand." + sd].head
            errs[sd].append((round((w - holds[f][sd][0]).length, 4), f))
rep = {"remaining": remaining, "hand_err_cm": {sd: [round(max(e)[0] * 100, 2), max(e)[1]] if e else None for sd, e in errs.items()}, "leg_follow": str(lf)}
json.dump(rep, open(r"C:\ai\unedited\anim_cache5\ma5_pass_report.json", "w"))
sc.frame_set(1)
sc.render.filepath = r"C:\ai\unedited\renders\anim_open6\f_"
bpy.ops.wm.save_as_mainfile(filepath=r"C:\ai\unedited\relic_v15_opening_cut6_pass.blend")
log("saved", rep)
LOG.close()
