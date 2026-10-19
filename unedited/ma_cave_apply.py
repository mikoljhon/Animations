# Cave opening pass (cut 4), the whole thing from the saved cut 3 original. Exec after ma_cave_pass.py.
# Order matters: overlap on the free parts first, then the release targets (they read the free arms' animation), then Fix Joints.
import json
import time

import importlib

pkg = importlib.import_module(MA)
importlib.reload(pkg)
pkg.dev_reload()


def apply_pass(max_turn=None, name="MenderC_RigAction.003"):
    sc, rig, s = setup()
    grip, autopose, overlap, analysis, bodym = mod("grip"), mod("autopose"), mod("overlap"), mod("analysis"), mod("body")
    if max_turn is not None:
        autopose.ELBOW_TURN = max_turn
    ad = rig.animation_data
    prev = ad.action
    work = bpy.data.actions["MenderC_cut3_original"].copy()
    work.name = "MenderC_work_tmp"
    ad.action, ad.action_slot = work, work.slots[0]
    if prev is not None and prev.name == name:
        bpy.data.actions.remove(prev)
    work.name = name
    t0 = time.time()
    overlap.bake_overlap(sc, rig, ["clavicle.L", "clavicle.R"], 1, 216, freq=6.0, damp=0.7, resp=0.0, strength=0.3, use_limits=True)
    overlap.bake_overlap(sc, rig, ["forearm.L", "forearm.R"], 108, 216, freq=6.0, damp=0.6, resp=0.0, strength=0.5, use_limits=True)
    overlap.bake_overlap(sc, rig, ["hand.L", "hand.R"], 108, 216, freq=7.0, damp=0.55, resp=0.0, strength=0.55, use_limits=True)
    rec, grips, _old = grip.load(r"C:\ai\unedited\anim_cache\grips_cut4.json")
    pick = bpy.data.objects["MenderC_Pick"]
    props = {"L": pick, "R": pick}
    holds = grip.targets(sc, rig, rec, props, grips)
    leans = autopose.reach_lean(sc, rig, 1, 216, holds, max_lean=math.radians(22.0))
    grip.save(r"C:\ai\unedited\anim_cache\grips_cut4b.json", rec, grips, holds)
    json.dump({str(f): v for f, v in leans.items()}, open(r"C:\ai\unedited\anim_cache\leans_cut4b.json", "w"))
    autopose.fix_animation(sc, rig, 1, 216, holds=holds, leans=leans)
    sc.frame_set(1)
    data = analysis.analyze(sc, rig, s)
    legs = [ch for ch in data["info"]["chains"] if "foot" in ch["target"].lower()]
    body = bodym.get_body(rig)
    probe_bones = np.array(body.names)[body.pbone]
    free = {ch["target"]: ~data["touch"][:, np.isin(probe_bones, ch["below"])].any(axis=1) for ch in legs}
    lf = overlap.bake_leg_follow(sc, rig, legs, data["f0"], data["f1"], free, strength=1.0)
    autopose.fix_animation(sc, rig, 106, 125, parts=("legs",))
    bad = autopose.check_animation(sc, rig, 1, 216)
    sc.frame_set(1)
    return {"secs": round(time.time() - t0, 1), "leg_follow": lf,
            "remaining": {b: {"frames": n, "worst_deg": round(w, 1)} for b, (n, w, f) in bad.items()}}


def pop_report(tag_new=None):
    """peak change of angular speed (deg/frame^2) per bone for the original, the first attempt and the current action, and the
    arms' speeds around the release"""
    sc, rig, s = setup()
    ad = rig.animation_data
    cur_act, cur_slot = ad.action, ad.action_slot
    P, Mw = rig.pose.bones, rig.matrix_world
    bones = ["upperarm.L", "forearm.L", "hand.L", "upperarm.R", "forearm.R", "hand.R"]
    out, speeds = {}, {}
    try:
        for tag, key in (("MenderC_cut3_original", "orig"), (cur_act.name, "new")):
            a = bpy.data.actions[tag]
            ad.action, ad.action_slot = a, a.slots[0]
            ax = {b: [] for b in bones}
            for f in range(1, 217):
                sc.frame_set(f)
                for b in bones:
                    ax[b].append(np.array((Mw @ P[b].matrix).to_3x3().col[1].normalized()))
            for b in bones:
                A = np.array(ax[b])
                d = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", A[1:], A[:-1]), -1, 1)))
                j = np.abs(d[1:] - d[:-1])
                out.setdefault(b, {})[key] = (round(float(j.max()), 1), int(j.argmax()) + 2)
                speeds.setdefault(b, {})[key] = [round(float(x), 0) for x in d[103:116]]
    finally:
        ad.action, ad.action_slot = cur_act, cur_slot
    sc.frame_set(1)
    return out, speeds
