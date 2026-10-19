# Regression run for Motion Assist (v0.4) on last session's jump and AutoPose demo file. Background:
#   blender.exe -b motion_assist_test.blend --python ma_regress.py
# Prints a JSON line with what Check Joints says before and after Fix Joints on every demo action.
import importlib
import json
import math
import sys

import bpy
import addon_utils

MA = "bl_ext.user_default.motion_assist"
addon_utils.enable(MA, default_set=False, persistent=False)
exec(open(r"C:\ai\unedited\ma_demo_jump.py").read())          # assign(), clear_pose()


def mod(n):
    return importlib.import_module(MA + "." + n)


autopose, limits, body, analysis = mod("autopose"), mod("limits"), mod("body"), mod("analysis")
sc = bpy.data.scenes["Mender_Studio"]
bpy.context.window.scene = sc if bpy.context.window else None
rig = bpy.data.objects["Mender_Rig"]
out = {"limits_version_check": {b: [round(v, 1) for v in l.ranges] for b, l in limits.rig_limits(rig).items() if b in ("shin.L", "forearm.L", "foot.L", "hand.L")}}
for name in ("Mender_jump_naive_joints", "Mender_jump_fixed3", "Mender_jump_v4", "Mender_ik_sweep", "Mender_ai_sweep", "Mender_autohips_sweep"):
    act = bpy.data.actions.get(name)
    if act is None:
        out[name] = "missing"
        continue
    work = act.copy()
    work.name = "REGRESS_" + name
    assign(rig, work)
    f0, f1 = int(work.frame_range[0]), int(work.frame_range[1])
    sc.frame_start, sc.frame_end = f0, f1
    bad0 = autopose.check_animation(sc, rig, f0, f1)
    moved = autopose.fix_animation(sc, rig, f0, f1)
    bad1 = autopose.check_animation(sc, rig, f0, f1)
    out[name] = {"frames": (f0, f1), "before": {b: [n, round(w, 1)] for b, (n, w, f) in bad0.items()},
                 "after": {b: [n, round(w, 1)] for b, (n, w, f) in bad1.items()}}
    assign(rig, act)
    bpy.data.actions.remove(work)
print("REGRESS_JSON " + json.dumps(out))
