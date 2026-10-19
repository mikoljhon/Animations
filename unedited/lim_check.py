import bpy, importlib, json
ap = importlib.import_module("bl_ext.user_default.motion_assist.autopose")
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
sp = json.load(open(r"C:\ai\unedited\anim_cache5\fm_spans_relic.json"))
res = ap.check_animation(sc, rig, 1, 303, tolerance=3.0)
for b, (n, w, wf) in sorted(res.items(), key=lambda kv: -kv[1][1]):
    mv = [k for k,(s,d,h) in sp.items() if s-1 <= wf <= s+d+1]
    print("LIM", b, "frames over", n, "worst deg", round(w), "at", wf, mv)
