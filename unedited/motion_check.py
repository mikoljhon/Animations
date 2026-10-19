# motion_check.py : does the body move like a body?  Legs: per-frame turn spikes, knee bend, IK foot error.  Joints: how far outside the human
# joint limits (Motion Assist) each bone goes.
#   blender.exe -b file.blend --python motion_check.py -- spans.json F_END [RigName]
import sys, json, math, importlib
import bpy
a = sys.argv.index("--") + 1
sp = json.load(open(sys.argv[a]))
NF = int(sys.argv[a + 1])
RIG = sys.argv[a + 2] if len(sys.argv) > a + 2 else "Relic_AI_Rig"
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects[RIG]
ap = importlib.import_module("bl_ext.user_default.motion_assist.autopose")


def seg(f):
    for k, (s, d, h) in sp.items():
        if s - 1 <= f <= s + d + 1:
            return k
    return "gap"


prev = {}
rows = []
agg = {}
W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
for f in range(1, NF + 1):
    sc.frame_set(f)
    m = seg(f)
    agg.setdefault(m, dict(knee=[0, 0], err=[0, 0], spike=0, worst=0))
    for i, s in enumerate("LR"):
        hip, kn, an = W("thigh." + s), W("shin." + s), W("shin." + s, True)
        flex = 180.0 - math.degrees((hip - kn).normalized().angle((an - kn).normalized()))
        err = (W("ik_foot." + s) - an).length
        agg[m]["knee"][i] = max(agg[m]["knee"][i], flex)
        agg[m]["err"][i] = max(agg[m]["err"][i], err)
        q = [(rig.matrix_world @ rig.pose.bones[n].matrix).to_quaternion() for n in ("thigh." + s, "shin." + s, "foot." + s)]
        if s in prev:
            for n, qq, pq in zip(("thigh", "shin", "foot"), q, prev[s]):
                ang = math.degrees(qq.rotation_difference(pq).angle)
                ang = min(ang, 360 - ang)
                if ang > 45:
                    agg[m]["spike"] += 1
                    agg[m]["worst"] = max(agg[m]["worst"], ang)
        prev[s] = q
print("LEGS (turn spikes = more than 45 degrees in one frame)")
for m, v in agg.items():
    print("LEGMAX %-14s knee bend L/R %3d %3d   foot error L/R %.3f %.3f m   leg spikes %d (worst %d deg)" % (m, v["knee"][0], v["knee"][1], v["err"][0], v["err"][1], v["spike"], v["worst"]))
res = ap.check_animation(sc, rig, 1, NF, tolerance=3.0)
print("JOINT LIMITS (frames outside the human range, worst excess in degrees)")
for b, (n, w, wf) in sorted(res.items(), key=lambda kv: -kv[1][1]):
    print("LIM %-12s frames %3d worst %3d deg at frame %d (%s)" % (b, n, w, wf, seg(wf)))
if not res:
    print("LIM none")
