# arm_diag.py : per frame, where each hand and elbow is relative to Relic's own chest (forward / up / out), vs the real capture.
#   Flags arms swung behind the body or elbows pointing up-and-back.  blender.exe -b pipe_<tag>.blend --python arm_diag.py -- <tag> [f0 f1]
import bpy, sys, math, json
from mathutils import Vector, Matrix
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
pb = rig.pose.bones
a = sys.argv.index("--") + 1
tag = sys.argv[a]
f0 = int(sys.argv[a + 1]) if len(sys.argv) > a + 1 else sc.frame_start
f1 = int(sys.argv[a + 2]) if len(sys.argv) > a + 2 else sc.frame_end
W = lambda n, t=False: rig.matrix_world @ (pb[n].tail if t else pb[n].head)
E = lambda n: bpy.data.objects["ref_" + n].matrix_world.translation.copy()
spans = json.load(open(r"C:\ai\unedited\anim_cache5\spans_%s.json" % tag))
inclip = lambda f: any(s <= f <= s + n for s, n, m in spans.values())


def frame_of(L, R, up):
    x = (L - R).normalized()                   # toward the left shoulder
    y = (up - x * up.dot(x)).normalized()
    z = x.cross(y)                             # forward
    return x, y, z


rows = []
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    cx, cy, cz = frame_of(W('upperarm.L'), W('upperarm.R'), W('neck') - W('pelvis'))
    rx, ry, rz = frame_of(E('shoulder_L'), E('shoulder_R'), E('neck') - E('hips'))
    out = [f, "clip" if inclip(f) else "BLEND"]
    for sd in "LR":
        s, e, h = W('upperarm.' + sd), W('forearm.' + sd), W('forearm.' + sd, True)
        ln = (e - s).length + (h - e).length
        hf, ef, eu = (h - s).dot(cz) / ln, (e - s).dot(cz) / (e - s).length, (e - s).dot(cy) / (e - s).length
        bend = 180 - math.degrees((s - e).angle(h - e))
        rs, re_, rh = E('shoulder_' + sd), E('elbow_' + sd), E('wrist_' + sd)
        rln = (re_ - rs).length + (rh - re_).length
        rhf, ref = (rh - rs).dot(rz) / rln, (re_ - rs).dot(rz) / (re_ - rs).length
        out += [sd, round(hf, 2), round(rhf, 2), round(ef, 2), round(ref, 2), round(eu, 2), round(bend)]
    rows.append(out)
    print("ARM f%d %-5s | L hand fwd %5.2f (real %5.2f) elbow fwd %5.2f (real %5.2f) elbow up %5.2f bend %3d | R hand fwd %5.2f (real %5.2f) elbow fwd %5.2f (real %5.2f) elbow up %5.2f bend %3d" % tuple(out[:2] + out[3:9] + out[10:16]))
json.dump(rows, open(r"C:\ai\unedited\anim_cache5\arm_diag_%s.json" % tag, "w"))
