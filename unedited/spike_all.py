import sys, json, math, bpy
a = sys.argv.index("--") + 1
sp = json.load(open(sys.argv[a])); NF = int(sys.argv[a + 1]); RIG = sys.argv[a + 2] if len(sys.argv) > a + 2 else "Relic_AI_Rig"
sc = bpy.data.scenes["Street"]; rig = bpy.data.objects[RIG]
def seg(f):
    for k, (s, d, h) in sp.items():
        if s - 1 <= f <= s + d + 1: return k
    return "gap"
prev = {}; hits = {}
for f in range(1, NF + 1):
    sc.frame_set(f)
    for pb in rig.pose.bones:
        if pb.name.startswith(("ik_", "fist")): continue
        q = (rig.matrix_world @ pb.matrix).to_quaternion()
        if pb.name in prev:
            ang = math.degrees(q.rotation_difference(prev[pb.name]).angle); ang = min(ang, 360 - ang)
            if ang > 40:
                hits.setdefault(pb.name, []).append((f, seg(f), round(ang)))
        prev[pb.name] = q
for b, v in sorted(hits.items(), key=lambda kv: -max(x[2] for x in kv[1])):
    print("SPIKE %-12s %2d frames, worst %3d deg, big: %s" % (b, len(v), max(x[2] for x in v), [x for x in v if x[2] > 80]))
if not hits: print("SPIKE none")
