import sys, json, bpy
a = sys.argv.index("--") + 1
sp = json.load(open(sys.argv[a])); NF = int(sys.argv[a + 1]); RIG = sys.argv[a + 2] if len(sys.argv) > a + 2 else "Relic_AI_Rig"
sc = bpy.data.scenes["Street"]; rig = bpy.data.objects[RIG]
ank0 = rig.data.bones["foot.L"].head_local.z
def seg(f):
    for k, (s, d, h) in sp.items():
        if s - 1 <= f <= s + d + 1: return k
    return "gap"
prev = {}
res = {}
for f in range(1, NF + 1):
    sc.frame_set(f)
    m = seg(f)
    for s in "LR":
        p = rig.matrix_world @ rig.pose.bones["foot." + s].head
        if s in prev and p.z < ank0 + 0.04 and prev[s].z < ank0 + 0.04:
            d = (p - prev[s]).length
            r = res.setdefault(m, [0.0, 0.0, 0])
            r[0] += d; r[1] = max(r[1], d); r[2] += 1
        prev[s] = p
for m, (tot, mx, n) in res.items():
    print("SLIDE %-14s grounded foot-frames %3d   total slide %.2f m   worst per frame %.3f m (%.1f cm)" % (m, n, tot, mx, mx * 100))
zs = []
for f in range(1, NF + 1):
    sc.frame_set(f)
    for s in "LR":
        zs.append((rig.matrix_world @ rig.pose.bones["foot." + s].head).z)
print("LOWEST ankle height over the whole clip: %.3f (rest %.3f)" % (min(zs), ank0))
