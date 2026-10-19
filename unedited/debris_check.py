# debris_check.py : find loose mesh pieces that float AWAY from the body on given frames (a piece whose nearest point on any other piece is
#   further than a gap). Reports size, where it sits at rest, and the bones it follows.   blender.exe -b pipe_<tag>.blend --python debris_check.py -- f1 f2 ...
import bpy, sys, numpy as np
from mathutils import kdtree
sc = bpy.data.scenes["Street"]
ob = bpy.data.objects["Relic_AI"]
me0 = ob.data
n = len(me0.vertices)
edges = np.zeros(len(me0.edges) * 2, dtype=int); me0.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
parent = np.arange(n)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a_, b_ in edges:
    ra, rb = find(a_), find(b_)
    if ra != rb:
        parent[ra] = rb
isl = np.array([find(i) for i in range(n)])
ids, counts = np.unique(isl, return_counts=True)
small = [i for i, c in zip(ids, counts) if c < 400]
rest = np.zeros(n * 3); me0.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
gi = {g.index: g.name for g in ob.vertex_groups}
a = sys.argv.index("--") + 1
for f in [int(x) for x in sys.argv[a:]]:
    sc.frame_set(f)
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    m = ev.to_mesh()
    co = np.zeros(len(m.vertices) * 3); m.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    ev.to_mesh_clear()
    kd = kdtree.KDTree(n)
    for i in range(n):
        kd.insert(co[i], i)
    kd.balance()
    out = []
    for k in small:
        idx = np.where(isl == k)[0]
        best = 1e9
        for i in idx[:: max(1, len(idx) // 20)]:
            for (p, j, d) in kd.find_n(co[i], 40):
                if isl[j] != k:
                    best = min(best, d)
                    break
        if best > 0.04:
            ws = {}
            for i in idx:
                for g in me0.vertices[int(i)].groups:
                    ws[gi[g.group]] = ws.get(gi[g.group], 0) + g.weight
            out.append((round(best, 3), int(k), len(idx), tuple(np.round(rest[idx].mean(0), 3)), sorted(ws.items(), key=lambda kv: -kv[1])[:3]))
    out.sort(reverse=True)
    print("DEBRIS frame", f, "floating pieces:", len(out))
    for r in out[:10]:
        print("   gap %.3f m  piece %d  %d verts  rest centre %s  bones %s" % r)
