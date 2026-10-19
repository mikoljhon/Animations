# sliver_dbg.py : find mesh edges that stretch (ratio vs rest) on given frames, grouped by mesh piece, with the bones they hang on.
#   blender.exe -b pipe_<tag>.blend --python sliver_dbg.py -- f1 f2 ...
import bpy, numpy as np, sys
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
ob = bpy.data.objects["Relic_AI"]
me0 = ob.data
n = len(me0.vertices)
rest = np.zeros(n * 3); me0.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
edges = np.zeros(len(me0.edges) * 2, dtype=int); me0.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
gi = {g.index: g.name for g in ob.vertex_groups}
VW = [{gi[g.group]: round(g.weight, 2) for g in v.groups if g.weight > 0.01} for v in me0.vertices]
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
rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
print("mods", [(m.type, m.name) for m in ob.modifiers])
a = sys.argv.index("--") + 1
for f in [int(x) for x in sys.argv[a:]]:
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    ev.to_mesh_clear()
    L = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    ratio = L / np.maximum(rest_len, 1e-6)
    bad = np.where((ratio > 1.8) & (L > 0.04))[0]
    print("FRAME", f, "edges ratio>1.8 & >4cm:", len(bad), " ratio>3 & >8cm:", int(((ratio > 3) & (L > 0.08)).sum()))
    by = {}
    for e in bad:
        k = isl[edges[e, 0]]
        by.setdefault(k, []).append(e)
    for k, es in sorted(by.items(), key=lambda kv: -max(L[e] for e in kv[1]))[:8]:
        e = max(es, key=lambda e: L[e])
        i, j = edges[e]
        nv = int((isl == k).sum())
        print("  piece %d (%d verts, %d bad edges): worst edge %.3f m (rest %.3f, x%.1f)  v%d %s  ->  v%d %s" % (
            k, nv, len(es), L[e], rest_len[e], ratio[e], i, VW[i], j, VW[j]))
