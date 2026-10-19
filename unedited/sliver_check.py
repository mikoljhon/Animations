import bpy, numpy as np, sys
sc = bpy.data.scenes["Street"]
ob = bpy.data.objects["Relic_AI"]
rig = bpy.data.objects["Relic_AI_Rig"]
me0 = ob.data
n = len(me0.vertices)
rest = np.zeros(n * 3); me0.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
edges = np.zeros(len(me0.edges) * 2, dtype=int); me0.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
gi = {g.index: g.name for g in ob.vertex_groups}
dom = []
for v in me0.vertices:
    best = max(v.groups, key=lambda g: g.weight) if v.groups else None
    dom.append(gi[best.group] if best else "none")
dom = np.array(dom)
rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
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
    bad = np.where((ratio > 6) & (L > 0.15))[0]
    pairs = {}
    for e in bad:
        k = tuple(sorted((dom[edges[e, 0]], dom[edges[e, 1]])))
        pairs[k] = pairs.get(k, 0) + 1
    print("SLIVER frame", f, "stretched edges", len(bad), "longest %.2f m" % (L[bad].max() if len(bad) else 0), sorted(pairs.items(), key=lambda kv: -kv[1])[:6])
