import bpy, numpy as np, sys
sc = bpy.data.scenes["Street"]
ob = bpy.data.objects["Relic_AI"]; rig = bpy.data.objects["Relic_AI_Rig"]
me0 = ob.data; n = len(me0.vertices)
rest = np.zeros(n * 3); me0.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
edges = np.zeros(len(me0.edges) * 2, dtype=int); me0.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
parent = np.arange(n)
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in edges:
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
isl = np.array([find(i) for i in range(n)])
gi = {g.index: g.name for g in ob.vertex_groups}
dom = np.array([gi[max(v.groups, key=lambda g: g.weight).group] if v.groups else "none" for v in me0.vertices])
rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
agg = {}
for f in [int(x) for x in sys.argv[sys.argv.index("--") + 1:]]:
    sc.frame_set(f)
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3); ev.to_mesh_clear()
    L = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    bad = np.where((L / np.maximum(rest_len, 1e-6) > 6) & (L > 0.15))[0]
    for e in bad:
        k = isl[edges[e, 0]]
        agg[k] = agg.get(k, 0) + 1
for k, c in sorted(agg.items(), key=lambda kv: -kv[1])[:10]:
    idx = np.where(isl == k)[0]
    names, cnt = np.unique(dom[idx], return_counts=True)
    ext = rest[idx].max(0) - rest[idx].min(0)
    print("ISL", k, "stretched", c, "verts", len(idx), "groups", dict(zip(names.tolist(), cnt.tolist())), "size", np.round(ext, 2).tolist(), "center", np.round(rest[idx].mean(0), 2).tolist())
