import bpy, numpy as np
ob = bpy.data.objects["Relic_AI"]
me = ob.data
n = len(me.vertices)
edges = np.zeros(len(me.edges) * 2, dtype=int); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
parent = np.arange(n)
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in edges:
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
roots = np.array([find(i) for i in range(n)])
u, c = np.unique(roots, return_counts=True)
order = np.argsort(-c)
print("ISLANDS", len(u), "largest", c[order][:10].tolist(), "verts", n)
print("ISLAND sizes >1000:", int((c > 1000).sum()), " 100-1000:", int(((c > 100) & (c <= 1000)).sum()), " <100:", int((c <= 100).sum()))
gi = {g.index: g.name for g in ob.vertex_groups}
W = {}
dom = np.empty(n, dtype=object)
for v in me.vertices:
    if v.groups:
        g = max(v.groups, key=lambda g: g.weight); dom[v.index] = gi[g.group]
    else:
        dom[v.index] = "none"
rows = []
for r in u:
    idx = np.where(roots == r)[0]
    names, cnt = np.unique(dom[idx], return_counts=True)
    top = names[np.argmax(cnt)]; cov = cnt.max() / len(idx)
    rows.append((len(idx), top, cov, dict(zip(names.tolist(), cnt.tolist()))))
rows.sort(key=lambda r: -r[0])
rigid = sum(r[0] for r in rows if r[2] >= 0.8)
span = sum(r[0] for r in rows if r[2] < 0.8)
print("ISLAND rigid-candidate verts", rigid, "spanning verts", span)
for r in rows[:25]:
    print("ISLAND", r[0], r[1], "%.2f" % r[2], {k: v for k, v in sorted(r[3].items(), key=lambda kv: -kv[1])[:4]})
