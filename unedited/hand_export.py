# hand_export.py : export each hand's vertices (hand-local coordinates, piece id, hand weight) to anim_cache5\hand_<L|R>.npz
#   for designing the finger rig outside Blender.   blender.exe -b fight_v04.blend --python hand_export.py
import bpy, numpy as np
exec(open(r"C:\ai\unedited\skin_rules.py", encoding="utf-8").read())
rig = bpy.data.objects["Relic_AI_Rig"]
ob = bpy.data.objects["Relic_AI"]
me = ob.data
n = len(me.vertices)
co = np.zeros(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
isl = islands_of(me)
gi = {g.index: g.name for g in ob.vertex_groups}
edges = np.zeros(len(me.edges) * 2, dtype=int); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
for sd in "LR":
    hb = rig.data.bones["hand." + sd]
    M = np.array(hb.matrix_local.inverted())
    hw = np.zeros(n)
    for v in me.vertices:
        for g in v.groups:
            if gi[g.group] == "hand." + sd:
                hw[v.index] = g.weight
    idx = np.where(hw > 0)[0]
    loc = (np.c_[co[idx], np.ones(len(idx))] @ M.T)[:, :3]
    sel = np.zeros(n, bool); sel[idx] = True
    e = edges[sel[edges[:, 0]] & sel[edges[:, 1]]]
    np.savez(r"C:\ai\unedited\anim_cache5\hand_%s.npz" % sd, idx=idx, loc=loc, isl=isl[idx], w=hw[idx], edges=e,
             length=hb.length, mat=np.array(hb.matrix_local))
    print("EXPORT", sd, len(idx), "verts", len(e), "edges")
print("HAND EXPORT DONE")
