# hand_scan.py : what Relic's hands are made of. Lists every mesh piece that follows a hand bone, in hand-local coordinates
#   (x across the palm, y along the hand toward the fingertips, z the palm normal), so the fingers can be rigged.
#   blender.exe -b fight_v04.blend --python hand_scan.py
import bpy, numpy as np, json
exec(open(r"C:\ai\unedited\skin_rules.py", encoding="utf-8").read())
rig = bpy.data.objects["Relic_AI_Rig"]
ob = bpy.data.objects["Relic_AI"]
me = ob.data
n = len(me.vertices)
co = np.zeros(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
isl = islands_of(me)
gi = {g.index: g.name for g in ob.vertex_groups}
out = {}
for sd in "LR":
    hb = rig.data.bones["hand." + sd]
    fb = rig.data.bones["forearm." + sd]
    M = hb.matrix_local.inverted()
    hw = np.zeros(n)
    for v in me.vertices:
        for g in v.groups:
            if gi[g.group] == "hand." + sd:
                hw[v.index] = g.weight
    loc = (np.c_[co, np.ones(n)] @ np.array(M).T)[:, :3]
    print("HAND", sd, "bone length %.3f" % hb.length, "head", tuple(round(x, 3) for x in hb.head_local), "tail", tuple(round(x, 3) for x in hb.tail_local))
    rows = []
    for k in np.unique(isl[hw > 0.5]):
        idx = np.where(isl == k)[0]
        frac = float((hw[idx] > 0.5).mean())
        L = loc[idx]
        lo, hi = L.min(0), L.max(0)
        rows.append(dict(piece=int(k), verts=len(idx), frac_hand=round(frac, 2), lo=[round(float(x), 3) for x in lo], hi=[round(float(x), 3) for x in hi],
                         mid=[round(float(x), 3) for x in L.mean(0)]))
    rows.sort(key=lambda r: r['mid'][1])
    for r in rows:
        print("  piece %(piece)7d  verts %(verts)5d  hand %(frac_hand).2f  x %(lo)s..%(hi)s  mid %(mid)s" % dict(r, lo=(r['lo'][0], r['lo'][1], r['lo'][2]), hi=(r['hi'][0], r['hi'][1], r['hi'][2])))
    out[sd] = rows
json.dump(out, open(r"C:\ai\unedited\anim_cache5\hand_scan.json", "w"), indent=1)
print("HAND SCAN DONE")
