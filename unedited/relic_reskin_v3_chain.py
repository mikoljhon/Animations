# relic_reskin.py (v3) : clean limb skinning for Relic, the way a rigger would do it.
#   The AI-made mesh came with weights that leak across the body (a vertex on the upper arm partly following the clavicle, plates over the elbow
#   half on each bone...), which stretched faces into long slivers (up to 0.37 m) at the shoulders, elbows and hips.
#   Here every limb vertex is re-weighted ALONG ITS OWN BONE CHAIN: rigid along each bone, blending only inside a short band at each joint
#   (chain_weights, as used for Mender). The mesh stays whole (no cuts, no holes) and bends at the joints like a sleeve over a joint.
#   Torso, neck and head keep their original smooth weights.
#   blender.exe -b fight_v01.blend --python relic_reskin.py   -> fight_v03.blend
#   (v1 cut the plates apart -> no stretch but ragged holes: relic_reskin_v1_split.py)
import bpy, numpy as np, math

OUT = r"C:\ai\unedited"


def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def chain_weights(P, joints, bones, blends):
    """weights for points P along a joint chain; bones[k] runs joints[k] -> joints[k+1]; blends[k] = half width of the band at joint k+1"""
    P = np.asarray(P, float)
    Jn = np.array([np.asarray(j, float) for j in joints])
    seg = Jn[1:] - Jn[:-1]
    L = np.linalg.norm(seg, axis=1)
    S = np.concatenate([[0.0], np.cumsum(L)])
    best_s = np.zeros(len(P))
    best_d = np.full(len(P), 1e9)
    for k in range(len(seg)):
        d = seg[k] / L[k]
        t = np.clip((P - Jn[k]) @ d, 0.0, L[k])
        q = Jn[k] + t[:, None] * d
        dist = np.linalg.norm(P - q, axis=1)
        m = dist < best_d
        best_d[m] = dist[m]
        best_s[m] = S[k] + t[m]
    w, n = {}, len(bones)
    for k, b in enumerate(bones):
        bl_in = blends[k - 1] if k > 0 else 0.0
        bl_out = blends[k] if k < n - 1 else 0.0
        up = _sstep((best_s - (S[k] - bl_in)) / (2 * bl_in)) if k > 0 else np.ones(len(P))
        down = 1.0 - _sstep((best_s - (S[k + 1] - bl_out)) / (2 * bl_out)) if k < n - 1 else np.ones(len(P))
        w[b] = up * down
    tot = sum(w.values())
    return {b: w[b] / np.maximum(tot, 1e-6) for b in w}


def reskin(ob, rig):
    me = ob.data
    B = rig.data.bones
    gi = {g.index: g.name for g in ob.vertex_groups}
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    W = []
    dom = []
    for v in me.vertices:
        w = {gi[g.group]: g.weight for g in v.groups}
        W.append(w)
        dom.append(max(w, key=w.get) if w else "none")
    dom = np.array(dom)
    chains = []
    for sd in "LR":
        arm = ["clavicle." + sd, "upperarm." + sd, "forearm." + sd, "hand." + sd]
        aj = [B["clavicle." + sd].head_local, B["upperarm." + sd].head_local, B["forearm." + sd].head_local, B["hand." + sd].head_local,
              B["hand." + sd].tail_local]
        chains.append((arm, aj, [0.05, 0.055, 0.03]))
        # the leg chain starts at the pelvis so hip plates that cover the hip joint bend smoothly there instead of stretching
        leg = ["pelvis", "thigh." + sd, "shin." + sd, "foot." + sd, "toe." + sd]
        lj = [B["pelvis"].head_local, B["thigh." + sd].head_local, B["shin." + sd].head_local, B["foot." + sd].head_local, B["toe." + sd].head_local,
              B["toe." + sd].tail_local]
        chains.append((leg, lj, [0.07, 0.065, 0.045, 0.03]))
    newW = {}
    counts = {}
    pel = B["pelvis"].head_local
    for bones, joints, blends in chains:
        if bones[0] == "pelvis":
            side = 1.0 if B[bones[1]].head_local.x - pel.x > 0 else -1.0
            limb = np.isin(dom, bones[1:])
            hipzone = (dom == "pelvis") & ((co[:, 0] - pel.x) * side > 0.02) & (co[:, 2] < pel.z + 0.08)
            idx = np.where(limb | hipzone)[0]
        else:
            idx = np.where(np.isin(dom, bones))[0]
        if len(idx) == 0:
            continue
        cw = chain_weights(co[idx], joints, bones, blends)
        for b, arr in cw.items():
            for i, x in zip(idx, arr):
                if x > 1e-3:
                    newW.setdefault(int(i), {})[b] = float(x)
        counts[bones[1]] = len(idx)
    # small separate pieces (shoulder caps, knee pads, plates under 22 cm) move as ONE block with the bone most of them follows: a piece that is
    # not connected to anything cannot stretch if it is rigid. Long pieces that really cover a joint keep the chain weights and bend there.
    edges = np.zeros(len(me.edges) * 2, dtype=int); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
    parent = np.arange(len(me.vertices))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a_, b_ in edges:
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[ra] = rb
    isl = np.array([find(i) for i in range(len(me.vertices))])
    rigid_islands = 0
    for k in np.unique(isl):
        idx = np.where(isl == k)[0]
        ext = co[idx].max(0) - co[idx].min(0)
        if ext.max() > 0.22:
            continue
        votes = {}
        for i in idx:
            w = newW.get(int(i), W[int(i)])
            if w:
                b = max(w, key=w.get)
                votes[b] = votes.get(b, 0) + 1
        if len(votes) < 2:
            continue
        top = max(votes, key=votes.get)
        for i in idx:
            newW[int(i)] = {top: 1.0}
        rigid_islands += 1
    print("RESKIN small pieces made rigid:", rigid_islands)
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    groups = {}

    def grp(n):
        if n not in groups:
            groups[n] = ob.vertex_groups.new(name=n)
        return groups[n]
    for v in me.vertices:
        w = newW.get(v.index, W[v.index])
        for n, x in w.items():
            grp(n).add([v.index], x, 'REPLACE')
    print("RESKIN v3 chain-weighted limb verts:", counts, "of", len(me.vertices))


rig = bpy.data.objects["Relic_AI_Rig"]
body = bpy.data.objects["Relic_AI"]
reskin(body, rig)
cr = bpy.data.objects.get("Relic_AI_Crust")
if cr is not None:
    for g in list(cr.vertex_groups):
        cr.vertex_groups.remove(g)
    for g in body.vertex_groups:
        cr.vertex_groups.new(name=g.name)
    for v in body.data.vertices:
        for g in v.groups:
            cr.vertex_groups[body.vertex_groups[g.group].name].add([v.index], g.weight, 'REPLACE')
bpy.ops.wm.save_as_mainfile(filepath=OUT + "/fight_v03.blend")
print("RESKIN SAVED")
