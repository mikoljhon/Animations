# relic_reskin.py (v5) : clean skinning for Relic's limbs, decided PIECE BY PIECE.
#   Relic's AI-made mesh is ~220 separate pieces. Its original weights leaked across the body and, worst, pieces were split between different
#   bones in a noisy way, so faces stretched into long slivers (up to 0.37 m) at the shoulders, elbows and hips.
#   Step 1 (v5, skin_rules.py): remove LEAKS. A vertex may only follow bones at most two joints apart. Both hands' finger pieces were also
#   weighted to the thigh beside them (up to 50%), so the fingers stretched into a long spike back toward the hip whenever the arm moved,
#   which read as a second forearm bent backwards; a helmet piece was pulled by the right clavicle.
#   Step 2, the rule for every separate piece:
#     * small piece (under 22 cm): moves as ONE rigid block with the bone most of it follows (a separate rigid piece cannot stretch)
#     * piece that wraps a limb joint (hip plate, knee pad, sleeve over the elbow): every vertex of it is weighted ALONG that limb's bone chain,
#       rigid along each bone and bending inside one band at each joint, wide enough that nothing stretches into a sliver
#     * anything else (the torso skin): keeps its original smooth weights
#   A piece is never split between two rules (that split edge is exactly what stretched).
#   blender.exe -b fight_v01.blend --python relic_reskin.py   -> fight_v06s.blend (v6: + hip zones, dual-quaternion)
#   history: v1 cut plates apart (ragged holes, relic_reskin_v1_split.py), v2 one-ring bands, v3 chain weights per vertex (relic_reskin_v3_chain.py),
#   v4 piece rules without the leak cleaning (fight_v03.blend).
import bpy, numpy as np, math

OUT = r"C:\ai\unedited"
exec(open(OUT + r"\skin_rules.py", encoding="utf-8").read())
SMALL = 0.22
# half widths (m) of the bend band at each joint of a chain, in chain order
ARM_BANDS = [0.07, 0.08, 0.04]          # clavicle|upperarm (shoulder), upperarm|forearm (elbow), forearm|hand (wrist)
LEG_BANDS = [0.11, 0.10, 0.05, 0.03]    # pelvis|thigh (hip), thigh|shin (knee), shin|foot (ankle), foot|toe


def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def chain_weights(P, joints, bones, blends):
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


SMOOTH_SIGMA = 0.04      # m: weights on the torso skin pieces are blurred over this distance, so no bone weight jumps between neighbouring
                         # vertices (a hip armour piece went from 62% thigh to 6% across one edge and tore open 2.4x on high kicks)


def smooth_piece(co, idx, W, sigma=SMOOTH_SIGMA):
    """Gaussian blur of the bone weights inside one piece (rest positions). Returns {vertex: {bone: w}} for the piece."""
    from mathutils import kdtree
    bones = sorted({b for i in idx for b in W[i]})
    if len(bones) < 2:
        return None
    bi = {b: k for k, b in enumerate(bones)}
    M = np.zeros((len(idx), len(bones)))
    for r, i in enumerate(idx):
        for b, x in W[i].items():
            M[r, bi[b]] = x
    kd = kdtree.KDTree(len(idx))
    for r, i in enumerate(idx):
        kd.insert(co[i], r)
    kd.balance()
    out = np.zeros_like(M)
    for r, i in enumerate(idx):
        nb = kd.find_range(co[i], 2.5 * sigma)
        js = np.fromiter((n[1] for n in nb), int, len(nb))
        ds = np.fromiter((n[2] for n in nb), float, len(nb))
        k = np.exp(-0.5 * (ds / sigma) ** 2)
        out[r] = k @ M[js] / k.sum()
    out[out < 0.01] = 0.0
    out /= np.maximum(out.sum(1, keepdims=True), 1e-9)
    return {int(i): {bones[c]: float(out[r, c]) for c in np.nonzero(out[r])[0]} for r, i in enumerate(idx)}


def islands_of(me):
    n = len(me.vertices)
    edges = np.zeros(len(me.edges) * 2, dtype=int); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
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
    return np.array([find(i) for i in range(n)])


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
    chains = {}
    for sd in "LR":
        chains["arm" + sd] = (["clavicle." + sd, "upperarm." + sd, "forearm." + sd, "hand." + sd],
                              [B["clavicle." + sd].head_local, B["upperarm." + sd].head_local, B["forearm." + sd].head_local, B["hand." + sd].head_local,
                               B["hand." + sd].tail_local], ARM_BANDS)
        chains["leg" + sd] = (["pelvis", "thigh." + sd, "shin." + sd, "foot." + sd, "toe." + sd],
                              [B["pelvis"].head_local, B["thigh." + sd].head_local, B["shin." + sd].head_local, B["foot." + sd].head_local,
                               B["toe." + sd].head_local, B["toe." + sd].tail_local], LEG_BANDS)
    isl = islands_of(me)
    newW = {}
    stats = dict(rigid=0, chain=0, kept=0, smoothed=0)
    for k in np.unique(isl):
        idx = np.where(isl == k)[0]
        ext = co[idx].max(0) - co[idx].min(0)
        names, cnt = np.unique(dom[idx], return_counts=True)
        votes = dict(zip(names.tolist(), cnt.tolist()))
        top = max(votes, key=votes.get)
        LIMBN = ("clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe")
        if top.split(".")[0] in LIMBN and votes[top] >= 0.80 * len(idx) and len(votes) >= 2:
            for i in idx:
                newW[int(i)] = {top: 1.0}                  # a limb piece that belongs (80%+) to one bone moves rigidly with it
            stats['rigid'] += 1
            continue
        if ext.max() <= SMALL and len(votes) >= 2:
            for i in idx:
                newW[int(i)] = {top: 1.0}
            stats['rigid'] += 1
            continue
        best = None
        for cname, (bones, joints, bands) in chains.items():
            limb_bones = bones[1:] if bones[0] == "pelvis" else bones
            in_limb = sum(votes.get(b, 0) for b in limb_bones)
            in_chain = in_limb + (votes.get("pelvis", 0) if bones[0] == "pelvis" else 0)
            frac_chain = in_chain / float(len(idx))
            frac_limb = in_limb / float(len(idx))
            spans = len([b for b in bones if votes.get(b, 0) > 0.05 * len(idx)])
            if frac_chain >= 0.85 and frac_limb >= 0.2 and spans >= 2:
                if best is None or frac_chain > best[0]:
                    best = (frac_chain, cname)
        if best is not None:
            bones, joints, bands = chains[best[1]]
            ranked = sorted(((votes.get(b, 0), b) for b in bones), reverse=True)
            if ranked[1][0] < 0.35 * len(idx):
                # one bone owns most of this piece (a knee pad on the thigh...): rigid to it
                for i in idx:
                    newW[int(i)] = {ranked[0][1]: 1.0}
                stats['rigid'] += 1
                continue
            if {ranked[0][1], ranked[1][1]} == {"pelvis", bones[1]}:
                # a hip plate: the chain turns a sharp corner at the hip, so blend by HEIGHT around the hip joint instead
                hz = B[bones[1]].head_local.z
                band = LEG_BANDS[0]
                wt = _sstep((hz + band - co[idx, 2]) / (2 * band))
                for i, x in zip(idx, wt):
                    d = {}
                    if 1.0 - x > 1e-3:
                        d["pelvis"] = float(1.0 - x)
                    if x > 1e-3:
                        d[bones[1]] = float(x)
                    newW[int(i)] = d
                stats['chain'] += 1
                continue
            cw = chain_weights(co[idx], joints, bones, bands)
            for b, arr in cw.items():
                for i, x in zip(idx, arr):
                    if x > 1e-3:
                        newW.setdefault(int(i), {})[b] = float(x)
            stats['chain'] += 1
        else:
            sm = smooth_piece(co, idx, W)
            if sm is not None:
                newW.update(sm)
                stats['smoothed'] += 1
            stats['kept'] += 1
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
    print("RESKIN v5 pieces: rigid %d, bent along a limb chain %d, kept %d (smoothed %d) (of %d)" % (stats['rigid'], stats['chain'], stats['kept'], stats['smoothed'], len(np.unique(isl))))


HIP_ZONE_R = 0.22       # m: the hip region = a capsule along each thigh, from 10 cm above the hip joint to 30 cm down the thigh
HIP_BAND = 0.07         # m: half width of the pelvis <-> thigh blend across the joint (along the thigh)
HIP_MIN_INSIDE = 0.2    # a piece is a hip piece when at least this share of it lies in a hip region
HIP_LAT = 0.10          # m: across the crotch, how softly a vertex is shared between the two thighs


def hip_zones(ob, rig):
    """v6: the waist plates and the thigh plates were separate rigid blocks and came apart on high kicks ("the leg disconnects").
    Every HIP PIECE (armour of the pelvis / upper thighs that reaches into a hip region) now gets its weights from POSITION on ALL its vertices:
    above the joint it follows the pelvis, below it the thigh, blending across +-HIP_BAND; a piece over the crotch is shared smoothly between
    both thighs. Whole pieces only: a piece is never torn between new weights and old ones (the first try cut pieces at a sphere's rim and
    tore them 4.6x on kicks)."""
    me = ob.data
    B = rig.data.bones
    n = len(me.vertices)
    co = np.zeros(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    isl = islands_of(me)
    gi = {g.index: g.name for g in ob.vertex_groups}
    W = [{gi[g.group]: g.weight for g in v.groups if g.weight > 1e-4} for v in me.vertices]
    side = {}
    for sd in "LR":
        th = B["thigh." + sd]
        H = np.array(th.head_local)
        ax = np.array((th.tail_local - th.head_local).normalized())
        d = co - H
        s = d @ ax
        sc = np.clip(s, -0.10, 0.30)
        dist = np.linalg.norm(d - np.outer(sc, ax), axis=1)
        side[sd] = (s, dist)
    inside = (side["L"][1] < HIP_ZONE_R) | (side["R"][1] < HIP_ZONE_R)
    OK = {"pelvis", "spine", "thigh.L", "thigh.R"}
    G = {g.name: g for g in ob.vertex_groups}
    for nm in ("pelvis", "thigh.L", "thigh.R"):
        if nm not in G:
            G[nm] = ob.vertex_groups.new(name=nm)
    n_p = n_v = 0
    for k in np.unique(isl[inside]):
        idx = np.where(isl == k)[0]
        if inside[idx].mean() < HIP_MIN_INSIDE:
            continue
        tot = {}
        for i in idx:
            for b, x in W[i].items():
                tot[b] = tot.get(b, 0.0) + x
        if any(b not in OK for b, x in tot.items() if x > 0.02 * len(idx)):
            continue                                   # part of an arm / lower leg: not a hip piece
        if tot.get("spine", 0.0) > 0.2 * len(idx):
            continue                                   # belly skin: leave it
        wt, lat = {}, {}
        for sd in "LR":
            s, dist = side[sd][0][idx], side[sd][1][idx]
            x = np.clip((s + HIP_BAND) / (2 * HIP_BAND), 0, 1)
            wt[sd] = x * x * (3 - 2 * x)
            lat[sd] = np.exp(-0.5 * (dist / HIP_LAT) ** 2) + 1e-9
        ls = lat["L"] + lat["R"]
        tl, tr = wt["L"] * lat["L"] / ls, wt["R"] * lat["R"] / ls
        for j, i in enumerate(idx):
            for b in list(W[i]):
                G[b].remove([int(i)])
            new = {"pelvis": max(0.0, 1.0 - tl[j] - tr[j]), "thigh.L": float(tl[j]), "thigh.R": float(tr[j])}
            for b, x in new.items():
                if x > 1e-3:
                    G[b].add([int(i)], x, 'REPLACE')
        n_p += 1
        n_v += len(idx)
    return n_p, n_v


rig = bpy.data.objects["Relic_AI_Rig"]
body = bpy.data.objects["Relic_AI"]
print("LEAKS before:", [(r['piece'], r['region'], r['verts'], r['bones']) for r in leak_report(body, rig)])
print("LEAKS cleaned (pieces, vertices):", clean_leaks(body, rig))
reskin(body, rig)
print("HIP ZONES (v6) hip pieces, vertices:", hip_zones(body, rig))
print("LEAKS after smoothing, cleaned again:", clean_leaks(body, rig))
print("LEAKS after:", len(leak_report(body, rig)))
for o in (body, bpy.data.objects.get("Relic_AI_Crust")):
    for m in (o.modifiers if o else []):
        if m.type == 'ARMATURE':
            m.use_deform_preserve_volume = False             # plain linear skinning: dual-quaternion was tried and stretched the fists more (fist_spike.py: 36 vs 16 edges)
cr = bpy.data.objects.get("Relic_AI_Crust")
if cr is not None:
    for g in list(cr.vertex_groups):
        cr.vertex_groups.remove(g)
    for g in body.vertex_groups:
        cr.vertex_groups.new(name=g.name)
    for v in body.data.vertices:
        for g in v.groups:
            cr.vertex_groups[body.vertex_groups[g.group].name].add([v.index], g.weight, 'REPLACE')
bpy.ops.wm.save_as_mainfile(filepath=OUT + "/fight_v06s.blend")
print("RESKIN SAVED")
