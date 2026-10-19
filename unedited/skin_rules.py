# skin_rules.py : which bones may pull on the same vertex.   exec'd by relic_reskin.py (cleaning), leak_scan.py and fight_pipeline.py (QA)
#   Rule: one vertex may only follow bones that are at most TWO joints apart in the skeleton (hand+forearm, upperarm+chest, thigh+thigh
#   at the crotch...). A vertex pulled by the hand AND the thigh (Relic's hands hang beside his thighs in the rest pose, and the original
#   auto-weights mixed them) stretches into a spike as soon as the arm leaves the leg. fist1/fist2 count as the hand.
import numpy as np

MAX_JOINTS = 2


def _canon(b):
    return b.replace("fist1", "hand").replace("fist2", "hand")


def bone_distances(rig):
    names = [b.name for b in rig.data.bones]
    nb = {n: set() for n in names}
    for b in rig.data.bones:
        if b.parent:
            nb[b.name].add(b.parent.name)
            nb[b.parent.name].add(b.name)
    D = {}
    for s in names:
        d = {s: 0}
        q = [s]
        while q:
            x = q.pop(0)
            for y in nb[x]:
                if y not in d:
                    d[y] = d[x] + 1
                    q.append(y)
        D[s] = d
    return D


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


def _weights(ob):
    gi = {g.index: g.name for g in ob.vertex_groups}
    return [{gi[g.group]: g.weight for g in v.groups if g.weight > 1e-4} for v in ob.data.vertices]


def _dist(D, a, b):
    a, b = _canon(a), _canon(b)
    if a not in D or b not in D[a]:
        return 0 if a == b else 99
    return D[a][b]


def _fix_vertex(w, T, D):
    """w: {bone: weight} of one vertex, T: total weight of each bone over its piece. Returns cleaned weights or None if clean."""
    bones = list(w)
    if len(bones) < 2:
        return None
    keep = max(bones, key=lambda b: (T.get(_canon(b), 0.0), w[b]))      # the bone this piece mostly belongs to
    out = {b: x for b, x in w.items() if _dist(D, b, keep) <= MAX_JOINTS}
    if len(out) == len(w):
        return None
    s = sum(out.values())
    return {b: x / s for b, x in out.items()}


def _piece_totals(W, isl):
    T = {}
    for i, w in enumerate(W):
        t = T.setdefault(isl[i], {})
        for b, x in w.items():
            t[_canon(b)] = t.get(_canon(b), 0.0) + x
    return T


def leak_report(ob, rig=None):
    rig = rig or next(m.object for m in ob.modifiers if m.type == 'ARMATURE')
    D = bone_distances(rig)
    W = _weights(ob)
    isl = islands_of(ob.data)
    T = _piece_totals(W, isl)
    rep = {}
    for i, w in enumerate(W):
        fx = _fix_vertex(w, T[isl[i]], D)
        if fx is None:
            continue
        lost = {b: x for b, x in w.items() if b not in fx}
        r = rep.setdefault(isl[i], dict(piece=int(isl[i]), region=max(T[isl[i]], key=T[isl[i]].get), verts=0, weight=0.0, max=0.0, bones={}))
        r['verts'] += 1
        r['weight'] += sum(lost.values())
        r['max'] = max(r['max'], sum(lost.values()) / max(1e-6, sum(w.values())))
        for b in lost:
            r['bones'][b] = r['bones'].get(b, 0) + 1
    return sorted(rep.values(), key=lambda r: -r['weight'])


def clean_leaks(ob, rig=None):
    """remove leaked weights in place; returns (pieces, vertices) fixed"""
    rig = rig or next(m.object for m in ob.modifiers if m.type == 'ARMATURE')
    D = bone_distances(rig)
    W = _weights(ob)
    isl = islands_of(ob.data)
    T = _piece_totals(W, isl)
    G = {g.name: g for g in ob.vertex_groups}
    pieces, nv = set(), 0
    for i, w in enumerate(W):
        fx = _fix_vertex(w, T[isl[i]], D)
        if fx is None:
            continue
        for b in w:
            if b in fx:
                G[b].add([i], fx[b], 'REPLACE')
            else:
                G[b].remove([i])
        pieces.add(isl[i]); nv += 1
    return len(pieces), nv
