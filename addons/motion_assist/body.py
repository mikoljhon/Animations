# Motion Assist: the body model.
# Every bone gets a mass, a centre of mass and an inertia tensor, estimated from the convex hulls of the mesh parts it
# carries: skinned vertices grouped by their strongest bone, plus whole meshes parented to a bone (gear, props).
# The hull vertices double as contact probes (the points that can touch the ground).

import json

import bmesh
import bpy
import numpy as np

PROP = "ma_body"
_cache = {}


def parse_ignore(text):
    return tuple(s.strip().lower() for s in text.split(",") if s.strip())


def ignored(ob, ignore=()):
    if ob.get("ma_ignore"):
        return True
    n = ob.name.lower()
    return any(s in n for s in ignore)


def character_parts(rig, ignore=()):
    """(mesh object, 'SKIN' or 'BONE') for meshes deformed by the rig or riding one of its bones"""
    parts = []
    for ob in bpy.data.objects:
        if ob.type != 'MESH' or ignored(ob, ignore):
            continue
        if any(m.type == 'ARMATURE' and m.object == rig for m in ob.modifiers):
            parts.append((ob, 'SKIN'))
        elif ob.parent == rig and ob.parent_type == 'BONE' and ob.parent_bone in rig.pose.bones:
            parts.append((ob, 'BONE'))
    return parts


def _coords(ob):
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    M = np.array(ob.matrix_world, dtype=float)
    return co.reshape(n, 3) @ M[:3, :3].T + M[:3, 3]


def _to_local(P, M4):
    M = np.array(M4, dtype=float)
    return (P - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T


def _skin_groups(ob, rig):
    """rest positions of a skinned mesh, split by strongest bone, in each bone's local frame"""
    P = _coords(ob)
    names = {g.index: g.name for g in ob.vertex_groups if g.name in rig.pose.bones}
    best = np.full(len(P), -1)
    bw = np.zeros(len(P))
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group in names and g.weight > bw[v.index]:
                bw[v.index] = g.weight
                best[v.index] = g.group
    out = {}
    for gi, bname in names.items():
        sel = best == gi
        if sel.sum() >= 4:
            out[bname] = _to_local(P[sel], rig.matrix_world @ rig.data.bones[bname].matrix_local)
    return out


def _bone_group(ob, rig):
    """a mesh parented to a bone: all of it rides that bone (pose independent, so the current pose is fine)"""
    pb = rig.pose.bones[ob.parent_bone]
    return {ob.parent_bone: _to_local(_coords(ob), rig.matrix_world @ pb.matrix)}


def hull_mass(P):
    """volume, centroid, inertia tensor about the centroid (per unit density) and hull vertices of the convex hull of P"""
    P = np.asarray(P, dtype=float)
    if len(P) < 4:
        return None
    o = P.mean(axis=0)
    bm = bmesh.new()
    try:
        for p in P - o:
            bm.verts.new(p)
        bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
        if len(bm.faces) < 4:
            return None
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.verts.index_update()
        tri = np.array([[tuple(v.co) for v in f.verts] for f in bm.faces], dtype=float)
        used = {v.index: tuple(v.co) for f in bm.faces for v in f.verts}
        hull_pts = np.array(list(used.values()), dtype=float) + o
    except Exception:
        return None
    finally:
        bm.free()
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))          # 6 x signed volume of the tetrahedron (0, a, b, c)
    V = det.sum() / 6.0
    if abs(V) < 1e-10:
        return None
    if V < 0:
        det, V = -det, -V
    s = a + b + c
    cen = (det[:, None] * s).sum(axis=0) / (24.0 * V)
    C = (np.einsum("i,ij,ik->jk", det, a, a) + np.einsum("i,ij,ik->jk", det, b, b)
         + np.einsum("i,ij,ik->jk", det, c, c) + np.einsum("i,ij,ik->jk", det, s, s)) / 120.0
    C -= V * np.outer(cen, cen)                                # second moment about the centroid
    J = np.trace(C) * np.eye(3) - C
    return V, cen + o, J, hull_pts


def farthest_points(P, k):
    P = np.asarray(P, dtype=float)
    if len(P) <= k:
        return P
    idx = [0]
    d = np.linalg.norm(P - P[0], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return P[idx]


def build_body(rig, total_mass=75.0, ignore=(), gear_density=0.35, probes_per_bone=24):
    """measure the character and store the body model on the rig; returns a short report"""
    bpy.context.view_layer.update()
    acc = {}
    nparts = 0
    for ob, how in character_parts(rig, ignore):
        dens = float(ob.get("ma_density", 1.0 if how == 'SKIN' else gear_density))
        if dens <= 0.0:
            continue
        groups = _skin_groups(ob, rig) if how == 'SKIN' else _bone_group(ob, rig)
        for bname, L in groups.items():
            h = hull_mass(L)
            if h is None:
                continue
            V, cen, J, hp = h
            acc.setdefault(bname, []).append((dens * V, cen, dens * J, hp))
            nparts += 1
    if not acc:
        raise RuntimeError("no meshes found on this rig (skinned or parented to its bones)")
    bones, probes = {}, {}
    for bname, items in acc.items():
        m = sum(i[0] for i in items)
        if m <= 0.0:
            continue
        c = sum(i[0] * i[1] for i in items) / m
        J = np.zeros((3, 3))
        for mi, ci, Ji, _hp in items:
            d = ci - c
            J += Ji + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        bones[bname] = (m, c, J)
        probes[bname] = farthest_points(np.vstack([i[3] for i in items]), probes_per_bone)
    k = total_mass / sum(v[0] for v in bones.values())
    out = {"version": 1, "total": float(total_mass), "bones": {}, "probes": {}}
    for bname, (m, c, J) in bones.items():
        out["bones"][bname] = {"m": float(m * k), "c": [float(x) for x in c], "I": (J * k).tolist()}
        out["probes"][bname] = [[float(x) for x in p] for p in probes[bname]]
    rig[PROP] = json.dumps(out)
    _cache.pop(rig.name, None)
    heavy = sorted(out["bones"].items(), key=lambda kv: -kv[1]["m"])[:6]
    return {"bones": len(bones), "parts": nparts, "probes": sum(len(v) for v in out["probes"].values()),
            "heaviest": [(n, round(v["m"], 1)) for n, v in heavy]}


class Body:
    """the stored body model as arrays: masses m, local centres c, local inertia I, contact probes"""

    def __init__(self, d, rig):
        self.names = [n for n in d["bones"] if n in rig.pose.bones]
        self.m = np.array([d["bones"][n]["m"] for n in self.names], dtype=float)
        self.c = np.array([d["bones"][n]["c"] for n in self.names], dtype=float)
        self.I = np.array([d["bones"][n]["I"] for n in self.names], dtype=float)
        self.M = float(self.m.sum())
        pb, pl = [], []
        for i, n in enumerate(self.names):
            for p in d["probes"].get(n, []):
                pb.append(i)
                pl.append(p)
        self.pbone = np.array(pb, dtype=int)
        self.plocal = np.array(pl, dtype=float).reshape(-1, 3)

    def mats(self, rig):
        """world matrices (n, 4, 4) of the bones with mass, in the current evaluated pose"""
        Mw = np.array(rig.matrix_world, dtype=float)
        pbs = rig.pose.bones
        A = np.array([np.array(pbs[n].matrix, dtype=float) for n in self.names])
        return np.einsum("ij,njk->nik", Mw, A)

    def centres(self, W):
        return np.einsum("nij,nj->ni", W[:, :3, :3], self.c) + W[:, :3, 3]

    def com(self, W):
        X = self.centres(W)
        return (self.m[:, None] * X).sum(axis=0) / self.M, X

    def probes_world(self, W):
        Wp = W[self.pbone]
        return np.einsum("nij,nj->ni", Wp[:, :3, :3], self.plocal) + Wp[:, :3, 3]


def get_body(rig):
    s = rig.get(PROP) if rig is not None else None
    if not s:
        return None
    key = (len(s), hash(s))
    hit = _cache.get(rig.name)
    if hit is not None and hit[0] == key:
        return hit[1]
    b = Body(json.loads(s), rig)
    _cache[rig.name] = (key, b)
    return b
