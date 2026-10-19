# Cut 5: the cover rocks over Relic fall instead of vanishing. Background Blender:
#   blender.exe -b relic_v13_opening_cut5.blend --python ma5_rocks.py   (saves relic_v13_rocks.blend)
# Each of the 48 cover rocks and slabs is held in place until the shock of the big strike (frame C3) reaches it (about 3 m/s
# outward from the strike point), then it is thrown a little up and out, falls under gravity onto the surface as it is at that frame
# (the debris shape key lowers the pit while it falls), bounces, rolls and comes to rest. Sphere-like rocks roll, flat slabs slide and
# lie along the ground. The result is keyed per frame, the old "hide when dug" drivers are removed.
import math
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

T0 = time.time()
LOG = open(r"C:\ai\unedited\anim_cache5\ma5_rocks_log.txt", "w")


def log(*a):
    LOG.write("%7.1f  %s\n" % (time.time() - T0, " ".join(str(x) for x in a)))
    LOG.flush()


sc = bpy.data.scenes["Cave"]
dg = bpy.context.evaluated_depsgraph_get()
C3 = 146
F_STOP = 248
F_SETTLED = 182          # the pit has finished lowering (dug = 1)
SUB = 8
DT = 1.0 / (24.0 * SUB)
G = Vector((0.0, 0.0, -9.81))
rng = np.random.default_rng(77)

deb = bpy.data.objects["Cave_Debris"]
cover = [o for o in bpy.data.collections["Cave_Cover"].objects if o.type == "MESH"]
log("cover rocks", len(cover))

# the strike point (rim target) and the middle of the pit, from the animation's own targets
sc.frame_set(1)
T_strike = None
try:
    pp = bpy.data.objects["MenderC_Puff_strike3"]
    T_strike = pp.matrix_world.translation.copy() if pp.parent is None else pp.matrix_world.translation.copy()
except Exception:
    pass
ring = bpy.data.objects.get("Relic_AI_Ring")
pit_c = (sum((ring.matrix_world @ Vector(c) for c in ring.bound_box), Vector()) / 8.0) if ring else Vector((0.1, 1.3, 1.2))
if T_strike is None:
    T_strike = Vector((0.9, 0.9, 1.0))
log("strike", tuple(T_strike), "pit centre", tuple(pit_c))


def mesh_bvh_world(objs, depsgraph):
    """one BVH of the evaluated world-space geometry of many mesh objects"""
    verts, polys, off = [], [], 0
    for o in objs:
        oe = o.evaluated_get(depsgraph)
        me = oe.to_mesh()
        if len(me.polygons) == 0:
            oe.to_mesh_clear()
            continue
        M = o.matrix_world
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        R = np.array(M.to_3x3())
        w = co @ R.T + np.array(M.translation)
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        verts.append(w)
        polys.append(tri.reshape(-1, 3) + off)
        off += len(w)
        oe.to_mesh_clear()
    V = np.concatenate(verts)
    P = np.concatenate(polys)
    return BVHTree.FromPolygons([Vector(v) for v in V.tolist()], [tuple(p) for p in P.tolist()])


# static scenery near the pit: loose rocks, floor and slope (not the cover rocks, not the debris sheet: that one changes)
rocks_near = [o for o in bpy.data.collections["Cave_Rocks"].all_objects
              if o.type == "MESH" and not o.hide_viewport and (o.matrix_world.translation - pit_c).length < 3.0 and o.name not in {c.name for c in cover}]
static_objs = rocks_near + [bpy.data.objects["Cave_Floor"], bpy.data.objects["Cave_Slope"]]
sc.frame_set(C3)
dg = bpy.context.evaluated_depsgraph_get()
static_bvh = mesh_bvh_world(static_objs, dg)
log("static bvh from", len(static_objs), "objects")
relic_objs = [o for o in bpy.data.collections["Relic_AI"].all_objects if o.type == "MESH" and "Eyes" not in o.name]
relic_bvh = mesh_bvh_world(relic_objs, dg)
log("relic bvh")

# bodies
bodies = []
for o in cover:
    o.rotation_mode = "QUATERNION"
    M0 = o.matrix_world.copy()
    bb = [M0 @ Vector(c) for c in o.bound_box]
    c0 = sum(bb, Vector()) / 8.0
    dims = sorted(o.dimensions)
    flat = dims[0] < 0.4 * dims[2] and dims[2] > 0.3
    r = 0.14 if flat else max(0.03, 0.46 * float(np.median(o.dimensions)))
    dist = (c0 - T_strike).length
    release = C3 + dist * 5.0 + float(rng.uniform(0.0, 4.0))        # frames: the shock ripples out at about 5 frames per metre
    out_h = Vector((c0.x - pit_c.x, c0.y - pit_c.y, 0.0))
    out_h = out_h.normalized() if out_h.length > 1e-4 else Vector((1.0, 0.0, 0.0))
    bodies.append(dict(ob=o, M0=M0, c0=c0, p=c0.copy(), v=Vector(), q=Quaternion((1, 0, 0, 0)), w=Vector(), r=r, flat=flat, rel=release, out=out_h,
                       live=False, rest=0, name=o.name, track=[]))
log("flat bodies", sum(1 for b in bodies if b["flat"]), "release frames", round(min(b["rel"] for b in bodies), 1), round(max(b["rel"] for b in bodies), 1))


def contact(b, bvhs):
    p, r = b["p"], b["r"]
    hit = False
    n_acc = Vector()
    for bvh in bvhs:
        res = bvh.find_nearest(p, r * 1.6)
        if res[0] is None:
            continue
        loc, n, idx, dist = res
        n = Vector(n).normalized()
        inside = (p - loc).dot(n) < 0.0
        if inside:
            n = -n
            pen = r + dist
        else:
            pen = r - dist
        if pen <= 0.0:
            continue
        p = p + n * pen
        vn = b["v"].dot(n)
        if vn < 0.0:
            e = 0.28 if vn < -0.9 else 0.0
            b["v"] = b["v"] - (1.0 + e) * vn * n
            vt = b["v"] - b["v"].dot(n) * n
            b["v"] = b["v"] - vt * (0.10 if not b["flat"] else 0.25)
        hit = True
        n_acc += n
    b["p"] = p
    return hit, (n_acc.normalized() if n_acc.length > 1e-6 else Vector((0, 0, 1)))


frames = list(range(C3 - 1, F_STOP + 1))
relic_every = 3
t_loop = time.time()
for f in frames:
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    if f >= 206 and f % 4 == 0:
        relic_bvh = mesh_bvh_world(relic_objs, dg)
    deb_bvh = BVHTree.FromObject(deb, dg)
    bvhs = [deb_bvh, static_bvh, relic_bvh]
    for b in bodies:
        if not b["live"] and f >= b["rel"]:
            b["live"] = True
            v_h = float(rng.uniform(0.35, 0.95))
            hit, n = contact(b, bvhs)
            down = Vector((0, 0, -1)) - n * Vector((0, 0, -1)).dot(n)
            down = down.normalized() if down.length > 1e-3 else Vector()
            b["v"] = b["out"] * v_h + down * float(rng.uniform(0.2, 0.7)) + Vector((0, 0, float(rng.uniform(0.5, 1.3))))
            ax = Vector(rng.normal(size=3)).normalized()
            b["w"] = ax * float(rng.uniform(1.5, 6.0) if not b["flat"] else rng.uniform(0.4, 1.4))
            if b["flat"]:
                b["w"].x = b["w"].y = 0.0
        if b["live"]:
            for _s in range(SUB):
                b["v"] = b["v"] + G * DT
                b["p"] = b["p"] + b["v"] * DT
                hit, n = contact(b, bvhs)
                if hit:
                    vt = b["v"] - b["v"].dot(n) * n
                    k = 9.0 if b["flat"] else 4.0
                    b["v"] = b["v"] - vt * min(1.0, k * DT)
                    if b["flat"]:
                        cur = b["q"] @ Vector((0, 0, 1))
                        al = cur.rotation_difference(n)
                        b["q"] = (Quaternion((1, 0, 0, 0)).slerp(al, min(1.0, 24.0 * DT)) @ b["q"]).normalized()
                    else:
                        wr = n.cross(vt) / b["r"]
                        b["w"] = b["w"].lerp(wr, 0.15)
                else:
                    b["w"] = b["w"] * (1.0 - 0.3 * DT)
                ang = b["w"].length * DT
                if ang > 1e-9:
                    b["q"] = (Quaternion(b["w"].normalized(), ang) @ b["q"]).normalized()
                if f >= F_SETTLED + 6 and b["v"].length < 0.10:
                    b["v"] = Vector()
                    b["w"] = Vector()
        M = Matrix.Translation(b["p"]) @ b["q"].to_matrix().to_4x4() @ Matrix.Translation(-b["c0"]) @ b["M0"]
        b["track"].append((f, M))
    if f % 10 == 0:
        log("frame", f, "elapsed", round(time.time() - t_loop, 1), "live", sum(1 for b in bodies if b["live"]))

# key the bodies, remove the hide drivers
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
for b in bodies:
    o = b["ob"]
    for prop in ("hide_render", "hide_viewport"):
        try:
            o.driver_remove(prop)
        except Exception:
            pass
        setattr(o, prop, False)
    if o.animation_data and o.animation_data.action:
        o.animation_data.action = None
    M0 = b["M0"]
    l, q, s = M0.decompose()
    o.location, o.rotation_quaternion, o.scale = l, q, s
    o.keyframe_insert("location", frame=1)
    o.keyframe_insert("rotation_quaternion", frame=1)
    o.keyframe_insert("scale", frame=1)
    prev = q
    for f, M in b["track"]:
        l, q, s = M.decompose()
        if prev.dot(q) < 0:
            q.negate()
        prev = q.copy()
        o.location, o.rotation_quaternion, o.scale = l, q, s
        o.keyframe_insert("location", frame=f)
        o.keyframe_insert("rotation_quaternion", frame=f)
        o.keyframe_insert("scale", frame=f)
sc.frame_set(1)
fin = {b["name"]: tuple(round(x, 2) for x in b["p"]) for b in bodies}
log("final positions sample", list(fin.items())[:6])
speeds = [round(b["v"].length, 2) for b in bodies]
log("max final speed", max(speeds))
bpy.ops.wm.save_as_mainfile(filepath=r"C:\ai\unedited\relic_v13_rocks.blend")
log("saved")
LOG.close()
