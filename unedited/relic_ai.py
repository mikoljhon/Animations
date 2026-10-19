# relic_ai.py : rig + extras for the AI-generated (Tripo) Relic mesh named "Relic_AI".
# Exec after relic_build.py (reuses SCENE/COLL, render_shot, make_mat ...).
import bpy, bmesh, math
import numpy as np
from mathutils import Vector, Matrix

AI_OBJ = "Relic_AI"
AI_RIG = "Relic_AI_Rig"


def ai_bones():
    """(name, head, tail, parent, radius). Left (+X) values measured on the mesh, right = mirrored."""
    B = [
        ("root", (0, 0, 0), (0, -0.25, 0), None, 0.0),
        ("pelvis", (0, -0.10, 1.02), (0, -0.10, 1.22), "root", 0.24),
        ("spine", (0, -0.10, 1.22), (0, -0.08, 1.48), "pelvis", 0.28),
        ("chest", (0, -0.08, 1.48), (0, -0.05, 1.78), "spine", 0.32),
        ("neck", (0, -0.05, 1.78), (0, -0.06, 1.86), "chest", 0.10),
        ("head", (0, -0.06, 1.86), (0, -0.075, 2.10), "neck", 0.20),
    ]
    L = {
        "clavicle": ((0.06, -0.04, 1.72), (0.36, 0.02, 1.62), "chest", 0.14),
        "upperarm": ((0.36, 0.02, 1.62), (0.45, 0.0, 1.33), "clavicle", 0.10),
        "forearm": ((0.45, 0.0, 1.33), (0.47, -0.115, 1.0), "upperarm", 0.09),
        "hand": ((0.47, -0.115, 1.0), (0.47, -0.13, 0.78), "forearm", 0.10),
        "thigh": ((0.26, -0.07, 1.02), (0.31, -0.15, 0.66), "pelvis", 0.15),
        "shin": ((0.31, -0.15, 0.66), (0.335, 0.03, 0.20), "thigh", 0.12),
        "foot": ((0.335, 0.03, 0.20), (0.35, -0.10, 0.05), "shin", 0.15),
        "toe": ((0.35, -0.10, 0.05), (0.36, -0.25, 0.05), "foot", 0.10),
    }
    for side, sx in (("L", 1), ("R", -1)):
        for nm, (h, t, par, r) in L.items():
            parent = (par + "." + side) if par not in ("chest", "pelvis") else par
            B.append((nm + "." + side, (h[0] * sx, h[1], h[2]), (t[0] * sx, t[1], t[2]), parent, r))
        B.append(("ik_foot." + side, (0.335 * sx, 0.03, 0.20), (0.35 * sx, -0.10, 0.05), "root", 0.0))
        B.append(("ik_knee." + side, (0.31 * sx, -0.90, 0.66), (0.31 * sx, -0.90, 0.76), "root", 0.0))
    return B


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def ai_rig(k=2.5, top=4):
    scn = bpy.data.scenes[SCENE]
    coll = bpy.data.collections["Relic_AI"]
    ob = bpy.data.objects[AI_OBJ]
    for o in list(bpy.data.objects):
        if o.name == AI_RIG:
            bpy.data.objects.remove(o, do_unlink=True)
    arm = bpy.data.armatures.new(AI_RIG)
    rig = bpy.data.objects.new(AI_RIG, arm)
    coll.objects.link(rig)
    vl = scn.view_layers[0]
    win = bpy.context.window_manager.windows[0]
    defs = ai_bones()
    with bpy.context.temp_override(window=win, scene=scn, view_layer=vl, active_object=rig, object=rig):
        vl.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        eb = arm.edit_bones
        for name, head, tail, parent, r in defs:
            b = eb.new(name)
            b.head = head
            b.tail = tail
        for name, head, tail, parent, r in defs:
            if parent:
                eb[name].parent = eb[parent]
                if (Vector(eb[parent].tail) - Vector(eb[name].head)).length < 1e-4:
                    eb[name].use_connect = True
        bpy.ops.object.mode_set(mode='OBJECT')
    arm.display_type = 'OCTAHEDRAL'
    rig.show_in_front = False

    # ---- weights: smooth falloff by distance to each deforming bone, one side only for limbs
    me = ob.data
    N = len(me.vertices)
    V = np.empty(N * 3, dtype=np.float32)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3).astype(np.float64)
    deform = [d for d in defs if d[4] > 0.0]
    names = [d[0] for d in deform]
    W = np.zeros((N, len(deform)))
    # surface graph so influence flows along the mesh, not through the air between neighbouring limbs
    NE = len(me.edges)
    E = np.empty(NE * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    adj = [[] for _ in range(N)]
    for (a, b), l in zip(E.tolist(), el.tolist()):
        adj[a].append((b, l))
        adj[b].append((a, l))
    import heapq
    for j, (name, head, tail, parent, r) in enumerate(deform):
        h_, t_ = np.array(head, float), np.array(tail, float)
        d = seg_dist(V, h_, t_)
        side_ok = np.ones(N, bool)
        if name.endswith(".L"):
            side_ok = V[:, 0] > -0.04
        elif name.endswith(".R"):
            side_ok = V[:, 0] < 0.04
        seeds = np.where((d < 0.8 * r) & side_ok)[0]
        if len(seeds) == 0:
            seeds = np.where(side_ok)[0][np.argsort(d[side_ok])[:50]]
        dist = np.full(N, np.inf)
        heap = []
        for s_ in seeds.tolist():
            dist[s_] = d[s_]
            heap.append((d[s_], s_))
        heapq.heapify(heap)
        cutoff = 3.0 * r
        while heap:
            dv, v = heapq.heappop(heap)
            if dv > dist[v] or dv > cutoff:
                continue
            for u, l in adj[v]:
                nd = dv + l
                if nd < dist[u] and nd <= cutoff:
                    dist[u] = nd
                    heapq.heappush(heap, (nd, u))
        w = np.where(np.isfinite(dist), np.exp(-k * (dist / r) ** 2), 0.0)
        w[~side_ok] = 0.0
        W[:, j] = w
    # keep the strongest influences per vertex
    order = np.argsort(-W, axis=1)
    keep = np.zeros_like(W)
    rows = np.arange(N)
    for t in range(top):
        keep[rows, order[:, t]] = W[rows, order[:, t]]
    s = keep.sum(axis=1, keepdims=True)
    s[s < 1e-9] = 1.0
    keep = keep / s
    # vertices the surface graph never reached (loose shells, bolts, slivers): copy the weights of the nearest reached vertex
    far = (W.max(axis=1) < 1e-4)
    if far.any():
        ridx = np.where(~far)[0]
        cell = 0.03
        grid = {}
        keys = np.floor(V[ridx] / cell).astype(int)
        for i, kk in zip(ridx.tolist(), map(tuple, keys.tolist())):
            grid.setdefault(kk, []).append(i)
        for fi in np.where(far)[0].tolist():
            c0 = np.floor(V[fi] / cell).astype(int)
            best, bd = None, 1e9
            for rad in (1, 2, 4, 8):
                cand = []
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        for dz in range(-rad, rad + 1):
                            cand += grid.get((c0[0] + dx, c0[1] + dy, c0[2] + dz), [])
                if cand:
                    cand = np.array(cand)
                    dd = np.linalg.norm(V[cand] - V[fi], axis=1)
                    j = int(dd.argmin())
                    best, bd = int(cand[j]), float(dd[j])
                    break
            if best is not None:
                keep[fi] = keep[best]

    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    levels = 32
    q = np.rint(keep * levels).astype(int)
    for j, name in enumerate(names):
        vg = ob.vertex_groups.new(name=name)
        for lv in range(1, levels + 1):
            idx = np.where(q[:, j] == lv)[0]
            if len(idx):
                vg.add(idx.tolist(), lv / levels, 'REPLACE')
    for m in list(ob.modifiers):
        if m.type == 'ARMATURE':
            ob.modifiers.remove(m)
    md = ob.modifiers.new("Armature", 'ARMATURE')
    md.object = rig
    ob.parent = rig
    return {"bones": len(arm.bones), "deform_bones": len(deform), "verts": N, "unreached": int(far.sum())}


def _color_image(me):
    for n in me.materials[0].node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image and n.image.name.startswith("Color"):
            return n.image
    return None


def ai_visor_paint(center=(0.0, -0.235, 1.862), radius=0.078, dark=(0.012, 0.013, 0.016)):
    """darken the baked porthole glass and eye beads in the base-colour texture (the new eyes glow on a dark glass)"""
    ob = bpy.data.objects[AI_OBJ]
    me = ob.data
    N = len(me.vertices)
    V = np.empty(N * 3, dtype=np.float32)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    c = np.array(center, np.float32)
    sel = (np.hypot(V[:, 0] - c[0], V[:, 2] - c[2]) < radius) & (V[:, 1] < -0.17) & (V[:, 1] > -0.30)
    img = _color_image(me)
    W, H = img.size
    px = np.empty(W * H * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tv = np.empty(nt * 3, dtype=np.int32)
    tl = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tv)
    me.loop_triangles.foreach_get("loops", tl)
    tv, tl = tv.reshape(-1, 3), tl.reshape(-1, 3)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers[0].uv.foreach_get("vector", uv)
    uv = uv.reshape(-1, 2)
    use = np.where(sel[tv].all(axis=1))[0]
    painted = 0
    d = np.array(dark + (1.0,), np.float32)
    for t in use.tolist():
        p = uv[tl[t]] * np.array([W - 1, H - 1], np.float32)
        x0, x1 = int(np.floor(p[:, 0].min())) - 1, int(np.ceil(p[:, 0].max())) + 1
        y0, y1 = int(np.floor(p[:, 1].min())) - 1, int(np.ceil(p[:, 1].max())) + 1
        if x1 - x0 > 200 or y1 - y0 > 200:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a, b, c3 = p
        den = (b[1] - c3[1]) * (a[0] - c3[0]) + (c3[0] - b[0]) * (a[1] - c3[1])
        if abs(den) < 1e-9:
            continue
        l1 = ((b[1] - c3[1]) * (xs - c3[0]) + (c3[0] - b[0]) * (ys - c3[1])) / den
        l2 = ((c3[1] - a[1]) * (xs - c3[0]) + (a[0] - c3[0]) * (ys - c3[1])) / den
        l3 = 1.0 - l1 - l2
        m = (l1 >= -0.05) & (l2 >= -0.05) & (l3 >= -0.05)
        yy, xx = ys[m], xs[m]
        ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        px[yy[ok], xx[ok]] = d
        painted += int(ok.sum())
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    return {"glass_verts": int(sel.sum()), "triangles": int(len(use)), "pixels": painted}


def _bind(ob, bone, rig):
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    vg = ob.vertex_groups.new(name=bone)
    vg.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
    for m in list(ob.modifiers):
        ob.modifiers.remove(m)
    md = ob.modifiers.new("Armature", 'ARMATURE')
    md.object = rig
    ob.parent = rig


def ai_extras():
    """cyan chest ring (Tally) on the yoke and glowing eyes with expression shape keys"""
    scn = bpy.data.scenes[SCENE]
    coll = bpy.data.collections["Relic_AI"]
    rig = bpy.data.objects[AI_RIG]
    body = bpy.data.objects[AI_OBJ]
    me0 = body.data
    N = len(me0.vertices)
    V = np.empty(N * 3, dtype=np.float32)
    me0.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    for o in list(coll.objects):
        if o.name.startswith("Relic_AI_Eyes") or o.name.startswith("Relic_AI_Ring"):
            bpy.data.objects.remove(o, do_unlink=True)
    gm = bpy.data.materials["Relic_Glyph"]
    gb = next(n for n in gm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    gb.inputs['Roughness'].default_value = 0.12
    gb.inputs['Emission Color'].default_value = (1.0, 0.52, 0.05, 1.0)
    gb.inputs['Emission Strength'].default_value = 2.8

    def mk(name, bm, mat, bone):
        for f in bm.faces:
            f.smooth = True
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        o = bpy.data.objects.new(name, me)
        coll.objects.link(o)
        _bind(o, bone, rig)
        return o

    # --- chest ring, centred on the smooth middle of the yoke
    zc, xc = 1.652, 0.0
    near = (np.abs(V[:, 0] - xc) < 0.02) & (np.abs(V[:, 2] - zc) < 0.02)
    yf = float(V[near, 1].min()) if near.any() else -0.31
    ring_m = bpy.data.materials["Relic_RingLight"]
    gun = bpy.data.materials["Relic_Gunmetal"]
    dark = bpy.data.materials["Relic_Dark"]
    mk("Relic_AI_RingBezel", bm_torus((xc, yf + 0.004, zc), 0.058, 0.058, 0.0075, 'Y', 56, 10), gun, "chest")
    mk("Relic_AI_RingCore", bm_cyl((xc, yf - 0.002, zc), 0.046, 0.006, 'Y', 40), dark, "chest")
    mk("Relic_AI_Ring", bm_torus((xc, yf - 0.007, zc), 0.043, 0.043, 0.0085, 'Y', 56, 12), ring_m, "chest")

    # --- eyes
    ey = -0.262
    ez = 1.862
    EX = 0.0335
    pts = patch_local(0.0245, 0.0245, 2.0, 3, 28)
    fcs = patch_faces(3, 28)
    n1 = len(pts)

    def coords(kind):
        out = []
        for ex in (EX, -EX):
            inner = -1.0 if ex > 0 else 1.0
            for lx, lz in pts:
                x, z = lx, lz
                if kind == 'Blink':
                    z = lz * 0.10
                elif kind == 'Wide':
                    x, z = lx * 1.15, lz * 1.25
                elif kind == 'Angry':
                    z = lz * 0.75 - 0.45 * (lx * inner) - 0.004
                elif kind == 'Sad':
                    z = lz * 0.85 + 0.45 * (lx * inner) - 0.001
                elif kind == 'Happy':
                    z = lz * 0.50 - 22.0 * lx * lx + 0.006
                out.append((ex + x, ey, ez + z))
        return out

    faces_all = list(fcs) + [tuple(i + n1 for i in f) for f in fcs]
    me = bpy.data.meshes.new("Relic_AI_Eyes")
    me.from_pydata(coords('Neutral'), [], faces_all)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(bpy.data.materials["Relic_Glyph"])
    eo = bpy.data.objects.new("Relic_AI_Eyes", me)
    coll.objects.link(eo)
    eo.shape_key_add(name="Basis")
    for kind in ("Blink", "Wide", "Angry", "Sad", "Happy"):
        kb = eo.shape_key_add(name=kind)
        kb.value = 0.0
        for i, c in enumerate(coords(kind)):
            kb.data[i].co = c
    _bind(eo, "head", rig)
    return {"ring_front_y": round(yf, 4)}


def ai_ik():
    scn = bpy.data.scenes[SCENE]
    rig = bpy.data.objects[AI_RIG]
    vl = scn.view_layers[0]
    for side in ("L", "R"):
        pb = rig.pose.bones["shin." + side]
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        con = pb.constraints.new('IK')
        con.target = rig
        con.subtarget = "ik_foot." + side
        con.pole_target = rig
        con.pole_subtarget = "ik_knee." + side
        con.chain_count = 2
        con.pole_angle = math.radians(-90)
        fb = rig.pose.bones["foot." + side]
        for c in list(fb.constraints):
            fb.constraints.remove(c)
        cc = fb.constraints.new('COPY_ROTATION')
        cc.target = rig
        cc.subtarget = "ik_foot." + side
        cc.target_space = 'WORLD'
        cc.owner_space = 'WORLD'

    def knee_y(angle):
        rig.pose.bones["shin.L"].constraints[0].pole_angle = math.radians(angle)
        rig.pose.bones["ik_foot.L"].matrix = Matrix.Translation((0, -0.05, 0.25)) @ rig.pose.bones["ik_foot.L"].matrix
        vl.update()
        y = (rig.matrix_world @ rig.pose.bones["shin.L"].head).y
        rig.pose.bones["ik_foot.L"].location = (0, 0, 0)
        vl.update()
        return y
    res = {a: round(knee_y(a), 3) for a in (-90, 0, 90, 180)}
    best = min(res, key=lambda a: res[a])
    for side in ("L", "R"):
        rig.pose.bones["shin." + side].constraints[0].pole_angle = math.radians(best)
    vl.update()
    return {"knee_y_by_angle": res, "pole_angle": best}
