# relic_model3.py : Relic v03 = reference-driven redesign: heavy black-chrome armor, gold caps, porthole visor.
# Exec AFTER relic_build.py and relic_model2.py in the same namespace (reuses their helpers).
# It overrides the layout functions (arm_pts, leg_pts, bone_defs, head constants) used by stage_rig.
import types

HEAD_C = Vector((0.0, 0.0, 1.86))
HEAD_R = (0.165, 0.180, 0.215)
HEAD_LOW = 0.06
HS = 2.0

FINGERS = [
    ("index", -0.027, 0.040, 0.033, 0.0115),
    ("middle", -0.009, 0.044, 0.036, 0.0115),
    ("ring", 0.009, 0.040, 0.033, 0.0110),
    ("pinky", 0.026, 0.033, 0.027, 0.0100),
]


def arm_pts(sx):
    S = Vector((sx * 0.37, 0.0, 1.60))
    t1, t2 = math.radians(18), math.radians(10)
    E = S + Vector((sx * 0.34 * math.sin(t1), 0.0, -0.34 * math.cos(t1)))
    W = E + Vector((sx * 0.32 * math.sin(t2), 0.0, -0.32 * math.cos(t2)))
    return S, E, W, t2


def leg_pts(sx):
    H = Vector((sx * 0.135, 0.0, 0.95))
    K = Vector((sx * 0.135, -0.02, 0.54))
    A = Vector((sx * 0.135, 0.0, 0.155))
    B = Vector((sx * 0.135, -0.125, 0.05))
    T = Vector((sx * 0.135, -0.25, 0.05))
    return H, K, A, B, T


def bone_defs():
    B = []
    B.append(("root", (0, 0, 0), (0, -0.25, 0), None))
    B.append(("pelvis", (0, 0, 0.98), (0, 0, 1.10), "root"))
    B.append(("spine", (0, 0, 1.10), (0, 0, 1.30), "pelvis"))
    B.append(("chest", (0, 0, 1.30), (0, 0, 1.72), "spine"))
    B.append(("neck", (0, 0, 1.72), (0, 0, 1.80), "chest"))
    B.append(("head", (0, 0, 1.80), (0, 0, 2.10), "neck"))
    for side, sx in (("L", 1), ("R", -1)):
        S, E, W, t2 = arm_pts(sx)
        Mh = hand_frame(sx)
        B.append(("clavicle." + side, (sx * 0.09, -0.01, 1.64), tuple(S), "chest"))
        B.append(("upperarm." + side, tuple(S), tuple(E), "clavicle." + side))
        B.append(("forearm." + side, tuple(E), tuple(W), "upperarm." + side))
        B.append(("hand." + side, tuple(W), tuple(Mh @ Vector((0, 0, -0.09))), "forearm." + side))
        for f in FINGERS:
            p0, p1, p2 = finger_pts(sx, f)
            B.append(("%s_1.%s" % (f[0], side), tuple(Mh @ p0), tuple(Mh @ p1), "hand." + side))
            B.append(("%s_2.%s" % (f[0], side), tuple(Mh @ p1), tuple(Mh @ p2), "%s_1.%s" % (f[0], side)))
        p0, p1, p2 = thumb_pts(sx)
        B.append(("thumb_1." + side, tuple(Mh @ p0), tuple(Mh @ p1), "hand." + side))
        B.append(("thumb_2." + side, tuple(Mh @ p1), tuple(Mh @ p2), "thumb_1." + side))
        H, K, A, Bl, T = leg_pts(sx)
        B.append(("thigh." + side, tuple(H), tuple(K), "pelvis"))
        B.append(("shin." + side, tuple(K), tuple(A), "thigh." + side))
        B.append(("foot." + side, tuple(A), tuple(Bl), "shin." + side))
        B.append(("toe." + side, tuple(Bl), tuple(T), "foot." + side))
        B.append(("ik_foot." + side, tuple(A), tuple(Bl), "root"))
        B.append(("ik_knee." + side, (sx * 0.135, -0.90, 0.54), (sx * 0.135, -0.90, 0.64), "root"))
    return B


# ------------------------------------------------------------------ extra geometry
def bm_cone(c, r1, r2, h, axis='Z', segs=16):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r1, radius2=r2, depth=h)
    bm.transform(Matrix.Translation(Vector(c)) @ rot_z_to(axis))
    return bm


def bm_loop_tube(points, radius, sides=8):
    bm = bmesh.new()
    n = len(points)
    tans = [(points[(i + 1) % n] - points[(i - 1) % n]).normalized() for i in range(n)]
    ref = Vector((0, 1, 0)) if abs(tans[0].y) < 0.9 else Vector((1, 0, 0))
    nrm = tans[0].cross(ref).normalized()
    rings = []
    for i, p in enumerate(points):
        t = tans[i]
        nrm = (nrm - t * nrm.dot(t)).normalized()
        bi = t.cross(nrm)
        rings.append([bm.verts.new(p + (nrm * math.cos(2 * math.pi * k / sides) + bi * math.sin(2 * math.pi * k / sides)) * radius)
                      for k in range(sides)])
    best, bs = 1e9, 0
    for s in range(sides):
        d = sum((rings[-1][k].co - rings[0][(k + s) % sides].co).length for k in range(sides))
        if d < best:
            best, bs = d, s
    for i in range(n):
        a = rings[i]
        if i < n - 1:
            b, sh = rings[i + 1], 0
        else:
            b, sh = rings[0], bs
        for k in range(sides):
            bm.faces.new((a[k], a[(k + 1) % sides], b[((k + 1) % sides + sh) % sides], b[(k + sh) % sides]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _nh(nt):
    N, L = nt.nodes, nt.links
    tc = N.new('ShaderNodeTexCoord')

    def mapping(sc):
        mp = N.new('ShaderNodeMapping')
        mp.inputs['Scale'].default_value = sc
        L.new(tc.outputs['Object'], mp.inputs['Vector'])
        return mp

    def noise(mp, scale, detail, rgh=0.6):
        nz = N.new('ShaderNodeTexNoise')
        nz.inputs['Scale'].default_value = scale
        nz.inputs['Detail'].default_value = detail
        nz.inputs['Roughness'].default_value = rgh
        L.new(mp.outputs['Vector'], nz.inputs['Vector'])
        return nz.outputs['Fac']

    def ramp(src, lo, hi):
        cr = N.new('ShaderNodeValToRGB')
        cr.color_ramp.elements[0].position = lo
        cr.color_ramp.elements[1].position = hi
        L.new(src, cr.inputs['Fac'])
        return cr.outputs['Color']

    def fm(op, a, b2=None, clamp=False):
        mn = N.new('ShaderNodeMath')
        mn.operation = op
        mn.use_clamp = clamp
        for i, x in enumerate((a, b2)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                mn.inputs[i].default_value = x
            else:
                L.new(x, mn.inputs[i])
        return mn.outputs['Value']

    def mixc(fac, a, b2):
        mx = N.new('ShaderNodeMix')
        mx.data_type = 'RGBA'
        for idx, x in ((6, a), (7, b2)):
            if isinstance(x, tuple):
                mx.inputs[idx].default_value = (x[0], x[1], x[2], 1.0)
            else:
                L.new(x, mx.inputs[idx])
        L.new(fac, mx.inputs[0])
        return mx.outputs[2]

    def mixf(fac, a, b2):
        mx = N.new('ShaderNodeMix')
        mx.data_type = 'FLOAT'
        for idx, x in ((2, a), (3, b2)):
            if isinstance(x, (int, float)):
                mx.inputs[idx].default_value = x
            else:
                L.new(x, mx.inputs[idx])
        L.new(fac, mx.inputs[0])
        return mx.outputs[0]

    return types.SimpleNamespace(N=N, L=L, tc=tc, mapping=mapping, noise=noise, ramp=ramp, fm=fm, mixc=mixc, mixf=mixf)


def make_chrome(name, base, rough=0.2, edge=(0.38, 0.39, 0.41), edge_gain=1.0, grime=(0.006, 0.006, 0.008),
                grime_amt=0.22, scratch_amt=0.16, metallic=1.0):
    """polished metal: brighter worn edges, fine scratches that roughen the surface, dirt in crevices"""
    m = make_mat(name, base, metallic, rough)
    try:
        nt, b = _principled(m)
        if not b:
            return m
        h = _nh(nt)
        N, L = h.N, h.L
        wear = N.new('ShaderNodeAttribute')
        wear.attribute_type = 'GEOMETRY'
        wear.attribute_name = 'wear'
        brk = h.ramp(h.noise(h.mapping((1, 1, 1)), 30.0, 4), 0.25, 0.60)
        edge_m = h.fm('MULTIPLY', h.fm('MULTIPLY', wear.outputs['Fac'], brk), edge_gain, clamp=True)
        s1 = h.ramp(h.noise(h.mapping((170, 9, 9)), 1.0, 1.0), 0.617, 0.627)
        s2 = h.ramp(h.noise(h.mapping((9, 170, 9)), 1.0, 1.0), 0.617, 0.627)
        scr = h.fm('MULTIPLY', h.fm('ADD', s1, s2, clamp=True), scratch_amt)
        light = h.fm('MAXIMUM', edge_m, scr)
        smudge = h.fm('MULTIPLY', h.ramp(h.noise(h.mapping((1, 1, 0.3)), 9.0, 6), 0.52, 0.80), grime_amt)
        g = smudge
        try:
            ao = N.new('ShaderNodeAmbientOcclusion')
            ao.inputs['Distance'].default_value = 0.05
            crev = h.fm('MULTIPLY', h.fm('SUBTRACT', 1.0, ao.outputs['AO'], clamp=True), 0.85)
            g = h.fm('MAXIMUM', smudge, crev)
        except Exception as e:
            NOTES.append("AO node: %s" % e)
        col = h.mixc(light, base, edge)
        col = h.mixc(g, col, grime)
        L.new(col, b.inputs['Base Color'])
        r1 = h.mixf(light, rough, min(0.9, rough + 0.16))
        L.new(h.mixf(g, r1, min(0.9, rough + 0.14)), b.inputs['Roughness'])
    except Exception as e:
        NOTES.append("make_chrome failed for %s: %s" % (name, e))
    return m


def setup_studio():
    """studio world: dark backdrop for the camera, bright soft-boxes that only show up in reflections and lighting"""
    scn = bpy.data.scenes[SCENE]
    w = scn.world
    nt = getattr(w, "node_tree", None)
    if nt is None:
        NOTES.append("no world node tree")
        return
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputWorld')
    bg = N.new('ShaderNodeBackground')
    L.new(bg.outputs['Background'], out.inputs['Surface'])
    tc = N.new('ShaderNodeTexCoord')
    sep = N.new('ShaderNodeSeparateXYZ')
    L.new(tc.outputs['Generated'], sep.inputs['Vector'])
    mr = N.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = -1.0
    mr.inputs['From Max'].default_value = 1.0
    L.new(sep.outputs['Z'], mr.inputs['Value'])
    cr = N.new('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].position = 0.38
    cr.color_ramp.elements[0].color = (0.008, 0.010, 0.013, 1.0)
    cr.color_ramp.elements[1].position = 0.70
    cr.color_ramp.elements[1].color = (0.09, 0.11, 0.135, 1.0)
    L.new(mr.outputs['Result'], cr.inputs['Fac'])
    acc = cr.outputs['Color']
    boxes = [((-0.50, -0.62, 0.60), 0.93, 0.965, (1.0, 0.95, 0.88), 16.0),
             ((0.62, 0.62, 0.45), 0.95, 0.975, (0.70, 0.90, 1.0), 12.0),
             ((0.90, -0.25, 0.15), 0.92, 0.95, (0.90, 0.95, 1.0), 4.0),
             ((-0.10, 0.0, 1.0), 0.90, 0.95, (1.0, 1.0, 1.0), 5.0),
             ((-0.85, 0.45, 0.25), 0.95, 0.975, (1.0, 0.70, 0.40), 7.0)]
    for d, c0, c1, col, s in boxes:
        d = Vector(d).normalized()
        dot = N.new('ShaderNodeVectorMath')
        dot.operation = 'DOT_PRODUCT'
        dot.inputs[1].default_value = (d.x, d.y, d.z)
        L.new(tc.outputs['Generated'], dot.inputs[0])
        r = N.new('ShaderNodeValToRGB')
        r.color_ramp.elements[0].position = c0
        r.color_ramp.elements[1].position = c1
        L.new(dot.outputs['Value'], r.inputs['Fac'])
        mul = N.new('ShaderNodeVectorMath')
        mul.operation = 'MULTIPLY'
        mul.inputs[1].default_value = (col[0] * s, col[1] * s, col[2] * s)
        L.new(r.outputs['Color'], mul.inputs[0])
        ad = N.new('ShaderNodeVectorMath')
        ad.operation = 'ADD'
        L.new(acc, ad.inputs[0])
        L.new(mul.outputs['Vector'], ad.inputs[1])
        acc = ad.outputs['Vector']
    lp = N.new('ShaderNodeLightPath')
    mx = N.new('ShaderNodeMix')
    mx.data_type = 'RGBA'
    L.new(lp.outputs['Is Camera Ray'], mx.inputs[0])
    L.new(acc, mx.inputs[6])
    mx.inputs[7].default_value = (0.030, 0.038, 0.048, 1.0)
    L.new(mx.outputs[2], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = 1.0
    for prop, val in (("use_raytracing", True), ("ray_tracing_method", 'SCREEN')):
        try:
            if hasattr(scn.eevee, prop):
                setattr(scn.eevee, prop, val)
        except Exception as e:
            NOTES.append("eevee %s: %s" % (prop, e))
    try:
        scn.view_settings.exposure = -0.3
    except Exception:
        pass
    make_mat("Relic_FloorMat", (0.018, 0.020, 0.024), 0.0, 0.5)
    for o in bpy.data.collections[COLL].objects:
        if o.type == 'LIGHT':
            o.data.energy = {"Key": 600, "Fill": 160, "Rim": 500, "Rim2": 260}.get(o.name, o.data.energy)


# ------------------------------------------------------------------------ the model
def stage_model3():
    scn = bpy.data.scenes[SCENE]
    coll = bpy.data.collections[COLL]
    for o in list(coll.objects):
        if o.type == 'MESH' and o.get("relic_part"):
            bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)

    chrome = make_chrome("Relic_Chrome", (0.17, 0.175, 0.19), rough=0.20)
    chrome2 = make_chrome("Relic_ChromeDark", (0.075, 0.078, 0.086), rough=0.27)
    gold = make_chrome("Relic_Gold", (0.82, 0.52, 0.13), rough=0.25, edge=(1.0, 0.80, 0.34), grime=(0.07, 0.035, 0.008), grime_amt=0.3, scratch_amt=0.12)
    gun = make_chrome("Relic_Gunmetal", (0.040, 0.042, 0.048), rough=0.42, edge=(0.22, 0.23, 0.25), scratch_amt=0.15)
    rubber = make_mat("Relic_Rubber", (0.012, 0.013, 0.015), 0.0, 0.45)
    braid = make_mat("Relic_Braid", (0.30, 0.15, 0.06), 0.9, 0.45)
    wire_t = make_mat("Relic_WireTeal", (0.02, 0.22, 0.24), 0.0, 0.35)
    brass = make_mat("Relic_Brass", (0.40, 0.33, 0.20), 0.9, 0.38)
    glass = make_mat("Relic_Glass", (0.002, 0.002, 0.003), 0.0, 0.04)
    glyph_m = make_mat("Relic_Glyph", (0.0, 0.0, 0.0), 0.0, 0.5, emit=(1.0, 0.36, 0.0), emit_strength=1.7)
    ring_m = make_mat("Relic_RingLight", (0.0, 0.0, 0.0), 0.0, 0.5, emit=(0.0, 0.62, 1.0), emit_strength=3.0)
    dark = make_mat("Relic_Dark", (0.006, 0.007, 0.009), 0.3, 0.4)
    patchm = make_worn("Relic_Patch", (0.02, 0.09, 0.105), (0.15, 0.15, 0.16), (0.14, 0.05, 0.016), rough=0.5, edge_gain=1.6)

    def finish(name, me, mat, bone):
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me)
        ob["relic_part"] = True
        ob["bone"] = bone
        coll.objects.link(ob)
        return ob

    def add(name, bm, mat, bone, sharp=38, wear=True):
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(sharp):
                e.smooth = False
        for f in bm.faces:
            f.smooth = True
        wv = curvature_vals(bm) if wear else None
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        if wv is not None:
            at = me.attributes.new("wear", 'FLOAT', 'POINT')
            at.data.foreach_set("value", [max(0.0, min(1.0, (c - 24.0) / 36.0)) for c in wv])
        return finish(name, me, mat, bone)

    def add_patch(name, coords, faces, mat, bone):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(c) for c in coords], [], faces)
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        return finish(name, me, mat, bone)

    def add_hose(name, p0, c, p2, radius, bone_a, bone_b, mat, blend=(0.35, 0.65), n=22, sides=10, fer=None, fr=1.5):
        P0, C, P2 = Vector(p0), Vector(c), Vector(p2)
        pts = bezier2(P0, C, P2, n)
        bm, ts = bm_tube(pts, radius, sides, True)
        ob = add(name, bm, mat, bone_a, wear=False)
        ob["bones"] = [bone_a, bone_b]
        ob["wts"] = [1.0 - sstep((t - blend[0]) / (blend[1] - blend[0])) for t in ts]
        f = fer if fer is not None else gun
        t0 = (pts[1] - pts[0]).normalized()
        t1 = (pts[-1] - pts[-2]).normalized()
        add(name + "_F0", bm_cyl(pts[0] + t0 * 0.010, radius * fr, 0.024, t0, 16), f, bone_a, wear=False)
        add(name + "_F1", bm_cyl(pts[-1] - t1 * 0.010, radius * fr, 0.024, t1, 16), f, bone_b, wear=False)
        return ob

    def bolt_ring(name, center, axis, radius, n, rb, mat, bone, a0=0.0):
        ax = Vector(axis).normalized()
        ref = Vector((0, 0, 1)) if abs(ax.z) < 0.9 else Vector((1, 0, 0))
        u = ax.cross(ref).normalized()
        v = ax.cross(u).normalized()
        for i in range(n):
            a = a0 + 2 * math.pi * i / n
            add(name, bm_sphere(Vector(center) + (u * math.cos(a) + v * math.sin(a)) * radius, rb, 10, 6), mat, bone, wear=False)

    # ============================================================ head
    rx, ry, rz = HEAD_R
    zc = HEAD_C.z
    base_z = zc - HEAD_LOW
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co.x, v.co.y, v.co.z
        if z >= -1e-9:
            v.co = Vector((x * rx, y * ry, zc + z * rz))
        else:
            r = math.hypot(x, y)
            if r > 1e-9:
                x, y = x / r, y / r
            v.co = Vector((x * rx, y * ry, zc + z * HEAD_LOW))
    add("Head_Dome", bm, chrome, "head")
    add("Head_Collar", bm_cyl((0, 0, base_z), 1.0, 0.046, 'Z', 48, (rx * 1.09, ry * 1.09)), gun, "head")
    add("Head_CollarTrim", bm_torus((0, 0, base_z + 0.026), rx * 1.09, ry * 1.09, 0.0095, 'Z', 64, 10), gold, "head")
    add("Head_CollarTrimLow", bm_torus((0, 0, base_z - 0.024), rx * 1.08, ry * 1.08, 0.0075, 'Z', 64, 10), gold, "head")
    for deg in (30, 90, 150, 210, 270, 330):
        th = math.radians(deg)
        lx = (rx * 1.09 + 0.010) * math.cos(th)
        ly = (ry * 1.09 + 0.010) * math.sin(th)
        add("Collar_Latch", bm_rbox((lx, ly, base_z), (0.030, 0.036, 0.062), 0.008, 2, rot=Matrix.Rotation(th, 4, 'Z')), gun, "head")
        add("Collar_Bolt", bm_sphere((lx + math.cos(th) * 0.016, ly + math.sin(th) * 0.016, base_z + 0.014), 0.0068, 10, 6), gold, "head", wear=False)
    for sx in (1, -1):
        ex = sx * (rx + 0.004)
        add("Head_Ear", bm_ellipsoid((ex, 0.0, base_z + 0.05), (0.030, 0.062, 0.062), 28, 14), gold, "head")
        bolt_ring("Ear_Bolt", (ex + sx * 0.022, 0.0, base_z + 0.05), (sx, 0, 0), 0.046, 8, 0.0058, chrome2, "head")
    for zz in (1.800, 1.828, 1.856):
        yy = ry * math.sqrt(max(0.0, 1.0 - (max(zz - zc, 0.0) / rz) ** 2))
        add("Head_Vent", bm_rbox((0, yy, zz), (0.090, 0.006, 0.005), 0.0015, 1), dark, "head", wear=False)

    # porthole visor: dark glass disc, raised gold rim, two round glowing eyes
    vz = zc + 0.030
    WR = 0.088
    pts = patch_local(WR, WR, 2.0, 4, 64)
    fcs = patch_faces(4, 64)
    pr_gl = head_proj(1.012)
    add_patch("Visor", [pr_gl(x, vz + z) for x, z in pts], fcs, glass, "head")
    pr_rim = head_proj(1.018)
    loop = [pr_rim(0.098 * math.cos(2 * math.pi * i / 56), vz + 0.098 * math.sin(2 * math.pi * i / 56)) for i in range(56)]
    add("Visor_Rim", bm_loop_tube(loop, 0.0095, 8), gold, "head", wear=False)
    eye_pts = patch_local(0.023, 0.023, 2.0, 3, 28)
    eye_faces = patch_faces(3, 28)
    n1 = len(eye_pts)
    pr_eye = head_proj(1.017)
    ez = vz + 0.002
    EX = 0.037

    def eye_coords(kind):
        out = []
        for ex in (EX, -EX):
            inner = -1.0 if ex > 0 else 1.0
            for lx, lz in eye_pts:
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
                out.append(pr_eye(ex + x, ez + z))
        return out

    faces_all = list(eye_faces) + [tuple(i + n1 for i in f) for f in eye_faces]
    eye_ob = add_patch("Eyes", eye_coords('Neutral'), faces_all, glyph_m, "head")
    eye_ob.shape_key_add(name="Basis")
    for kind in ("Blink", "Wide", "Angry", "Sad", "Happy"):
        kb = eye_ob.shape_key_add(name=kind)
        kb.value = 0.0
        for i, c in enumerate(eye_coords(kind)):
            kb.data[i].co = c

    # ============================================================ neck + torso
    add("Neck", bm_cyl((0, 0, 1.75), 0.062, 0.10, 'Z', 24), gun, "neck")
    add("Neck_Ring", bm_torus((0, 0, 1.742), 0.07, 0.07, 0.012, 'Z', 40, 10), gold, "neck", wear=False)

    YC, YS, YT = 1.66, (0.62, 0.34, 0.20), (0.80, 0.86)

    def yoke_front(z):
        t = max(0.0, min(1.0, (z - (YC - YS[2] / 2)) / YS[2]))
        return -(YS[1] / 2) * (YT[1] + (1 - YT[1]) * t)

    CC, CS, CT = 1.43, (0.46, 0.30, 0.32), (0.74, 0.84)

    def chest_y(z):
        t = max(0.0, min(1.0, (z - (CC - CS[2] / 2)) / CS[2]))
        return -(CS[1] / 2) * (CT[1] + (1 - CT[1]) * t)

    add("Yoke", bm_rbox((0, 0, YC), YS, 0.07, 5, taper=YT), chrome, "chest")
    add("Chest", bm_rbox((0, 0, CC), CS, 0.07, 5, taper=CT), chrome2, "chest")
    for sx in (1, -1):
        yp = yoke_front(1.645)
        add("Port_Housing", bm_cyl((sx * 0.18, yp - 0.004, 1.645), 0.052, 0.03, 'Y', 32), gun, "chest")
        add("Port_Ring", bm_torus((sx * 0.18, yp - 0.016, 1.645), 0.046, 0.046, 0.009, 'Y', 40, 10), gold, "chest", wear=False)
        add("Port_Glass", bm_cyl((sx * 0.18, yp - 0.0155, 1.645), 0.038, 0.004, 'Y', 28), glass, "chest", wear=False)
        bolt_ring("Port_Bolt", (sx * 0.18, yp - 0.018, 1.645), (0, 1, 0), 0.058, 6, 0.0055, chrome2, "chest")
    fy = chest_y(1.46)
    add("Chest_Housing", bm_cyl((0, fy - 0.003, 1.46), 0.068, 0.030, 'Y', 40), gun, "chest")
    add("Chest_Ring", bm_torus((0, fy - 0.0195, 1.46), 0.047, 0.047, 0.009, 'Y', 56, 12), ring_m, "chest", wear=False)
    add("Chest_RingCore", bm_cyl((0, fy - 0.0185, 1.46), 0.037, 0.004, 'Y', 40), dark, "chest", wear=False)
    bolt_ring("Ring_Bolt", (0, fy - 0.012, 1.46), (0, 1, 0), 0.080, 8, 0.0065, gold, "chest")
    py = chest_y(1.33)
    add("Serial_Plate", bm_rbox((0.13, py - 0.0015, 1.33), (0.075, 0.006, 0.040), 0.002, 2), brass, "chest")
    for i, zz in enumerate((1.341, 1.330, 1.319)):
        add("Serial_Line%d" % i, bm_rbox((0.127, py - 0.0045, zz), (0.050 - 0.008 * (i % 2), 0.002, 0.0035), 0.001, 1), dark, "chest", wear=False)
    add("Chest_Seam", bm_rbox((0, py - 0.001, 1.34), (0.004, 0.003, 0.14), 0.001, 1), dark, "chest", wear=False)

    cy = -chest_y(1.45)
    add("Back_Hatch", bm_rbox((0, cy, 1.43), (0.24, 0.020, 0.26), 0.012, 3), gun, "chest")
    add("Power_Cell", bm_rbox((0, cy + 0.045, 1.43), (0.22, 0.08, 0.30), 0.014, 3), chrome2, "chest")
    add("Cell_Band", bm_rbox((0, cy + 0.0855, 1.43), (0.225, 0.004, 0.05), 0.001, 1), gold, "chest", wear=False)
    for bx in (-0.05, 0.0, 0.05):
        add("Cell_Terminal", bm_cyl((bx, cy + 0.05, 1.596), 0.014, 0.034, 'Z', 16), gold, "chest", wear=False)

    # segmented abdomen
    for i, zz in enumerate((1.098, 1.140, 1.182, 1.224, 1.262)):
        d = abs(i - 2)
        add("Abdomen_Disc", bm_cyl((0, 0, zz), 1.0, 0.032, 'Z', 40, (0.120 - 0.005 * d, 0.097 - 0.004 * d)), chrome2, "spine")
        add("Abdomen_Ring", bm_torus((0, 0, zz + 0.019), 0.114 - 0.005 * d, 0.092 - 0.004 * d, 0.0045, 'Z', 40, 8), gold, "spine", wear=False)
    add("Abdomen_Core", bm_cyl((0, 0, 1.18), 1.0, 0.20, 'Z', 32, (0.090, 0.072)), dark, "spine", wear=False)
    for sx in (1, -1):
        add("Abdomen_Sleeve", cyl_between((sx * 0.150, -0.01, 1.10), (sx * 0.150, -0.01, 1.20), 0.021), gun, "pelvis")
        add("Abdomen_Rod", cyl_between((sx * 0.150, -0.01, 1.17), (sx * 0.150, -0.01, 1.285), 0.012), gold, "chest", wear=False)

    add("Pelvis", bm_rbox((0, 0, 0.99), (0.40, 0.26, 0.16), 0.05, 4), chrome2, "pelvis")
    add("Pelvis_Low", bm_rbox((0, 0, 0.915), (0.34, 0.22, 0.10), 0.03, 3), gun, "pelvis")
    add("Belt_Plate", bm_rbox((0, -0.132, 1.0), (0.12, 0.024, 0.105), 0.012, 3), gold, "pelvis")
    add("Belt_Core", bm_cyl((0, -0.1465, 1.0), 0.030, 0.005, 'Y', 24), dark, "pelvis", wear=False)
    for sx in (1, -1):
        add("Hip_Plate", bm_rbox((sx * 0.215, 0.0, 1.0), (0.055, 0.20, 0.15), 0.022, 3), chrome, "pelvis")

    # ============================================================ arms
    for side, sx in (("L", 1), ("R", -1)):
        S, E, W, t2 = arm_pts(sx)
        Mh = hand_frame(sx)
        bu, bf, bh, bc = "upperarm." + side, "forearm." + side, "hand." + side, "clavicle." + side
        d1 = (E - S).normalized()
        d2 = (W - E).normalized()
        add("Clavicle." + side, bm_capsule((sx * 0.10, -0.005, 1.62), S, 0.040), gun, bc)
        P = Vector((sx * 0.385, 0.0, 1.64))
        PR = (0.125, 0.110, 0.100)
        add("Pauldron." + side, bm_ellipsoid(P, PR, 44, 22), chrome, bc)
        add("PauldronCap." + side, bm_ellipsoid(Vector((sx * 0.395, 0.0, 1.715)), (0.108, 0.095, 0.046), 36, 16), chrome2, bc)
        add("PauldronGold." + side, bm_ellipsoid(Vector((sx * (0.385 + 0.118), 0.0, 1.64)), (0.030, 0.058, 0.058), 28, 14), gold, bc)
        bolt_ring("Pauldron_Bolt", (sx * (0.385 + 0.1385), 0.0, 1.64), (sx, 0, 0), 0.046, 8, 0.0058, chrome2, bc)
        add("ShoulderJoint." + side, bm_sphere(S, 0.072), gun, bu)
        add("UpperArm." + side, bm_capsule(S, E, 0.064), chrome, bu)
        add("ArmRingTop." + side, bm_torus(S + d1 * 0.075, 0.068, 0.068, 0.010, d1, 40, 10), gold, bu, wear=False)
        add("ArmRingMid." + side, bm_torus(S + (E - S) * 0.50, 0.067, 0.067, 0.008, d1, 40, 10), gun, bu, wear=False)
        add("ElbowJoint." + side, bm_sphere(E, 0.068), gun, bf)
        add("ElbowCap." + side, bm_ellipsoid(E + Vector((sx * 0.066, 0.0, 0.0)), (0.026, 0.056, 0.056), 28, 14), gold, bf)
        bolt_ring("Elbow_Bolt", E + Vector((sx * 0.085, 0.0, 0.0)), (sx, 0, 0), 0.042, 6, 0.0055, chrome2, bf)
        add("Forearm." + side, bm_capsule(E, W, 0.058), chrome, bf)
        gc = E + (W - E) * 0.46 + Vector((sx * 0.062, 0.0, 0.0))
        add("ForearmGuard." + side, bm_rbox(gc, (0.040, 0.100, 0.22), 0.016, 3, rot=rot_z_to(d2)), chrome2, bf)
        add("ForearmFront." + side, bm_rbox(E + (W - E) * 0.40 + Vector((0.0, -0.058, 0.0)), (0.075, 0.034, 0.17), 0.014, 3, rot=rot_z_to(d2)), chrome2, bf)
        add("WristRing." + side, bm_torus(W, 0.056, 0.056, 0.011, d2, 40, 10), gold, bh, wear=False)

        bm = bm_rbox((0, 0, -0.045), (0.034, 0.080, 0.092), 0.013, 3)
        bm.transform(Mh)
        add("Palm." + side, bm, gun, bh)
        for f in FINGERS:
            p0, p1, p2 = finger_pts(sx, f)
            for k, (a, b, rr) in enumerate(((p0, p1, f[4]), (p1, p2, f[4] * 0.92))):
                bm = bm_capsule(a, b, rr, segs=14, rings=8)
                bm.transform(Mh)
                add("%s_%d.%s" % (f[0], k + 1, side), bm, gun, "%s_%d.%s" % (f[0], k + 1, side))
            dtip = (p2 - p1).normalized()
            bm = bm_cone(p2 + dtip * 0.007, f[4] * 0.86, 0.0025, 0.020, dtip, 12)
            bm.transform(Mh)
            add("%s_Claw.%s" % (f[0], side), bm, gold, "%s_2.%s" % (f[0], side), wear=False)
            d1f = (p1 - p0).normalized()
            bm = bm_torus(p0 + d1f * 0.006, f[4] * 1.18, f[4] * 1.18, 0.0036, d1f, 16, 6)
            bm.transform(Mh)
            add("%s_Knuckle.%s" % (f[0], side), bm, gold, "%s_1.%s" % (f[0], side), wear=False)
        p0, p1, p2 = thumb_pts(sx)
        for k, (a, b, rr) in enumerate(((p0, p1, 0.0105), (p1, p2, 0.0098))):
            bm = bm_capsule(a, b, rr, segs=14, rings=8)
            bm.transform(Mh)
            add("thumb_%d.%s" % (k + 1, side), bm, gun, "thumb_%d.%s" % (k + 1, side))
        dtip = (p2 - p1).normalized()
        bm = bm_cone(p2 + dtip * 0.007, 0.0085, 0.0025, 0.020, dtip, 12)
        bm.transform(Mh)
        add("thumb_Claw." + side, bm, gold, "thumb_2." + side, wear=False)

        # hoses on this arm
        for k, off in enumerate((-0.016, 0.016)):
            ox = sx * off
            U = S + (E - S) * 0.55 + Vector((ox, 0.058, 0.0))
            C = E + Vector((ox, 0.108, 0.015))
            Fp = E + (W - E) * 0.32 + Vector((ox, 0.048, 0.0))
            add_hose("Hose_Elbow%d.%s" % (k, side), U, C, Fp, 0.0115, bu, bf, rubber if k == 0 else wire_t, (0.40, 0.60))
        wp0 = E + (W - E) * 0.70 + Vector((sx * 0.060, 0.0, 0.0))
        wp2 = Mh @ Vector((sx * 0.010, 0.020, -0.045))
        wc = (wp0 + wp2) / 2 + Vector((sx * 0.040, 0.030, 0.0))
        add_hose("Hose_Wrist." + side, wp0, wc, wp2, 0.0085, bf, bh, braid, (0.45, 0.70), fer=gold, fr=1.6)
        yp = yoke_front(1.645)
        add_hose("Hose_PortLine." + side, (sx * 0.20, yp - 0.006, 1.645), (sx * 0.285, yp - 0.040, 1.675), (sx * 0.330, -0.060, 1.640),
                 0.0095, "chest", bc, wire_t, (0.45, 0.65), fer=gold, fr=1.6)
        add_hose("Hose_NeckSide." + side, (sx * 0.165, 0.0, 1.80), (sx * 0.235, -0.015, 1.795), (sx * 0.140, -0.03, 1.715),
                 0.0125, "head", "chest", braid, (0.3, 0.7), fer=gold)

    # repair patch on the left pauldron (character's left = +X)
    P = Vector((0.385, 0.0, 1.64))
    PR = (0.125, 0.110, 0.100)
    dvec = Vector((0.30, -0.40, 0.87)).normalized()
    sp = P + Vector((PR[0] * dvec.x, PR[1] * dvec.y, PR[2] * dvec.z))

    def surf(q, off):
        w = q - P
        t = 1.0 / math.sqrt((w.x / PR[0]) ** 2 + (w.y / PR[1]) ** 2 + (w.z / PR[2]) ** 2)
        p = P + w * t
        n = Vector(((p.x - P.x) / PR[0] ** 2, (p.y - P.y) / PR[1] ** 2, (p.z - P.z) / PR[2] ** 2)).normalized()
        return p + n * off

    nrm = Vector(((sp.x - P.x) / PR[0] ** 2, (sp.y - P.y) / PR[1] ** 2, (sp.z - P.z) / PR[2] ** 2)).normalized()
    tu = Vector((0, 0, 1)).cross(nrm).normalized()
    tv = nrm.cross(tu).normalized()
    pl = patch_local(0.056, 0.040, 4.0, 3, 32)
    pf = patch_faces(3, 32)
    top = [surf(sp + tu * x + tv * z, 0.008) for x, z in pl]
    outer = list(range(len(pl) - 32, len(pl)))
    bottom = [surf(sp + tu * pl[i][0] + tv * pl[i][1], 0.0) for i in outer]
    nb = len(top)
    pfaces = list(pf)
    for k in range(32):
        a, b = outer[k], outer[(k + 1) % 32]
        pfaces.append((a, nb + k, nb + (k + 1) % 32, b))
    add_patch("Repair_Patch", top + bottom, pfaces, patchm, "clavicle.L")
    for ox in (-0.042, 0.042):
        for oy in (-0.024, 0.024):
            add("Patch_Bolt", bm_sphere(surf(sp + tu * ox + tv * oy, 0.009), 0.0062, 12, 8), gun, "clavicle.L", wear=False)

    # ============================================================ legs
    for side, sx in (("L", 1), ("R", -1)):
        H, K, A, Bl, T = leg_pts(sx)
        bt, bs, bf, bto = "thigh." + side, "shin." + side, "foot." + side, "toe." + side
        xs = sx * 0.135
        add("HipJoint." + side, bm_sphere(H, 0.080), gun, "pelvis")
        add("Thigh." + side, bm_capsule(H, K, 0.090), chrome, bt)
        add("ThighRing." + side, bm_cyl((xs, -0.004, 0.88), 0.0925, 0.022, 'Z', 40), gold, bt, wear=False)
        add("ThighPlate." + side, bm_rbox((xs, -0.082, 0.74), (0.130, 0.050, 0.26), 0.022, 3), chrome2, bt)
        add("KneeJoint." + side, bm_sphere(K, 0.080), gun, bt)
        add("KneeCap." + side, bm_ellipsoid((xs, K.y - 0.088, 0.54), (0.082, 0.034, 0.082), 32, 16), gold, bt)
        add("KneeRing." + side, bm_torus((xs, K.y - 0.078, 0.54), 0.090, 0.090, 0.011, 'Y', 40, 10), chrome2, bt, wear=False)
        bolt_ring("Knee_Bolt", (xs, K.y - 0.118, 0.54), (0, -1, 0), 0.058, 6, 0.0058, chrome2, bt)
        add("Shin." + side, bm_capsule(K, A, 0.070), chrome, bs)
        add("ShinPlate." + side, bm_rbox((xs, -0.090, 0.33), (0.135, 0.058, 0.30), 0.024, 3), chrome2, bs)
        add("ShinCap." + side, bm_ellipsoid((xs + sx * 0.074, -0.004, 0.27), (0.028, 0.052, 0.052), 28, 14), gold, bs)
        add("AnkleJoint." + side, bm_sphere(A, 0.060), gun, bs)
        add("AnkleRing." + side, bm_torus((xs, 0.0, 0.118), 0.070, 0.070, 0.010, 'Z', 40, 10), gold, bs, wear=False)
        add("AnkleCollar." + side, bm_cyl((xs, 0.0, 0.098), 0.070, 0.05, 'Z', 28), chrome2, bs)
        bm = bm_rbox((xs, -0.075, 0.050), (0.165, 0.360, 0.100), 0.042, 4)
        for v in bm.verts:
            tt = max(0.0, min(1.0, (-0.115 - v.co.y) / 0.14))
            v.co.x = xs + (v.co.x - xs) * (1.0 - 0.20 * tt)
            if v.co.z > 0.05:
                v.co.z -= 0.026 * tt
        sh = add("Shoe." + side, bm, gun, bf)
        sh["toe_bone"] = bto
        sh["split_y"] = -0.12
        add("ToePlate." + side, bm_rbox((xs, -0.19, 0.088), (0.120, 0.100, 0.026), 0.012, 3), chrome2, bto)
        dclaw = Vector((0.0, -1.0, -0.28)).normalized()
        for k in (-1, 0, 1):
            cb = Vector((xs + k * 0.052, -0.250, 0.046))
            add("Claw", bm_cone(cb + dclaw * 0.045, 0.021, 0.0035, 0.090, dclaw, 14), gold, bto, wear=False)
        add("Heel_Plate." + side, bm_rbox((xs, 0.082, 0.062), (0.130, 0.040, 0.075), 0.016, 3), chrome2, bf)

        # outer hydraulic piston at the knee
        add("Piston_Sleeve." + side, cyl_between((sx * 0.238, -0.010, 0.86), (sx * 0.230, -0.010, 0.62), 0.020), gun, bt)
        add("Piston_Rod." + side, cyl_between((sx * 0.230, -0.010, 0.66), (sx * 0.214, -0.012, 0.40), 0.0115), gold, bs, wear=False)
        add("Piston_MountTop." + side, bm_sphere((sx * 0.236, -0.010, 0.865), 0.026, 14, 8), gun, bt)
        add("Piston_MountLow." + side, bm_sphere((sx * 0.212, -0.012, 0.395), 0.026, 14, 8), gun, bs)

        # hoses on this leg
        add_hose("Hose_Hip." + side, (sx * 0.190, 0.050, 1.03), (sx * 0.250, 0.090, 0.94), (sx * 0.205, 0.040, 0.84),
                 0.0135, "pelvis", bt, rubber, (0.3, 0.6))
        for k, off in enumerate((-0.020, 0.020)):
            xk = xs + off
            add_hose("Hose_Knee%d.%s" % (k, side), (xk, 0.070, 0.78), (xk, 0.140, 0.54), (xk, 0.045, 0.34),
                     0.0135, bt, bs, rubber if k == 0 else braid, (0.40, 0.60), fer=gold if k else None)
        add_hose("Hose_Ankle." + side, (xs, 0.055, 0.27), (xs, 0.140, 0.17), (xs, 0.090, 0.080),
                 0.0125, bs, bf, rubber, (0.45, 0.65))
        add_hose("Hose_Cell." + side, (sx * 0.080, 0.20, 1.35), (sx * 0.115, 0.27, 1.18), (sx * 0.080, 0.10, 1.04),
                 0.0155, "chest", "pelvis", rubber, (0.35, 0.65))
        add_hose("Hose_WaistFront." + side, (sx * 0.090, -0.095, 1.30), (sx * 0.105, -0.125, 1.18), (sx * 0.090, -0.095, 1.04),
                 0.0115, "chest", "pelvis", rubber, (0.35, 0.65))
        add_hose("Hose_WaistSide." + side, (sx * 0.130, 0.0, 1.30), (sx * 0.152, 0.0, 1.18), (sx * 0.130, 0.0, 1.04),
                 0.0115, "chest", "pelvis", braid, (0.35, 0.65), fer=gold)

    for i, x in enumerate((-0.05, 0.0, 0.05)):
        add_hose("Hose_NeckRear%d" % i, (x, 0.180, 1.80), (x * 1.15, 0.275, 1.70), (x, 0.190, 1.56),
                 0.0135, "head", "chest", rubber if i != 1 else wire_t, (0.25, 0.75))

    n = len([o for o in coll.objects if o.get("relic_part")])
    return {"parts": n, "notes": NOTES}
