# relic_model2.py : Relic v02 = darker, chipped/worn paint, hoses and more mechanical detail.
# Exec this AFTER relic_build.py in the same namespace (it reuses that file's helpers).


def curvature_vals(bm):
    """signed curvature per vertex (positive on convex, tight edges) used to drive edge wear"""
    bm.normal_update()
    vals = []
    for v in bm.verts:
        s, n = 0.0, 0
        nv = v.normal
        for e in v.link_edges:
            u = e.other_vert(v)
            d = u.co - v.co
            l2 = d.length_squared
            if l2 < 1e-12:
                continue
            s += (u.normal - nv).dot(d) / l2
            n += 1
        vals.append(s / n if n else 0.0)
    return vals


def sstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def bezier2(p0, c, p2, n):
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(p0 * ((1 - t) ** 2) + c * (2 * (1 - t) * t) + p2 * (t * t))
    return out


def bm_tube(points, radius, sides=10, caps=True):
    bm = bmesh.new()
    n = len(points)
    tans = []
    for i in range(n):
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, n - 1)]
        tans.append((b - a).normalized())
    t0 = tans[0]
    ref = Vector((0, 0, 1)) if abs(t0.z) < 0.9 else Vector((1, 0, 0))
    nrm = t0.cross(ref).normalized()
    rings, ts = [], []
    for i, p in enumerate(points):
        t = tans[i]
        nrm = (nrm - t * nrm.dot(t)).normalized()
        bi = t.cross(nrm)
        ring = []
        for k in range(sides):
            a = 2 * math.pi * k / sides
            ring.append(bm.verts.new(p + (nrm * math.cos(a) + bi * math.sin(a)) * radius))
            ts.append(i / (n - 1))
        rings.append(ring)
    for i in range(n - 1):
        for k in range(sides):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % sides], rings[i + 1][(k + 1) % sides], rings[i + 1][k]))
    if caps:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm, ts


def cyl_between(p0, p1, r, segs=20):
    p0, p1 = Vector(p0), Vector(p1)
    return bm_cyl((p0 + p1) / 2, r, (p1 - p0).length, p1 - p0, segs)


def make_worn(name, paint, metal, rust, rough=0.46, rust_amt=0.8, chip_scale=30.0, edge_gain=1.0,
              zr=(0.1, 1.7), grime=(0.02, 0.019, 0.017), grime_amt=0.6, scratch_amt=0.9):
    """chipped paint over bare metal: edge wear from the baked 'wear' attribute, random chips, scratches, rust, grime"""
    m = make_mat(name, paint, 0.0, rough)
    try:
        nt, b = _principled(m)
        if not b:
            return m
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

        wear = N.new('ShaderNodeAttribute')
        wear.attribute_type = 'GEOMETRY'
        wear.attribute_name = 'wear'
        sep = N.new('ShaderNodeSeparateXYZ')
        L.new(tc.outputs['Object'], sep.inputs['Vector'])
        hm = N.new('ShaderNodeMapRange')
        hm.inputs['From Min'].default_value = zr[0]
        hm.inputs['From Max'].default_value = zr[1]
        hm.inputs['To Min'].default_value = 1.0
        hm.inputs['To Max'].default_value = 0.30
        L.new(sep.outputs['Z'], hm.inputs['Value'])
        height = hm.outputs['Result']

        chips = ramp(noise(mapping((1, 1, 1)), chip_scale, 7), 0.645, 0.675)
        brk = ramp(noise(mapping((1, 1, 1)), 30.0, 4), 0.30, 0.62)
        edge = fm('MULTIPLY', wear.outputs['Fac'], brk)
        edge = fm('MULTIPLY', edge, edge_gain, clamp=True)
        s1 = ramp(noise(mapping((170, 9, 9)), 1.0, 1.0), 0.617, 0.625)
        s2 = ramp(noise(mapping((9, 170, 9)), 1.0, 1.0), 0.617, 0.625)
        scr = fm('MULTIPLY', fm('ADD', s1, s2, clamp=True), scratch_amt)
        bare = fm('MAXIMUM', fm('MAXIMUM', chips, edge), scr)

        rust_n = ramp(noise(mapping((1, 1, 1)), 7.0, 7), 0.58, 0.75)
        rust_h = fm('MULTIPLY', fm('MULTIPLY', rust_n, height), rust_amt)
        rust_mask = fm('ADD', rust_h, fm('MULTIPLY', bare, 0.30), clamp=True)
        g_mask = fm('MULTIPLY', ramp(noise(mapping((1, 1, 0.28)), 14.0, 6), 0.56, 0.80), grime_amt)
        try:
            ao = N.new('ShaderNodeAmbientOcclusion')
            ao.inputs['Distance'].default_value = 0.06
            crev = fm('MULTIPLY', fm('SUBTRACT', 1.0, ao.outputs['AO'], clamp=True), 0.9)
            g_mask = fm('MAXIMUM', g_mask, crev)
        except Exception as e:
            NOTES.append("AO node: %s" % e)

        col = mixc(bare, paint, metal)
        col = mixc(rust_mask, col, rust)
        col = mixc(g_mask, col, grime)
        L.new(col, b.inputs['Base Color'])
        L.new(fm('MULTIPLY', fm('MULTIPLY', bare, 0.55), fm('SUBTRACT', 1.0, rust_mask, clamp=True)), b.inputs['Metallic'])
        r1 = mixf(bare, rough, 0.38)
        L.new(mixf(rust_mask, r1, 0.85), b.inputs['Roughness'])
    except Exception as e:
        NOTES.append("make_worn failed for %s: %s" % (name, e))
    return m


def stage_model2():
    scn = bpy.data.scenes[SCENE]
    coll = bpy.data.collections[COLL]
    for o in list(coll.objects):
        if o.type == 'MESH' and o.get("relic_part"):
            bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)

    METAL = (0.15, 0.155, 0.165)
    RUST = (0.14, 0.05, 0.016)
    paint = make_worn("Relic_Paint", (0.052, 0.049, 0.043), METAL, RUST, rough=0.55, edge_gain=1.8, scratch_amt=0.5)
    slate = make_worn("Relic_Slate", (0.021, 0.025, 0.028), METAL, RUST, rough=0.55, edge_gain=2.0, scratch_amt=0.5)
    orange = make_worn("Relic_Stripe", (0.45, 0.10, 0.0), METAL, RUST, rough=0.55, chip_scale=24.0, edge_gain=1.6)
    patchm = make_worn("Relic_Patch", (0.02, 0.09, 0.105), METAL, RUST, rough=0.5, edge_gain=1.6)
    graphite = make_mat("Relic_Graphite", (0.030, 0.034, 0.040), 0.8, 0.45)
    copper = make_mat("Relic_Copper", (0.38, 0.19, 0.08), 1.0, 0.38)
    brass = make_mat("Relic_Brass", (0.30, 0.25, 0.15), 0.8, 0.5)
    rubber = make_mat("Relic_Rubber", (0.014, 0.015, 0.017), 0.0, 0.5)
    braid = make_mat("Relic_Braid", (0.20, 0.10, 0.045), 0.9, 0.5)
    wire_t = make_mat("Relic_WireTeal", (0.02, 0.22, 0.24), 0.0, 0.4)
    visor_m = make_mat("Relic_Visor", (0.004, 0.005, 0.007), 0.0, 0.08)
    trim_m = make_mat("Relic_VisorTrim", (0.30, 0.15, 0.06), 1.0, 0.35)
    glyph_m = make_mat("Relic_Glyph", (0.0, 0.0, 0.0), 0.0, 0.5, emit=(1.0, 0.32, 0.0), emit_strength=2.2)
    ring_m = make_mat("Relic_RingLight", (0.0, 0.0, 0.0), 0.0, 0.5, emit=(0.0, 0.62, 1.0), emit_strength=2.4)
    dark = make_mat("Relic_Dark", (0.008, 0.009, 0.011), 0.3, 0.4)

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
        f = fer if fer is not None else graphite
        t0 = (pts[1] - pts[0]).normalized()
        t1 = (pts[-1] - pts[-2]).normalized()
        add(name + "_F0", bm_cyl(pts[0] + t0 * 0.010, radius * fr, 0.024, t0, 16), f, bone_a, wear=False)
        add(name + "_F1", bm_cyl(pts[-1] - t1 * 0.010, radius * fr, 0.024, t1, 16), f, bone_b, wear=False)
        return ob

    # ================================================================ head
    rx, ry, rz = HEAD_R
    zc = HEAD_C.z
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
    add("Head_Dome", bm, paint, "head")
    base_z = zc - HEAD_LOW
    add("Head_Collar", bm_cyl((0, 0, base_z), 1.0, 0.038, 'Z', 48, (rx * 1.06, ry * 1.06)), graphite, "head")
    add("Head_CollarTrim", bm_torus((0, 0, base_z + 0.021), rx * 1.06, ry * 1.06, 0.0075, 'Z', 64, 10), copper, "head")
    for deg in (35, 145, 215, 325):
        th = math.radians(deg)
        lx = (rx * 1.06 + 0.012) * math.cos(th)
        ly = (ry * 1.06 + 0.012) * math.sin(th)
        add("Collar_Latch", bm_rbox((lx, ly, base_z + 0.002), (0.03, 0.036, 0.052), 0.008, 2, rot=Matrix.Rotation(th, 4, 'Z')), graphite, "head")
        add("Collar_Bolt", bm_sphere((lx + math.cos(th) * 0.016, ly + math.sin(th) * 0.016, base_z + 0.012), 0.0065, 10, 6), copper, "head", wear=False)
    for sx in (1, -1):
        px = sx * (rx + 0.004)
        add("Head_Port", bm_cyl((px, 0.0, 1.772), 0.034, 0.014, 'X', 28), graphite, "head")
        add("Head_PortRing", bm_torus((px + sx * 0.005, 0.0, 1.772), 0.034, 0.034, 0.0045, 'X', 32, 8), copper, "head", wear=False)
        add("Head_PortCore", bm_cyl((px + sx * 0.0075, 0.0, 1.772), 0.018, 0.004, 'X', 20), dark, "head", wear=False)
    for zz in (1.795, 1.822, 1.849):
        yy = ry * math.sqrt(max(0.0, 1.0 - (max(zz - zc, 0.0) / rz) ** 2))
        add("Head_Vent", bm_rbox((0, yy, zz), (0.082, 0.006, 0.0045), 0.0015, 1), dark, "head", wear=False)

    vz = zc + 0.040
    pts = patch_local(0.124, 0.046, 3.2, 4, 56)
    fcs = patch_faces(4, 56)
    pr_trim = head_proj(1.007)
    add_patch("Visor_Trim", [pr_trim(x, vz + z) for x, z in pts], fcs, trim_m, "head")
    pts2 = patch_local(0.116, 0.039, 3.2, 4, 56)
    pr_vis = head_proj(1.012)
    add_patch("Visor", [pr_vis(x, vz + z) for x, z in pts2], fcs, visor_m, "head")
    eye_pts = patch_local(0.022, 0.0125, 3.0, 3, 28)
    eye_faces = patch_faces(3, 28)
    n1 = len(eye_pts)
    pr_eye = head_proj(1.017)
    ez = vz + 0.003
    EX = 0.046

    def eye_coords(kind):
        out = []
        for ex in (EX, -EX):
            inner = -1.0 if ex > 0 else 1.0
            for lx, lz in eye_pts:
                x, z = lx, lz
                if kind == 'Blink':
                    z = lz * 0.12
                elif kind == 'Wide':
                    x, z = lx * 1.12, lz * 1.7
                elif kind == 'Angry':
                    z = lz * 0.8 - 0.35 * (lx * inner) - 0.003
                elif kind == 'Sad':
                    z = lz * 0.9 + 0.35 * (lx * inner) - 0.001
                elif kind == 'Happy':
                    z = lz * 0.55 - 20.0 * lx * lx + 0.004
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

    # ================================================================ neck + torso
    add("Neck", bm_cyl((0, 0, 1.68), 0.055, 0.13, 'Z', 24), graphite, "neck")
    add("Neck_Ring", bm_torus((0, 0, 1.665), 0.066, 0.066, 0.011, 'Z', 40, 10), copper, "neck")

    add("Chest", bm_rbox((0, 0, 1.42), (0.41, 0.25, 0.48), 0.085, 5, taper=(0.72, 0.84)), paint, "chest")
    fy = chest_front(1.46)
    add("Chest_Housing", bm_cyl((0, fy - 0.003, 1.46), 0.064, 0.030, 'Y', 40), graphite, "chest")
    add("Chest_Ring", bm_torus((0, fy - 0.0195, 1.46), 0.044, 0.044, 0.0085, 'Y', 56, 12), ring_m, "chest", wear=False)
    add("Chest_RingCore", bm_cyl((0, fy - 0.0185, 1.46), 0.035, 0.004, 'Y', 40), dark, "chest", wear=False)
    py = chest_front(1.30)
    add("Serial_Plate", bm_rbox((0.115, py - 0.0015, 1.30), (0.075, 0.006, 0.040), 0.002, 2), brass, "chest")
    for i, zz in enumerate((1.311, 1.300, 1.289)):
        add("Serial_Line%d" % i, bm_rbox((0.112, py - 0.0045, zz), (0.050 - 0.008 * (i % 2), 0.002, 0.0035), 0.001, 1), dark, "chest", wear=False)
    add("Chest_Seam", bm_rbox((0, py - 0.001, 1.285), (0.004, 0.003, 0.20), 0.001, 1), dark, "chest", wear=False)
    for zz in (1.395, 1.265):
        add("Panel_Line", bm_rbox((0, chest_front(zz) - 0.001, zz), (0.22, 0.003, 0.004), 0.001, 1), dark, "chest", wear=False)
        for xx in (-0.11, 0.11):
            add("Panel_Rivet", bm_sphere((xx, chest_front(zz) - 0.002, zz), 0.0048, 10, 6), copper, "chest", wear=False)
    add("Back_Hatch", bm_rbox((0, 0.118, 1.42), (0.21, 0.020, 0.23), 0.012, 3), graphite, "chest")
    add("Power_Cell", bm_rbox((0, 0.162, 1.44), (0.17, 0.07, 0.26), 0.012, 3), slate, "chest")
    add("Cell_Band", bm_rbox((0, 0.1975, 1.44), (0.175, 0.004, 0.05), 0.001, 1), orange, "chest")
    for bx in (-0.045, 0.0, 0.045):
        add("Cell_Terminal", bm_cyl((bx, 0.165, 1.585), 0.0125, 0.032, 'Z', 16), copper, "chest", wear=False)

    add("Waist", bm_cyl((0, 0, 1.13), 1.0, 0.15, 'Z', 40, (0.10, 0.085)), graphite, "spine")
    for zz in (1.098, 1.118, 1.138, 1.158, 1.178):
        add("Waist_Rib", bm_torus((0, 0, zz), 0.104, 0.089, 0.0055, 'Z', 48, 8), brass, "spine", wear=False)
    add("Waist_Ring_Lo", bm_torus((0, 0, 1.075), 0.104, 0.089, 0.0095, 'Z', 48, 10), copper, "spine")
    add("Waist_Ring_Hi", bm_torus((0, 0, 1.197), 0.104, 0.089, 0.0095, 'Z', 48, 10), copper, "spine")

    add("Pelvis", bm_rbox((0, 0, 0.995), (0.34, 0.22, 0.17), 0.055, 4), paint, "pelvis")
    add("Pelvis_Low", bm_rbox((0, 0, 0.925), (0.28, 0.19, 0.09), 0.03, 3), graphite, "pelvis")
    add("Belt_Plate", bm_rbox((0, -0.112, 1.0), (0.13, 0.02, 0.075), 0.01, 2), slate, "pelvis")
    add("Belt_Buckle", bm_torus((0, -0.1235, 1.0), 0.022, 0.022, 0.005, 'Y', 28, 8), brass, "pelvis", wear=False)
    for sx in (1, -1):
        add("Pouch", bm_rbox((sx * 0.200, 0.0, 0.99), (0.075, 0.088, 0.125), 0.024, 3), slate, "pelvis")
        add("Pouch_Strap", bm_rbox((sx * 0.200, 0.0, 1.04), (0.080, 0.092, 0.014), 0.004, 2), graphite, "pelvis")
        add("Pouch_Buckle", bm_rbox((sx * 0.2395, 0.0, 1.04), (0.004, 0.026, 0.026), 0.002, 1), brass, "pelvis", wear=False)

    # ================================================================ arms
    for side, sx in (("L", 1), ("R", -1)):
        S, E, W, t2 = arm_pts(sx)
        Mh = hand_frame(sx)
        bu, bf, bh, bc = "upperarm." + side, "forearm." + side, "hand." + side, "clavicle." + side
        d1 = (E - S).normalized()
        d2 = (W - E).normalized()
        add("Clavicle." + side, bm_capsule((sx * 0.09, -0.005, 1.60), S, 0.034), graphite, bc)
        P = Vector((sx * 0.285, 0.0, 1.63))
        PR = (0.100, 0.092, 0.078)
        add("Pauldron." + side, bm_ellipsoid(P, PR, 40, 20), paint, bc)
        add("ShoulderJoint." + side, bm_sphere(S, 0.058), graphite, bu)
        add("UpperArm." + side, bm_capsule(S, E, 0.052), paint, bu)
        for k, (f, wd) in enumerate(((0.28, 0.034), (0.42, 0.014))):
            add("ArmBand%d.%s" % (k, side), bm_cyl(S + (E - S) * f, 0.0548, wd, d1, 40), orange, bu)
        add("ElbowJoint." + side, bm_sphere(E, 0.054), copper, bf)
        add("ElbowPad." + side, bm_rbox(E + Vector((sx * 0.058, 0.0, 0.0)), (0.034, 0.066, 0.076), 0.014, 3, rot=rot_z_to(d2)), slate, bf)
        add("Forearm." + side, bm_capsule(E, W, 0.047), slate, bf)
        gc = E + (W - E) * 0.48 + Vector((sx * 0.052, 0.0, 0.0))
        add("ForearmGuard." + side, bm_rbox(gc, (0.028, 0.076, 0.15), 0.012, 3, rot=rot_z_to(d2)), paint, bf)
        add("WristRing." + side, bm_torus(W, 0.046, 0.046, 0.0095, d2, 40, 10), copper, bh, wear=False)

        bm = bm_rbox((0, 0, -0.045), (0.032, 0.078, 0.090), 0.012, 3)
        bm.transform(Mh)
        add("Palm." + side, bm, graphite, bh)
        for f in FINGERS:
            p0, p1, p2 = finger_pts(sx, f)
            for k, (a, b, rr) in enumerate(((p0, p1, f[4]), (p1, p2, f[4] * 0.92))):
                bm = bm_capsule(a, b, rr, segs=14, rings=8)
                bm.transform(Mh)
                add("%s_%d.%s" % (f[0], k + 1, side), bm, graphite, "%s_%d.%s" % (f[0], k + 1, side))
        p0, p1, p2 = thumb_pts(sx)
        for k, (a, b, rr) in enumerate(((p0, p1, 0.0105), (p1, p2, 0.0098))):
            bm = bm_capsule(a, b, rr, segs=14, rings=8)
            bm.transform(Mh)
            add("thumb_%d.%s" % (k + 1, side), bm, graphite, "thumb_%d.%s" % (k + 1, side))

        # hoses on this arm
        add_hose("Hose_NeckSide." + side, (sx * 0.162, 0.0, 1.738), (sx * 0.205, -0.015, 1.705), (sx * 0.125, -0.02, 1.632),
                 0.011, "head", "chest", braid, (0.3, 0.7))
        add_hose("Hose_Shoulder." + side, (sx * 0.185, 0.04, 1.56), (sx * 0.250, 0.100, 1.555), (sx * 0.298, 0.042, 1.505),
                 0.012, "chest", bu, rubber, (0.35, 0.65))
        add_hose("Hose_ChestLine." + side, (sx * 0.060, fy - 0.004, 1.485), (sx * 0.150, fy - 0.022, 1.590), (sx * 0.235, -0.040, 1.620),
                 0.0085, "chest", bc, wire_t, (0.45, 0.65), fr=1.7)
        for k, off in enumerate((-0.014, 0.014)):
            ox = sx * off
            U = S + (E - S) * 0.55 + Vector((ox, 0.046, 0.0))
            C = E + Vector((ox, 0.088, 0.015))
            Fp = E + (W - E) * 0.32 + Vector((ox, 0.040, 0.0))
            add_hose("Hose_Elbow%d.%s" % (k, side), U, C, Fp, 0.0105, bu, bf, rubber if k == 0 else wire_t, (0.40, 0.60))
        wp0 = E + (W - E) * 0.70 + Vector((sx * 0.050, 0.0, 0.0))
        wp2 = Mh @ Vector((sx * 0.010, 0.020, -0.045))
        wc = (wp0 + wp2) / 2 + Vector((sx * 0.035, 0.025, 0.0))
        add_hose("Hose_Wrist." + side, wp0, wc, wp2, 0.0075, bf, bh, braid, (0.45, 0.70), fr=1.6)

    # repair patch on the left pauldron (character's left = +X)
    P = Vector((0.285, 0.0, 1.63))
    PR = (0.100, 0.092, 0.078)
    dvec = Vector((0.35, -0.45, 0.82)).normalized()
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
    pl = patch_local(0.046, 0.032, 4.0, 3, 32)
    pf = patch_faces(3, 32)
    top = [surf(sp + tu * x + tv * z, 0.0075) for x, z in pl]
    outer = list(range(len(pl) - 32, len(pl)))
    bottom = [surf(sp + tu * pl[i][0] + tv * pl[i][1], 0.0) for i in outer]
    nb = len(top)
    pfaces = list(pf)
    for k in range(32):
        a, b = outer[k], outer[(k + 1) % 32]
        pfaces.append((a, nb + k, nb + (k + 1) % 32, b))
    add_patch("Repair_Patch", top + bottom, pfaces, patchm, "clavicle.L")
    for ox in (-0.034, 0.034):
        for oy in (-0.020, 0.020):
            add("Patch_Bolt", bm_sphere(surf(sp + tu * ox + tv * oy, 0.0085), 0.0055, 12, 8), graphite, "clavicle.L", wear=False)

    # ================================================================ legs
    for side, sx in (("L", 1), ("R", -1)):
        H, K, A, Bl, T = leg_pts(sx)
        bt, bs, bf, bto = "thigh." + side, "shin." + side, "foot." + side, "toe." + side
        add("HipJoint." + side, bm_sphere(H, 0.064), graphite, "pelvis")
        add("Thigh." + side, bm_capsule(H, K, 0.074), paint, bt)
        add("ThighBand." + side, bm_cyl((sx * 0.115, -0.0034, 0.87), 0.0768, 0.036, 'Z', 40), orange, bt)
        add("ThighPlate." + side, bm_rbox((sx * 0.115, -0.068, 0.74), (0.095, 0.038, 0.20), 0.016, 3), slate, bt)
        add("KneeJoint." + side, bm_sphere(K, 0.063), copper, bt)
        add("KneeCap." + side, bm_rbox((sx * 0.115, -0.082, 0.55), (0.088, 0.036, 0.09), 0.02, 3), paint, bt)
        add("Shin." + side, bm_capsule(K, A, 0.057), paint, bs)
        add("ShinBand." + side, bm_cyl((sx * 0.115, -0.0045, 0.27), 0.0592, 0.030, 'Z', 40), orange, bs)
        add("ShinGuard." + side, bm_rbox((sx * 0.115, -0.062, 0.34), (0.092, 0.040, 0.26), 0.016, 3), slate, bs)
        add("AnkleJoint." + side, bm_sphere(A, 0.049), graphite, bs)
        add("AnkleCollar." + side, bm_cyl((sx * 0.115, 0.0, 0.112), 0.056, 0.05, 'Z', 28), slate, bs)
        bm = bm_rbox((sx * 0.115, -0.07, 0.0425), (0.125, 0.30, 0.085), 0.035, 4)
        cx = sx * 0.115
        for v in bm.verts:
            tt = max(0.0, min(1.0, (-0.09 - v.co.y) / 0.13))
            v.co.x = cx + (v.co.x - cx) * (1.0 - 0.22 * tt)
            if v.co.z > 0.045:
                v.co.z -= 0.022 * tt
        sh = add("Shoe." + side, bm, graphite, bf)
        sh["toe_bone"] = bto
        sh["split_y"] = -0.105
        add("ToeCap." + side, bm_rbox((sx * 0.115, -0.172, 0.074), (0.092, 0.085, 0.022), 0.009, 3), slate, bto)
        # hydraulic piston on the outside of the knee
        add("Piston_Sleeve." + side, cyl_between((sx * 0.205, -0.010, 0.80), (sx * 0.198, -0.010, 0.60), 0.0165), graphite, bt)
        add("Piston_Rod." + side, cyl_between((sx * 0.198, -0.010, 0.64), (sx * 0.182, -0.012, 0.42), 0.009), copper, bs, wear=False)
        add("Piston_MountTop." + side, bm_sphere((sx * 0.203, -0.010, 0.805), 0.02, 14, 8), graphite, bt)
        add("Piston_MountLow." + side, bm_sphere((sx * 0.182, -0.012, 0.415), 0.02, 14, 8), graphite, bs)

        # hoses on this leg
        add_hose("Hose_Hip." + side, (sx * 0.160, 0.045, 1.02), (sx * 0.222, 0.075, 0.93), (sx * 0.172, 0.030, 0.84),
                 0.012, "pelvis", bt, rubber, (0.3, 0.6))
        for k, off in enumerate((-0.018, 0.018)):
            xk = sx * 0.115 + off
            add_hose("Hose_Knee%d.%s" % (k, side), (xk, 0.060, 0.76), (xk, 0.118, 0.55), (xk, 0.040, 0.36),
                     0.0125, bt, bs, rubber if k == 0 else braid, (0.40, 0.60))
        add_hose("Hose_Ankle." + side, (sx * 0.115, 0.050, 0.27), (sx * 0.115, 0.118, 0.17), (sx * 0.115, 0.070, 0.078),
                 0.011, bs, bf, rubber, (0.45, 0.65))
        add_hose("Hose_Cell." + side, (sx * 0.070, 0.185, 1.33), (sx * 0.100, 0.245, 1.17), (sx * 0.070, 0.095, 1.04),
                 0.014, "chest", "pelvis", rubber, (0.35, 0.65))
        add_hose("Hose_WaistFront." + side, (sx * 0.075, -0.082, 1.225), (sx * 0.092, -0.118, 1.135), (sx * 0.075, -0.082, 1.045),
                 0.0105, "chest", "pelvis", rubber, (0.35, 0.65))
        add_hose("Hose_WaistSide." + side, (sx * 0.112, 0.0, 1.22), (sx * 0.135, 0.0, 1.135), (sx * 0.112, 0.0, 1.05),
                 0.0105, "chest", "pelvis", braid, (0.35, 0.65))

    for i, x in enumerate((-0.045, 0.0, 0.045)):
        add_hose("Hose_NeckRear%d" % i, (x, 0.172, 1.742), (x * 1.15, 0.255, 1.665), (x, 0.165, 1.56),
                 0.0125, "head", "chest", rubber if i != 1 else wire_t, (0.25, 0.75))

    n = len([o for o in coll.objects if o.get("relic_part")])
    return {"parts": n, "notes": NOTES}
