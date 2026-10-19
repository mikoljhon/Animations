# relic_cave4.py : ceiling crack + sky, hanging roots, props (rebar, pipes, cables), lights, buried / dug-out states.
# Exec after relic_cave.py .. relic_cave3.py.

SLIT = dict(center=(-2.3, 2.6), ang=math.radians(35.0), length=4.4)


def tube_arrays(P, r, sides=6):
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1)[:, None], 1e-9)
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    Nn = np.cross(T[0], a)
    Nn /= np.linalg.norm(Nn)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = []
    for i in range(n):
        if i > 0:
            Nn = Nn - T[i] * np.dot(Nn, T[i])
            Nn /= np.linalg.norm(Nn)
        B = np.cross(T[i], Nn)
        ri = float(r[i]) if hasattr(r, "__len__") else float(r)
        rings.append(P[i] + ri * (np.outer(np.cos(ang), Nn) + np.outer(np.sin(ang), B)))
    V = np.concatenate(rings)
    F = []
    for i in range(n - 1):
        for k in range(sides):
            a_ = i * sides + k
            b_ = i * sides + (k + 1) % sides
            c_ = (i + 1) * sides + (k + 1) % sides
            d_ = (i + 1) * sides + k
            F.append((a_, b_, c_, d_))
    return V, np.array(F, dtype=np.int32)


def rubber_material(name="Cave_Rubber"):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])
    pb.inputs['Base Color'].default_value = (0.012, 0.012, 0.014, 1)
    pb.inputs['Roughness'].default_value = 0.45
    return m


def ceiling_z(x, y):
    ob = bpy.data.objects["Cave_Shell"]
    hit = ob.ray_cast(Vector((x, y, 0.5)), Vector((0, 0, 1)), distance=30.0)
    return hit[1].z if hit[0] else 6.5


def cut_slit():
    """open a jagged crack in the ceiling so a shaft of daylight can fall into the cave"""
    ob = bpy.data.objects["Cave_Shell"]
    me = ob.data
    cx, cy = SLIT["center"]
    a, L = SLIT["ang"], SLIT["length"]
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    t = np.linspace(-0.5, 0.5, 48)
    wig = 0.35 * np.sin(t * 7.0 + 0.8) + 0.18 * np.sin(t * 17.0)
    px = cx + t * L * ux + wig * vx
    py = cy + t * L * uy + wig * vy
    base = 0.17 + 0.12 * (fbm(np.stack([t * 6, np.zeros_like(t), np.zeros_like(t)], 1), 1.0, 2, 51) - 0.5) * 2
    wid = base * (0.45 + 0.55 * np.sin(np.pi * (t + 0.5)))
    bm = bmesh.new()
    bm.from_mesh(me)
    dele = []
    for f in bm.faces:
        c = f.calc_center_median()
        if c.z < 3.0:
            continue
        d = np.sqrt((px - c.x) ** 2 + (py - c.y) ** 2)
        k = int(d.argmin())
        if d[k] < wid[k] + 0.07:
            dele.append(f)
    bmesh.ops.delete(bm, geom=dele, context='FACES')
    bm.to_mesh(me)
    bm.free()
    me.update()
    return {"deleted_faces": len(dele), "ceiling_z": round(ceiling_z(cx - 0.5, cy), 2)}


def build_sky():
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Props", cave)
    _drop("Cave_Sky")
    cx, cy = SLIT["center"]
    me = mesh_from_arrays("Cave_Sky", [(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)], [(0, 1, 2, 3)])
    m = bpy.data.materials.get("Cave_SkyMat") or bpy.data.materials.new("Cave_SkyMat")
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (0.70, 0.85, 1.0, 1)
    em.inputs['Strength'].default_value = 6.0
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    me.materials.append(m)
    ob = bpy.data.objects.new("Cave_Sky", me)
    coll.objects.link(ob)
    ob.location = (cx, cy, 11.5)
    ob.visible_shadow = False
    ob.visible_glossy = False
    return "ok"


def build_roots(seed=3, n=30):
    rng = np.random.default_rng(seed)
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Roots", cave)
    wipe_collection("Cave_Roots")
    coll = get_coll("Cave_Roots", cave)
    mat = root_material()
    cx, cy = SLIT["center"]
    a, L = SLIT["ang"], SLIT["length"]
    ux, uy = math.cos(a), math.sin(a)
    made = 0
    for i in range(n):
        t = rng.uniform(-0.45, 0.45)
        wig = 0.35 * math.sin(t * 7.0 + 0.8) + 0.18 * math.sin(t * 17.0)
        x = cx + t * L * ux + wig * (-uy) + rng.normal(0, 0.12)
        y = cy + t * L * uy + wig * ux + rng.normal(0, 0.12)
        zc = ceiling_z(x, y)
        length = rng.uniform(0.7, 4.2) if i < n - 8 else rng.uniform(0.3, 1.2)
        steps = 16
        s = np.linspace(0, 1, steps)
        sway = 0.25 * np.stack([np.sin(s * 5 + rng.uniform(0, 6)), np.cos(s * 4 + rng.uniform(0, 6))], axis=1) * s[:, None] * min(length, 2.5) * 0.35
        P = np.stack([x + sway[:, 0], y + sway[:, 1], zc + 0.1 - s * length], axis=1)
        r0 = rng.uniform(0.012, 0.045) if i < n - 8 else rng.uniform(0.004, 0.009)
        rad = r0 * (1.0 - 0.85 * s ** 0.8)
        V, F = tube_arrays(P, rad, sides=6)
        me = mesh_from_arrays("Cave_Root_%d" % i, V, F)
        set_smooth(me)
        me.materials.append(mat)
        o = bpy.data.objects.new("Cave_Root_%d" % i, me)
        coll.objects.link(o)
        made += 1
    return made


def add_props(seed=9):
    rng = np.random.default_rng(seed)
    cave = get_coll(CAVE)
    wipe_collection("Cave_Debris_Props")
    coll = get_coll("Cave_Debris_Props", cave)
    rebar = metal_material("Cave_Rebar", base=(0.10, 0.04, 0.02), rough=0.7, metallic=0.35)
    pipe = metal_material("Cave_Pipe", base=(0.16, 0.13, 0.10), rough=0.6, metallic=0.55)
    rub = rubber_material()
    cx, cz = DEB["win"]
    made = {"rebar": 0, "pipe": 0, "cable": 0}

    def add_mesh(name, V, F, mat):
        me = mesh_from_arrays(name, V, F)
        set_smooth(me)
        me.materials.append(mat)
        o = bpy.data.objects.new(name, me)
        coll.objects.link(o)
        o.matrix_world = rig_matrix()
        return o

    for i in range(18):
        x, z = rng.uniform(-2.3, 2.3), rng.uniform(-0.7, 3.1)
        if math.hypot(x - cx, z - cz) < 0.5 or surf_pit(x, z) > 0.02:
            continue
        n = surf_normal(x, z, 0)
        y = BACK - surf_H(x, z, 0)
        base = np.array([x, y, z]) - n * 0.08
        d = n + rng.normal(0, 0.4, 3)
        d /= np.linalg.norm(d)
        length = rng.uniform(0.4, 1.5)
        bend = rng.normal(0, 1, 3)
        bend -= d * (bend @ d)
        bend /= np.linalg.norm(bend)
        t = np.linspace(0, 1, 14)
        P = np.array([base + d * length * tt + bend * 0.45 * max(0.0, tt - 0.55) ** 1.5 * length for tt in t])
        V, F = tube_arrays(P, rng.uniform(0.010, 0.016), sides=5)
        add_mesh("rebar%d" % i, V, F, rebar)
        made["rebar"] += 1
    for i, (x, z) in enumerate([(-4.2, 4.0), (3.8, 5.2), (-1.0, 6.4)]):
        n = surf_normal(x, z, 0)
        y = BACK - surf_H(x, z, 0)
        yaw = rng.uniform(0, 6.28)
        dirv = np.array([math.cos(yaw), 0, math.sin(yaw)])
        dirv -= n * (dirv @ n)
        dirv /= np.linalg.norm(dirv)
        L = rng.uniform(2.0, 3.5)
        base = np.array([x, y, z]) - n * 0.02
        P = [base + dirv * L * tt + n * (0.10 + 0.25 * max(0, tt - 0.7) * 3) for tt in np.linspace(0, 1, 18)]
        V, F = tube_arrays(P, rng.uniform(0.05, 0.10), sides=10)
        add_mesh("pipe%d" % i, V, F, pipe)
        made["pipe"] += 1
    for i in range(5):
        x, z = rng.uniform(-3.5, 3.5), rng.uniform(-1.8, 3.2)
        if FINE_X[0] < x < FINE_X[1] and FINE_Z[0] < z < FINE_Z[1]:
            continue
        n = surf_normal(x, z, 0)
        y = BACK - surf_H(x, z, 0)
        dirv = np.array([rng.normal(), 0, rng.normal()])
        dirv -= n * (dirv @ n)
        dirv /= np.linalg.norm(dirv)
        side = np.cross(n, dirv)
        L = rng.uniform(1.0, 2.6)
        for k in range(3):
            off = side * rng.normal(0, 0.03)
            P = []
            for tt in np.linspace(0, 1, 22):
                p = np.array([x, y, z]) + dirv * L * tt + off + side * 0.18 * math.sin(tt * 5 + k) + n * (0.02 + 0.03 * math.sin(tt * 9))
                P.append(p)
            V, F = tube_arrays(P, 0.014, sides=6)
            add_mesh("cable%d_%d" % (i, k), V, F, rub)
        made["cable"] += 1
    return made


# ------------------------------------------------------------------ lights
def ring_world():
    rig = bpy.data.objects[RIG]
    ring = bpy.data.objects["Relic_AI_Ring"]
    bb = np.mean([np.array(v[:]) for v in ring.bound_box], axis=0)
    rel = rig.matrix_world.inverted() @ ring.matrix_world
    c_rest = rel @ Vector(bb)
    pb = rig.pose.bones["chest"]
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    posed = delta @ c_rest
    front = (delta.to_3x3() @ Vector((0, -1, 0))).normalized()
    Mw = rig.matrix_world
    return Mw @ posed, (Mw.to_3x3() @ front).normalized()


def get_win():
    if "win" in DEB:
        return DEB["win"]
    ob = bpy.data.objects["Cave_Debris"]
    return (ob["win_x"], ob["win_z"])


def window_world():
    cx, cz = get_win()
    if "H0" in DEB:
        y = BACK - surf_H(cx, cz, 0)
    else:
        y = BACK - 0.30
    return Vector(to_world(np.array([[cx, y - 0.03, cz]]))[0])


def build_lights(sun_target=(-1.9, 1.3, 0.9), head_pos=(2.7, -2.5, 2.3), sun_energy=9.0, head_energy=2200.0, free_headlamp=False):
    cave = get_coll(CAVE)
    wipe_collection("Cave_Lights")
    lc = get_coll("Cave_Lights", cave)
    cx, cy = SLIT["center"]
    zc = ceiling_z(cx, cy - 1.0) if bpy.data.objects.get("Cave_Shell") else 6.5
    S3 = Vector((cx, cy, zc))
    d = (Vector(sun_target) - S3).normalized()
    sd = bpy.data.lights.new("Shaft", 'SUN')
    sd.energy = sun_energy
    sd.angle = math.radians(0.7)
    sd.color = (0.80, 0.89, 1.0)
    so = bpy.data.objects.new("Shaft", sd)
    lc.objects.link(so)
    so.location = S3
    so.rotation_euler = (-d).to_track_quat('Z', 'Y').to_euler()
    # headlamp: since v08 Mender stands in the scene with his own spot light on the hat (MenderC_Headlamp, see mender_cave.py), so the
    # free-floating stand-in light is only built on request (free_headlamp=True), for shots without the Mender copy
    tgt = window_world()
    if free_headlamp:
        hd = bpy.data.lights.new("Mender_Headlamp", 'SPOT')
        hd.energy = head_energy
        hd.spot_size = math.radians(30)
        hd.spot_blend = 0.45
        hd.shadow_soft_size = 0.04
        hd.color = (1.0, 0.92, 0.78)
        ho = bpy.data.objects.new("Mender_Headlamp", hd)
        lc.objects.link(ho)
        ho.location = head_pos
        ho.rotation_euler = (tgt - Vector(head_pos)).to_track_quat('-Z', 'Y').to_euler()
    # the chest ring's cyan glow on the rocks around it
    rp, rf = ring_world()
    gd = bpy.data.lights.new("RingGlow", 'POINT')
    gd.energy = 0.0
    gd.shadow_soft_size = 0.06
    gd.color = (0.35, 0.85, 1.0)
    go = bpy.data.objects.new("RingGlow", gd)
    lc.objects.link(go)
    go.location = rp + rf * 0.10
    # a very faint cool fill so the walls read
    fd = bpy.data.lights.new("CaveFill", 'AREA')
    fd.energy = 230.0
    fd.size = 10.0
    fd.color = (0.50, 0.60, 0.85)
    fo = bpy.data.objects.new("CaveFill", fd)
    lc.objects.link(fo)
    fo.location = (-0.5, -1.0, 4.9)
    fo.rotation_euler = (math.radians(8), 0, 0)
    return {"sun_dir": [round(x, 3) for x in d], "ring": [round(x, 3) for x in rp], "window": [round(x, 3) for x in tgt]}


# ------------------------------------------------------------------ dust in the air (a bounded volume, not the world)
def build_volume(density=0.030):
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Props", cave)
    _drop("Cave_Volume")
    me = bpy.data.meshes.new("Cave_Volume")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Cave_Volume", me)
    coll.objects.link(ob)
    ob.location = (0.0, 1.0, 3.0)
    ob.scale = (18.0, 22.0, 8.0)
    ob.display_type = 'BOUNDS'
    ob.visible_shadow = False
    m = bpy.data.materials.get("Cave_Dust") or bpy.data.materials.new("Cave_Dust")
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputMaterial')
    vol = N.new('ShaderNodeVolumePrincipled')
    vol.inputs['Color'].default_value = (0.85, 0.80, 0.72, 1.0)
    vol.inputs['Anisotropy'].default_value = 0.6
    L.new(vol.outputs['Volume'], out.inputs['Volume'])
    tc = N.new('ShaderNodeTexCoord')
    mp = N.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (0.35, 0.35, 0.55)
    L.new(tc.outputs['Object'], mp.inputs['Vector'])
    nz = N.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 2.2
    nz.inputs['Detail'].default_value = 3.0
    L.new(mp.outputs['Vector'], nz.inputs['Vector'])
    mr = N.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = 0.30
    mr.inputs['From Max'].default_value = 0.75
    mr.inputs['To Min'].default_value = 0.25
    mr.inputs['To Max'].default_value = 1.6
    L.new(nz.outputs['Fac'], mr.inputs['Value'])
    mul = N.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = density
    L.new(mr.outputs['Result'], mul.inputs[0])
    L.new(mul.outputs['Value'], vol.inputs['Density'])
    me.materials.append(m)
    return "ok"


# ------------------------------------------------------------------ states
def _rest_pos(N, L, label):
    """node output giving the undeformed (rest) position of the shading point in rig space. 'Object' coordinates follow the
    deformation, 'Generated' does not, so we un-normalise Generated with the rest-mesh bounding box."""
    me = bpy.data.objects["Relic_AI"].data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3)
    mn, mx = P.min(0), P.max(0)
    tc = N.new('ShaderNodeTexCoord')
    tc.label = label
    mp = N.new('ShaderNodeMapping')
    mp.label = label + " map"
    mp.vector_type = 'POINT'
    mp.inputs['Location'].default_value = tuple(float(x) for x in mn)
    mp.inputs['Scale'].default_value = tuple(float(x) for x in (mx - mn))
    L.new(tc.outputs['Generated'], mp.inputs['Vector'])
    return mp.outputs['Vector']


def clean_face(radius=0.15, fade=0.10):
    """the settled dust skips the visor glass so the eyes read when they light up"""
    ob = bpy.data.objects["Relic_AI"]
    nt = ob.data.materials[0].node_tree
    N, L = nt.nodes, nt.links
    pbn = next(n for n in N if n.type == 'BSDF_PRINCIPLED')
    # undo a previous run first so this stays idempotent
    for n in list(N):
        if n.label == "FC mul" and n.inputs[0].links:
            for l in list(L):
                if l.from_node == n:
                    L.new(n.inputs[0].links[0].from_socket, l.to_node.inputs[0])
        if n.label == "FC dark" and n.inputs[6].links:
            L.new(n.inputs[6].links[0].from_socket, pbn.inputs['Base Color'])
        if n.label == "FC rough" and n.inputs[2].links:
            L.new(n.inputs[2].links[0].from_socket, pbn.inputs['Roughness'])
        if n.label == "FC metal" and n.inputs[2].links:
            L.new(n.inputs[2].links[0].from_socket, pbn.inputs['Metallic'])
    for n in list(N):
        if n.label.startswith("FC"):
            N.remove(n)
    target = None
    for n in N:
        if n.type == 'MIX' and getattr(n, "data_type", "") == 'RGBA':
            c = n.inputs[7].default_value
            if abs(c[0] - 0.32) < 0.01 and abs(c[1] - 0.29) < 0.01 and abs(c[2] - 0.24) < 0.01:
                target = n
    if target is None:
        return None
    if target.inputs[0].links:
        src = target.inputs[0].links[0].from_socket
    else:
        return None
    rp = _rest_pos(N, L, "FC rest")
    vm = N.new('ShaderNodeVectorMath')
    vm.operation = 'DISTANCE'
    vm.label = "FC dist"
    vm.inputs[1].default_value = (0.0, -0.235, 1.862)
    L.new(rp, vm.inputs[0])
    mr = N.new('ShaderNodeMapRange')
    mr.label = "FC map"
    mr.clamp = True
    mr.inputs['From Min'].default_value = radius
    mr.inputs['From Max'].default_value = radius + fade
    L.new(vm.outputs['Value'], mr.inputs['Value'])
    mul = N.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    mul.label = "FC mul"
    L.new(src, mul.inputs[0])
    L.new(mr.outputs['Result'], mul.inputs[1])
    L.new(mul.outputs['Value'], target.inputs[0])
    # the glass inside the porthole goes dark so the glyph eyes pop when they light up
    pb = next(n for n in N if n.type == 'BSDF_PRINCIPLED')
    if pb.inputs['Base Color'].links:
        base_src = pb.inputs['Base Color'].links[0].from_socket
        vm2 = N.new('ShaderNodeVectorMath')
        vm2.operation = 'DISTANCE'
        vm2.label = "FC dist2"
        vm2.inputs[1].default_value = (0.0, -0.235, 1.862)
        L.new(rp, vm2.inputs[0])
        mr2 = N.new('ShaderNodeMapRange')
        mr2.label = "FC glass"
        mr2.clamp = True
        mr2.inputs['From Min'].default_value = 0.082
        mr2.inputs['From Max'].default_value = 0.100
        mr2.inputs['To Min'].default_value = 1.0
        mr2.inputs['To Max'].default_value = 0.0
        L.new(vm2.outputs['Value'], mr2.inputs['Value'])
        mx = N.new('ShaderNodeMix')
        mx.data_type = 'RGBA'
        mx.label = "FC dark"
        mx.inputs[7].default_value = (0.006, 0.007, 0.009, 1.0)
        L.new(mr2.outputs['Result'], mx.inputs[0])
        L.new(base_src, mx.inputs[6])
        L.new(mx.outputs[2], pb.inputs['Base Color'])
        # and it turns glassy: smooth and non-metal, so it shows small highlights instead of a pale wash
        for sock, lab, val in (('Roughness', "FC rough", 0.10), ('Metallic', "FC metal", 0.0)):
            if pb.inputs[sock].links:
                src_s = pb.inputs[sock].links[0].from_socket
                m2 = N.new('ShaderNodeMix')
                m2.data_type = 'FLOAT'
                m2.label = lab
                m2.inputs[3].default_value = val
                L.new(mr2.outputs['Result'], m2.inputs[0])
                L.new(src_s, m2.inputs[2])
                L.new(m2.outputs[0], pb.inputs[sock])
    return "ok"


def fix_head_weights(z0=1.78, dz=0.10, rad=0.21, rfade=0.09):
    """the porthole sat right on the head/neck pivot and was weighted 85% to the chest, so it lagged behind the glyph eyes
    whenever the head turned. Everything from the collar up and inside the helmet radius now follows the head bone."""
    out = {}
    for ob_name in ("Relic_AI", "Relic_AI_Crust"):
        ob = bpy.data.objects.get(ob_name)
        if not ob:
            continue
        me = ob.data
        if me.get("head_fixed"):
            out[ob_name] = "already"
            continue
        n = len(me.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        P = co.reshape(-1, 3).astype(np.float64)
        r = np.sqrt(P[:, 0] ** 2 + (P[:, 1] + 0.075) ** 2)
        w = smooth((P[:, 2] - z0) / dz) * (1.0 - smooth((r - rad) / rfade))
        idx = np.where(w > 0.01)[0]
        head_i = ob.vertex_groups["head"].index
        for i in idx:
            i = int(i)
            wi = float(w[i])
            cur = {g.group: g.weight for g in me.vertices[i].groups}
            new = {gi: wt * (1.0 - wi) for gi, wt in cur.items() if gi != head_i}
            new[head_i] = cur.get(head_i, 0.0) * (1.0 - wi) + wi
            for gi, wt in new.items():
                vg = ob.vertex_groups[gi]
                if wt > 0.002:
                    vg.add([i], wt, 'REPLACE')
                else:
                    try:
                        vg.remove([i])
                    except RuntimeError:
                        pass
        me["head_fixed"] = True
        out[ob_name] = int(len(idx))
    bpy.context.view_layer.update()
    return out


def tune_crust(thr=0.30, face_clear=0.17, ring_clear=0.13):
    """bigger thr = the mud and stone crust covers less of Relic (0.18 was the first-pass coverage).
    face_clear / ring_clear: radius (rig rest space) around the visor and the chest ring that stays free of crust"""
    m = bpy.data.materials.get("Relic_Crust")
    if not m:
        return None
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    nt.nodes["Math.007"].inputs[1].default_value = thr
    for n in list(N):
        if n.label.startswith("CV"):
            N.remove(n)
    alpha = nt.nodes["Math.008"]
    mix = next(n for n in N if n.type == 'MIX_SHADER')
    rp = _rest_pos(N, L, "CV rest")
    masks = []
    for (cx_, cy_, cz_, rad) in ((0.0, -0.235, 1.862, face_clear), (0.0, -0.215, 1.652, ring_clear), (0.40, 0.0, 1.66, 0.12)):
        vm = N.new('ShaderNodeVectorMath')
        vm.operation = 'DISTANCE'
        vm.inputs[1].default_value = (cx_, cy_, cz_)
        vm.label = "CV dist"
        L.new(rp, vm.inputs[0])
        mr = N.new('ShaderNodeMapRange')
        mr.label = "CV map"
        mr.clamp = True
        mr.inputs['From Min'].default_value = rad
        mr.inputs['From Max'].default_value = rad * 1.6
        L.new(vm.outputs['Value'], mr.inputs['Value'])
        masks.append(mr.outputs['Result'])
    cur = alpha.outputs[0]
    for mk in masks:
        mul = N.new('ShaderNodeMath')
        mul.operation = 'MULTIPLY'
        mul.label = "CV mul"
        L.new(cur, mul.inputs[0])
        L.new(mk, mul.inputs[1])
        cur = mul.outputs['Value']
    L.new(cur, mix.inputs[0])
    return thr


def build_controls():
    """One slider drives the whole opening: Cave_Controls["dug"], 0 = Relic buried and dormant, 1 = dug out and awake.
    It blends the 'Dug' shape key of the debris, hides the cover rocks at 0.5, and fades the ring, the glyph eyes and
    the cyan glow on the rocks. Select Cave_Controls > Object properties > Custom Properties, or keyframe it."""
    cave = get_coll(CAVE)
    ob = bpy.data.objects.get("Cave_Controls")
    if ob is None:
        ob = bpy.data.objects.new("Cave_Controls", None)
        ob.empty_display_type = 'ARROWS'
        ob.empty_display_size = 0.6
        cave.objects.link(ob)
        ob["dug"] = 0.0
    ob.location = (2.2, -1.0, 0.4)
    try:
        ob.id_properties_ui("dug").update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0,
                                           description="0 = Relic buried and dormant, 1 = dug out and awake")
    except Exception:
        pass

    # v09: the chest ring and the glyph eyes have their own sliders (ring, eyes) so the wake-up can be timed apart from the digging;
    # cave_state() still moves all three together
    for prop, desc in (("ring", "0 = chest ring dark, 1 = full glow"), ("eyes", "0 = visor eyes dark, 1 = lit")):
        if prop not in ob.keys():
            ob[prop] = 0.0
        try:
            ob.id_properties_ui(prop).update(min=0.0, max=2.0, soft_min=0.0, soft_max=1.0, description=desc)
        except Exception:
            pass

    def drive(idb, path, expr, prop="dug"):
        try:
            idb.driver_remove(path)
        except Exception:
            pass
        fc = idb.driver_add(path)
        drv = fc.driver
        drv.type = 'SCRIPTED'
        drv.expression = expr
        v = drv.variables.new()
        v.name = prop
        v.type = 'SINGLE_PROP'
        v.targets[0].id = ob
        v.targets[0].data_path = '["%s"]' % prop
        return fc

    deb = bpy.data.objects["Cave_Debris"]
    drive(deb.data.shape_keys, 'key_blocks["Dug"].value', "dug")
    n = 0
    for o in bpy.data.collections["Cave_Cover"].objects:
        drive(o, "hide_render", "dug > 0.5")
        drive(o, "hide_viewport", "dug > 0.5")
        n += 1
    g = bpy.data.objects.get("RingGlow")
    if g:
        drive(g.data, "energy", "ring * 8", "ring")
    for mname, st, prop in (("Relic_RingLight", 3.0, "ring"), ("Relic_Glyph", 1.7, "eyes")):
        m = bpy.data.materials[mname]
        b = next(nd for nd in m.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED')
        drive(m.node_tree, 'nodes["%s"].inputs["Emission Strength"].default_value' % b.name, "%s * %s" % (prop, st), prop)
    return {"cover_rocks": n}


def cave_state(dug):
    ob = bpy.data.objects["Cave_Controls"]
    ob["dug"] = 1.0 if dug else 0.0
    for k in ("ring", "eyes"):
        if k in ob.keys():
            ob[k] = 1.0 if dug else 0.0
    crust = bpy.data.objects.get("Relic_AI_Crust")
    if crust:
        crust.hide_viewport = False
        crust.hide_render = False
    ob.update_tag()
    bpy.context.view_layer.update()
    return "dug" if dug else "buried"
