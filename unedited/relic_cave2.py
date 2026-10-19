# relic_cave2.py : stage 3 of the cave set (materials, ground field, shell, floor, slope). Exec after relic_cave.py.


# ------------------------------------------------------------------ rig frame helpers
def rig_matrix():
    return bpy.data.objects[RIG].matrix_world.copy()


def to_world(P):
    """rig-local points (n,3) -> world (n,3)"""
    M = np.array(rig_matrix())
    P = np.asarray(P, dtype=np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


# ------------------------------------------------------------------ materials
def rock_material(name, c1, c2, dust=(0.20, 0.17, 0.14), dust_amt=0.8, rough=0.88, big=3.5, bump=0.9,
                  bump_scale=14.0, strata=0.0, rand_amt=0.35):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputMaterial')
    pb = N.new('ShaderNodeBsdfPrincipled')
    L.new(pb.outputs['BSDF'], out.inputs['Surface'])
    h = _nh(nt)
    info = N.new('ShaderNodeObjectInfo')
    rnd = info.outputs['Random']
    sx = (1.0, 1.0, 1.0 / (1.0 + strata * 6.0))
    nbig = h.noise(h.mapping(sx), big, 4)
    nmid = h.noise(h.mapping((1, 1, 1)), big * 4.0, 5)
    nfine = h.noise(h.mapping((1, 1, 1)), big * 22.0, 3)
    col = h.mixc(h.ramp(nbig, 0.35, 0.65), c1, c2)
    # per-instance brightness shift: colour * (1 - rand/2 + rand * random)
    shift = h.fm('ADD', 1.0 - rand_amt * 0.5, h.fm('MULTIPLY', rnd, rand_amt))
    grey = N.new('ShaderNodeCombineColor')
    L.new(shift, grey.inputs[0])
    L.new(shift, grey.inputs[1])
    L.new(shift, grey.inputs[2])
    mul = N.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs[0].default_value = 1.0
    L.new(col, mul.inputs[6])
    L.new(grey.outputs['Color'], mul.inputs[7])
    col = mul.outputs[2]
    # dust settles on upward-facing surfaces and in crevices
    geo = N.new('ShaderNodeNewGeometry')
    sepn = N.new('ShaderNodeSeparateXYZ')
    L.new(geo.outputs['Normal'], sepn.inputs['Vector'])
    um = N.new('ShaderNodeMapRange')
    um.clamp = True
    um.inputs['From Min'].default_value = 0.30
    um.inputs['From Max'].default_value = 0.95
    L.new(sepn.outputs['Z'], um.inputs['Value'])
    up = um.outputs['Result']
    ao = N.new('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.25
    crev = h.fm('SUBTRACT', 1.0, ao.outputs['AO'], clamp=True)
    dmask = h.fm('MULTIPLY', h.fm('ADD', h.fm('MULTIPLY', up, 0.9), h.fm('MULTIPLY', crev, 0.55)),
                 h.ramp(nmid, 0.30, 0.62), clamp=True)
    dmask = h.fm('MULTIPLY', dmask, dust_amt, clamp=True)
    col = h.mixc(dmask, col, dust)
    L.new(col, pb.inputs['Base Color'])
    pb.inputs['Metallic'].default_value = 0.0
    L.new(h.mixf(dmask, h.fm('ADD', rough - 0.12, h.fm('MULTIPLY', nfine, 0.2)), 0.97), pb.inputs['Roughness'])
    bp = N.new('ShaderNodeBump')
    bp.inputs['Distance'].default_value = 0.03
    bp.inputs['Strength'].default_value = bump
    hn = h.fm('ADD', h.fm('MULTIPLY', h.noise(h.mapping((1, 1, 1)), bump_scale, 5), 0.6),
              h.fm('MULTIPLY', nfine, 0.4))
    L.new(hn, bp.inputs['Height'])
    L.new(bp.outputs['Normal'], pb.inputs['Normal'])
    return m


def metal_material(name, base=(0.25, 0.09, 0.035), rough=0.55, metallic=0.7):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputMaterial')
    pb = N.new('ShaderNodeBsdfPrincipled')
    L.new(pb.outputs['BSDF'], out.inputs['Surface'])
    h = _nh(nt)
    n1 = h.noise(h.mapping((1, 1, 1)), 6.0, 5)
    L.new(h.mixc(h.ramp(n1, 0.35, 0.7), base, (base[0] * 0.35, base[1] * 0.35, base[2] * 0.35)), pb.inputs['Base Color'])
    pb.inputs['Metallic'].default_value = metallic
    L.new(h.mixf(h.ramp(n1, 0.4, 0.7), rough - 0.15, rough + 0.3), pb.inputs['Roughness'])
    return m


def root_material(name="Cave_Root"):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputMaterial')
    pb = N.new('ShaderNodeBsdfPrincipled')
    L.new(pb.outputs['BSDF'], out.inputs['Surface'])
    h = _nh(nt)
    n1 = h.noise(h.mapping((1, 1, 6)), 30.0, 3)
    L.new(h.mixc(h.ramp(n1, 0.3, 0.7), (0.32, 0.25, 0.16), (0.12, 0.085, 0.05)), pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.85
    return m


# ------------------------------------------------------------------ ground field (rig-local)
BODY_RECT = (-0.55, 0.55, 0.0, 2.2)      # x0, x1, z0, z1 footprint of the body in rig-local x,z


def body_dist(x, z):
    x0, x1, z0, z1 = BODY_RECT
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dz = np.maximum(np.maximum(z0 - z, z - z1), 0.0)
    return np.sqrt(dx * dx + dz * dz)


def make_lumps(seed=11, n=40):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.uniform(-8.5, 8.5)
        z = rng.uniform(1.2, 10.5)
        if body_dist(np.array([x]), np.array([z]))[0] < 1.3:
            continue
        out.append((x, z, rng.uniform(0.35, 1.5), rng.uniform(0.35, 1.5), rng.uniform(0.18, 0.75)))
    return out


LUMPS = make_lumps()


def ground_h(x, z):
    """height of the rubble ground above the slope plane, toward the camera (>0 piles up, <0 dips)"""
    P = np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=1)
    h = 0.55 * (fbm(P, 0.22, 3, 1) - 0.5) * 2 + 0.20 * (fbm(P, 0.9, 3, 2) - 0.5) * 2 + 0.045 * (fbm(P, 4.0, 2, 3) - 0.5) * 2
    for (cx, cz, rx, rz, a) in LUMPS:
        h = h + a * np.exp(-(((P[:, 0] - cx) / rx) ** 2 + ((P[:, 2] - cz) / rz) ** 2))
    h = h.reshape(x.shape)
    w = smooth(body_dist(x, z) / 0.9)
    return h * w


# ------------------------------------------------------------------ stage 3: cave shell, floor, slope
SHELL_C = np.array([0.0, 1.0, 1.5])
SHELL_R = np.array([8.5, 10.5, 5.2])
FLOOR_Z = -0.6


def _drop(name):
    o = bpy.data.objects.get(name)
    if o:
        d = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if d is not None and d.users == 0 and d.name in bpy.data.meshes:
            bpy.data.meshes.remove(d)


def build_shell():
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Shell", cave)
    _drop("Cave_Shell")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=7, radius=1.0)
    U = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int32)
    bm.free()
    p0 = SHELL_C + U * SHELL_R
    nrm = U / SHELL_R
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    s = (1.4 * (fbm(p0, 0.11, 3, 21) - 0.5) * 2 + 0.5 * (fbm(p0, 0.42, 3, 22) - 0.5) * 2 + 0.10 * (fbm(p0, 1.9, 3, 23) - 0.5) * 2)
    rg = ridge(p0 * np.array([1.0, 1.0, 0.30]), 0.30, 3, 24)
    s = s - 0.55 * smooth((rg - 0.78) / 0.2)                  # narrow vertical fissures
    s = s + 0.12 * (fbm(p0 * np.array([0.4, 0.4, 3.2]), 1.0, 2, 25) - 0.5)
    p = p0 + nrm * s[:, None]
    me = mesh_from_arrays("Cave_Shell", p, F[:, ::-1])       # reversed: normals face the inside
    set_smooth(me)
    ob = bpy.data.objects.new("Cave_Shell", me)
    coll.objects.link(ob)
    return {"verts": len(me.vertices), "faces": len(me.polygons)}


def build_floor():
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Ground", cave)
    _drop("Cave_Floor")
    nx = 190
    xs = np.linspace(-15, 15, nx)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    z = FLOOR_Z + 0.42 * (fbm(P, 0.18, 3, 31) - 0.5) * 2 + 0.10 * (fbm(P, 0.9, 3, 32) - 0.5) * 2 + 0.03 * (fbm(P, 4.0, 2, 33) - 0.5) * 2
    V = np.stack([X.ravel(), Y.ravel(), z], axis=1)
    me = mesh_from_arrays("Cave_Floor", V, grid_faces(nx, nx, flip=True))
    set_smooth(me)
    ob = bpy.data.objects.new("Cave_Floor", me)
    coll.objects.link(ob)
    return {"verts": len(me.vertices)}


def build_slope():
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Ground", cave)
    _drop("Cave_Slope")
    step = 0.12
    xs = np.arange(-11.0, 11.0 + 1e-6, step)
    zs = np.arange(-3.0, 11.0 + 1e-6, step)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    H = ground_h(X, Z)
    Y = BACK - H + 0.03            # 3 cm behind the fine debris patch so they never z-fight
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    me = mesh_from_arrays("Cave_Slope", V, grid_faces(len(xs), len(zs), flip=False))
    set_smooth(me)
    ob = bpy.data.objects.new("Cave_Slope", me)
    coll.objects.link(ob)
    ob.matrix_world = rig_matrix()
    return {"verts": len(me.vertices)}


def assign_cave_materials():
    wall = rock_material("Cave_Wall", (0.020, 0.018, 0.017), (0.060, 0.052, 0.045), dust=(0.16, 0.13, 0.10), dust_amt=0.45,
                         rough=0.9, big=0.9, bump=1.0, bump_scale=3.0, strata=0.6)
    floor = rock_material("Cave_FloorMat", (0.025, 0.022, 0.020), (0.070, 0.060, 0.050), dust=(0.20, 0.17, 0.13), dust_amt=0.8,
                          rough=0.92, big=1.4, bump=1.0, bump_scale=8.0)
    slope = rock_material("Cave_SlopeMat", (0.018, 0.016, 0.014), (0.055, 0.047, 0.040), dust=(0.15, 0.12, 0.095), dust_amt=0.55,
                          rough=0.9, big=2.2, bump=1.2, bump_scale=18.0)
    for name, mat in (("Cave_Shell", wall), ("Cave_Floor", floor), ("Cave_Slope", slope), ("Cave_Debris", slope)):
        o = bpy.data.objects.get(name)
        if o:
            o.data.materials.clear()
            o.data.materials.append(mat)
    return "ok"
