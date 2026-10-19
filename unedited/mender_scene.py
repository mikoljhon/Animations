# mender_scene.py : instantiate Mender in Blender (objects, skinning, bone parenting), materials, studio scene, renders, design sheet.
# Exec after mender_build.py. build_mender() then make_studio() then design_sheet().


def _nh(nt):
    N, L = nt.nodes, nt.links
    tc = N.new('ShaderNodeTexCoord')

    def mapping(sc, out='Object'):
        mp = N.new('ShaderNodeMapping')
        mp.inputs['Scale'].default_value = sc
        L.new(tc.outputs[out], mp.inputs['Vector'])
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


# ------------------------------------------------------------------ materials
def _new_mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])
    return m, nt, pb


def _crev(h, dist=0.05):
    ao = h.N.new('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = dist
    return h.fm('SUBTRACT', 1.0, ao.outputs['AO'], clamp=True)


def rest_mapping(h, ob):
    """mapping node giving the REST position in metres (Generated un-normalised by the mesh bounding box), for skinned objects"""
    me = ob.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3)
    mn, mx = P.min(0), P.max(0)
    mp = h.N.new('ShaderNodeMapping')
    mp.vector_type = 'POINT'
    mp.inputs['Location'].default_value = tuple(float(x) for x in mn)
    mp.inputs['Scale'].default_value = tuple(float(x) for x in (mx - mn))
    h.L.new(h.tc.outputs['Generated'], mp.inputs['Vector'])
    return mp


def pbr(name, c1, c2, metal=0.0, rough=0.6, nscale=6.0, bump=0.25, bscale=60.0, dirt=0.35, dirt_col=(0.035, 0.028, 0.022),
        wear_col=None, wear=0.0, spec=0.5, ob=None):
    m, nt, pb = _new_mat(name)
    h = _nh(nt)
    mp = rest_mapping(h, ob) if ob is not None else h.mapping((1, 1, 1))
    n1 = h.noise(mp, nscale, 5)
    col = h.mixc(h.ramp(n1, 0.35, 0.65), c1, c2)
    crev = _crev(h, 0.04)
    n2 = h.noise(mp, nscale * 3.1, 4)
    dmask = h.fm('MULTIPLY', h.fm('MULTIPLY', crev, dirt * 2.0), h.ramp(n2, 0.30, 0.62), clamp=True)
    col = h.mixc(dmask, col, dirt_col)
    if wear_col is not None:
        n3 = h.noise(mp, nscale * 5.0, 5)
        wm = h.fm('MULTIPLY', h.ramp(n3, 0.58, 0.72), wear, clamp=True)
        col = h.mixc(wm, col, wear_col)
    nb = h.fm('ADD', h.fm('MULTIPLY', h.noise(mp, bscale, 4), 0.6), h.fm('MULTIPLY', h.noise(mp, bscale * 4.0, 3), 0.4))
    bp = h.N.new('ShaderNodeBump')
    bp.inputs['Distance'].default_value = 0.004
    bp.inputs['Strength'].default_value = bump
    h.L.new(nb, bp.inputs['Height'])
    h.L.new(col, pb.inputs['Base Color'])
    pb.inputs['Metallic'].default_value = metal
    h.L.new(h.mixf(dmask, rough, min(rough + 0.25, 1.0)), pb.inputs['Roughness'])
    pb.inputs['Specular IOR Level'].default_value = spec
    h.L.new(bp.outputs['Normal'], pb.inputs['Normal'])
    return m


def suit_material(name, ob, c1, c2, fade_col, dirt_col, weave=True, dirt=0.9, bump=0.7):
    """cloth for the skinned suit: texture coordinates are the REST position (Generated un-normalised), so nothing swims when he moves"""
    me = ob.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3)
    mn, mx = P.min(0), P.max(0)
    m, nt, pb = _new_mat(name)
    h = _nh(nt)
    N, L = h.N, h.L
    tcg = N.new('ShaderNodeTexCoord')
    mp = N.new('ShaderNodeMapping')
    mp.vector_type = 'POINT'
    mp.inputs['Location'].default_value = tuple(float(x) for x in mn)
    mp.inputs['Scale'].default_value = tuple(float(x) for x in (mx - mn))
    L.new(tcg.outputs['Generated'], mp.inputs['Vector'])
    n_big = h.noise(mp, 3.0, 4)
    n_mid = h.noise(mp, 11.0, 5)
    n_fine = h.noise(mp, 70.0, 3)
    col = h.mixc(h.ramp(n_big, 0.35, 0.65), c1, c2)
    geo = N.new('ShaderNodeNewGeometry')
    sepn = N.new('ShaderNodeSeparateXYZ')
    L.new(geo.outputs['Normal'], sepn.inputs['Vector'])
    um = N.new('ShaderNodeMapRange')
    um.clamp = True
    um.inputs['From Min'].default_value = 0.25
    um.inputs['From Max'].default_value = 0.95
    L.new(sepn.outputs['Z'], um.inputs['Value'])
    fade = h.fm('MULTIPLY', um.outputs['Result'], h.ramp(n_mid, 0.25, 0.7), clamp=True)
    col = h.mixc(h.fm('MULTIPLY', fade, 0.55), col, fade_col)
    sepr = N.new('ShaderNodeSeparateXYZ')
    L.new(mp.outputs['Vector'], sepr.inputs['Vector'])
    low = N.new('ShaderNodeMapRange')
    low.clamp = True
    low.inputs['From Min'].default_value = 0.10
    low.inputs['From Max'].default_value = 0.85
    low.inputs['To Min'].default_value = 1.0
    low.inputs['To Max'].default_value = 0.0
    L.new(sepr.outputs['Z'], low.inputs['Value'])
    crev = _crev(h, 0.05)
    dm = h.fm('MULTIPLY', h.fm('ADD', h.fm('MULTIPLY', low.outputs['Result'], 0.85), h.fm('MULTIPLY', crev, 0.7)),
              h.ramp(n_mid, 0.22, 0.60), clamp=True)
    col = h.mixc(h.fm('MULTIPLY', dm, dirt, clamp=True), col, dirt_col)
    stain = h.fm('MULTIPLY', h.ramp(n_big, 0.60, 0.72), 0.8, clamp=True)
    col = h.mixc(stain, col, (0.025, 0.02, 0.018))
    # long vertical grime streaks (dust and rain running down the cloth)
    mp2 = N.new('ShaderNodeMapping')
    mp2.vector_type = 'POINT'
    mp2.inputs['Scale'].default_value = (1.0, 1.0, 0.12)
    L.new(mp.outputs['Vector'], mp2.inputs['Vector'])
    streak = h.fm('MULTIPLY', h.ramp(h.noise(mp2, 16.0, 4), 0.52, 0.72), 0.55, clamp=True)
    col = h.mixc(h.fm('MULTIPLY', streak, dirt, clamp=True), col, dirt_col)
    L.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.92
    pb.inputs['Specular IOR Level'].default_value = 0.25
    hgt = h.fm('MULTIPLY', n_fine, 0.4)
    if weave:
        wx = N.new('ShaderNodeTexWave')
        wx.wave_type = 'BANDS'
        wx.bands_direction = 'X'
        wx.inputs['Scale'].default_value = 420.0
        wy = N.new('ShaderNodeTexWave')
        wy.wave_type = 'BANDS'
        wy.bands_direction = 'Y'
        wy.inputs['Scale'].default_value = 420.0
        L.new(mp.outputs['Vector'], wx.inputs['Vector'])
        L.new(mp.outputs['Vector'], wy.inputs['Vector'])
        hgt = h.fm('ADD', hgt, h.fm('MULTIPLY', h.fm('ADD', wx.outputs['Fac'], wy.outputs['Fac']), 0.3))
    bp = N.new('ShaderNodeBump')
    bp.inputs['Distance'].default_value = 0.003
    bp.inputs['Strength'].default_value = bump
    L.new(hgt, bp.inputs['Height'])
    L.new(bp.outputs['Normal'], pb.inputs['Normal'])
    return m


def glow_mat(name, color, strength):
    m, nt, pb = _new_mat(name)
    pb.inputs['Base Color'].default_value = (color[0] * 0.2, color[1] * 0.2, color[2] * 0.2, 1)
    pb.inputs['Emission Color'].default_value = (color[0], color[1], color[2], 1)
    pb.inputs['Emission Strength'].default_value = strength
    pb.inputs['Roughness'].default_value = 0.3
    return m


def make_materials(suit_ob, boots_ob=None, hand_ob=None):
    M = {}
    if boots_ob is not None:
        M['boot0'] = pbr('Mender_BootUpper', (0.060, 0.034, 0.016), (0.135, 0.075, 0.036), rough=0.55, nscale=7, bump=0.22, bscale=110,
                         dirt=1.1, dirt_col=(0.04, 0.032, 0.025), wear_col=(0.24, 0.16, 0.09), wear=0.45, ob=boots_ob)
        M['boot1'] = pbr('Mender_BootToeCap', (0.10, 0.10, 0.11), (0.22, 0.22, 0.23), metal=1.0, rough=0.5, nscale=8, bump=0.35,
                         dirt=0.9, wear_col=(0.20, 0.08, 0.03), wear=0.7, ob=boots_ob)
        M['boot2'] = pbr('Mender_BootSole', (0.012, 0.012, 0.013), (0.035, 0.033, 0.030), rough=0.8, nscale=8, bump=0.4,
                         dirt=0.6, dirt_col=(0.08, 0.07, 0.06), ob=boots_ob)
        M['boot3'] = pbr('Mender_BootLace', (0.18, 0.15, 0.09), (0.30, 0.25, 0.15), rough=0.85, nscale=30, bump=0.7, bscale=150,
                         dirt=0.8, ob=boots_ob)
        M['boot4'] = pbr('Mender_BootEyelet', (0.30, 0.30, 0.31), (0.50, 0.50, 0.52), metal=1.0, rough=0.35, nscale=8, bump=0.1,
                         dirt=0.5, ob=boots_ob)
        M['boot5'] = pbr('Mender_BootLeatherDark', (0.035, 0.020, 0.012), (0.080, 0.045, 0.025), rough=0.62, nscale=9, bump=0.45,
                         bscale=40, dirt=0.9, wear_col=(0.20, 0.13, 0.07), wear=0.45, ob=boots_ob)
    if hand_ob is not None:
        M['hskin'] = pbr('Mender_HandSkin', (0.22, 0.125, 0.080), (0.30, 0.175, 0.115), rough=0.5, nscale=14, bump=0.18, bscale=120,
                         dirt=0.9, dirt_col=(0.05, 0.035, 0.025), ob=hand_ob)
        M['hglove'] = pbr('Mender_GloveSuede', (0.17, 0.105, 0.055), (0.29, 0.185, 0.10), rough=0.85, nscale=9, bump=0.6, bscale=100,
                          dirt=1.0, dirt_col=(0.05, 0.035, 0.02), wear_col=(0.40, 0.30, 0.18), wear=0.4, ob=hand_ob)
        M['hglove2'] = pbr('Mender_GlovePalm', (0.05, 0.03, 0.018), (0.11, 0.07, 0.04), rough=0.7, nscale=9, bump=0.5, bscale=80,
                           dirt=1.0, ob=hand_ob)
        M['hstrap'] = pbr('Mender_GloveStrap', (0.035, 0.030, 0.025), (0.080, 0.070, 0.055), rough=0.8, nscale=8, bump=0.3,
                          dirt=0.6, ob=hand_ob)
        M['hnail'] = pbr('Mender_Nail', (0.27, 0.17, 0.13), (0.34, 0.22, 0.17), rough=0.4, nscale=20, bump=0.05, dirt=0.5, dirt_col=(0.08, 0.06, 0.04), ob=hand_ob)
    M['coverall'] = suit_material('Mender_Coverall', suit_ob, (0.30, 0.085, 0.020), (0.46, 0.16, 0.040), (0.62, 0.34, 0.14), (0.045, 0.032, 0.024), dirt=1.35)
    M['patch'] = suit_material('Mender_Patch', suit_ob, (0.028, 0.115, 0.115), (0.05, 0.20, 0.19), (0.25, 0.38, 0.34), (0.035, 0.04, 0.035), dirt=1.0)
    M['pad'] = suit_material('Mender_Pad', suit_ob, (0.040, 0.030, 0.024), (0.075, 0.056, 0.042), (0.20, 0.16, 0.12), (0.03, 0.025, 0.02), weave=False, dirt=0.6, bump=0.5)
    M['patch2'] = suit_material('Mender_PatchOlive', suit_ob, (0.060, 0.065, 0.035), (0.11, 0.12, 0.065), (0.30, 0.30, 0.20), (0.04, 0.04, 0.03), dirt=1.0)
    M['patch3'] = suit_material('Mender_PatchMustard', suit_ob, (0.34, 0.22, 0.025), (0.50, 0.34, 0.055), (0.70, 0.55, 0.25), (0.05, 0.04, 0.03), dirt=1.0)
    M['steelDark'] = pbr('Mender_SteelDark', (0.09, 0.09, 0.10), (0.20, 0.20, 0.21), metal=1.0, rough=0.5, nscale=8, bump=0.35, dirt=0.8, wear_col=(0.20, 0.08, 0.03), wear=0.7)
    M['skin'] = pbr('Mender_Skin', (0.24, 0.135, 0.085), (0.32, 0.185, 0.12), rough=0.55, nscale=9, bump=0.1, dirt=0.2)
    M['hat'] = pbr('Mender_Hat', (0.40, 0.27, 0.045), (0.55, 0.40, 0.085), rough=0.45, nscale=5, bump=0.55, bscale=25, dirt=1.0, dirt_col=(0.04, 0.03, 0.02), wear_col=(0.06, 0.055, 0.05), wear=0.7)
    M['steel'] = pbr('Mender_Steel', (0.32, 0.32, 0.33), (0.50, 0.50, 0.52), metal=1.0, rough=0.42, nscale=8, bump=0.3, dirt=0.6, wear_col=(0.22, 0.09, 0.04), wear=0.5)
    M['paint'] = pbr('Mender_Paint', (0.60, 0.42, 0.04), (0.72, 0.55, 0.08), metal=0.15, rough=0.5, nscale=6, bump=0.35, dirt=0.8, wear_col=(0.30, 0.30, 0.31), wear=0.7)
    M['rubber'] = pbr('Mender_Rubber', (0.012, 0.012, 0.014), (0.03, 0.03, 0.033), rough=0.72, nscale=7, bump=0.35, dirt=0.5, dirt_col=(0.06, 0.055, 0.05))
    M['strap'] = pbr('Mender_Strap', (0.050, 0.045, 0.036), (0.10, 0.085, 0.065), rough=0.75, nscale=8, bump=0.3, dirt=0.5)
    M['leather'] = pbr('Mender_Leather', (0.10, 0.050, 0.022), (0.20, 0.10, 0.045), rough=0.55, nscale=8, bump=0.45, bscale=45, dirt=0.8, wear_col=(0.30, 0.20, 0.10), wear=0.5)
    M['leatherB'] = pbr('Mender_BootLeather', (0.07, 0.04, 0.02), (0.15, 0.08, 0.04), rough=0.6, nscale=9, bump=0.5, bscale=40, dirt=1.0, dirt_col=(0.05, 0.04, 0.03), wear_col=(0.22, 0.15, 0.08), wear=0.5)
    M['canvas'] = pbr('Mender_PackCanvas', (0.07, 0.085, 0.045), (0.14, 0.16, 0.09), rough=0.9, nscale=4, bump=0.6, bscale=120, dirt=1.0, dirt_col=(0.04, 0.035, 0.025))
    M['canvas2'] = pbr('Mender_PackCanvas2', (0.10, 0.075, 0.040), (0.19, 0.14, 0.075), rough=0.9, nscale=4, bump=0.6, bscale=120, dirt=1.0, dirt_col=(0.04, 0.035, 0.025))
    M['rope'] = pbr('Mender_Rope', (0.20, 0.15, 0.08), (0.34, 0.26, 0.14), rough=0.9, nscale=40, bump=0.9, bscale=150, dirt=0.6)
    M['mat'] = pbr('Mender_Mat', (0.10, 0.12, 0.13), (0.18, 0.20, 0.20), rough=0.9, nscale=6, bump=0.4, dirt=0.8)
    M['scarf'] = pbr('Mender_Scarf', (0.22, 0.035, 0.03), (0.34, 0.06, 0.045), rough=0.9, nscale=5, bump=0.5, bscale=100, dirt=0.7)
    M['glove'] = pbr('Mender_Glove', (0.18, 0.12, 0.07), (0.30, 0.20, 0.11), rough=0.8, nscale=7, bump=0.5, bscale=60, dirt=1.0)
    M['wood'] = pbr('Mender_Wood', (0.14, 0.08, 0.035), (0.26, 0.15, 0.07), rough=0.65, nscale=20, bump=0.4, bscale=30, dirt=0.4)
    M['filter'] = pbr('Mender_Filter', (0.50, 0.48, 0.40), (0.68, 0.66, 0.58), rough=0.55, nscale=6, bump=0.3, dirt=0.9)
    M['port'] = pbr('Mender_RegistryPort', (0.70, 0.72, 0.75), (0.85, 0.87, 0.90), metal=1.0, rough=0.18, nscale=3, bump=0.05, dirt=0.0)
    M['lens'] = pbr('Mender_Lens', (0.004, 0.005, 0.005), (0.012, 0.014, 0.012), rough=0.06, nscale=14, bump=0.1, dirt=0.35, spec=0.9)
    M['led'] = glow_mat('Mender_LED', (0.25, 0.65, 1.0), 6.0)
    M['lamp'] = glow_mat('Mender_Lamp', (1.0, 0.92, 0.75), 5.0)
    M['lamp2'] = glow_mat('Mender_Lantern', (1.0, 0.62, 0.25), 2.5)
    M['screen'] = glow_mat('Mender_Screen', (0.35, 1.0, 0.55), 2.2)
    return M


# ------------------------------------------------------------------ instantiate
def _coll(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def wipe(name):
    c = bpy.data.collections.get(name)
    if not c:
        return
    for o in list(c.objects):
        d = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if d is not None and getattr(d, "users", 1) == 0:
            for coll in (bpy.data.meshes, bpy.data.armatures):
                if d.name in coll and coll[d.name] == d:
                    coll.remove(d)
                    break
    bpy.data.collections.remove(c)


def make_obj(name, V, polys, coll, smooth=True):
    V = np.asarray(V, float) * SC
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], [tuple(int(i) for i in p) for p in polys])
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def make_skinned(name, S, coll, rig):
    """one mesh object from a Suit accumulator: material slots by index, vertex groups from the weights, Armature modifier"""
    V, polys, mats, W = suit_arrays(S)
    ob = make_obj(name, V, polys, coll)
    for i, mi in enumerate(mats):
        ob.data.polygons[i].material_index = mi
    for b, arr in W.items():
        vg = ob.vertex_groups.new(name=b)
        q = np.round(arr * 32).astype(int)
        for lv in np.unique(q):
            if lv <= 0:
                continue
            vg.add([int(i) for i in np.where(q == lv)[0]], lv / 32.0, 'REPLACE')
    mod = ob.modifiers.new("Armature", 'ARMATURE')
    mod.object = rig
    ob.parent = rig
    return ob


def leg_cuffs():
    parts = []
    for s, sx in (('L', 1.0), ('R', -1.0)):
        V, p = torus((0.110 * sx, 0.015, 0.176), 0.0835, 0.0075, axis=(0, 0, 1), nu=30, nv=8)
        parts.append(('LegCuff' + s, V, p, 'strap', 'shin.' + s))
    return parts


def build_mender():
    for n in list(bpy.data.collections.keys()):
        if n == MC:
            wipe(n)
    for me in list(bpy.data.meshes):
        if me.name.startswith("Mender_") and me.users == 0:
            bpy.data.meshes.remove(me)
    coll = _coll(MC)
    rig = make_rig(coll)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'XYZ'
    setup_ik(rig)
    # the skinned suit
    S, chains = build_suit()
    S = build_suit_details(S)
    V, polys, mats, W = suit_arrays(S)
    suit = make_skinned("Mender_Suit", S, coll, rig)
    boots_ob = make_skinned("Mender_Boots", build_boots_skin(), coll, rig)
    hand_ob = make_skinned("Mender_HandL", build_hand_skin(), coll, rig)
    M = make_materials(suit, boots_ob, hand_ob)
    for k in ('coverall', 'patch', 'pad', 'patch2', 'patch3'):
        suit.data.materials.append(M[k])
    for k in ('boot0', 'boot1', 'boot2', 'boot3', 'boot4', 'boot5'):
        boots_ob.data.materials.append(M[k])
    for k in ('hskin', 'hglove', 'hglove2', 'hstrap', 'hnail'):
        hand_ob.data.materials.append(M[k])
    # rigid gear, parented to bones
    parts = []
    for fn in (head_gear, hose_and_cable, scarf, backpack, belt_and_pouches, leg_cuffs, mech_arm_parts):
        parts += fn()
    tree = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(q) for q in polys])
    parts += shoulder_straps(tree)
    objs = []
    for name, Vp, Pp, mk, bone in parts:
        ob = make_obj("Mender_" + name, Vp, Pp, coll)
        ob.data.materials.append(M[mk])
        ob.parent = rig
        ob.parent_type = 'BONE'
        ob.parent_bone = bone
        objs.append(ob)
    bpy.context.view_layer.update()
    for ob in objs:
        ob.matrix_parent_inverse = ob.matrix_world.inverted()
    # props: the pick (stowed across the pack) and the scanner (in its hip holster); both can be re-attached for hand poses
    pp = props()
    pick = make_prop("Mender_Pick", [q for q in pp if q[0].startswith("Pick")], M, coll)
    scanner = make_prop("Mender_Scanner", [q for q in pp if q[0].startswith("Scanner")], M, coll)
    stow_props(rig, pick, scanner)
    return {"suit_verts": len(suit.data.vertices), "suit_polys": len(suit.data.polygons), "rigid_parts": len(objs), "props": 2,
            "bones": len(rig.data.bones)}


def make_prop(name, parts, M, coll):
    keys, Vs, Ps, mi, off = [], [], [], [], 0
    for (pn, V, polys, mk, bone) in parts:
        if mk not in keys:
            keys.append(mk)
        Vs.append(V)
        Ps.extend(tuple(int(i) + off for i in q) for q in polys)
        mi.extend([keys.index(mk)] * len(polys))
        off += len(V)
    ob = make_obj(name, np.concatenate(Vs), Ps, coll)
    for i, m in enumerate(mi):
        ob.data.polygons[i].material_index = m
    for k in keys:
        ob.data.materials.append(M[k])
    return ob


def attach(ob, rig, bone, W):
    """parent `ob` to a bone while keeping the world matrix W (the rig must be in its rest pose and at the origin)"""
    ob.parent = rig
    ob.parent_type = 'BONE'
    ob.parent_bone = bone
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    Pm = ob.matrix_world.copy()
    ob.matrix_parent_inverse = Pm.inverted()
    ob.matrix_basis = W


def frame4(origin, R):
    M4 = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            M4[r][c] = float(R[r, c])
        M4[r][3] = float(origin[r])
    return M4


def stow_props(rig, pick, scanner):
    scv = np.asarray(SC, float)
    B = np.array([-0.17, 0.338, 0.86]) * scv
    T = np.array([0.15, 0.338, 1.78]) * scv
    zdir = unit(T - B)
    attach(pick, rig, 'chest', frame4(B, basis(zdir, (0, 1, 0))))
    a = math.radians(-8.0)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    attach(scanner, rig, 'pelvis', frame4(np.array([0.118, -0.150, 0.925]) * scv, Rx))


# ------------------------------------------------------------------ studio
def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def make_studio():
    sc = bpy.data.scenes.get("Mender_Studio") or bpy.data.scenes.new("Mender_Studio")
    sc.render.engine = 'BLENDER_EEVEE'
    ee = sc.eevee
    ee.use_raytracing = True
    ee.ray_tracing_method = 'SCREEN'
    ee.taa_render_samples = 48
    ee.shadow_ray_count = 2
    ee.shadow_step_count = 8
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = 0.0
    sc.render.resolution_x, sc.render.resolution_y = 700, 1000
    sc.render.resolution_percentage = 100
    w = bpy.data.worlds.get("Mender_World") or bpy.data.worlds.new("Mender_World")
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.11, 0.115, 0.125, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    sc.world = w
    mc = bpy.data.collections.get(MC)
    if mc is not None and mc.name not in [c.name for c in sc.collection.children]:
        sc.collection.children.link(mc)
    sc_coll = _coll("Studio_Set", sc.collection)
    for o in list(sc_coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    # seamless backdrop: floor curving up into a wall
    prof = [(-14.0, 0.0), (0.0, 0.0), (4.0, 0.0)]
    for t in np.linspace(0.0, math.pi / 2, 14)[1:]:
        prof.append((4.0 + 3.0 * math.sin(t), 3.0 - 3.0 * math.cos(t)))
    prof += [(7.0, 8.0), (7.0, 14.0)]
    Vc, Pc = [], []
    for (yy, zz) in prof:
        Vc += [(-25.0, yy, zz), (25.0, yy, zz)]
    for i in range(len(prof) - 1):
        Pc.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
    me = bpy.data.meshes.new("Studio_Floor")
    me.from_pydata(Vc, [], Pc)
    me.update()
    for p_ in me.polygons:
        p_.use_smooth = True
    fo = bpy.data.objects.new("Studio_Floor", me)
    sc_coll.objects.link(fo)
    fm, nt2, pb = _new_mat("Studio_FloorMat")
    pb.inputs['Base Color'].default_value = (0.20, 0.20, 0.21, 1)
    pb.inputs['Roughness'].default_value = 0.9
    me.materials.append(fm)
    # lights
    def area(name, size, energy, loc, color=(1, 1, 1), target=(0, 0, 0.95)):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy
        ld.size = size
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        sc_coll.objects.link(lo)
        lo.location = loc
        look_at(lo, target)
        return lo
    area("Key", 2.6, 650, (-2.8, -3.6, 3.0), (1.0, 0.95, 0.88))
    area("Fill", 3.4, 220, (3.6, -3.0, 1.6), (0.75, 0.85, 1.0))
    area("Rim", 1.6, 520, (1.8, 3.6, 2.8), (1.0, 0.85, 0.7))
    area("Rim2", 1.6, 300, (-2.6, 3.0, 2.2), (0.8, 0.9, 1.0))
    area("Top", 1.4, 160, (0.0, -0.4, 4.2), (1, 1, 1))
    return sc


def mshot(sc, name, loc, target, lens=85.0, res=(700, 1000), samples=48, fstop=None, folder="renders"):
    cam = bpy.data.objects.get("Mender_Cam")
    if cam is None:
        cd = bpy.data.cameras.new("Mender_Cam")
        cam = bpy.data.objects.new("Mender_Cam", cd)
        sc.collection.objects.link(cam)
    elif cam.name not in sc.objects:
        sc.collection.objects.link(cam)
    cam.data.lens = lens
    cam.data.clip_start = 0.05
    cam.data.clip_end = 100.0
    cam.location = Vector(loc)
    look_at(cam, target)
    if fstop:
        cam.data.dof.use_dof = True
        cam.data.dof.aperture_fstop = fstop
        cam.data.dof.focus_distance = (Vector(target) - cam.location).length
    else:
        cam.data.dof.use_dof = False
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.eevee.taa_render_samples = samples
    sc.render.image_settings.file_format = 'PNG'
    os.makedirs(os.path.join(OUT_DIR, folder), exist_ok=True)
    path = os.path.join(OUT_DIR, folder, name + ".png")
    sc.render.filepath = path
    win = bpy.context.window_manager.windows[0]
    prefs = bpy.context.preferences.view
    old = prefs.render_display_type
    prefs.render_display_type = 'NONE'
    try:
        with bpy.context.temp_override(window=win, scene=sc):
            bpy.ops.render.render(write_still=True)
    finally:
        prefs.render_display_type = old
    return path


def contact_sheet(paths, out_name, folder="renders"):
    arrs = []
    for p in paths:
        img = bpy.data.images.load(p, check_existing=False)
        w, h = img.size
        a = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(a)
        arrs.append(a.reshape(h, w, 4))
        bpy.data.images.remove(img)
    H = min(a.shape[0] for a in arrs)
    sheet = np.concatenate([a[:H] for a in arrs], axis=1)
    sheet[..., 3] = 1.0
    out = os.path.join(OUT_DIR, folder, out_name + ".png")
    im = bpy.data.images.new("tmp_sheet", sheet.shape[1], sheet.shape[0], alpha=False)
    im.pixels.foreach_set(sheet.ravel())
    im.filepath_raw = out
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)
    return out


def spin_shots(sc, tag, views, samples=48, res=(700, 1000), lens=95.0, cam=(0.0, -5.0, 0.95), target=(0.0, 0.0, 0.90)):
    """fixed camera and lights, the character turns on the spot (so the backdrop never changes): views = [(name, degrees about Z), ...]"""
    rig = bpy.data.objects[RIGN]
    paths = []
    for k, ang in views:
        rig.rotation_euler = (0, 0, math.radians(ang))
        bpy.context.view_layer.update()
        paths.append(mshot(sc, "%s_%s" % (tag, k), cam, target, lens=lens, res=res, samples=samples))
    rig.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    return paths


def turnaround(sc, tag, samples=48, res=(700, 1000)):
    return spin_shots(sc, tag, [("front", 0), ("q3", -40), ("side", -90), ("back", 180)], samples, res)
