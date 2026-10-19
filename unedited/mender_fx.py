# mender_fx.py : dust puffs (volumetric spheres that bloom and fade) at the strikes, the slab sliding off, the pick landing and the boots scrabbling,
# plus pebbles that fall off Relic's shoulders as his head lifts. Exec after mender_anim6.py; window scene = Cave; call after the keys exist.

def puff_base_material(name="Cave_Puff"):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputMaterial')
    vol = N.new('ShaderNodeVolumePrincipled')
    vol.inputs['Color'].default_value = (0.62, 0.56, 0.48, 1.0)
    vol.inputs['Anisotropy'].default_value = 0.55
    L.new(vol.outputs['Volume'], out.inputs['Volume'])
    tc = N.new('ShaderNodeTexCoord')
    # soft sphere: 1 at the centre, 0 at the rim (generated coordinates run 0..1 over the bounding box)
    sub = N.new('ShaderNodeVectorMath')
    sub.operation = 'SUBTRACT'
    sub.inputs[1].default_value = (0.5, 0.5, 0.5)
    L.new(tc.outputs['Generated'], sub.inputs[0])
    ln = N.new('ShaderNodeVectorMath')
    ln.operation = 'LENGTH'
    L.new(sub.outputs['Vector'], ln.inputs[0])
    mr = N.new('ShaderNodeMapRange')
    mr.clamp = True
    mr.inputs['From Min'].default_value = 0.12
    mr.inputs['From Max'].default_value = 0.50
    mr.inputs['To Min'].default_value = 1.0
    mr.inputs['To Max'].default_value = 0.0
    L.new(ln.outputs['Value'], mr.inputs['Value'])
    # billows
    mp = N.new('ShaderNodeMapping')
    L.new(tc.outputs['Object'], mp.inputs['Vector'])
    nz = N.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 3.0
    nz.inputs['Detail'].default_value = 5.0
    nz.inputs['Roughness'].default_value = 0.6
    L.new(mp.outputs['Vector'], nz.inputs['Vector'])
    nmr = N.new('ShaderNodeMapRange')
    nmr.clamp = True
    nmr.inputs['From Min'].default_value = 0.30
    nmr.inputs['From Max'].default_value = 0.72
    nmr.inputs['To Min'].default_value = 0.10
    nmr.inputs['To Max'].default_value = 1.0
    L.new(nz.outputs['Fac'], nmr.inputs['Value'])
    strength = N.new('ShaderNodeValue')
    strength.label = "Strength"
    strength.name = "Strength"
    strength.outputs[0].default_value = 0.0
    m1 = N.new('ShaderNodeMath')
    m1.operation = 'MULTIPLY'
    L.new(mr.outputs['Result'], m1.inputs[0])
    L.new(nmr.outputs['Result'], m1.inputs[1])
    m2 = N.new('ShaderNodeMath')
    m2.operation = 'MULTIPLY'
    L.new(m1.outputs['Value'], m2.inputs[0])
    L.new(strength.outputs[0], m2.inputs[1])
    L.new(m2.outputs['Value'], vol.inputs['Density'])
    return m


def puff_mesh():
    me = bpy.data.meshes.get("Puff_Sphere")
    if me is None:
        me = bpy.data.meshes.new("Puff_Sphere")
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        bm.to_mesh(me)
        bm.free()
    return me


def add_puff(sc, coll, name, loc, r0, r1, f0, dur, peak, seed=1):
    base = bpy.data.materials.get("Cave_Puff") or puff_base_material()
    mat = base.copy()
    mat.name = "Cave_Puff_" + name
    ob = bpy.data.objects.get(CP + "Puff_" + name)
    if ob is not None:
        bpy.data.objects.remove(ob, do_unlink=True)
    ob = bpy.data.objects.new(CP + "Puff_" + name, puff_mesh())
    coll.objects.link(ob)
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    ob.display_type = 'BOUNDS'
    ob.visible_shadow = False
    ob.location = Vector(loc)
    rng_ = np.random.default_rng(seed)
    ob.rotation_euler = tuple(rng_.uniform(0, 6.28, 3))
    ob.scale = (r0, r0, r0)
    nodes = mat.node_tree.nodes
    node = nodes["Strength"]
    # visibility only while the puff lives
    for f, hide in ((1, True), (f0 - 1, False), (f0 + dur + 2, True)):
        ob.hide_render = hide
        ob.keyframe_insert('hide_render', frame=f)
    for fc in iter_fcurves(ob):
        if fc.data_path == 'hide_render':
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'
    for k in range(0, dur + 1, 2):
        u = k / float(dur)
        r = r0 + (r1 - r0) * (1.0 - (1.0 - u) ** 3)
        ob.scale = (r, r, r)
        ob.keyframe_insert('scale', frame=f0 + k)
        att = smoothstep(u / 0.15)
        dec = (1.0 - smoothstep((u - 0.15) / 0.85)) ** 1.5
        node.outputs[0].default_value = peak * att * dec
        node.outputs[0].keyframe_insert('default_value', frame=f0 + k)
    return ob


def rim_target():
    C = DIG["C"]
    fxy = Vector((GLINT.x - C[0], GLINT.y - C[1])).normalized()
    txy = Vector((GLINT.x, GLINT.y)) - fxy * DIG["dist"]
    g = surface_top(txy.x, txy.y)
    return Vector((txy.x, txy.y, g[0].z + 0.02))


def key_puffs(sc, rig):
    coll = bpy.data.collections[CCOLL]
    T = rim_target()
    ring_p = Vector(ring_world()[0])
    sc.frame_set(1)
    for ob in list(coll.objects):
        if ob.name.startswith(CP + "Puff_"):
            bpy.data.objects.remove(ob, do_unlink=True)
    made = []
    made.append(add_puff(sc, coll, "strike1", T, 0.10, 0.38, 8, 20, 4.8, 1).name)
    made.append(add_puff(sc, coll, "strike2", T + Vector((0.02, 0.0, 0.0)), 0.10, 0.42, 31, 22, 5.4, 2).name)
    made.append(add_puff(sc, coll, "strike3", T + Vector((0.0, 0.0, 0.08)), 0.18, 0.95, 58, 30, 12.0, 3).name)
    made.append(add_puff(sc, coll, "slab", ring_p + Vector((0.0, 0.0, 0.12)), 0.15, 0.80, 62, 40, 8.0, 4).name)
    sc.frame_set(DROP + 7)
    pk = bpy.data.objects[CP + "Pick"]
    made.append(add_puff(sc, coll, "pickdrop", pk.matrix_world.translation + Vector((0.0, 0.0, 0.04)), 0.10, 0.36, DROP + 7, 20, 3.8, 5).name)
    sc.frame_set(118)
    foot = rig.matrix_world @ rig.pose.bones['foot.L'].head
    made.append(add_puff(sc, coll, "scuff", foot + Vector((0.0, 0.0, 0.02)), 0.10, 0.42, 118, 22, 3.6, 6).name)
    sc.frame_set(1)
    return made


# ------------------------------------------------------------------ pebbles
def pebble_mesh():
    me = bpy.data.meshes.get("Pebble_Mesh")
    if me is None:
        me = bpy.data.meshes.new("Pebble_Mesh")
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.5)
        rng_ = np.random.default_rng(5)
        for v in bm.verts:
            v.co *= rng_.uniform(0.75, 1.25)
            v.co.z *= 0.8
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = False
    return me


def key_pebbles(sc, n=16, f0=DROP - 2, f1=190):
    coll = bpy.data.collections[CCOLL]
    for ob in list(coll.objects):
        if ob.name.startswith(CP + "Pebble"):
            bpy.data.objects.remove(ob, do_unlink=True)
    rng_ = np.random.default_rng(21)
    mat = bpy.data.materials["Cave_Rock"]
    sc.frame_set(f0)
    rr = bpy.data.objects[RIG]
    head = rr.matrix_world @ rr.pose.bones['head'].head
    sh = rr.matrix_world @ rr.pose.bones['upperarm.L'].head
    items = []
    for k in range(n):
        base = head if k % 2 == 0 else sh
        xy = Vector((base.x + rng_.uniform(-0.22, 0.22), base.y + rng_.uniform(-0.10, 0.10)))
        g = surface_top(xy.x, xy.y)
        s = float(rng_.uniform(0.022, 0.060))
        ob = bpy.data.objects.new(CP + "Pebble_%02d" % k, pebble_mesh())
        coll.objects.link(ob)
        ob.data = pebble_mesh()
        ob.scale = (s * 2, s * 2, s * 2)
        ob.rotation_mode = 'QUATERNION'
        ob.visible_shadow = True
        if mat.name not in [m.name for m in ob.data.materials]:
            ob.data.materials.append(mat)
        p = Vector((xy.x, xy.y, (g[0].z if g else base.z) + s))
        v = Vector((rng_.uniform(-0.25, 0.25), rng_.uniform(-0.25, 0.10), rng_.uniform(0.0, 0.3)))
        items.append(dict(ob=ob, p=p, v=v, r=s, q=Quaternion((1, 0, 0, 0)), rest=False))
    h = (1.0 / 24.0) / 4.0
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        for it in items:
            for _s in range(4):
                if it['rest'] and f > f0 + 40:
                    break
                it['v'] += Vector((0, 0, -9.81)) * h
                it['p'] += it['v'] * h
                g = surface_top(it['p'].x, it['p'].y)
                if g:
                    loc, nrm = g
                    nrm = Vector(nrm).normalized()
                    if it['p'].z - it['r'] < loc.z:
                        it['p'] = loc + nrm * it['r']
                        vn = it['v'].dot(nrm)
                        if vn < 0:
                            it['v'] -= (1.25) * vn * nrm
                        vt = it['v'] - it['v'].dot(nrm) * nrm
                        it['v'] -= vt * min(1.0, 5.0 * h)
                    axis = nrm.cross(it['v'])
                    if axis.length > 1e-6:
                        ang = it['v'].length * h / max(it['r'], 1e-3)
                        it['q'] = (Quaternion(axis.normalized(), ang) @ it['q']).normalized()
                it['rest'] = it['v'].length < 0.03
            ob = it['ob']
            ob.location = it['p']
            ob.rotation_quaternion = it['q']
            ob.keyframe_insert('location', frame=f)
            ob.keyframe_insert('rotation_quaternion', frame=f)
    for it in items:
        ob = it['ob']
        for f, hide in ((1, True), (f0, False)):
            ob.hide_render = hide
            ob.keyframe_insert('hide_render', frame=f)
        for fc in iter_fcurves(ob):
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT' if fc.data_path == 'hide_render' else 'LINEAR'
    sc.frame_set(1)
    return len(items)
