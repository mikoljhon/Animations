# relic_aged.py : "100 years old" look for the AI-generated Relic (object "Relic_AI").
# Exec after relic_build.py, relic_model3.py (for _nh) and relic_ai.py in the same namespace.
# One material, one "Age" value node: 0 = brand new (the approved look, used for flashbacks), 1 = 100 years buried.
import bpy, math


def ai_aging(age=1.0):
    ob = bpy.data.objects["Relic_AI"]
    mat = ob.data.materials[0]
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    for n in list(N):
        if n.label.startswith("AG"):
            N.remove(n)
    pb = N["Principled BSDF"]
    tex_c = N["Image Texture"]
    orm = N["Separate Color"]
    nmap = N["Normal Map"]
    before = set(N)
    h = _nh(nt)

    age_n = N.new('ShaderNodeValue')
    age_n.name = "Age"
    age_n.outputs[0].default_value = age
    A = age_n.outputs[0]

    col = tex_c.outputs['Color']
    sc = N.new('ShaderNodeSeparateColor')
    L.new(col, sc.inputs['Color'])
    # gold mask from the base colour (red well above blue)
    g = h.fm('SUBTRACT', h.fm('MULTIPLY', h.fm('SUBTRACT', sc.outputs['Red'], sc.outputs['Blue']), 3.0), 0.25, clamp=True)

    ao = N.new('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.08
    crev = h.fm('SUBTRACT', 1.0, ao.outputs['AO'], clamp=True)

    geo = N.new('ShaderNodeNewGeometry')
    sepn = N.new('ShaderNodeSeparateXYZ')
    L.new(geo.outputs['Normal'], sepn.inputs['Vector'])
    um = N.new('ShaderNodeMapRange')
    um.clamp = True
    um.inputs['From Min'].default_value = 0.2
    um.inputs['From Max'].default_value = 0.9
    L.new(sepn.outputs['Z'], um.inputs['Value'])
    up = um.outputs['Result']

    sepp = N.new('ShaderNodeSeparateXYZ')
    L.new(h.tc.outputs['Object'], sepp.inputs['Vector'])
    hm = N.new('ShaderNodeMapRange')
    hm.inputs['From Min'].default_value = 0.0
    hm.inputs['From Max'].default_value = 1.6
    hm.inputs['To Min'].default_value = 1.0
    hm.inputs['To Max'].default_value = 0.35
    L.new(sepp.outputs['Z'], hm.inputs['Value'])
    low = hm.outputs['Result']

    big = h.noise(h.mapping((1, 1, 1)), 3.0, 4)
    mid = h.noise(h.mapping((1, 1, 1)), 9.0, 6)
    fine = h.noise(h.mapping((1, 1, 1)), 45.0, 3)

    rust = h.fm('MULTIPLY', h.ramp(mid, 0.43, 0.61), h.fm('ADD', 0.30, h.fm('MULTIPLY', crev, 0.70)))
    rust = h.fm('MULTIPLY', h.fm('MULTIPLY', h.fm('MULTIPLY', rust, low), 1.4), A, clamp=True)

    dsrc = h.fm('ADD', h.fm('MULTIPLY', up, 1.0), h.fm('MULTIPLY', crev, 1.2))
    dirt = h.fm('MULTIPLY', dsrc, h.ramp(big, 0.24, 0.52))
    dirt = h.fm('MULTIPLY', h.fm('MULTIPLY', h.fm('MULTIPLY', dirt, low), 1.7), A, clamp=True)

    # pale mineral dust that settles on upward-facing surfaces, and drip streaks of crust
    streak = h.noise(h.mapping((1, 1, 0.22)), 11.0, 5)
    dust = h.fm('MULTIPLY', h.fm('MULTIPLY', up, h.ramp(streak, 0.46, 0.70)), 0.75)
    dust = h.fm('MULTIPLY', dust, A, clamp=True)

    pits = h.fm('MULTIPLY', h.fm('MULTIPLY', h.ramp(fine, 0.64, 0.70), 0.6), A, clamp=True)

    c = h.mixc(h.fm('MULTIPLY', h.fm('MULTIPLY', g, A), 0.65), col, (0.10, 0.065, 0.025))      # tarnished gold
    ver = h.fm('MULTIPLY', h.fm('MULTIPLY', g, crev), h.fm('MULTIPLY', h.ramp(mid, 0.45, 0.65), A))
    c = h.mixc(h.fm('MULTIPLY', ver, 0.5), c, (0.09, 0.20, 0.16))                              # verdigris in the crevices
    film = h.fm('MULTIPLY', h.fm('SUBTRACT', 1.0, g, clamp=True), h.fm('MULTIPLY', A, 0.45))
    c = h.mixc(film, c, (0.045, 0.045, 0.05))                                                  # oxidised chrome film
    c = h.mixc(rust, c, (0.22, 0.08, 0.025))
    c = h.mixc(dirt, c, (0.12, 0.095, 0.07))
    c = h.mixc(dust, c, (0.32, 0.29, 0.24))
    c = h.mixc(pits, c, (0.01, 0.01, 0.012))
    L.new(c, pb.inputs['Base Color'])

    cover = h.fm('ADD', h.fm('ADD', h.fm('MULTIPLY', rust, 0.9), dirt), h.fm('MULTIPLY', dust, 0.8), clamp=True)
    L.new(h.fm('MULTIPLY', orm.outputs['Blue'], h.fm('SUBTRACT', 1.0, cover, clamp=True)), pb.inputs['Metallic'])
    rvar = h.fm('MULTIPLY', h.fm('MULTIPLY', h.ramp(mid, 0.30, 0.75), A), 0.9)
    r1 = h.mixf(rvar, orm.outputs['Green'], 0.72)
    L.new(h.mixf(cover, r1, 0.95), pb.inputs['Roughness'])

    dent = h.noise(h.mapping((1, 1, 1)), 5.0, 3)
    pit_n = h.noise(h.mapping((1, 1, 1)), 80.0, 2)
    hgt = h.fm('ADD', h.fm('MULTIPLY', dent, 0.6), h.fm('MULTIPLY', pit_n, 0.4))
    bump = N.new('ShaderNodeBump')
    bump.inputs['Distance'].default_value = 0.006
    L.new(h.fm('MULTIPLY', A, 0.9), bump.inputs['Strength'])
    L.new(hgt, bump.inputs['Height'])
    L.new(nmap.outputs['Normal'], bump.inputs['Normal'])
    L.new(bump.outputs['Normal'], pb.inputs['Normal'])

    for n in set(N) - before:
        if n.name != "Age":
            n.label = "AG " + (n.label or n.name)
        else:
            n.label = "AG Age (0 new, 1 = 100 years)"
    return {"nodes_added": len(set(N) - before)}


def ai_power(on=True):
    """chest ring (Tally) and eyes: on = glowing, off = dormant (a faint ember so the glass still reads)"""
    def setem(name, strength):
        m = bpy.data.materials[name]
        b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        b.inputs['Emission Strength'].default_value = strength
    setem("Relic_RingLight", 3.0 if on else 0.0)
    setem("Relic_Glyph", 2.8 if on else 0.0)


def ai_crust(thickness=0.009, show=True):
    """a patchy shell of caked mud and stone that follows the body; hide or delete it once he is cleaned up"""
    coll = bpy.data.collections["Relic_AI"]
    body = bpy.data.objects["Relic_AI"]
    for o in list(bpy.data.objects):
        if o.name == "Relic_AI_Crust":
            old = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if old.users == 0:
                bpy.data.meshes.remove(old)
    crust = body.copy()
    crust.data = body.data.copy()
    crust.name = "Relic_AI_Crust"
    crust.data.name = "Relic_AI_Crust"
    coll.objects.link(crust)
    for m in list(crust.modifiers):
        if m.type != 'ARMATURE':
            crust.modifiers.remove(m)
    inflate = crust.modifiers.new("Inflate", 'DISPLACE')
    inflate.strength = thickness
    inflate.mid_level = 0.0
    inflate.direction = 'NORMAL'

    m = bpy.data.materials.get("Relic_Crust") or bpy.data.materials.new("Relic_Crust")
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputMaterial')
    pb = N.new('ShaderNodeBsdfPrincipled')
    tr = N.new('ShaderNodeBsdfTransparent')
    mix = N.new('ShaderNodeMixShader')
    L.new(tr.outputs['BSDF'], mix.inputs[1])
    L.new(pb.outputs['BSDF'], mix.inputs[2])
    L.new(mix.outputs['Shader'], out.inputs['Surface'])
    h = _nh(nt)
    ao = N.new('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.10
    crev = h.fm('SUBTRACT', 1.0, ao.outputs['AO'], clamp=True)
    sepp = N.new('ShaderNodeSeparateXYZ')
    L.new(h.tc.outputs['Object'], sepp.inputs['Vector'])
    hm = N.new('ShaderNodeMapRange')
    hm.inputs['From Min'].default_value = 0.0
    hm.inputs['From Max'].default_value = 1.8
    hm.inputs['To Min'].default_value = 1.0
    hm.inputs['To Max'].default_value = 0.25
    L.new(sepp.outputs['Z'], hm.inputs['Value'])
    big = h.noise(h.mapping((1, 1, 1)), 4.0, 5)
    mid = h.noise(h.mapping((1, 1, 1)), 16.0, 6)
    fine = h.noise(h.mapping((1, 1, 1)), 70.0, 4)
    cover = h.fm('MULTIPLY', h.ramp(big, 0.48, 0.60), h.fm('ADD', 0.25, h.fm('MULTIPLY', hm.outputs['Result'], 0.75)))
    cover = h.fm('ADD', cover, h.fm('MULTIPLY', crev, 0.55))
    cover = h.fm('MULTIPLY', cover, h.ramp(mid, 0.38, 0.55))
    alpha = h.fm('MULTIPLY', h.fm('SUBTRACT', cover, 0.18), 3.0, clamp=True)
    L.new(alpha, mix.inputs[0])
    col = h.mixc(h.ramp(mid, 0.52, 0.74), (0.030, 0.021, 0.014), (0.105, 0.088, 0.066))
    L.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.96
    pb.inputs['Metallic'].default_value = 0.0
    bump = N.new('ShaderNodeBump')
    bump.inputs['Distance'].default_value = 0.01
    bump.inputs['Strength'].default_value = 0.9
    L.new(h.fm('ADD', h.fm('MULTIPLY', mid, 0.5), h.fm('MULTIPLY', fine, 0.5)), bump.inputs['Height'])
    L.new(bump.outputs['Normal'], pb.inputs['Normal'])
    crust.data.materials.clear()
    crust.data.materials.append(m)
    crust.hide_render = not show
    crust.hide_viewport = not show
    return {"crust": crust.name, "verts": len(crust.data.vertices)}


def set_age(v):
    mat = bpy.data.objects["Relic_AI"].data.materials[0]
    mat.node_tree.nodes["Age"].outputs[0].default_value = v
