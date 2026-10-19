# style_mats.py : the shared UNEDITED character look, matched to Relic (realistic, worn, layered PBR that works in EEVEE and Cycles).
# Exec after mender_scene.py (uses _new_mat, _nh, _crev, rest_mapping, glow_mat).
#
# House style for every character (Relic is the reference):
#   - real materials at real scale: painted steel, bare steel, brass, chrome, rubber; nothing toy-clean, no pillow shapes
#   - small crisp bevels (1 to 2.5 cm on a 2.8 m machine); parts are bolted, hinged, piped and cabled, so they read as built
#   - every surface is aged: paint worn off convex edges to bare steel, scratches, grime in crevices, rust streaks running down,
#     dust on upward faces, mud near the ground, and a normal made of pitting, scratches and soft dents
#   - shared accents: Relic's brass joint caps, white stencils; light only where there is a lamp or a lens
#   - soft goods (Mender's coveralls, canvas, webbing, leather, rubber) use the same layers with cloth=1: nothing turns metallic, worn
#     edges scuff lighter instead of polishing, plus weave, sun fade, caked mud and folds (all off by default, so hard surfaces are unchanged)
# Coordinates: rigid parts use Object coordinates (their vertices are stored in the rest pose, so nothing swims when they move);
# skinned meshes pass ob= and get rest-mapped Generated coordinates. Edge wear needs edge_attr(ob) on each mesh.
import bpy, math
import numpy as np


def edge_attr(ob, lo=6.0, hi=30.0, blur=1):
    """per-vertex mean curvature (1/m) from the mesh itself: 'edge' = convex bevels and corners (0..1), 'cavity' = concave folds"""
    me = ob.data
    nv, ne = len(me.vertices), len(me.edges)
    if nv == 0 or ne == 0:
        return
    P = np.empty(nv * 3)
    me.vertices.foreach_get("co", P)
    P = P.reshape(-1, 3)
    Nn = np.empty(nv * 3)
    try:
        me.vertex_normals.foreach_get("vector", Nn)
    except Exception:
        me.vertices.foreach_get("normal", Nn)
    Nn = Nn.reshape(-1, 3)
    E = np.empty(ne * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    D = P[E[:, 0]] - P[E[:, 1]]
    L2 = np.maximum(np.einsum('ij,ij->i', D, D), 1e-12)
    ka = 2.0 * np.einsum('ij,ij->i', Nn[E[:, 0]], D) / L2
    kb = -2.0 * np.einsum('ij,ij->i', Nn[E[:, 1]], D) / L2
    s, deg = np.zeros(nv), np.zeros(nv)
    np.add.at(s, E[:, 0], ka)
    np.add.at(s, E[:, 1], kb)
    np.add.at(deg, E[:, 0], 1.0)
    np.add.at(deg, E[:, 1], 1.0)
    k = s / np.maximum(deg, 1.0)
    for _ in range(blur):
        t = np.zeros(nv)
        np.add.at(t, E[:, 0], k[E[:, 1]])
        np.add.at(t, E[:, 1], k[E[:, 0]])
        k = 0.5 * k + 0.5 * t / np.maximum(deg, 1.0)
    for nm, val in (("edge", np.clip((k - lo) / (hi - lo), 0.0, 1.0)), ("cavity", np.clip((-k - lo) / (hi - lo), 0.0, 1.0))):
        a = me.attributes.get(nm) or me.attributes.new(nm, 'FLOAT', 'POINT')
        a.data.foreach_set("value", val.astype(np.float32))


def _attr(h, name):
    a = h.N.new('ShaderNodeAttribute')
    a.attribute_type = 'GEOMETRY'
    a.attribute_name = name
    return a.outputs['Fac']


def _mr(h, src, a, b, c=0.0, d=1.0):
    m = h.N.new('ShaderNodeMapRange')
    m.clamp = True
    m.inputs['From Min'].default_value = a
    m.inputs['From Max'].default_value = b
    m.inputs['To Min'].default_value = c
    m.inputs['To Max'].default_value = d
    h.L.new(src, m.inputs['Value'])
    return m.outputs['Result']


def _remap(h, mp, scale):
    m = h.N.new('ShaderNodeMapping')
    m.inputs['Scale'].default_value = scale
    h.L.new(mp.outputs['Vector'], m.inputs['Vector'])
    return m


def _nz(h, mp, scale, detail=4.0, rough=0.55):
    nz = h.N.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = scale
    nz.inputs['Detail'].default_value = detail
    nz.inputs['Roughness'].default_value = rough
    h.L.new(mp.outputs['Vector'], nz.inputs['Vector'])
    return nz.outputs['Fac']


def worn(name, base, base2=None, metal=0.0, rough=0.5, ob=None, paint=True, under=(0.30, 0.30, 0.31), wear=1.0, chips=0.3,
         scratches=0.6, grime=1.0, rust=0.35, dust=0.7, mud=1.0, nscale=1.0, bump=0.35, stripes=None, spec=0.5,
         cloth=0.0, weave=0.0, fade=0.0, fade_col=None, caked=0.0, folds=0.0, zoff=0.0, sheen=0.0, mud_col=None, dust_col=None):
    """the layered house material. paint=True: a paint layer over bare steel (`under`); paint=False: bare metal (edges polish).
    stripes=(colour A, colour B, scale) paints diagonal hazard bands as the base colour.
    Soft goods (all off by default, so the hard-surface graph is unchanged):
      cloth=1   fabric, leather or rubber with paint=False: convex edges and seams scuff lighter and duller, scratches and oil stay non-metallic
      weave     canvas / twill weave in the normal (about 1.2 mm threads); folds: crumple and wrinkle normal
      fade      sun bleaching on upward faces and in old washed-out patches, toward fade_col
      caked     raised mud crust wherever the mud layer is; sheen: cloth sheen tinted by the surface colour
      zoff      added to the rest height for the road grime and mud bands (props whose own coordinates do not start at the floor)
      mud_col, dust_col   override the mud and settled-dust colours (default: the dry grey-brown of the KillDroid's feet)"""
    m, nt, pb = _new_mat(name)
    h = _nh(nt)
    mp = rest_mapping(h, ob) if ob is not None else h.mapping((1, 1, 1))
    base2 = base2 or tuple(min(1.0, c * 1.25) for c in base)
    col = h.mixc(h.ramp(_nz(h, mp, 1.7 * nscale, 4), 0.2, 0.8), base, base2)
    if stripes is not None:
        wv = h.N.new('ShaderNodeTexWave')
        wv.wave_type = 'BANDS'
        wv.bands_direction = 'DIAGONAL'
        wv.inputs['Scale'].default_value = stripes[2]
        h.L.new(mp.outputs['Vector'], wv.inputs['Vector'])
        col = h.mixc(h.fm('GREATER_THAN', wv.outputs['Fac'], 0.5), stripes[0], stripes[1])
    # mottling at the hand-span scale, and faded / touched-up patches at the panel scale
    col = h.mixc(h.fm('MULTIPLY', _mr(h, _nz(h, mp, 9.0 * nscale, 6), 0.3, 0.7), 0.50), col, tuple(c * 0.70 for c in base))
    col = h.mixc(h.fm('MULTIPLY', _mr(h, _nz(h, mp, 2.2 * nscale, 3), 0.50, 0.72), 0.55), col, tuple(min(1.0, c * 1.35) for c in base2))
    if fade > 0:
        # sun-bleached where it faces the sky, plus old washed-out patches anywhere
        fc = fade_col or tuple(min(1.0, 0.5 * c + 0.8 * sum(base2) / 3.0) for c in base2)
        gf = h.N.new('ShaderNodeNewGeometry')
        sf = h.N.new('ShaderNodeSeparateXYZ')
        h.L.new(gf.outputs['Normal'], sf.inputs['Vector'])
        up_ = h.fm('MULTIPLY', _mr(h, sf.outputs['Z'], 0.10, 0.90), _mr(h, _nz(h, mp, 2.6 * nscale, 4), 0.25, 0.65))
        old_ = h.fm('MULTIPLY', _mr(h, _nz(h, mp, 1.1 * nscale, 3), 0.52, 0.70), 0.75)
        col = h.mixc(h.fm('MULTIPLY', h.fm('MAXIMUM', up_, old_), fade, clamp=True), col, fc)
    met = h.fm('ADD', metal, 0.0)
    rgh = h.fm('ADD', rough, h.fm('MULTIPLY', h.fm('SUBTRACT', _nz(h, mp, 6.0 * nscale, 4), 0.5), 0.16))
    # 1. edge wear: paint chipped off convex edges (bare steel shows) / bare metal polished on edges; broken up, never a clean rim
    edge = _attr(h, 'edge')
    w = h.fm('ADD', h.fm('MULTIPLY', edge, 0.70 * wear), h.fm('MULTIPLY', h.fm('SUBTRACT', _nz(h, mp, 26.0 * nscale, 10, 0.7), 0.5), 2.2))
    wm = h.fm('MAXIMUM', _mr(h, w, 0.46, 0.51), h.fm('MULTIPLY', _mr(h, _nz(h, mp, 11.0 * nscale, 8, 0.7), 0.685, 0.705), chips))
    bright = tuple(min(1.0, c * 1.45) for c in under)
    if paint:
        ucol = h.mixc(h.ramp(_nz(h, mp, 40.0, 3), 0.3, 0.7), under, bright)
        col = h.mixc(wm, col, ucol)
        met = h.mixf(wm, met, 1.0)
        rgh = h.mixf(wm, rgh, 0.30)
    elif cloth > 0:
        # scuffed: the nap and the dye wear off convex edges, seams and fold crests, so they read lighter, greyer and duller.
        # A soft mask (cloth wears in gradients, it does not chip)
        soft = _mr(h, h.fm('ADD', h.fm('MULTIPLY', edge, 0.70 * wear), h.fm('MULTIPLY', h.fm('SUBTRACT', _nz(h, mp, 7.0 * nscale, 4), 0.5), 0.6)),
                   0.35, 0.85, 0.0, 0.75)
        grey = sum(base2) / 3.0
        col = h.mixc(soft, col, tuple(min(1.0, 0.55 * c * 1.6 + 0.45 * grey * 1.5) for c in base2))
        rgh = h.mixf(soft, rgh, min(1.0, rough + 0.06))
    else:
        wm2 = h.fm('MULTIPLY', wm, 0.65)
        col = h.mixc(wm2, col, tuple(min(1.0, c * 1.4) for c in base2))
        rgh = h.mixf(wm2, rgh, max(rough * 0.55, 0.10))
    # 2. scratches: scattered fragments of thin cell edges at two sizes (most of each network masked away, so no crackle pattern)
    if scratches > 0:
        sm = None
        for sc_, keep, th, ns in ((4.0, 0.64, 0.0045, 2.5), (11.0, 0.68, 0.0035, 3.7)):
            vo = h.N.new('ShaderNodeTexVoronoi')
            vo.feature = 'DISTANCE_TO_EDGE'
            vo.inputs['Scale'].default_value = sc_ * nscale
            h.L.new(mp.outputs['Vector'], vo.inputs['Vector'])
            part = h.fm('MULTIPLY', _mr(h, vo.outputs['Distance'], 0.0, th, 1.0, 0.0), _mr(h, _nz(h, mp, ns * nscale, 4), keep, keep + 0.05))
            sm = part if sm is None else h.fm('MAXIMUM', sm, part)
        sm = h.fm('MULTIPLY', sm, scratches, clamp=True)
        col = h.mixc(sm, col, bright)
        met = h.mixf(sm, met, 0.0 if cloth > 0 else 1.0)
        rgh = h.mixf(sm, rgh, 0.25)
    # 3. crevice grime: AO at two reaches plus the mesh's own concave folds
    crev = h.fm('ADD', h.fm('MULTIPLY', h.fm('MAXIMUM', _crev(h, 0.06), _attr(h, 'cavity')), 2.4), h.fm('MULTIPLY', _crev(h, 0.22), 0.9))
    gm = h.fm('MULTIPLY', h.fm('MULTIPLY', crev, grime), _mr(h, _nz(h, mp, 8.0, 4), 0.20, 0.70), clamp=True)
    col = h.mixc(gm, col, (0.026, 0.022, 0.018))
    rgh = h.mixf(gm, rgh, 0.85)
    met = h.mixf(gm, met, 0.0)
    # grime streaks running down from ledges
    gs = h.fm('MULTIPLY', _mr(h, _nz(h, _remap(h, mp, (6.0, 6.0, 0.4)), 7.0, 6), 0.58, 0.80), _mr(h, _nz(h, mp, 1.8, 3), 0.40, 0.70))
    gsm = h.fm('MULTIPLY', gs, 0.65 * grime, clamp=True)
    col = h.mixc(gsm, col, (0.032, 0.028, 0.022))
    rgh = h.mixf(gsm, rgh, 0.80)
    # oil and hydraulic-fluid stains: dark, a little glossy
    om = h.fm('MULTIPLY', _mr(h, _nz(h, mp, 5.5 * nscale, 6, 0.6), 0.66, 0.74), 0.7 * grime, clamp=True)
    col = h.mixc(om, col, (0.018, 0.016, 0.013))
    rgh = h.mixf(om, rgh, 0.62 if cloth > 0 else 0.30)
    # 4. rust streaks running down
    if rust > 0:
        streak = _mr(h, _nz(h, _remap(h, mp, (5.0, 5.0, 0.35)), 6.0, 6), 0.56, 0.76)
        rm = h.fm('MULTIPLY', h.fm('MULTIPLY', streak, _mr(h, _nz(h, mp, 2.5, 3), 0.45, 0.70)), rust, clamp=True)
        col = h.mixc(rm, col, (0.15, 0.062, 0.024))
        rgh = h.mixf(rm, rgh, 0.80)
        met = h.mixf(rm, met, 0.0)
    # 5. dust on upward faces (world normal, so it stays on top whatever the pose)
    geo = h.N.new('ShaderNodeNewGeometry')
    sn = h.N.new('ShaderNodeSeparateXYZ')
    h.L.new(geo.outputs['Normal'], sn.inputs['Vector'])
    dcol = dust_col or (0.24, 0.215, 0.18)
    col = h.mixc(h.fm('ADD', 0.10 * dust, 0.0), col, (0.20, 0.18, 0.15) if dust_col is None else tuple(c * 0.85 for c in dcol))     # a thin dust film over everything (dulls the colours)
    dm = h.fm('MULTIPLY', h.fm('MULTIPLY', _mr(h, sn.outputs['Z'], 0.35, 0.95), _mr(h, _nz(h, mp, 3.5, 5), 0.30, 0.70)), dust, clamp=True)
    col = h.mixc(dm, col, dcol)
    rgh = h.mixf(dm, rgh, 0.95)
    met = h.mixf(dm, met, 0.0)
    # 6. road grime building up toward the ground, then mud and dust right at the bottom (rest height in metres)
    sp = h.N.new('ShaderNodeSeparateXYZ')
    h.L.new(mp.outputs['Vector'], sp.inputs['Vector'])
    zsrc = sp.outputs['Z'] if zoff == 0 else h.fm('ADD', sp.outputs['Z'], zoff)
    rg = h.fm('MULTIPLY', h.fm('MULTIPLY', _mr(h, zsrc, 1.40, 0.15), _mr(h, _nz(h, mp, 4.0, 5), 0.25, 0.70)), 0.45 * grime, clamp=True)
    col = h.mixc(rg, col, (0.050, 0.044, 0.034))
    rgh = h.mixf(rg, rgh, 0.85)
    md = h.fm('MULTIPLY', h.fm('MULTIPLY', _mr(h, zsrc, 0.60, 0.04), _mr(h, _nz(h, mp, 5.0, 5), 0.25, 0.65)), mud, clamp=True)
    col = h.mixc(md, col, mud_col or (0.080, 0.066, 0.048))
    rgh = h.mixf(md, rgh, 0.90)
    met = h.mixf(md, met, 0.0)
    h.L.new(col, pb.inputs['Base Color'])
    h.L.new(met, pb.inputs['Metallic'])
    h.L.new(rgh, pb.inputs['Roughness'])
    pb.inputs['Specular IOR Level'].default_value = spec
    if sheen > 0:
        try:
            pb.inputs['Sheen Weight'].default_value = sheen
            pb.inputs['Sheen Roughness'].default_value = 0.6
            h.L.new(col, pb.inputs['Sheen Tint'])
        except Exception:
            pass
    # normal: fine pitting and the scratches, then soft dents
    b1 = h.N.new('ShaderNodeBump')
    b1.inputs['Distance'].default_value = 0.0015
    b1.inputs['Strength'].default_value = bump
    hgt = h.fm('ADD', h.fm('MULTIPLY', _nz(h, mp, 320.0, 3), 0.6), h.fm('MULTIPLY', _mr(h, _nz(h, mp, 60.0, 4), 0.55, 0.75), 0.5))
    h.L.new(hgt, b1.inputs['Height'])
    if weave > 0 or caked > 0:
        # soft goods: the weave of the cloth and a raised, cracked mud crust, in a bump of their own ahead of the pitting
        b0 = h.N.new('ShaderNodeBump')
        b0.inputs['Distance'].default_value = 0.0010
        b0.inputs['Strength'].default_value = 1.0
        h0 = h.fm('ADD', 0.0, 0.0)
        if weave > 0:
            for dname, sc_ in (('X', 262.0), ('Y', 262.0), ('Z', 262.0), ('DIAGONAL', 300.0)):
                wv = h.N.new('ShaderNodeTexWave')
                wv.wave_type = 'BANDS'
                wv.bands_direction = dname
                wv.inputs['Scale'].default_value = sc_
                h.L.new(mp.outputs['Vector'], wv.inputs['Vector'])
                h0 = h.fm('ADD', h0, h.fm('MULTIPLY', wv.outputs['Fac'], 0.05 * weave * (0.8 if dname == 'DIAGONAL' else 1.0)))
        if caked > 0:
            crust = h.fm('MULTIPLY', _mr(h, _nz(h, mp, 55.0, 5), 0.42, 0.58), h.fm('MULTIPLY', md, 0.9 * caked))
            crack = _mr(h, _nz(h, mp, 140.0, 3), 0.62, 0.70, 0.0, 0.25)
            h0 = h.fm('ADD', h0, h.fm('MULTIPLY', h.fm('SUBTRACT', crust, crack), 1.0))
        h.L.new(h0, b0.inputs['Height'])
        h.L.new(b0.outputs['Normal'], b1.inputs['Normal'])
    b2 = h.N.new('ShaderNodeBump')
    b2.inputs['Distance'].default_value = 0.02
    b2.inputs['Strength'].default_value = 0.22
    h2 = h.fm('ADD', _nz(h, mp, 3.2, 3), h.fm('MULTIPLY', _nz(h, mp, 9.0, 4), 0.5))
    if folds > 0:
        # crumples: noise squashed vertically reads as horizontal wrinkles, plus a little fine crumple
        h2 = h.fm('ADD', h2, h.fm('MULTIPLY', _nz(h, _remap(h, mp, (1.0, 1.0, 4.5)), 7.0, 4), folds))
        h2 = h.fm('ADD', h2, h.fm('MULTIPLY', _nz(h, mp, 26.0, 3), 0.35 * folds))
    h.L.new(h2, b2.inputs['Height'])
    h.L.new(b1.outputs['Normal'], b2.inputs['Normal'])
    h.L.new(b2.outputs['Normal'], pb.inputs['Normal'])
    return m


def house_palette():
    """colours shared by the cast (linear RGB). Relic's steel and brass are the anchors."""
    return {
        'steel': ((0.36, 0.36, 0.37), (0.55, 0.55, 0.56)),
        'plate': ((0.19, 0.19, 0.20), (0.31, 0.31, 0.32)),
        'darksteel': ((0.085, 0.085, 0.09), (0.16, 0.16, 0.17)),
        'brass': ((0.42, 0.28, 0.09), (0.58, 0.40, 0.15)),
        'chrome': ((0.62, 0.62, 0.64), (0.75, 0.75, 0.77)),
        'rubber': ((0.010, 0.010, 0.011), (0.022, 0.021, 0.020)),
        'stencil': ((0.55, 0.54, 0.50), (0.66, 0.65, 0.60)),
        'hazard': ((0.62, 0.40, 0.012), (0.012, 0.012, 0.013)),
    }
