# street_build.py : the ruined street of UNEDITED episode 1 (scenes 4 and 5): a cracked road at dawn under fog, towers like broken teeth wrapped in vines,
# an arcing rusted rail bridge to drop from, concrete pillars, a wrecked tram, a length of rusted rail. World frame: the crack lip is at the origin, the street
# runs along +Y (about 160 m), x across (the road is 24 m wide), z up. Run:  blender.exe -b --factory-startup --python street_build.py
#   -> C:\ai\unedited\street_v01.blend and preview renders in renders\street_v01_*.png
import bpy, bmesh, math
import numpy as np
from mathutils import Vector, Matrix, Euler

OUT = r"C:\ai\unedited"
FOG_DENSITY = 0.0016
rng = np.random.default_rng(42)
SCN = "Street"


# ------------------------------------------------------------------ small helpers
def coll(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def mesh_obj(name, V, polys, c, mat=None, loc=(0, 0, 0), smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(int(i) for i in p) for p in polys])
    me.update()
    for p in me.polygons:
        p.use_smooth = smooth
    ob = bpy.data.objects.new(name, me)
    c.objects.link(ob)
    ob.location = loc
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


def box_vp(cx, cy, cz, sx, sy, sz):
    h = np.array([sx, sy, sz]) / 2.0
    V = np.array([[cx + a * h[0], cy + b * h[1], cz + c_ * h[2]] for a in (-1, 1) for b in (-1, 1) for c_ in (-1, 1)])
    P = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return V, P


def merge_vp(items):
    Vs, Ps, off = [], [], 0
    for V, P in items:
        Vs.append(np.asarray(V, float))
        Ps.extend(tuple(i + off for i in p) for p in P)
        off += len(V)
    return np.concatenate(Vs), Ps


def rot_vp(V, rz=0.0, rx=0.0, ry=0.0, center=(0, 0, 0)):
    R = Euler((rx, ry, rz), 'XYZ').to_matrix()
    c = np.asarray(center, float)
    return (np.asarray(V, float) - c) @ np.array(R).T + c


# ------------------------------------------------------------------ materials (all procedural)
def new_mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])
    return m, nt, pb


def N(nt, kind, **kw):
    n = nt.nodes.new(kind)
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def noise(nt, scale, detail=5.0, coord='Object'):
    tc = N(nt, 'ShaderNodeTexCoord')
    nz = N(nt, 'ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = scale
    nz.inputs['Detail'].default_value = detail
    nt.links.new(tc.outputs[coord], nz.inputs['Vector'])
    return nz.outputs['Fac']


def ramp(nt, fac, lo, hi):
    cr = N(nt, 'ShaderNodeValToRGB')
    cr.color_ramp.elements[0].position = lo
    cr.color_ramp.elements[1].position = hi
    nt.links.new(fac, cr.inputs['Fac'])
    return cr.outputs['Color']


def mix(nt, fac, a, b):
    mx = N(nt, 'ShaderNodeMix', data_type='RGBA')
    nt.links.new(fac, mx.inputs[0])
    for idx, x in ((6, a), (7, b)):
        if isinstance(x, tuple):
            mx.inputs[idx].default_value = (x[0], x[1], x[2], 1.0)
        else:
            nt.links.new(x, mx.inputs[idx])
    return mx.outputs[2]


def mulf(nt, a, b):
    m = N(nt, 'ShaderNodeMath', operation='MULTIPLY', use_clamp=True)
    for i, x in enumerate((a, b)):
        if isinstance(x, (int, float)):
            m.inputs[i].default_value = x
        else:
            nt.links.new(x, m.inputs[i])
    return m.outputs['Value']


def bump(nt, pb, height, strength, dist=0.05):
    bp = N(nt, 'ShaderNodeBump')
    bp.inputs['Strength'].default_value = strength
    bp.inputs['Distance'].default_value = dist
    nt.links.new(height, bp.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], pb.inputs['Normal'])


def mat_road():
    m, nt, pb = new_mat('St_Road')
    n1 = noise(nt, 0.35, 6)
    col = mix(nt, ramp(nt, n1, 0.35, 0.65), (0.022, 0.022, 0.024), (0.060, 0.058, 0.054))
    # cracks: Voronoi distance to edge
    tc = N(nt, 'ShaderNodeTexCoord')
    vo = N(nt, 'ShaderNodeTexVoronoi', feature='DISTANCE_TO_EDGE')
    vo.inputs['Scale'].default_value = 0.65
    nt.links.new(tc.outputs['Object'], vo.inputs['Vector'])
    crack = N(nt, 'ShaderNodeMath', operation='LESS_THAN')
    nt.links.new(vo.outputs['Distance'], crack.inputs[0])
    crack.inputs[1].default_value = 0.022
    col = mix(nt, crack.outputs['Value'], col, (0.012, 0.012, 0.012))
    # moss in the cracks and low corners
    n2 = noise(nt, 0.9, 5)
    moss = mulf(nt, ramp(nt, n2, 0.55, 0.72), 0.8)
    col = mix(nt, moss, col, (0.045, 0.085, 0.020))
    nt.links.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.85
    nb = N(nt, 'ShaderNodeMath', operation='ADD')
    nt.links.new(noise(nt, 6.0, 6), nb.inputs[0])
    nt.links.new(noise(nt, 40.0, 4), nb.inputs[1])
    bump(nt, pb, nb.outputs['Value'], 0.5, 0.08)
    return m


def mat_concrete(name='St_Concrete', tint=(1.0, 1.0, 1.0), windows=False, wx=3.0, wz=3.6):
    m, nt, pb = new_mat(name)
    n1 = noise(nt, 0.25, 6)
    base = mix(nt, ramp(nt, n1, 0.3, 0.7), (0.075, 0.074, 0.070), (0.17, 0.16, 0.145))
    # rain streaks: noise stretched along z
    tc = N(nt, 'ShaderNodeTexCoord')
    mp = N(nt, 'ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (1.2, 1.2, 0.08)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nz = N(nt, 'ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 3.0
    nz.inputs['Detail'].default_value = 4
    nt.links.new(mp.outputs['Vector'], nz.inputs['Vector'])
    streak = mulf(nt, ramp(nt, nz.outputs['Fac'], 0.52, 0.72), 0.7)
    col = mix(nt, streak, base, (0.035, 0.034, 0.030))
    # moss low and dirty
    col = mix(nt, mulf(nt, ramp(nt, noise(nt, 0.6, 5), 0.58, 0.75), 0.6), col, (0.04, 0.07, 0.02))
    if windows:
        sep = N(nt, 'ShaderNodeSeparateXYZ')
        nt.links.new(tc.outputs['Object'], sep.inputs[0])
        xy = N(nt, 'ShaderNodeMath', operation='ADD')
        nt.links.new(sep.outputs['X'], xy.inputs[0])
        nt.links.new(sep.outputs['Y'], xy.inputs[1])
        mx = N(nt, 'ShaderNodeMath', operation='FLOORED_MODULO')
        nt.links.new(xy.outputs['Value'], mx.inputs[0])
        mx.inputs[1].default_value = wx
        ix = N(nt, 'ShaderNodeMath', operation='LESS_THAN')
        nt.links.new(mx.outputs['Value'], ix.inputs[0])
        ix.inputs[1].default_value = wx * 0.58
        mz = N(nt, 'ShaderNodeMath', operation='FLOORED_MODULO')
        nt.links.new(sep.outputs['Z'], mz.inputs[0])
        mz.inputs[1].default_value = wz
        iz = N(nt, 'ShaderNodeMath', operation='LESS_THAN')
        nt.links.new(mz.outputs['Value'], iz.inputs[0])
        iz.inputs[1].default_value = wz * 0.55
        horiz = ix
        win = mulf(nt, horiz.outputs['Value'], iz.outputs['Value'])
        col = mix(nt, win, col, (0.006, 0.008, 0.010))
    nt.links.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.88
    nb = N(nt, 'ShaderNodeMath', operation='ADD')
    nt.links.new(noise(nt, 4.0, 6), nb.inputs[0])
    nt.links.new(noise(nt, 30.0, 4), nb.inputs[1])
    bump(nt, pb, nb.outputs['Value'], 0.35, 0.06)
    return m


def mat_rust(name='St_Rust'):
    m, nt, pb = new_mat(name)
    n1 = noise(nt, 2.5, 6)
    col = mix(nt, ramp(nt, n1, 0.35, 0.65), (0.050, 0.020, 0.011), (0.20, 0.075, 0.030))
    col = mix(nt, mulf(nt, ramp(nt, noise(nt, 9.0, 5), 0.6, 0.78), 0.8), col, (0.015, 0.012, 0.011))
    nt.links.new(col, pb.inputs['Base Color'])
    pb.inputs['Metallic'].default_value = 0.65
    pb.inputs['Roughness'].default_value = 0.62
    nb = noise(nt, 18.0, 6)
    bump(nt, pb, nb, 0.5, 0.03)
    return m


def mat_vine():
    m, nt, pb = new_mat('St_Vine')
    n1 = noise(nt, 6.0, 6)
    col = mix(nt, ramp(nt, n1, 0.4, 0.65), (0.020, 0.060, 0.012), (0.075, 0.165, 0.030))
    nt.links.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.7
    # leaf holes: alpha from noise
    al = ramp(nt, noise(nt, 9.0, 4), 0.38, 0.52)
    ac = N(nt, 'ShaderNodeSeparateColor')
    nt.links.new(al, ac.inputs[0])
    nt.links.new(ac.outputs['Red'], pb.inputs['Alpha'])
    m.surface_render_method = 'DITHERED'
    return m


def mat_flower():
    m, nt, pb = new_mat('St_Flower')
    pb.inputs['Base Color'].default_value = (0.95, 0.70, 0.04, 1.0)
    pb.inputs['Emission Color'].default_value = (0.95, 0.65, 0.03, 1.0)
    pb.inputs['Emission Strength'].default_value = 0.25
    pb.inputs['Roughness'].default_value = 0.5
    return m


def mat_moss():
    m, nt, pb = new_mat('St_MossBlob')
    col = mix(nt, ramp(nt, noise(nt, 14.0, 5), 0.3, 0.7), (0.025, 0.07, 0.015), (0.08, 0.17, 0.035))
    nt.links.new(col, pb.inputs['Base Color'])
    pb.inputs['Roughness'].default_value = 0.9
    return m


# ------------------------------------------------------------------ geometry builders
def build_ground(c, mat):
    sx, sy, nx, ny = 160.0, 200.0, 160, 200
    xs = np.linspace(-sx / 2, sx / 2, nx + 1)
    ys = np.linspace(-6.0, sy - 6.0, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = 0.12 * np.sin(X * 0.21) * np.cos(Y * 0.17) + 0.09 * np.sin(X * 0.9 + Y * 0.7) * np.exp(-((np.abs(X) - 16) / 10.0) ** 2) * 0.0
    # kerb-ish rise at the sides and rubble banks
    Z += np.clip((np.abs(X) - 13.0) * 0.12, 0, 1.5) * (0.5 + 0.5 * np.sin(Y * 0.31))
    # the crack: the ground falls away at y < 0, a ragged lip
    lip = 0.35 * np.sin(X * 0.9) + 0.25 * np.sin(X * 2.3 + 1.0) + 0.1 * np.sin(X * 5.1)
    drop = np.clip((lip - Y) * 1.6, 0.0, 1.0)
    Z -= 30.0 * drop ** 1.5
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    P = [(i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1) for i in range(nx) for j in range(ny)]
    ob = mesh_obj("Street_Ground", V, P, c, mat, smooth=True)
    return ob


def build_tower(name, cx, cy, w, d, h, c, mat):
    nx, ny = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    tall = (int(rng.integers(0, nx)), int(rng.integers(0, ny)))
    items = []
    for i in range(nx):
        for j in range(ny):
            x0, x1 = -w / 2 + w * i / nx, -w / 2 + w * (i + 1) / nx
            y0, y1 = -d / 2 + d * j / ny, -d / 2 + d * (j + 1) / ny
            hh = h if (i, j) == tall else h * float(rng.uniform(0.45, 0.95))
            zt = [hh - float(rng.uniform(0, 0.12)) * hh for _ in range(4)]
            V = np.array([[x0, y0, -3.0], [x1, y0, -3.0], [x1, y1, -3.0], [x0, y1, -3.0], [x0, y0, zt[0]], [x1, y0, zt[1]], [x1, y1, zt[2]], [x0, y1, zt[3]]])
            P = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            items.append((V, P))
    V, P = merge_vp(items)
    ob = mesh_obj(name, V, P, c, mat, loc=(cx, cy, 0.0))
    return ob


def build_vines(c, mat, towers):
    items = []
    for (cx, cy, w, d, h) in towers:
        for _ in range(int(rng.integers(7, 12))):
            side = int(rng.integers(0, 4))
            u = float(rng.uniform(-0.45, 0.45))
            if side == 0:
                p = np.array([cx + u * w, cy - d / 2 - 0.1, 0]); n = np.array([0, -1.0, 0]); t = np.array([1.0, 0, 0])
            elif side == 1:
                p = np.array([cx + u * w, cy + d / 2 + 0.1, 0]); n = np.array([0, 1.0, 0]); t = np.array([1.0, 0, 0])
            elif side == 2:
                p = np.array([cx - w / 2 - 0.1, cy + u * d, 0]); n = np.array([-1.0, 0, 0]); t = np.array([0, 1.0, 0])
            else:
                p = np.array([cx + w / 2 + 0.1, cy + u * d, 0]); n = np.array([1.0, 0, 0]); t = np.array([0, 1.0, 0])
            top = float(rng.uniform(0.35, 0.9)) * h
            bot = max(top - float(rng.uniform(0.25, 0.7)) * h, 0.0)
            wd = float(rng.uniform(1.0, 3.2))
            m_ = 10
            V, P = [], []
            for k in range(m_ + 1):
                z = bot + (top - bot) * k / m_
                wob = 0.25 * math.sin(k * 1.3 + u * 7)
                for s in (-0.5, 0.5):
                    V.append(p + t * (s * wd * (1.0 - 0.3 * k / m_)) + n * wob + np.array([0, 0, z]))
            for k in range(m_):
                a = 2 * k
                P.append((a, a + 1, a + 3, a + 2))
            items.append((np.array(V), P))
    V, P = merge_vp(items)
    return mesh_obj("Street_Vines", V, P, c, mat, smooth=True)


def build_bridge(c, mat_deck, mat_conc, mat_rust_):
    """an arcing viaduct: deck sections along a curve in plan (centre of the arc far off at y=+260), rails, ties, a few girders and concrete piers"""
    R = 150.0
    cx0, cy0 = 0.0, 26.0 + R            # centre of the circle, so the deck passes through (0, 26)
    deck, rails, piers, girders = [], [], [], []
    angs = np.radians(np.linspace(-40, 40, 41))
    for k in range(len(angs) - 1):
        a0, a1 = angs[k], angs[k + 1]
        p0 = np.array([cx0 + R * math.sin(a0), cy0 - R * math.cos(a0)])
        p1 = np.array([cx0 + R * math.sin(a1), cy0 - R * math.cos(a1)])
        mid = (p0 + p1) / 2
        L = np.linalg.norm(p1 - p0) + 0.05
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        # deck slab, rails, ties
        for (sz, off, z0, lst, w) in ((None, None, None, None, None),):
            pass
        V, P = box_vp(0, 0, 0, L, 7.0, 1.4)
        V = rot_vp(V, rz=yaw) + np.array([mid[0], mid[1], 14.0])
        deck.append((V, P))
        for dy in (-1.1, 1.1):
            V, P = box_vp(0, dy, 0, L, 0.16, 0.22)
            V = rot_vp(V, rz=yaw) + np.array([mid[0], mid[1], 14.9])
            rails.append((V, P))
        V, P = box_vp(0, 0, 0, 0.35, 3.0, 0.14)
        V = rot_vp(V, rz=yaw) + np.array([mid[0], mid[1], 14.77])
        rails.append((V, P))
        for dy in (-3.7, 3.7):
            V, P = box_vp(0, dy, 0, L, 0.18, 1.0)
            V = rot_vp(V, rz=yaw) + np.array([mid[0], mid[1], 15.1])
            girders.append((V, P))
        if k % 8 == 4:
            V, P = box_vp(0, 0, 0, 3.4, 3.4, 15.0)
            V = rot_vp(V, rz=yaw) + np.array([mid[0], mid[1], 6.5])
            piers.append((V, P))
            V, P = box_vp(0, 0, 0, 5.0, 8.0, 1.2)
            V = rot_vp(V, rz=yaw) + np.array([mid[0], mid[1], 13.4])
            piers.append((V, P))
    mesh_obj("Bridge_Deck", *merge_vp(deck), c, mat_conc)
    mesh_obj("Bridge_Rails", *merge_vp(rails), c, mat_rust_)
    mesh_obj("Bridge_Girders", *merge_vp(girders), c, mat_rust_)
    mesh_obj("Bridge_Piers", *merge_vp(piers), c, mat_conc)


def build_pillars(c, mat_conc, mat_rust_, spots):
    for k, (x, y, h) in enumerate(spots):
        items = [box_vp(0, 0, h / 2 - 0.5, 1.7, 1.7, h + 1.0)]
        # a broken top: a slanted cut with rebar sticking out
        V, P = items[0]
        V[V[:, 2] > h * 0.5, 2] -= np.array([0.0 if (V[i, 0] < 0) else float(rng.uniform(0.2, 0.9)) for i in range(len(V))])[V[:, 2] > h * 0.5]
        ob = mesh_obj("Pillar%d" % k, V, P, c, mat_conc, loc=(x, y, 0.0))
        ob.rotation_euler[2] = float(rng.uniform(0, 3.14))
        bars = []
        for dx, dy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.4)):
            top = h + float(rng.uniform(0.0, 0.5))
            bars.append(box_vp(dx, dy, top, 0.04, 0.04, 1.1 + float(rng.uniform(0, 0.4))))
        bo = mesh_obj("PillarRebar%d" % k, *merge_vp(bars), c, mat_rust_, loc=(x, y, 0.0))
        bo.rotation_euler[2] = ob.rotation_euler[2]


def build_tram(c, mat_rust_, mat_conc, loc, yaw):
    items = []
    V, P = box_vp(0, 0, 1.75, 12.0, 2.8, 3.1)
    items.append((V, P))
    V, P = box_vp(0, 0, 0.35, 12.4, 2.9, 0.4)
    items.append((V, P))
    ob = mesh_obj("Tram_Body", *merge_vp(items), c, mat_conc, loc=loc)
    ob.rotation_euler = (0.0, math.radians(7), yaw)
    items = [box_vp(-6.0 + 12.0 * i / 4.0, 1.41, 2.3, 1.8, 0.05, 1.2) for i in range(5)]
    wob = mesh_obj("Tram_Windows", *merge_vp(items), c, mat_rust_, loc=loc)
    wob.rotation_euler = ob.rotation_euler
    return ob


def build_rail_beam(c, mat_rust_, loc, yaw):
    """a length of rusted rail, 2.6 m: the finishing weapon. Local x along the rail; origin at its middle."""
    items = [box_vp(0, 0, 0.0, 2.6, 0.14, 0.05), box_vp(0, 0, 0.115, 2.6, 0.035, 0.18), box_vp(0, 0, 0.215, 2.6, 0.12, 0.04)]
    ob = mesh_obj("RailBeam", *merge_vp(items), c, mat_rust_, loc=loc)
    ob.rotation_euler = (0.0, 0.0, yaw)
    return ob


def build_boulders(c, mat_conc, spots):
    for k, (x, y, r) in enumerate(spots):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r)
        for v in bm.verts:
            v.co *= Vector((float(rng.uniform(0.8, 1.3)), float(rng.uniform(0.8, 1.3)), float(rng.uniform(0.55, 0.9))))
            v.co += Vector((float(rng.normal(0, r * 0.08)),) * 3)
        me = bpy.data.meshes.new("Boulder%d" % k)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("Boulder%d" % k, me)
        c.objects.link(ob)
        ob.location = (x, y, r * 0.35)
        ob.rotation_euler = (0, 0, float(rng.uniform(0, 6.28)))
        ob.data.materials.append(mat_conc)


def build_cars(c, mat_rust_, spots):
    for k, (x, y, yaw) in enumerate(spots):
        items = [box_vp(0, 0, 0.55, 4.2, 1.8, 0.8), box_vp(-0.2, 0, 1.2, 2.2, 1.6, 0.6)]
        ob = mesh_obj("Car%d" % k, *merge_vp(items), c, mat_rust_, loc=(x, y, 0.0))
        ob.rotation_euler = (float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05)), yaw)


def build_lamps(c, mat_rust_, spots):
    for k, (x, y) in enumerate(spots):
        items = [box_vp(0, 0, 3.5, 0.22, 0.22, 7.0), box_vp(0.6 * (1 if x < 0 else -1), 0, 7.0, 1.5, 0.16, 0.16)]
        mesh_obj("Lamp%d" % k, *merge_vp(items), c, mat_rust_, loc=(x, y, 0.0))


def build_flowers(c, mat_f, mat_m, spots):
    bm = bmesh.new()
    for (x, y) in spots:
        for _ in range(int(rng.integers(2, 6))):
            r = 0.045
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r, matrix=Matrix.Translation((x + float(rng.normal(0, 0.35)), y + float(rng.normal(0, 0.35)), 0.12)))
    me = bpy.data.meshes.new("Flowers")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Street_Flowers", me)
    c.objects.link(ob)
    ob.data.materials.append(mat_f)
    bm = bmesh.new()
    for (x, y) in spots:
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=float(rng.uniform(0.18, 0.4)), matrix=Matrix.Translation((x + float(rng.normal(0, 0.25)), y + float(rng.normal(0, 0.25)), 0.05)))
    me = bpy.data.meshes.new("MossBlobs")
    bm.to_mesh(me)
    bm.free()
    mo = bpy.data.objects.new("Street_MossBlobs", me)
    c.objects.link(mo)
    mo.data.materials.append(mat_m)


# ------------------------------------------------------------------ world, lights, render
def setup_world(sc):
    w = bpy.data.worlds.get("St_World") or bpy.data.worlds.new("St_World")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new('ShaderNodeTexSky')
    try:
        sky.sky_type = 'NISHITA'
    except Exception:
        pass
    sky.sun_elevation = math.radians(9.0)
    sky.sun_rotation = math.radians(0.0)
    try:
        sky.sun_disc = False
    except Exception:
        pass
    for attr, val in (('air_density', 1.2), ('dust_density', 2.5), ('aerosol_density', 2.5), ('ozone_density', 1.0)):
        try:
            setattr(sky, attr, val)
        except Exception:
            pass
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 0.30
    hs = nt.nodes.new('ShaderNodeHueSaturation')
    hs.inputs['Saturation'].default_value = 0.25
    hs.inputs['Value'].default_value = 1.15
    nt.links.new(sky.outputs['Color'], hs.inputs['Color'])
    nt.links.new(hs.outputs['Color'], bg.inputs['Color'])
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    sc.world = w
    # the low sun shines down the street from the far end, a little to the right, so light reaches the road between the towers
    sun = bpy.data.lights.new("St_Sun", 'SUN')
    sun.energy = 5.0
    sun.color = (1.0, 0.74, 0.50)
    sun.angle = math.radians(1.5)
    so = bpy.data.objects.new("St_Sun", sun)
    sc.collection.objects.link(so)
    d = Vector((-0.18, -0.93, -0.32)).normalized()
    so.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    # a cool fill from the sky side so the shadow side is not black
    fill = bpy.data.lights.new("St_Fill", 'SUN')
    fill.energy = 0.16
    fill.color = (0.86, 0.89, 0.96)
    fo = bpy.data.objects.new("St_Fill", fill)
    sc.collection.objects.link(fo)
    fo.rotation_euler = Vector((0.4, 0.5, -0.75)).normalized().to_track_quat('-Z', 'Y').to_euler()
    return so


def build_fog(c):
    m = bpy.data.materials.get('St_Fog') or bpy.data.materials.new('St_Fog')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    vs = nt.nodes.new('ShaderNodeVolumeScatter')
    vs.inputs['Color'].default_value = (0.88, 0.80, 0.74, 1.0)
    vs.inputs['Density'].default_value = FOG_DENSITY
    vs.inputs['Anisotropy'].default_value = 0.4
    nt.links.new(vs.outputs['Volume'], out.inputs['Volume'])
    V, P = box_vp(0.0, 95.0, 25.0, 230.0, 260.0, 90.0)
    ob = mesh_obj("Street_Fog", V, P, c, m)
    ob.display_type = 'WIRE'
    ob.visible_shadow = False
    return ob


def build_street():
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        if o.name in ('Cube', 'Light', 'Camera'):
            bpy.data.objects.remove(o, do_unlink=True)
    sc.name = SCN
    sc.render.engine = 'BLENDER_EEVEE'
    sc.eevee.taa_render_samples = 24
    try:
        sc.eevee.use_raytracing = True
    except Exception:
        pass
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    c = coll("Street")
    M = {'road': mat_road(), 'conc': mat_concrete('St_Concrete'), 'rust': mat_rust(), 'vine': mat_vine(), 'flower': mat_flower(), 'moss': mat_moss()}
    M['tower'] = mat_concrete('St_Tower', windows=True)
    M['tower2'] = mat_concrete('St_Tower2', windows=True, wx=2.6, wz=3.2)
    build_ground(c, M['road'])
    towers = []
    for side in (-1, 1):
        y = 8.0
        while y < 170.0:
            w = float(rng.uniform(14, 28)); d = float(rng.uniform(14, 26)); h = float(rng.uniform(28, 95))
            x = side * (float(rng.uniform(16, 24)) + w / 2.0)
            build_tower("Tower_%d_%d" % (side, int(y)), x, y + d / 2, w, d, h, c, M['tower' if rng.random() < 0.55 else 'tower2'])
            towers.append((x, y + d / 2, w, d, h))
            y += d + float(rng.uniform(1.5, 7.0))
    # a far backdrop row of towers in the fog
    for k in range(7):
        x = float(rng.uniform(-70, 70))
        build_tower("Far_%d" % k, x, 175.0 + float(rng.uniform(0, 30)), float(rng.uniform(18, 30)), 18.0, float(rng.uniform(55, 110)), c, M['tower'])
    build_vines(c, M['vine'], towers)
    build_bridge(c, M['rust'], M['conc'], M['rust'])
    build_pillars(c, M['conc'], M['rust'], [(-6.5, 11.0, 6.0), (7.0, 13.5, 4.5), (-8.0, 22.0, 7.5), (6.0, 19.0, 5.0), (-5.0, 30.0, 6.5), (9.5, 31.0, 5.5)])
    build_tram(c, M['rust'], M['conc'], (9.5, 44.0, 1.6), math.radians(8))
    build_rail_beam(c, M['rust'], (-2.8, 9.0, 0.25), math.radians(-70))
    build_boulders(c, M['conc'], [(float(rng.uniform(-11, 11)), float(rng.uniform(4, 60)), float(rng.uniform(0.4, 1.8))) for _ in range(34)])
    build_cars(c, M['rust'], [(-7.0, 46.0, 0.3), (4.0, 54.0, -0.5), (-4.0, 66.0, 1.2), (8.5, 72.0, 0.1), (-9.0, 40.0, -0.2)])
    build_lamps(c, M['rust'], [(-12.5, 14.0), (12.5, 26.0), (-12.5, 40.0), (12.5, 56.0)])
    spots = [(float(rng.uniform(-11, 11)), float(rng.uniform(0.5, 70))) for _ in range(160)]
    build_flowers(c, M['flower'], M['moss'], spots)
    setup_world(sc)
    build_fog(c)
    cd = bpy.data.cameras.new("St_Cam")
    cam = bpy.data.objects.new("St_Cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return sc, cam


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def previews(sc, cam):
    views = [("fp_lip", (0.0, 0.6, 1.9), (0.0, 40.0, 7.0), 24, "eye level from the crack lip, looking up the street"),
             ("fight", (-12.0, -3.0, 2.3), (0.0, 16.0, 4.5), 26, "the fight zone"),
             ("high", (34.0, -22.0, 24.0), (0.0, 45.0, 8.0), 30, "establishing"),
             ("bridge", (0.0, 8.0, 1.9), (0.0, 26.0, 11.0), 24, "looking at the rail bridge from where Relic stands")]
    files = []
    for nm, loc, tgt, lens, _d in views:
        cam.location = loc
        cam.data.lens = lens
        cam.data.clip_end = 1500.0
        look_at(cam, tgt)
        sc.render.resolution_x, sc.render.resolution_y = 800, 450
        sc.render.filepath = OUT + r"\renders\street_v01_%s.png" % nm
        bpy.ops.render.render(write_still=True)
        files.append(sc.render.filepath)
    return files


if __name__ == "__main__" or True:
    sc, cam = build_street()
    print("BUILT street objects:", len(bpy.data.objects))
    print("RENDERED", previews(sc, cam))
    bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\street_v01.blend")
    print("SAVED")
