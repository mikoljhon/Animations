import bpy, bmesh, math, os
from mathutils import Vector, Matrix, Euler

SCENE = "Relic"
COLL = "Relic"
OUT_DIR = r"C:\ai\unedited"

AX = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}
HEAD_C = Vector((0.0, 0.0, 1.80))
HEAD_R = (0.160, 0.178, 0.215)
HEAD_LOW = 0.065
HS = 1.3          # hand scale
NOTES = []


def chest_front(z):
    """world y of the chest's front surface at height z (chest box 0.25 deep, tapered toward the waist)"""
    t = max(0.0, min(1.0, (z - 1.18) / 0.48))
    return -0.125 * (0.84 + 0.16 * t)


# ----------------------------------------------------------------- materials
def _principled(m):
    nt = m.node_tree
    if nt is None:
        try:
            m.use_nodes = True
        except Exception:
            pass
        nt = m.node_tree
    if nt is None:
        return None, None
    b = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None:
        nt.nodes.clear()
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        o = nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(b.outputs['BSDF'], o.inputs['Surface'])
    return nt, b


def make_mat(name, color, metallic=0.0, rough=0.5, emit=None, emit_strength=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (color[0], color[1], color[2], 1.0)
    nt, b = _principled(m)
    if b:
        b.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
        b.inputs['Metallic'].default_value = metallic
        b.inputs['Roughness'].default_value = rough
        if emit:
            b.inputs['Emission Color'].default_value = (emit[0], emit[1], emit[2], 1.0)
            b.inputs['Emission Strength'].default_value = emit_strength
    return m


def make_weathered(name, base, rust, rough=0.42, metallic=0.05, scale=5.0, ramp=(0.42, 0.62), zr=(0.1, 1.7),
                   grime=(0.05, 0.045, 0.04)):
    m = make_mat(name, base, metallic, rough)
    try:
        nt, b = _principled(m)
        if not b:
            return m
        N, L = nt.nodes, nt.links
        tc = N.new('ShaderNodeTexCoord')

        def ramp_node(lo, hi):
            cr = N.new('ShaderNodeValToRGB')
            cr.color_ramp.elements[0].position = lo
            cr.color_ramp.elements[1].position = hi
            return cr

        def noise_node(sc, det):
            nz = N.new('ShaderNodeTexNoise')
            nz.inputs['Scale'].default_value = sc
            nz.inputs['Detail'].default_value = det
            nz.inputs['Roughness'].default_value = 0.62
            L.new(tc.outputs['Object'], nz.inputs['Vector'])
            return nz

        sep = N.new('ShaderNodeSeparateXYZ')
        L.new(tc.outputs['Object'], sep.inputs['Vector'])
        hmap = N.new('ShaderNodeMapRange')          # more wear toward the ground
        hmap.inputs['From Min'].default_value = zr[0]
        hmap.inputs['From Max'].default_value = zr[1]
        hmap.inputs['To Min'].default_value = 1.0
        hmap.inputs['To Max'].default_value = 0.30
        L.new(sep.outputs['Z'], hmap.inputs['Value'])

        n1 = noise_node(scale, 7.0)
        cr1 = ramp_node(ramp[0], ramp[1])
        L.new(n1.outputs['Fac'], cr1.inputs['Fac'])
        rust_mask = N.new('ShaderNodeMath')
        rust_mask.operation = 'MULTIPLY'
        L.new(cr1.outputs['Color'], rust_mask.inputs[0])
        L.new(hmap.outputs['Result'], rust_mask.inputs[1])

        n2 = noise_node(16.0, 8.0)
        cr2 = ramp_node(0.52, 0.74)
        L.new(n2.outputs['Fac'], cr2.inputs['Fac'])
        grime_mask = N.new('ShaderNodeMath')
        grime_mask.operation = 'MULTIPLY'
        grime_mask.inputs[1].default_value = 0.75
        L.new(cr2.outputs['Color'], grime_mask.inputs[0])

        mix1 = N.new('ShaderNodeMix')
        mix1.data_type = 'RGBA'
        mix1.inputs[6].default_value = (base[0], base[1], base[2], 1.0)
        mix1.inputs[7].default_value = (rust[0], rust[1], rust[2], 1.0)
        L.new(rust_mask.outputs['Value'], mix1.inputs[0])
        mix2 = N.new('ShaderNodeMix')
        mix2.data_type = 'RGBA'
        mix2.inputs[7].default_value = (grime[0], grime[1], grime[2], 1.0)
        L.new(mix1.outputs[2], mix2.inputs[6])
        L.new(grime_mask.outputs['Value'], mix2.inputs[0])
        L.new(mix2.outputs[2], b.inputs['Base Color'])

        tot = N.new('ShaderNodeMath')
        tot.operation = 'ADD'
        tot.use_clamp = True
        L.new(rust_mask.outputs['Value'], tot.inputs[0])
        L.new(grime_mask.outputs['Value'], tot.inputs[1])
        rm = N.new('ShaderNodeMapRange')
        rm.inputs['To Min'].default_value = rough
        rm.inputs['To Max'].default_value = 0.88
        L.new(tot.outputs['Value'], rm.inputs['Value'])
        L.new(rm.outputs['Result'], b.inputs['Roughness'])
    except Exception as e:
        NOTES.append("weathering failed for %s: %s" % (name, e))
    return m


# ------------------------------------------------------------------ geometry
def make_weathered2(name, base, rust, rough=0.42, metallic=0.05, rust_scale=8.0, grime_amt=0.45,
                    scratch_amt=0.55, zr=(0.1, 1.7), grime=(0.05, 0.045, 0.04), scratch=(0.55, 0.53, 0.48)):
    m = make_mat(name, base, metallic, rough)
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

        def noise(mp, scale, detail):
            nz = N.new('ShaderNodeTexNoise')
            nz.inputs['Scale'].default_value = scale
            nz.inputs['Detail'].default_value = detail
            nz.inputs['Roughness'].default_value = 0.6
            L.new(mp.outputs['Vector'], nz.inputs['Vector'])
            return nz

        def ramp(src, lo, hi):
            cr = N.new('ShaderNodeValToRGB')
            cr.color_ramp.elements[0].position = lo
            cr.color_ramp.elements[1].position = hi
            L.new(src.outputs['Fac'], cr.inputs['Fac'])
            return cr

        def mul(a_out, b_val):
            mn = N.new('ShaderNodeMath')
            mn.operation = 'MULTIPLY'
            L.new(a_out, mn.inputs[0])
            if isinstance(b_val, (int, float)):
                mn.inputs[1].default_value = b_val
            else:
                L.new(b_val, mn.inputs[1])
            return mn

        def mixrgb(fac_out, a_col, b_col, a_out=None):
            mx = N.new('ShaderNodeMix')
            mx.data_type = 'RGBA'
            if a_out is not None:
                L.new(a_out, mx.inputs[6])
            else:
                mx.inputs[6].default_value = (a_col[0], a_col[1], a_col[2], 1.0)
            mx.inputs[7].default_value = (b_col[0], b_col[1], b_col[2], 1.0)
            L.new(fac_out, mx.inputs[0])
            return mx

        sep = N.new('ShaderNodeSeparateXYZ')
        L.new(tc.outputs['Object'], sep.inputs['Vector'])
        hmap = N.new('ShaderNodeMapRange')
        hmap.inputs['From Min'].default_value = zr[0]
        hmap.inputs['From Max'].default_value = zr[1]
        hmap.inputs['To Min'].default_value = 1.0
        hmap.inputs['To Max'].default_value = 0.30
        L.new(sep.outputs['Z'], hmap.inputs['Value'])

        r_nz = noise(mapping((1.0, 1.0, 1.0)), rust_scale, 7.0)
        r_cr = ramp(r_nz, 0.50, 0.68)
        r_mask = mul(r_cr.outputs['Color'], hmap.outputs['Result'])
        g_nz = noise(mapping((1.0, 1.0, 0.28)), 14.0, 6.0)
        g_cr = ramp(g_nz, 0.55, 0.78)
        g_mask = mul(g_cr.outputs['Color'], grime_amt)
        s1 = noise(mapping((60.0, 3.0, 3.0)), 1.0, 1.0)
        s1c = ramp(s1, 0.60, 0.62)
        s2 = noise(mapping((3.0, 60.0, 3.0)), 1.0, 1.0)
        s2c = ramp(s2, 0.60, 0.62)
        s_sum = N.new('ShaderNodeMath')
        s_sum.operation = 'ADD'
        s_sum.use_clamp = True
        L.new(s1c.outputs['Color'], s_sum.inputs[0])
        L.new(s2c.outputs['Color'], s_sum.inputs[1])
        s_mask = mul(s_sum.outputs['Value'], scratch_amt)

        c1 = mixrgb(r_mask.outputs['Value'], base, rust)
        c2 = mixrgb(g_mask.outputs['Value'], None, grime, a_out=c1.outputs[2])
        c3 = mixrgb(s_mask.outputs['Value'], None, scratch, a_out=c2.outputs[2])
        L.new(c3.outputs[2], b.inputs['Base Color'])

        tot = N.new('ShaderNodeMath')
        tot.operation = 'ADD'
        tot.use_clamp = True
        L.new(r_mask.outputs['Value'], tot.inputs[0])
        L.new(g_mask.outputs['Value'], tot.inputs[1])
        rm = N.new('ShaderNodeMapRange')
        rm.inputs['To Min'].default_value = rough
        rm.inputs['To Max'].default_value = 0.88
        L.new(tot.outputs['Value'], rm.inputs['Value'])
        L.new(rm.outputs['Result'], b.inputs['Roughness'])
    except Exception as e:
        NOTES.append("weathering2 failed for %s: %s" % (name, e))
    return m


def rot_z_to(d):
    if isinstance(d, str):
        d = AX[d]
    return Vector((0, 0, 1)).rotation_difference(Vector(d).normalized()).to_matrix().to_4x4()


def bm_ellipsoid(c, radii, segs=40, rings=20):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    for v in bm.verts:
        v.co.x *= radii[0]
        v.co.y *= radii[1]
        v.co.z *= radii[2]
    bm.transform(Matrix.Translation(Vector(c)))
    return bm


def bm_sphere(c, r, segs=24, rings=12):
    return bm_ellipsoid(c, (r, r, r), segs, rings)


def bm_capsule(p0, p1, r, rxy=(1.0, 1.0), segs=24, rings=12):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    h = max(d.length - 2 * r, 0.0) / 2.0
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=r)
    for v in bm.verts:
        if v.co.z > 1e-7:
            v.co.z += h
        elif v.co.z < -1e-7:
            v.co.z -= h
        v.co.x *= rxy[0]
        v.co.y *= rxy[1]
    bm.transform(Matrix.Translation((p0 + p1) / 2) @ rot_z_to(d))
    return bm


def bm_cyl(c, r, h, axis='Z', segs=32, rxy=(1.0, 1.0)):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r, radius2=r, depth=h)
    for v in bm.verts:
        v.co.x *= rxy[0]
        v.co.y *= rxy[1]
    bm.transform(Matrix.Translation(Vector(c)) @ rot_z_to(axis))
    return bm


def bm_torus(c, Rx, Ry, r, axis='Z', major=48, minor=12):
    bm = bmesh.new()
    rows = []
    for i in range(major):
        a = 2 * math.pi * i / major
        ca, sa = math.cos(a), math.sin(a)
        row = []
        for j in range(minor):
            b = 2 * math.pi * j / minor
            cb, sb = math.cos(b), math.sin(b)
            row.append(bm.verts.new(((Rx + r * cb) * ca, (Ry + r * cb) * sa, r * sb)))
        rows.append(row)
    for i in range(major):
        for j in range(minor):
            bm.faces.new((rows[i][j], rows[(i + 1) % major][j], rows[(i + 1) % major][(j + 1) % minor], rows[i][(j + 1) % minor]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.transform(Matrix.Translation(Vector(c)) @ rot_z_to(axis))
    return bm


def bm_rbox(c, size, bevel=0.02, segs=3, taper=None, rot=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size[0]
        v.co.y *= size[1]
        v.co.z *= size[2]
    b = min(bevel, min(size) * 0.49)
    if b > 1e-5:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=b, offset_type='OFFSET', segments=segs, profile=0.5, affect='EDGES')
    if taper:
        z0, z1 = -size[2] / 2.0, size[2] / 2.0
        for v in bm.verts:
            t = (v.co.z - z0) / (z1 - z0)
            v.co.x *= taper[0] + (1.0 - taper[0]) * t
            v.co.y *= taper[1] + (1.0 - taper[1]) * t
    M = Matrix.Translation(Vector(c))
    if rot is not None:
        M = M @ rot
    bm.transform(M)
    return bm


# decals (visor, eyes) projected onto the head dome
def patch_faces(rings, segs):
    def idx(k, i):
        return 1 + (k - 1) * segs + (i % segs)
    faces = []
    for i in range(segs):
        faces.append((0, idx(1, i), idx(1, i + 1)))
    for k in range(1, rings):
        for i in range(segs):
            faces.append((idx(k, i), idx(k + 1, i), idx(k + 1, i + 1), idx(k, i + 1)))
    return faces


def patch_local(a, b, n, rings, segs):
    pts = [(0.0, 0.0)]
    for k in range(1, rings + 1):
        rho = k / rings
        for i in range(segs):
            th = 2 * math.pi * i / segs
            c, s = math.cos(th), math.sin(th)
            pts.append((a * rho * math.copysign(abs(c) ** (2.0 / n), c), b * rho * math.copysign(abs(s) ** (2.0 / n), s)))
    return pts


def head_proj(scale):
    rx, ry, rz = HEAD_R[0] * scale, HEAD_R[1] * scale, HEAD_R[2] * scale

    def f(x, z):
        u = (x / rx) ** 2
        w = ((z - HEAD_C.z) / rz) ** 2
        arg = max(1.0 - u - w, 0.0)
        return Vector((x, -ry * math.sqrt(arg), z))
    return f


# ---------------------------------------------------------------- body layout
def arm_pts(sx):
    S = Vector((sx * 0.27, 0.0, 1.58))
    t1, t2 = math.radians(25), math.radians(15)
    E = S + Vector((sx * 0.30 * math.sin(t1), 0.0, -0.30 * math.cos(t1)))
    W = E + Vector((sx * 0.27 * math.sin(t2), 0.0, -0.27 * math.cos(t2)))
    return S, E, W, t2


def hand_frame(sx):
    S, E, W, t2 = arm_pts(sx)
    return Matrix.Translation(W) @ Matrix.Rotation(-sx * t2, 4, 'Y') @ Matrix.Scale(HS, 4)


FINGERS = [
    ("index", -0.027, 0.036, 0.030, 0.0095),
    ("middle", -0.009, 0.040, 0.032, 0.0095),
    ("ring", 0.009, 0.036, 0.030, 0.0092),
    ("pinky", 0.026, 0.030, 0.025, 0.0085),
]


def finger_pts(sx, f):
    name, y, L1, L2, r = f
    p0 = Vector((0.0, y, -0.088))
    a1, a2 = math.radians(8), math.radians(26)
    d1 = Vector((-sx * math.sin(a1), 0, -math.cos(a1)))
    d2 = Vector((-sx * math.sin(a2), 0, -math.cos(a2)))
    p1 = p0 + d1 * L1
    p2 = p1 + d2 * L2
    return p0, p1, p2


def thumb_pts(sx):
    p0 = Vector((-sx * 0.014, -0.040, -0.030))
    d1 = Vector((-sx * 0.18, -0.55, -0.82)).normalized()
    d2 = Vector((-sx * 0.30, -0.45, -0.84)).normalized()
    p1 = p0 + d1 * 0.034
    p2 = p1 + d2 * 0.028
    return p0, p1, p2


def leg_pts(sx):
    H = Vector((sx * 0.115, 0.0, 0.95))
    K = Vector((sx * 0.115, -0.015, 0.55))
    A = Vector((sx * 0.115, 0.0, 0.15))
    B = Vector((sx * 0.115, -0.115, 0.04))
    T = Vector((sx * 0.115, -0.22, 0.04))
    return H, K, A, B, T


def bone_defs():
    B = []
    B.append(("root", (0, 0, 0), (0, -0.25, 0), None))
    B.append(("pelvis", (0, 0, 0.98), (0, 0, 1.08), "root"))
    B.append(("spine", (0, 0, 1.08), (0, 0, 1.28), "pelvis"))
    B.append(("chest", (0, 0, 1.28), (0, 0, 1.62), "spine"))
    B.append(("neck", (0, 0, 1.62), (0, 0, 1.72), "chest"))
    B.append(("head", (0, 0, 1.72), (0, 0, 2.03), "neck"))
    for side, sx in (("L", 1), ("R", -1)):
        S, E, W, t2 = arm_pts(sx)
        Mh = hand_frame(sx)
        B.append(("clavicle." + side, (sx * 0.07, -0.01, 1.60), tuple(S), "chest"))
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
        B.append(("ik_knee." + side, (sx * 0.115, -0.85, 0.55), (sx * 0.115, -0.85, 0.65), "root"))
    return B


# ---------------------------------------------------------------------- stage
def stage_scene():
    scn = bpy.data.scenes.get(SCENE) or bpy.data.scenes.new(SCENE)
    coll = bpy.data.collections.get(COLL) or bpy.data.collections.new(COLL)
    if coll.name not in [c.name for c in scn.collection.children]:
        scn.collection.children.link(coll)

    w = bpy.data.worlds.get("Relic_World") or bpy.data.worlds.new("Relic_World")
    scn.world = w
    w.color = (0.055, 0.07, 0.085)
    try:
        nt = w.node_tree
        if nt is None:
            w.use_nodes = True
            nt = w.node_tree
        if nt:
            for n in nt.nodes:
                if n.type == 'BACKGROUND':
                    n.inputs['Color'].default_value = (0.055, 0.07, 0.085, 1.0)
                    n.inputs['Strength'].default_value = 1.0
    except Exception as e:
        NOTES.append("world nodes: %s" % e)

    scn.render.engine = 'BLENDER_EEVEE'
    scn.render.resolution_x, scn.render.resolution_y = 900, 1200
    scn.render.resolution_percentage = 100
    scn.render.fps = 24
    try:
        scn.eevee.taa_render_samples = 24
    except Exception as e:
        NOTES.append("eevee samples: %s" % e)
    try:
        scn.view_settings.exposure = -0.55
    except Exception as e:
        NOTES.append("exposure: %s" % e)

    for o in list(coll.objects):
        if o.type in {'CAMERA', 'LIGHT'} or o.name == "Relic_Floor":
            bpy.data.objects.remove(o, do_unlink=True)

    cam_d = bpy.data.cameras.new("Relic_Cam")
    cam_d.lens = 50
    cam = bpy.data.objects.new("Relic_Cam", cam_d)
    coll.objects.link(cam)
    scn.camera = cam

    def light(name, loc, energy, size, color):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy
        ld.size = size
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        coll.objects.link(lo)
        d = Vector((0, 0, 1.05)) - Vector(loc)
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return lo

    light("Key", (-2.6, -3.2, 3.3), 1150, 2.6, (1.0, 0.93, 0.84))
    light("Fill", (3.2, -2.6, 1.8), 330, 3.0, (0.75, 0.86, 1.0))
    light("Rim", (1.6, 3.2, 2.9), 850, 1.6, (0.6, 0.88, 1.0))
    light("Rim2", (-2.2, 2.6, 2.2), 480, 1.6, (1.0, 0.75, 0.5))

    bm = bmesh.new()
    s = 400.0
    bm.faces.new([bm.verts.new(v) for v in [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)]])
    me = bpy.data.meshes.new("Relic_Floor")
    bm.to_mesh(me)
    bm.free()
    fl = bpy.data.objects.new("Relic_Floor", me)
    me.materials.append(make_mat("Relic_FloorMat", (0.04, 0.045, 0.05), 0.0, 0.6))
    coll.objects.link(fl)

    for win in bpy.context.window_manager.windows:
        try:
            win.scene = scn
        except Exception as e:
            NOTES.append("window scene: %s" % e)
    return {"scene": scn.name, "notes": NOTES}


def place_camera(loc, target):
    scn = bpy.data.scenes[SCENE]
    cam = scn.camera
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


# ---------------------------------------------------------------------- model
def stage_model():
    scn = bpy.data.scenes[SCENE]
    coll = bpy.data.collections[COLL]
    for o in list(coll.objects):
        if o.type == 'MESH' and o.get("relic_part"):
            bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)

    ivory = make_weathered2("Relic_Ivory", (0.56, 0.52, 0.43), (0.30, 0.13, 0.06), rough=0.40)
    ivory2 = make_weathered2("Relic_IvoryDark", (0.45, 0.41, 0.34), (0.27, 0.12, 0.05), rough=0.45)
    graphite = make_mat("Relic_Graphite", (0.045, 0.050, 0.058), 0.75, 0.42)
    copper = make_mat("Relic_Copper", (0.62, 0.32, 0.13), 1.0, 0.30)
    brass = make_mat("Relic_Brass", (0.40, 0.34, 0.22), 0.8, 0.45)
    patch = make_weathered2("Relic_Patch", (0.10, 0.22, 0.24), (0.30, 0.15, 0.07), rough=0.5, scratch_amt=0.8)
    visor_m = make_mat("Relic_Visor", (0.004, 0.005, 0.007), 0.0, 0.08)
    trim_m = make_mat("Relic_VisorTrim", (0.40, 0.20, 0.09), 1.0, 0.35)
    glyph_m = make_mat("Relic_Glyph", (0.0, 0.0, 0.0), 0.0, 0.5, emit=(1.0, 0.32, 0.0), emit_strength=2.2)
    ring_m = make_mat("Relic_RingLight", (0.0, 0.0, 0.0), 0.0, 0.5, emit=(0.0, 0.62, 1.0), emit_strength=2.4)
    dark = make_mat("Relic_Dark", (0.010, 0.011, 0.013), 0.3, 0.4)

    def finish(name, me, mat, bone):
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me)
        ob["relic_part"] = True
        ob["bone"] = bone
        coll.objects.link(ob)
        return ob

    def add(name, bm, mat, bone, sharp=38):
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(sharp):
                e.smooth = False
        for f in bm.faces:
            f.smooth = True
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        return finish(name, me, mat, bone)

    def add_patch(name, coords, faces, mat, bone):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(c) for c in coords], [], faces)
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        return finish(name, me, mat, bone)

    # ---- head
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
    add("Head_Dome", bm, ivory, "head")
    base_z = zc - HEAD_LOW
    add("Head_Collar", bm_cyl((0, 0, base_z), 1.0, 0.038, 'Z', 48, (rx * 1.06, ry * 1.06)), graphite, "head")
    add("Head_CollarTrim", bm_torus((0, 0, base_z + 0.021), rx * 1.06, ry * 1.06, 0.0075, 'Z', 64, 10), copper, "head")

    # visor trim + visor + glyphs (projected onto the dome)
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

    base = eye_coords('Neutral')
    faces_all = list(eye_faces) + [tuple(i + n1 for i in f) for f in eye_faces]
    eye_ob = add_patch("Eyes", base, faces_all, glyph_m, "head")
    eye_ob.shape_key_add(name="Basis")
    for kind in ("Blink", "Wide", "Angry", "Sad", "Happy"):
        kb = eye_ob.shape_key_add(name=kind)
        kb.value = 0.0
        for i, c in enumerate(eye_coords(kind)):
            kb.data[i].co = c

    # ---- neck
    add("Neck", bm_cyl((0, 0, 1.68), 0.055, 0.13, 'Z', 24), graphite, "neck")
    add("Neck_Ring", bm_torus((0, 0, 1.665), 0.066, 0.066, 0.011, 'Z', 40, 10), copper, "neck")

    # ---- torso
    add("Chest", bm_rbox((0, 0, 1.42), (0.41, 0.25, 0.48), 0.085, 5, taper=(0.72, 0.84)), ivory, "chest")
    fy = chest_front(1.46)
    add("Chest_Housing", bm_cyl((0, fy - 0.003, 1.46), 0.064, 0.030, 'Y', 40), graphite, "chest")
    add("Chest_Ring", bm_torus((0, fy - 0.0195, 1.46), 0.044, 0.044, 0.0085, 'Y', 56, 12), ring_m, "chest")
    add("Chest_RingCore", bm_cyl((0, fy - 0.0185, 1.46), 0.035, 0.004, 'Y', 40), dark, "chest")
    py = chest_front(1.30)
    add("Serial_Plate", bm_rbox((0.115, py - 0.0015, 1.30), (0.075, 0.006, 0.040), 0.002, 2), brass, "chest")
    for i, zz in enumerate((1.311, 1.300, 1.289)):
        add("Serial_Line%d" % i, bm_rbox((0.112, py - 0.0045, zz), (0.050 - 0.008 * (i % 2), 0.002, 0.0035), 0.001, 1), dark, "chest")
    add("Chest_Seam", bm_rbox((0, py - 0.001, 1.285), (0.004, 0.003, 0.20), 0.001, 1), dark, "chest")
    add("Back_Hatch", bm_rbox((0, 0.118, 1.42), (0.21, 0.020, 0.23), 0.012, 3), graphite, "chest")
    for bx in (-0.08, 0.08):
        for bz in (1.32, 1.52):
            add("Back_Bolt", bm_sphere((bx, 0.130, bz), 0.0085, 12, 8), copper, "chest")

    add("Waist", bm_cyl((0, 0, 1.13), 1.0, 0.15, 'Z', 40, (0.10, 0.085)), graphite, "spine")
    add("Waist_Ring_Lo", bm_torus((0, 0, 1.085), 0.102, 0.087, 0.0095, 'Z', 48, 10), copper, "spine")
    add("Waist_Ring_Hi", bm_torus((0, 0, 1.175), 0.102, 0.087, 0.0095, 'Z', 48, 10), copper, "spine")

    add("Pelvis", bm_rbox((0, 0, 0.995), (0.34, 0.22, 0.17), 0.055, 4), ivory, "pelvis")
    add("Pelvis_Low", bm_rbox((0, 0, 0.925), (0.28, 0.19, 0.09), 0.03, 3), graphite, "pelvis")

    # ---- shoulders + arms
    for side, sx in (("L", 1), ("R", -1)):
        S, E, W, t2 = arm_pts(sx)
        Mh = hand_frame(sx)
        bone_up, bone_fore, bone_hand, bone_cl = "upperarm." + side, "forearm." + side, "hand." + side, "clavicle." + side
        add("Clavicle." + side, bm_capsule((sx * 0.09, -0.005, 1.60), S, 0.034), graphite, bone_cl)
        P = Vector((sx * 0.285, 0.0, 1.63))
        PR = (0.100, 0.092, 0.078)
        add("Pauldron." + side, bm_ellipsoid(P, PR, 40, 20), ivory, bone_cl)
        add("ShoulderJoint." + side, bm_sphere(S, 0.058), graphite, bone_up)
        add("UpperArm." + side, bm_capsule(S, E, 0.052), ivory, bone_up)
        add("ElbowJoint." + side, bm_sphere(E, 0.054), copper, bone_fore)
        add("Forearm." + side, bm_capsule(E, W, 0.047), ivory2, bone_fore)
        d2 = (W - E).normalized()
        gc = E + (W - E) * 0.48 + Vector((sx * 0.052, 0.0, 0.0))
        add("ForearmGuard." + side, bm_rbox(gc, (0.028, 0.076, 0.15), 0.012, 3, rot=rot_z_to(d2)), ivory, bone_fore)
        add("WristRing." + side, bm_torus(W, 0.046, 0.046, 0.0095, d2, 40, 10), copper, bone_hand)

        # hand (built in the hand's local frame, then moved into place)
        bm = bm_rbox((0, 0, -0.045), (0.032, 0.078, 0.090), 0.012, 3)
        bm.transform(Mh)
        add("Palm." + side, bm, graphite, bone_hand)
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
    add_patch("Repair_Patch", top + bottom, pfaces, patch, "clavicle.L")
    for ox in (-0.034, 0.034):
        for oy in (-0.020, 0.020):
            add("Patch_Bolt", bm_sphere(surf(sp + tu * ox + tv * oy, 0.0085), 0.0055, 12, 8), graphite, "clavicle.L")

    # ---- legs
    for side, sx in (("L", 1), ("R", -1)):
        H, K, A, Bl, T = leg_pts(sx)
        bt, bs, bf, bto = "thigh." + side, "shin." + side, "foot." + side, "toe." + side
        add("HipJoint." + side, bm_sphere(H, 0.064), graphite, "pelvis")
        add("Thigh." + side, bm_capsule(H, K, 0.074), ivory, bt)
        add("ThighPlate." + side, bm_rbox((sx * 0.115, -0.068, 0.77), (0.095, 0.038, 0.22), 0.016, 3), ivory2, bt)
        add("KneeJoint." + side, bm_sphere(K, 0.063), copper, bt)
        add("Shin." + side, bm_capsule(K, A, 0.057), ivory, bs)
        add("ShinGuard." + side, bm_rbox((sx * 0.115, -0.062, 0.34), (0.092, 0.040, 0.26), 0.016, 3), ivory2, bs)
        add("AnkleJoint." + side, bm_sphere(A, 0.049), graphite, bs)
        add("AnkleCollar." + side, bm_cyl((sx * 0.115, 0.0, 0.112), 0.056, 0.05, 'Z', 28), ivory2, bs)
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
        add("ToeCap." + side, bm_rbox((sx * 0.115, -0.175, 0.082), (0.100, 0.085, 0.020), 0.009, 3), ivory2, bto)

    n = len([o for o in coll.objects if o.get("relic_part")])
    return {"parts": n, "notes": NOTES}


# ------------------------------------------------------------------------ rig
def stage_rig():
    scn = bpy.data.scenes[SCENE]
    coll = bpy.data.collections[COLL]
    for o in list(coll.objects):
        if o.type == 'ARMATURE':
            bpy.data.objects.remove(o, do_unlink=True)
    for a in list(bpy.data.armatures):
        if a.users == 0:
            bpy.data.armatures.remove(a)

    arm = bpy.data.armatures.new("Relic_Rig")
    rig = bpy.data.objects.new("Relic_Rig", arm)
    coll.objects.link(rig)
    vl = scn.view_layers[0]
    win = bpy.context.window_manager.windows[0]
    defs = bone_defs()
    with bpy.context.temp_override(window=win, scene=scn, view_layer=vl, active_object=rig, object=rig):
        vl.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        eb = arm.edit_bones
        for name, head, tail, parent in defs:
            b = eb.new(name)
            b.head = head
            b.tail = tail
        for name, head, tail, parent in defs:
            if parent:
                eb[name].parent = eb[parent]
                if (Vector(eb[parent].tail) - Vector(eb[name].head)).length < 1e-4:
                    eb[name].use_connect = True
        bpy.ops.object.mode_set(mode='OBJECT')

    arm.display_type = 'OCTAHEDRAL'
    rig.show_in_front = True
    names = {b.name for b in arm.bones}
    bound, missing = 0, []
    for o in coll.objects:
        if o.type != 'MESH' or not o.get("relic_part"):
            continue
        bone = o["bone"]
        if bone not in names:
            missing.append((o.name, bone))
            continue
        for g in list(o.vertex_groups):
            o.vertex_groups.remove(g)
        wts = o.get("wts")
        bns = o.get("bones")
        tb = o.get("toe_bone")
        if wts is not None and bns is not None and bns[0] in names and bns[1] in names:
            vga = o.vertex_groups.new(name=bns[0])
            vgb = o.vertex_groups.new(name=bns[1])
            for i, w in enumerate(list(wts)):
                vga.add([i], float(w), 'REPLACE')
                vgb.add([i], 1.0 - float(w), 'REPLACE')
        elif tb and tb in names:
            sy = o["split_y"]
            vg_f = o.vertex_groups.new(name=bone)
            vg_t = o.vertex_groups.new(name=tb)
            for v in o.data.vertices:
                t = max(0.0, min(1.0, (v.co.y - (sy - 0.02)) / 0.04))
                vg_f.add([v.index], t, 'REPLACE')
                vg_t.add([v.index], 1.0 - t, 'REPLACE')
        else:
            vg = o.vertex_groups.new(name=bone)
            vg.add(list(range(len(o.data.vertices))), 1.0, 'REPLACE')
        for m in list(o.modifiers):
            o.modifiers.remove(m)
        md = o.modifiers.new("Armature", 'ARMATURE')
        md.object = rig
        o.parent = rig
        bound += 1
    return {"bones": len(names), "bound": bound, "missing": missing}


def stage_ik():
    scn = bpy.data.scenes[SCENE]
    rig = bpy.data.objects["Relic_Rig"]
    out = {}
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
    vl = scn.view_layers[0]

    def knee_y(angle):
        for side in ("L",):
            rig.pose.bones["shin." + side].constraints[0].pole_angle = math.radians(angle)
        rig.pose.bones["ik_foot.L"].matrix = Matrix.Translation((0, -0.05, 0.28)) @ rig.pose.bones["ik_foot.L"].matrix
        vl.update()
        y = (rig.matrix_world @ rig.pose.bones["shin.L"].head).y
        rig.pose.bones["ik_foot.L"].location = (0, 0, 0)
        vl.update()
        return y
    res = {}
    for a in (-90, 0, 90, 180):
        res[a] = round(knee_y(a), 3)
    best = min(res, key=lambda k: res[k])
    for side in ("L", "R"):
        rig.pose.bones["shin." + side].constraints[0].pole_angle = math.radians(best)
    vl.update()
    out["knee_y_by_angle"] = res
    out["pole_angle"] = best
    return out


def render_shot(name, loc, target, res=(900, 1200), samples=24):
    scn = bpy.data.scenes[SCENE]
    place_camera(loc, target)
    scn.render.resolution_x, scn.render.resolution_y = res
    try:
        scn.eevee.taa_render_samples = samples
    except Exception:
        pass
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".png")
    scn.render.filepath = path
    scn.render.image_settings.file_format = 'PNG'
    win = bpy.context.window_manager.windows[0]
    prefs = bpy.context.preferences.view
    old = prefs.render_display_type
    prefs.render_display_type = 'NONE'
    try:
        with bpy.context.temp_override(window=win, scene=scn):
            bpy.ops.render.render(write_still=True)
    finally:
        prefs.render_display_type = old
    return path
