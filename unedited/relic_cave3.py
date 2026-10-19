# relic_cave3.py : debris cover over Relic (with a "Dug" shape key), rock library + scatter, props, roots, ceiling crack,
# lights and the buried / dug-out states. Exec after relic_cave.py and relic_cave2.py.

FINE_X = (-2.4, 2.4)
FINE_Z = (-0.9, 3.3)
FINE_STEP = 0.025
DEB = {}
LIB = {}


def floor_z(x, y):
    P = np.stack([np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel(), np.zeros(np.size(x))], axis=1)
    z = FLOOR_Z + 0.42 * (fbm(P, 0.18, 3, 31) - 0.5) * 2 + 0.10 * (fbm(P, 0.9, 3, 32) - 0.5) * 2 + 0.03 * (fbm(P, 4.0, 2, 33) - 0.5) * 2
    return z.reshape(np.shape(x))


# ------------------------------------------------------------------ image-space helpers
def dilate(A, r):
    out = A.copy()
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            if (dx or dz) and dx * dx + dz * dz <= r * r:
                out = np.maximum(out, np.roll(np.roll(A, dx, 0), dz, 1))
    return out


def blur(A, sigma):
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(A, r, mode='edge')
    tmp = np.zeros_like(pad)
    for i, w in enumerate(k):
        tmp += w * np.roll(pad, i - r, 0)
    out = np.zeros_like(pad)
    for i, w in enumerate(k):
        out += w * np.roll(tmp, i - r, 1)
    return out[r:-r, r:-r]


# ------------------------------------------------------------------ the debris cover
def body_depth(xs, zs):
    """ray-cast the posed body from the front; D[x,z] = how far the front surface sits in front of the slope plane"""
    rig = bpy.data.objects[RIG]
    body = bpy.data.objects["Relic_AI"]
    deps = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(deps)
    me = ev.to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    V = co.reshape(-1, 3).astype(np.float64)
    M = np.array(rig.matrix_world.inverted() @ ev.matrix_world)
    V = V @ M[:3, :3].T + M[:3, 3]
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    T = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", T)
    tree = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(t) for t in T.reshape(-1, 3)])
    ev.to_mesh_clear()
    D = np.zeros((len(xs), len(zs)))
    d = Vector((0, 1, 0))
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            hit = tree.ray_cast(Vector((x, -3.0, z)), d)
            if hit[0] is not None and hit[0].y < BACK:
                D[i, j] = BACK - hit[0].y
    return D


def window_center():
    rig = bpy.data.objects[RIG]
    h = rig.pose.bones["upperarm.L"].head
    return (h.x + 0.04, h.z + 0.05)


def build_debris(window_r=0.15, window_dz=0.0):
    cave = get_coll(CAVE)
    coll = get_coll("Cave_Ground", cave)
    _drop("Cave_Debris")
    rig = bpy.data.objects[RIG]
    xs = np.arange(FINE_X[0], FINE_X[1] + 1e-9, FINE_STEP)
    zs = np.arange(FINE_Z[0], FINE_Z[1] + 1e-9, FINE_STEP)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    Draw = body_depth(xs, zs)
    Hg = ground_h(X, Z)
    Ddil = np.maximum(blur(dilate(Draw, 4), 2.0), Draw)
    P = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    sh = X.shape
    ex = (X / 1.15) ** 2 + ((Z - 1.15) / 1.75) ** 2
    mound = 0.62 * np.exp(-ex) * (0.75 + 0.5 * fbm(P, 2.4, 3, 41).reshape(sh))
    T = 0.04 + 0.16 * smooth(fbm(P, 1.7, 3, 42).reshape(sh) * 1.5 - 0.25) + 0.05 * fbm(P, 6.0, 2, 43).reshape(sh)
    Hc = np.maximum(Ddil + 0.035, mound) + T * np.where(Ddil > 0.02, 1.0, 0.5)
    # a wide shallow dish around the shoulder so the headlamp can reach the gold
    cx, cz = window_center()
    cz += window_dz
    rr = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2)
    Wd = smooth((0.78 - rr) / 0.45)
    Hdish = Ddil + 0.07 + 0.04 * fbm(P, 5.0, 2, 45).reshape(sh)
    Hc = Hc * (1 - Wd) + np.minimum(Hc, Hdish) * Wd
    edge = np.minimum(np.minimum(X - FINE_X[0], FINE_X[1] - X), np.minimum(Z - FINE_Z[0], FINE_Z[1] - Z))
    bw = smooth(edge / 0.7)
    H0 = Hg + bw * (np.maximum(Hc, Hg) - Hg)
    # window: the gold shoulder cap breaks the surface
    Wg = np.exp(-((((X - cx) ** 2 + (Z - cz) ** 2) / window_r ** 2) ** 2))     # flat-topped window
    Hwin = np.maximum(dilate(Draw, 1) - 0.05, 0.0)
    H0 = H0 * (1 - Wg) + Hwin * Wg
    # the pit that gets dug: chest, head, shoulders
    pb = rig.pose.bones

    def mid(n):
        h, t = pb[n].head, pb[n].tail
        return ((h.x + t.x) / 2, (h.z + t.z) / 2)

    pit = np.zeros(sh)
    ells = [(*mid("chest"), 0.78, 0.62), (*mid("head"), 0.45, 0.48), (*mid("spine"), 0.55, 0.30),
            (pb["upperarm.L"].head.x, pb["upperarm.L"].head.z, 0.38, 0.34),
            (pb["upperarm.R"].head.x, pb["upperarm.R"].head.z, 0.38, 0.34)]
    for (ex_, ez_, rx, rz) in ells:
        d = np.sqrt(((X - ex_) / rx) ** 2 + ((Z - ez_) / rz) ** 2)
        pit = np.maximum(pit, smooth((1.18 - d) / 0.38))
    berm = 0.15 * (4 * pit * (1 - pit)) * (0.5 + fbm(P, 3.0, 3, 44).reshape(sh))
    hdug = np.maximum(Hg, Draw - 0.045) + berm
    H1 = H0 + pit * (hdug - H0)
    DEB.update(xs=xs, zs=zs, H0=H0, H1=H1, pit=pit, Draw=Draw, win=(cx, cz))
    V0 = np.stack([X.ravel(), (BACK - H0).ravel(), Z.ravel()], axis=1)
    V1 = np.stack([X.ravel(), (BACK - H1).ravel(), Z.ravel()], axis=1)
    me = mesh_from_arrays("Cave_Debris", V0, grid_faces(len(xs), len(zs), flip=False))
    set_smooth(me)
    ob = bpy.data.objects.new("Cave_Debris", me)
    coll.objects.link(ob)
    ob.matrix_world = rig_matrix()
    ob["win_x"], ob["win_z"] = float(cx), float(cz)
    ob.shape_key_add(name="Basis", from_mix=False)
    sk = ob.shape_key_add(name="Dug", from_mix=False)
    sk.data.foreach_set("co", V1.astype(np.float32).ravel())
    sk.value = 0.0
    return {"verts": len(me.vertices), "win": (round(cx, 3), round(cz, 3)), "max_body_depth": round(float(Draw.max()), 3),
            "pit_cells": int((pit > 0.5).sum())}


# ------------------------------------------------------------------ surface lookups (rig-local)
def surf_H(x, z, state=0):
    if "H0" in DEB and FINE_X[0] <= x <= FINE_X[1] and FINE_Z[0] <= z <= FINE_Z[1]:
        H = DEB["H1"] if state else DEB["H0"]
        fx = (x - FINE_X[0]) / FINE_STEP
        fz = (z - FINE_Z[0]) / FINE_STEP
        i = min(int(fx), H.shape[0] - 2)
        j = min(int(fz), H.shape[1] - 2)
        tx, tz = fx - i, fz - j
        return float(H[i, j] * (1 - tx) * (1 - tz) + H[i + 1, j] * tx * (1 - tz) + H[i, j + 1] * (1 - tx) * tz + H[i + 1, j + 1] * tx * tz)
    return float(ground_h(np.array([x]), np.array([z]))[0]) - 0.03


def surf_pit(x, z):
    if "pit" in DEB and FINE_X[0] <= x <= FINE_X[1] and FINE_Z[0] <= z <= FINE_Z[1]:
        i = min(max(int((x - FINE_X[0]) / FINE_STEP), 0), DEB["pit"].shape[0] - 1)
        j = min(max(int((z - FINE_Z[0]) / FINE_STEP), 0), DEB["pit"].shape[1] - 1)
        return float(DEB["pit"][i, j])
    return 0.0


def surf_normal(x, z, state=0):
    e = 0.06
    hx = (surf_H(x + e, z, state) - surf_H(x - e, z, state)) / (2 * e)
    hz = (surf_H(x, z + e, state) - surf_H(x, z - e, state)) / (2 * e)
    n = np.array([-hx, -1.0, -hz])
    return n / np.linalg.norm(n)


# ------------------------------------------------------------------ rock library
def _ico():
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    U = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int32)
    bm.free()
    return U, F


def shape_rock(U, F, seed, kind):
    rng = np.random.default_rng(seed)
    if kind == "slab":
        ax = np.array([1.0, rng.uniform(0.55, 1.0), rng.uniform(0.07, 0.14)])
        ncut = int(rng.integers(6, 10))
    elif kind == "chunk":
        ax = np.array([1.0, rng.uniform(0.6, 1.0), rng.uniform(0.45, 0.9)])
        ncut = int(rng.integers(10, 15))
    else:
        ax = np.array([1.0, rng.uniform(0.6, 1.0), rng.uniform(0.45, 0.85)])
        ncut = int(rng.integers(12, 19))
    V = U * ax
    for _ in range(ncut):
        if kind == "slab":
            phi = rng.uniform(0, 2 * np.pi)
            n = np.array([np.cos(phi), np.sin(phi), rng.uniform(-0.12, 0.12)])
            dfrac = rng.uniform(0.62, 0.92)
        elif kind == "chunk":
            n = np.eye(3)[int(rng.integers(0, 3))] * rng.choice([-1.0, 1.0]) + rng.normal(0, 0.28, 3)
            dfrac = rng.uniform(0.6, 0.9)
        else:
            n = rng.normal(0, 1, 3)
            dfrac = rng.uniform(0.52, 0.9)
        n = n / np.linalg.norm(n)
        sup = np.sqrt(((ax * n) ** 2).sum())
        d = sup * dfrac
        ex = V @ n - d
        m = ex > 0
        V[m] -= ex[m, None] * n
    r = np.linalg.norm(V, axis=1)[:, None]
    V = V + (V / np.maximum(r, 1e-6)) * ((fbm(V * 1.9, 1.0, 3, seed) - 0.5) * 0.10)[:, None]
    V /= np.abs(V[:, 0]).max()
    me = mesh_from_arrays("CaveRock_%s_%d" % (kind, seed), V, F)
    me["zr"] = float(np.abs(V[:, 2]).max())
    for p in me.polygons:
        p.use_smooth = False           # faceted, broken-stone look
    return me


def build_rock_library():
    for me in list(bpy.data.meshes):
        if me.name.startswith("CaveRock_") and me.users == 0:
            bpy.data.meshes.remove(me)
    U, F = _ico()
    LIB.clear()
    LIB["rock"] = [shape_rock(U, F, 100 + i, "rock") for i in range(12)]
    LIB["slab"] = [shape_rock(U, F, 200 + i, "slab") for i in range(5)]
    LIB["chunk"] = [shape_rock(U, F, 300 + i, "chunk") for i in range(6)]
    rock = rock_material("Cave_Rock", (0.016, 0.014, 0.012), (0.058, 0.050, 0.042), dust=(0.15, 0.125, 0.10), dust_amt=0.5,
                         rough=0.86, big=2.0, bump=1.3, bump_scale=10.0)
    conc = rock_material("Cave_Concrete", (0.065, 0.062, 0.056), (0.145, 0.140, 0.130), dust=(0.20, 0.17, 0.14), dust_amt=0.55,
                         rough=0.92, big=2.6, bump=1.0, bump_scale=14.0, rand_amt=0.3)
    for k, mat in (("rock", rock), ("slab", conc), ("chunk", conc)):
        for me in LIB[k]:
            me.materials.clear()
            me.materials.append(mat)
    return {k: len(v) for k, v in LIB.items()}


# ------------------------------------------------------------------ placement
def _rot_to(n, yaw, tilt, rng):
    z = np.asarray(n, dtype=np.float64)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 0, 1.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(yaw), math.sin(yaw)
    x2 = x * c + y * s
    y2 = -x * s + y * c
    R = np.stack([x2, y2, z], axis=1)
    if tilt:
        ax_ = rng.normal(0, 1, 3)
        ax_ -= z * (ax_ @ z)
        ax_ /= np.linalg.norm(ax_)
        R = np.array(Matrix.Rotation(math.radians(rng.uniform(-tilt, tilt)), 3, Vector(ax_))) @ R
    return R


def place_rock(coll, kind, x, z, s, rng, name, sink=0.7, tilt=22.0, aspect=True, lift=0.0):
    me = LIB[kind][int(rng.integers(0, len(LIB[kind])))]
    n = surf_normal(x, z, 0)
    y = BACK - surf_H(x, z, 0)
    R = _rot_to(n, rng.uniform(0, 2 * math.pi), tilt, rng)
    u = rng.uniform(0.8, 1.25, 3) if aspect else np.ones(3)
    zr = float(me.get("zr", 0.5))
    half = s * u[2] * zr
    p_local = np.array([x, y, z]) + n * (half * (1.0 - sink) + lift)
    Mw = np.array(rig_matrix())
    pw = Mw[:3, :3] @ p_local + Mw[:3, 3]
    Rw = Mw[:3, :3] @ R
    S = np.diag([s * u[0], s * u[1], s * u[2]])
    M = np.eye(4)
    M[:3, :3] = Rw @ S
    M[:3, 3] = pw
    o = bpy.data.objects.new(name, me)
    o.matrix_world = Matrix(M.tolist())
    coll.objects.link(o)
    return o


def scatter_rocks(seed=5, density=1.0):
    rng = np.random.default_rng(seed)
    cave = get_coll(CAVE)
    wipe_collection("Cave_Rocks")
    wipe_collection("Cave_Cover")
    cr = get_coll("Cave_Rocks", cave)
    cc = get_coll("Cave_Cover", cave)
    cx, cz = DEB["win"]
    count = {"rock": 0, "cover": 0, "slab": 0, "floor": 0}

    def put(kind, x, z, s, name, **kw):
        # keep the glint window clear
        if math.hypot(x - cx, z - cz) < 0.17 + s * 0.8:
            return None
        pit = surf_pit(x, z)
        coll = cc if pit > 0.06 else cr
        o = place_rock(coll, kind, x, z, s, rng, name, **kw)
        o["pit"] = pit
        count["cover" if pit > 0.06 else "rock"] += 1
        return o

    def lg(a, b):
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))

    # near the body (the fine patch): gravel, small and medium stones
    for i in range(int(300 * density)):
        put("rock", rng.uniform(*FINE_X), rng.uniform(*FINE_Z), lg(0.025, 0.08), "g%d" % i, tilt=35.0)
    for i in range(int(120 * density)):
        put("rock", rng.uniform(*FINE_X), rng.uniform(*FINE_Z), lg(0.08, 0.2), "s%d" % i)
    for i in range(int(45 * density)):
        put("rock", rng.uniform(*FINE_X), rng.uniform(*FINE_Z), lg(0.2, 0.42), "m%d" % i)
    for i in range(int(8 * density)):
        put("chunk", rng.uniform(-2.0, 2.0), rng.uniform(-0.5, 3.0), lg(0.25, 0.5), "c%d" % i)
    # the wider slope
    for i in range(int(130 * density)):
        x, z = rng.uniform(-9, 9), rng.uniform(-2.5, 10)
        if FINE_X[0] < x < FINE_X[1] and FINE_Z[0] < z < FINE_Z[1]:
            continue
        put("rock", x, z, lg(0.12, 0.5), "sm%d" % i)
    for i in range(int(75 * density)):
        x, z = rng.uniform(-9, 9), rng.uniform(-2.0, 10)
        if FINE_X[0] - 0.5 < x < FINE_X[1] + 0.5 and FINE_Z[0] - 0.5 < z < FINE_Z[1] + 0.5:
            continue
        put("rock", x, z, lg(0.5, 1.2), "lg%d" % i)
    for i in range(int(16 * density)):
        x, z = rng.uniform(-9, 9), rng.uniform(0.5, 10)
        if abs(x) < 3.2 and z < 4.0:
            continue
        put("rock", x, z, lg(1.2, 2.2), "bo%d" % i, tilt=15.0)
    for i in range(int(26 * density)):
        x, z = rng.uniform(-9, 9), rng.uniform(-2, 10)
        if FINE_X[0] - 0.4 < x < FINE_X[1] + 0.4 and FINE_Z[0] - 0.4 < z < FINE_Z[1] + 0.4:
            continue
        put("chunk", x, z, lg(0.3, 0.9), "ch%d" % i)
    # concrete slabs: three lie across Relic, the rest are scattered
    keys = [(-0.20, 1.38, 0.52, 20), (0.06, 2.18, 0.40, -15), (0.00, 0.45, 0.68, 8), (cx + 0.66, cz - 0.04, 0.36, 35),
            (cx - 0.20, cz + 0.62, 0.33, -30), (-0.95, 1.0, 0.50, 70)]
    for i, (x, z, s, yaw) in enumerate(keys):
        o = place_rock(cc if surf_pit(x, z) > 0.06 else cr, "slab", x, z, s, rng, "slab_key%d" % i, sink=0.55, tilt=6.0, aspect=False)
        o["pit"] = surf_pit(x, z)
        count["slab"] += 1
    for i in range(int(22 * density)):
        x, z = rng.uniform(-8, 8), rng.uniform(-2, 9)
        if FINE_X[0] - 0.6 < x < FINE_X[1] + 0.6 and FINE_Z[0] - 0.6 < z < FINE_Z[1] + 0.6:
            continue
        place_rock(cr, "slab", x, z, lg(0.5, 1.5), rng, "slab%d" % i, sink=0.6, tilt=20.0)
        count["slab"] += 1
    # the cave floor
    for i in range(int(70 * density)):
        x, y = rng.uniform(-7, 7), rng.uniform(-8, -0.8)
        zf = float(floor_z(np.array([x]), np.array([y]))[0])
        me = LIB["rock" if rng.random() < 0.75 else "chunk"][0 if False else int(rng.integers(0, 6))]
        s = lg(0.06, 0.8)
        o = bpy.data.objects.new("fl%d" % i, me)
        R = _rot_to(np.array([0, 0, 1.0]), rng.uniform(0, 6.28), 30.0, rng)
        S = np.diag(rng.uniform(0.8, 1.25, 3) * s)
        M = np.eye(4)
        M[:3, :3] = R @ S
        M[:3, 3] = [x, y, zf + s * 0.2]
        o.matrix_world = Matrix(M.tolist())
        cr.objects.link(o)
        count["floor"] += 1
    return count
