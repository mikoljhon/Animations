# relic_cave5.py : one-call rebuild + camera presets. Exec after relic_cave.py .. relic_cave4.py.


def build_all():
    r = {}
    r["setup"] = cave_setup(save=False)
    r["pose"] = cave_pose(True)
    r["shell"] = build_shell()
    r["slit"] = cut_slit()
    r["sky"] = build_sky()
    r["floor"] = build_floor()
    r["slope"] = build_slope()
    r["debris"] = build_debris()
    r["lib"] = build_rock_library()
    r["rocks"] = scatter_rocks()
    r["roots"] = build_roots()
    r["props"] = add_props()
    r["vol"] = build_volume()
    assign_cave_materials()
    for n in ("Test_Spot", "Test_Fill", "TestSun"):
        o = bpy.data.objects.get(n)
        if o:
            bpy.data.objects.remove(o, do_unlink=True)
    r["lights"] = build_lights()
    r["controls"] = build_controls()
    r["state"] = cave_state(False)
    r["crust"] = tune_crust(0.30)
    r["face"] = clean_face()
    return r


def bone_world(name, t=0.5):
    rig = bpy.data.objects[RIG]
    pb = rig.pose.bones[name]
    p = pb.head * (1 - t) + pb.tail * t
    return rig.matrix_world @ p


def posed_dir_world(name, rest_dir):
    """a direction given in rig-local REST space, carried by the posed bone, in world space"""
    rig = bpy.data.objects[RIG]
    pb = rig.pose.bones[name]
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    return (rig.matrix_world.to_3x3() @ (delta.to_3x3() @ Vector(rest_dir))).normalized()


def visor_world():
    rig = bpy.data.objects[RIG]
    pb = rig.pose.bones["head"]
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    p = rig.matrix_world @ (delta @ Vector((0, -0.235, 1.862)))
    n = (rig.matrix_world.to_3x3() @ (delta.to_3x3() @ Vector((0, -1, 0)))).normalized()
    # the porthole glass faces a little differently from the head's forward axis: use the real surface normal
    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    o = p + n * 0.6
    for _ in range(6):
        ok, loc, nrm, idx, ob, mat = sc.ray_cast(deps, o, -n)
        if not ok:
            break
        if ob.name == "Relic_AI":
            nn = nrm.normalized()
            if nn.dot(n) < 0.35:        # a stray surface (rim, collar): trust the head's own forward axis
                return p, n
            return loc, nn
        o = loc - n * 0.001
    return p, n


def visor_cam(dist=0.95, spread=45, step=9):
    """search the directions around the visor for the one that shows both glyph eyes fully (they sit inside a porthole)"""
    rig = bpy.data.objects[RIG]
    pb = rig.pose.bones["head"]
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    Mw = rig.matrix_world
    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    vp, n0 = visor_world()
    pts = []
    for side in (-1, 1):
        cc = (side * 0.0335, -0.262, 1.862)
        for k in range(7):
            if k == 0:
                ox, oz = 0.0, 0.0
            else:
                a = 2 * math.pi * (k - 1) / 6
                ox, oz = 0.022 * math.cos(a), 0.022 * math.sin(a)
            pts.append(Mw @ (delta @ Vector((cc[0] + ox, cc[1], cc[2] + oz))))
    up = Vector((0, 0, 1))
    r = n0.cross(up).normalized()
    u = r.cross(n0).normalized()
    best = None
    for az in range(-spread, spread + 1, step):
        for el in range(-spread, spread + 1, step):
            d = (n0 + r * math.tan(math.radians(az)) + u * math.tan(math.radians(el))).normalized()
            cam = vp + d * dist
            vis = 0
            for p in pts:
                dv = p - cam
                L = dv.length
                dv.normalize()
                o = cam.copy()
                blocked = False
                for _ in range(5):
                    ok, loc, nrm, idx, ob, mat = sc.ray_cast(deps, o, dv)
                    if not ok or (loc - cam).length >= L - 0.002:
                        break
                    if ob.name == "Relic_AI_Crust" or "Puff_" in ob.name or "Pebble" in ob.name:      # dust spheres and pebbles never block a view
                        o = loc + dv * 0.0005
                        continue
                    if ob.name == "Relic_AI_Eyes":
                        break
                    blocked = True
                    break
                if not blocked:
                    vis += 1
            score = vis - 0.002 * (abs(az) + abs(el))
            if best is None or score > best[0]:
                best = (score, cam.copy(), vis, az, el)
    return best[1], vp, {"visible_pts": best[2], "of": len(pts), "az": best[3], "el": best[4]}


def presets():
    W = window_world()
    ring, ringf = ring_world()
    vp, vn = visor_world()
    right = vn.cross(Vector((0, 0, 1))).normalized()
    front = (rig_matrix().to_3x3() @ Vector((0, -1, 0))).normalized()    # world direction the slope faces
    return {
        "wide": dict(loc=(-4.6, -5.6, 2.1), target=(0.3, 1.3, 1.0), lens=26.0),
        "mid": dict(loc=(-2.2, -2.6, 1.6), target=(0.3, 1.2, 0.9), lens=32.0),
        "glint": dict(loc=tuple(W + front * 1.35 + Vector((0.55, -0.1, 0.0))), target=tuple(W), lens=60.0, fstop=3.2),
        "chest": dict(loc=tuple(ring + Vector((0.8, -1.55, 0.6))), target=tuple(ring), lens=42.0, fstop=4.0),
        "face": dict(loc=tuple(vp + vn * 1.0 + right * 0.18 + Vector((0, 0, 0.08))), target=tuple(vp), lens=70.0, fstop=3.2),
        "dug_wide": dict(loc=(-3.4, -4.2, 2.3), target=(0.1, 1.2, 1.1), lens=30.0),
    }


def take(name, key, res=(1280, 720), samples=48):
    if key == "face":
        cam, tgt, info = visor_cam()
        return shot(name, tuple(cam), tuple(tgt), lens=70.0, res=res, samples=samples, fstop=3.2)
    p = presets()[key]
    return shot(name, p["loc"], p["target"], lens=p.get("lens", 35.0), res=res, samples=samples, fstop=p.get("fstop"))
