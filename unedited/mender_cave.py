# mender_cave.py : Mender in the Cave scene.
# A second Mender (own rig, own pose, same mesh data) in collection "Mender_Cave" with a real spot-light headlamp on the head bone,
# plus pose helpers that work in WORLD space (look-at, foot placement on the rubble, two-bone arm reach, pick grip).
# Exec order in one execute_blender_code call: relic_cave.py .. relic_cave5.py, mender_build.py, mender_scene.py, mender_pose.py, this file.
# Names: rig MenderC_Rig, objects MenderC_*, light MenderC_Headlamp. The studio copy (Mender_Rig, scene Mender_Studio) is untouched.

CRIG = "MenderC_Rig"
CCOLL = "Mender_Cave"
CP = "MenderC_"


def w2l(rig, v):
    """world direction -> rig-local direction"""
    return rig.matrix_world.inverted().to_3x3() @ Vector(v)


def p2l(rig, p):
    """world point -> rig-local point"""
    return rig.matrix_world.inverted() @ Vector(p)


def l2w(rig, v):
    """rig-local direction -> world direction"""
    return rig.matrix_world.to_3x3() @ Vector(v)


def rest_studio():
    """the studio Mender back to the rest pose with the pick on the pack and the scanner in the holster (window scene must be Mender_Studio)"""
    src_rig = bpy.data.objects[RIGN]
    src_rig.rotation_euler = (0, 0, 0)
    src_rig.location = (0, 0, 0)
    reset_pose(src_rig)
    stow_props(src_rig, bpy.data.objects["Mender_Pick"], bpy.data.objects["Mender_Scanner"])
    bpy.context.view_layer.update()
    return "rest"


def dup_mender(scene_name="Cave", energy=140.0):
    """copy the studio Mender (rest pose, props stowed; call rest_studio() first) into the Cave scene: linked mesh data, own armature object
    with its own pose. The window scene must be the Cave scene."""
    sc = bpy.data.scenes[scene_name]
    wipe(CCOLL)
    for lt in list(bpy.data.lights):
        if lt.name.startswith(CP) and lt.users == 0:
            bpy.data.lights.remove(lt)
    dst = bpy.data.collections.new(CCOLL)
    sc.collection.children.link(dst)
    src = bpy.data.collections[MC]
    mp = {}
    for o in src.objects:
        n = o.copy()
        n.name = CP + o.name[len("Mender_"):]
        dst.objects.link(n)
        mp[o.name] = n
    for name, n in mp.items():
        o = bpy.data.objects[name]
        if o.parent is not None:
            # setting parent / parent_type / parent_bone each clears the parent inverse, so restore it last
            pt, pbone, inv = o.parent_type, o.parent_bone, o.matrix_parent_inverse.copy()
            n.parent = mp[o.parent.name]
            n.parent_type = pt
            n.parent_bone = pbone
            n.matrix_parent_inverse = inv
        for md in n.modifiers:
            if md.type == 'ARMATURE' and md.object is not None and md.object.name in mp:
                md.object = mp[md.object.name]
    rn = mp[RIGN]
    fixed = 0
    for pb in rn.pose.bones:
        for c in pb.constraints:
            t = getattr(c, "target", None)
            if t is not None and t.name == RIGN:
                c.target = rn
                fixed += 1
            t = getattr(c, "pole_target", None)
            if t is not None and t.name == RIGN:
                c.pole_target = rn
                fixed += 1
    bpy.context.view_layer.update()
    lamp = add_headlamp(rn, dst, energy)
    lantern = add_lantern_light(rn, dst)
    return {"objects": len(mp), "constraint_targets_fixed": fixed, "lamp": lamp.name, "lantern": lantern.name}


def add_lantern_light(rig, coll, energy=9.0):
    """a small warm point light inside the lantern on the pack (bone chest), so his back and the rocks beside him pick up a little amber"""
    for nm in (CP + "LanternLight",):
        old = bpy.data.objects.get(nm)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
    g = bpy.data.objects[CP + "LanternGlass"]
    co = np.empty(len(g.data.vertices) * 3, dtype=np.float32)
    g.data.vertices.foreach_get("co", co)
    c = co.reshape(-1, 3).astype(float).mean(0)           # rest rig-space centre (bone-parented parts keep rest coordinates)
    ld = bpy.data.lights.get(CP + "LanternLight") or bpy.data.lights.new(CP + "LanternLight", 'POINT')
    ld.energy = energy
    ld.color = (1.0, 0.62, 0.28)
    ld.shadow_soft_size = 0.05
    lo = bpy.data.objects.new(CP + "LanternLight", ld)
    coll.objects.link(lo)
    attach(lo, rig, 'chest', frame4(c, np.eye(3)))
    return lo


def verify_dup():
    """largest distance between the world position of each studio object and its Cave copy (both rigs at rest, at the origin)"""
    bpy.context.view_layer.update()
    worst, who = 0.0, None
    for o in bpy.data.collections[MC].objects:
        n = bpy.data.objects.get(CP + o.name[len("Mender_"):])
        if n is None:
            return {"missing": o.name}
        d = (o.matrix_world.translation - n.matrix_world.translation).length
        d += float(np.abs(np.array(o.matrix_world) - np.array(n.matrix_world)).max())
        if d > worst:
            worst, who = d, o.name
    return {"worst": round(worst, 6), "object": who}


def add_headlamp(rig, coll, energy=140.0, cone_deg=44.0):
    """a real spot light on the hard-hat lamp, parented to the head bone so it follows every head turn"""
    old = bpy.data.objects.get(CP + "Headlamp")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    ld = bpy.data.lights.get(CP + "Headlamp") or bpy.data.lights.new(CP + "Headlamp", 'SPOT')
    ld.energy = energy
    ld.spot_size = math.radians(cone_deg)
    ld.spot_blend = 0.55
    ld.shadow_soft_size = 0.03
    ld.color = (1.0, 0.92, 0.78)
    lo = bpy.data.objects.new(CP + "Headlamp", ld)
    coll.objects.link(lo)
    scv = np.asarray(SC, float)
    origin = np.array([0.0, -0.1945, 1.728]) * scv
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])      # light axis (-Z) = head forward (-Y)
    attach(lo, rig, 'head', frame4(origin, R))
    return lo


# ------------------------------------------------------------------ world-space posing
def aim_w(rig, name, d_world, twist=0.0):
    aim(rig, name, w2l(rig, d_world), twist)


def look_dir(rig, name, d_world, frac=1.0):
    """turn bone `name` about its head so that its forward axis (rest -Y) points a fraction `frac` of the way toward d_world"""
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    fwd = (delta.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()
    d = w2l(rig, d_world).normalized()
    q = Quaternion().slerp(fwd.rotation_difference(d), frac)
    m = pb.matrix.copy()
    pb.matrix = Matrix.Translation(m.translation) @ q.to_matrix().to_4x4() @ m.to_3x3().to_4x4()
    bpy.context.view_layer.update()


def _frame3(f, n):
    """columns (right, forward, up) for a forward vector f and a normal n"""
    f = Vector(f).normalized()
    n = Vector(n).normalized()
    f = (f - n * f.dot(n)).normalized()
    r = f.cross(n).normalized()
    return Matrix([[r.x, f.x, n.x], [r.y, f.y, n.y], [r.z, f.z, n.z]])


def place_foot(rig, s, ankle_w, fwd_w, normal_w=(0.0, 0.0, 1.0)):
    """ik foot s: the ankle (bone head) at ankle_w (world), the boot pointing along fwd_w, its sole turned to normal_w"""
    pb = rig.pose.bones['ik_foot.' + s]
    bpy.context.view_layer.update()
    rest = pb.bone.matrix_local.to_3x3()
    F0 = _frame3((0, -1, 0), (0, 0, 1))
    F = _frame3(w2l(rig, fwd_w), w2l(rig, normal_w))
    new3 = (F @ F0.inverted()) @ rest
    pb.matrix = Matrix.Translation(p2l(rig, ankle_w)) @ new3.to_4x4()
    bpy.context.view_layer.update()


def reach_arm(rig, s, wrist_w, pole_w):
    """FK arm s: the wrist (forearm tail) goes to wrist_w (world), the elbow bends toward the world direction pole_w; returns the miss distance"""
    bpy.context.view_layer.update()
    up = rig.pose.bones['upperarm.' + s]
    fo = rig.pose.bones['forearm.' + s]
    S = up.head.copy()
    L1, L2 = up.length, fo.length
    T = p2l(rig, wrist_w)
    d = T - S
    dist = d.length
    u = d / dist
    dc = min(max(dist, abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4)
    a = (L1 * L1 - L2 * L2 + dc * dc) / (2.0 * dc)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    pv = w2l(rig, pole_w)
    pv = pv - u * pv.dot(u)
    pv.normalize()
    E = S + u * a + pv * h
    Tc = S + u * dc
    aim(rig, 'upperarm.' + s, E - S)
    aim(rig, 'forearm.' + s, Tc - E)
    return dist - dc


def hand_world_frame(rig, s):
    """the posed hand frame (columns x across, y back of the hand, z along the fingers) in WORLD space, as a numpy 3x3"""
    pb = rig.pose.bones['hand.' + s]
    bpy.context.view_layer.update()
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    Rp = np.array(delta.to_3x3()) @ _rest_frame_world(s)
    return np.array(rig.matrix_world.to_3x3()) @ Rp


def orient_hand_w(rig, s, z_w, y_w):
    """hand s: fingers along z_w, the back of the hand toward y_w (world directions)"""
    zl = np.array(w2l(rig, z_w))
    yl = np.array(w2l(rig, y_w))
    orient_hand(rig, s, basis(zl, yl))


# ------------------------------------------------------------------ ground contact
_SKIP_PREFIX = ("MenderC_", "Mender_", "Relic")


def ground_at(x, y, z_top=3.4, skip_names=()):
    """world z and normal of the rubble under (x, y), looking down from z_top; lights, Mender and Relic are ignored"""
    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    o = Vector((x, y, z_top))
    d = Vector((0.0, 0.0, -1.0))
    for _ in range(16):
        ok, loc, nrm, idx, ob, mat = sc.ray_cast(deps, o, d)
        if not ok:
            return None
        if ob.name.startswith(_SKIP_PREFIX) or ob.name in skip_names:
            o = loc + d * 0.001
            continue
        return loc, nrm
    return None


def foot_plane(center_xy, fwd_xy, half_len=0.12, half_w=0.045, skip_names=()):
    """ground under a boot: returns (ankle height above the sole contact, normal) sampled at heel, toe, left and right"""
    cx, cy = center_xy
    f = Vector((fwd_xy[0], fwd_xy[1], 0.0)).normalized()
    r = f.cross(Vector((0, 0, 1))).normalized()
    pts = []
    for dl, dw in ((half_len, 0.0), (-half_len, 0.0), (0.0, half_w), (0.0, -half_w), (0.0, 0.0)):
        p = Vector((cx, cy, 0.0)) + f * dl + r * dw
        g = ground_at(p.x, p.y, skip_names=skip_names)
        pts.append(g[0] if g else Vector((p.x, p.y, -0.7)))
    zc = pts[4].z
    n = (pts[0] - pts[1]).cross(pts[3] - pts[2])
    if n.z < 0:
        n = -n
    n.normalize()
    zmax = max(p.z for p in pts[:4])
    return Vector((cx, cy, 0.5 * (zc + zmax))), n
