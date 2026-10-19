# Cut 6 fx tune (background Blender, after ma5_rocks.py): thinner dust, new camera aim, puffs, and pebbles that start ON Relic and slide off
# when his head lifts.   blender -b relic_v13_rocks.blend --python ma5_fxtune.py   -> relic_v13_final.blend
exec(open(r"C:\ai\unedited\ma5_load.py").read())
import time
from mathutils.bvhtree import BVHTree

T0 = time.time()
LOG = open(r"C:\ai\unedited\anim_cache5\ma5_fxtune_log.txt", "w")


def log(*a):
    LOG.write("%7.1f  %s\n" % (time.time() - T0, " ".join(str(x) for x in a)))
    LOG.flush()


sc = bpy.data.scenes["Cave"]
rig = bpy.data.objects[CRIG]
key_controls(sc)
log("controls + dust keyed")
face = face_cam_positions(frame=Gi(169))
build_cameras3(sc, face)
log("cameras", sc.camera.name if sc.camera else None)
key_puffs(sc, rig)
log("puffs")


def bvh_world(objs, dg):
    verts, polys, off = [], [], 0
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = o.matrix_world
        w = co @ np.array(M.to_3x3()).T + np.array(M.translation)
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        verts.append(w)
        polys.append(tri.reshape(-1, 3) + off)
        off += len(w)
        oe.to_mesh_clear()
    return BVHTree.FromPolygons([Vector(v) for v in np.concatenate(verts).tolist()], [tuple(p) for p in np.concatenate(polys).tolist()])


def key_pebbles2(sc, n=16, f_vis=None, f_end=None):
    coll = bpy.data.collections[CCOLL]
    for ob in list(coll.objects):
        if ob.name.startswith(CP + "Pebble"):
            bpy.data.objects.remove(ob, do_unlink=True)
    rng_ = np.random.default_rng(21)
    mat = bpy.data.materials["Cave_Rock"]
    f_vis = f_vis or (C3 + 26)                  # they appear inside the dust, lying on Relic
    f_end = f_end or Gi(190)
    f_lift = int(G(104))
    relic_objs = [o for o in bpy.data.collections["Relic_AI"].all_objects if o.type == "MESH" and "Eyes" not in o.name]
    cover_names = {o.name for o in bpy.data.collections["Cave_Cover"].objects}
    static_objs = [o for o in bpy.data.collections["Cave_Rocks"].all_objects if o.type == "MESH" and not o.hide_viewport and o.name not in cover_names
                   and (o.matrix_world.translation - Vector((0.1, 1.3, 1.1))).length < 3.0] + [bpy.data.objects["Cave_Floor"], bpy.data.objects["Cave_Slope"]]
    sc.frame_set(f_vis)
    dg = bpy.context.evaluated_depsgraph_get()
    static_bvh = bvh_world(static_objs, dg)
    relic_bvh = bvh_world(relic_objs, dg)
    rr = bpy.data.objects[RIG]
    head = rr.matrix_world @ rr.pose.bones['head'].head
    sh = rr.matrix_world @ rr.pose.bones['upperarm.L'].head
    chest = rr.matrix_world @ rr.pose.bones['chest'].head
    items = []
    for k in range(n):
        base = head if k % 3 == 0 else (sh if k % 3 == 1 else chest)
        hit = (None,)
        for _try in range(12):
            x, y = base.x + rng_.uniform(-0.2, 0.2), base.y + rng_.uniform(-0.12, 0.12)
            hit = relic_bvh.ray_cast(Vector((x, y, base.z + 1.0)), Vector((0, 0, -1)), 2.5)
            if hit[0] is not None:
                break
        if hit[0] is None:
            continue
        s = float(rng_.uniform(0.016, 0.040))
        ob = bpy.data.objects.new(CP + "Pebble_%02d" % k, pebble_mesh())
        coll.objects.link(ob)
        ob.data = pebble_mesh()
        ob.scale = (s * 2, s * 2, s * 2)
        ob.rotation_mode = 'QUATERNION'
        if mat.name not in [m.name for m in ob.data.materials]:
            ob.data.materials.append(mat)
        p = Vector(hit[0]) + Vector(hit[1]).normalized() * s
        items.append(dict(ob=ob, p=p, v=Vector(), r=s, q=Quaternion((1, 0, 0, 0)), w=Vector()))
    log("pebbles placed", len(items))
    DT = 1.0 / (24.0 * 8)
    for f in range(f_vis, f_end + 1):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        if f >= f_lift - 3:
            relic_bvh = bvh_world(relic_objs, dg)
        deb_bvh = BVHTree.FromObject(bpy.data.objects["Cave_Debris"], dg)
        bvhs = [relic_bvh, deb_bvh, static_bvh]
        for it in items:
            for _s in range(8):
                it['v'] += Vector((0, 0, -9.81)) * DT
                it['p'] += it['v'] * DT
                hit_n = None
                for bvh in bvhs:
                    res = bvh.find_nearest(it['p'], it['r'] * 1.6)
                    if res[0] is None:
                        continue
                    loc, n, idx, dist = res
                    n = Vector(n).normalized()
                    if (it['p'] - loc).dot(n) < 0:
                        n, pen = -n, it['r'] + dist
                    else:
                        pen = it['r'] - dist
                    if pen <= 0:
                        continue
                    it['p'] += n * pen
                    vn = it['v'].dot(n)
                    if vn < 0:
                        it['v'] -= (1.0 + (0.35 if vn < -0.7 else 0.0)) * vn * n
                    hit_n = n
                if hit_n is not None:
                    vt = it['v'] - it['v'].dot(hit_n) * hit_n
                    tilt = math.degrees(hit_n.angle(Vector((0, 0, 1))))
                    if vt.length < 0.2 and tilt < 38.0:
                        it['v'] = Vector()                            # sticks: dust and a little friction hold it on the armour
                    else:
                        it['v'] -= vt * min(1.0, 3.5 * DT)
                        it['w'] = it['w'].lerp(hit_n.cross(it['v']) / it['r'], 0.2)
                ang = it['w'].length * DT
                if ang > 1e-9:
                    it['q'] = (Quaternion(it['w'].normalized(), ang) @ it['q']).normalized()
            ob = it['ob']
            ob.location = it['p']
            ob.rotation_quaternion = it['q']
            ob.keyframe_insert('location', frame=f)
            ob.keyframe_insert('rotation_quaternion', frame=f)
    for it in items:
        ob = it['ob']
        for f, hide in ((1, True), (f_vis, False)):
            ob.hide_render = hide
            ob.keyframe_insert('hide_render', frame=f)
        for fc in iter_fcurves(ob):
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT' if fc.data_path == 'hide_render' else 'LINEAR'
    sc.frame_set(1)
    return len(items)


bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
log("pebbles", key_pebbles2(sc))
sc.frame_set(1)
sc.render.filepath = r"C:\ai\unedited\renders\anim_open6\f_"
bpy.ops.wm.save_as_mainfile(filepath=r"C:\ai\unedited\relic_v13_final.blend")
log("saved")
LOG.close()
