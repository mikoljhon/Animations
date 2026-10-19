# relic_cave.py : the buried-cave set for UNEDITED episode 1 (scene "Cave" in relic_v06_cave.blend).
# Exec it, then call the stage functions. Everything is idempotent (a stage deletes what it built before).
# Frames: "rig-local" = the armature's own space (Relic standing, facing -Y, +X = his left, feet at z=0).
# The rig object is tilted by TILT about X so he lies back on a rubble slope; the slope plane is y = BACK in rig-local.
import bpy, bmesh, math, os, types, random
import numpy as np
from mathutils import Vector, Matrix, Euler, Quaternion
from mathutils.bvhtree import BVHTree

OUT_DIR = r"C:\ai\unedited"
RIG = "Relic_AI_Rig"
TILT = math.radians(-55.0)          # reclined, head up the slope
BACK = 0.22                          # rig-local y of the slope plane behind his back
CAVE = "Cave"


# ------------------------------------------------------------------ vectorised value noise
_TAB = np.random.default_rng(7).random(1 << 20).astype(np.float64)
_MASK = (1 << 20) - 1


def _hash(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ ((seed * 2654435761) & 0xFFFFFFFF)
    return h & _MASK


def vnoise(P, seed=0):
    P = np.asarray(P, dtype=np.float64)
    i = np.floor(P).astype(np.int64)
    f = P - i
    u = f * f * (3.0 - 2.0 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out = out + w * _TAB[_hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)]
    return out


def fbm(P, freq=1.0, octaves=4, seed=0, gain=0.5, lac=2.03):
    P = np.asarray(P, dtype=np.float64) * freq
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise(P + o * 17.31, seed + o * 31)
        norm += amp
        amp *= gain
        P = P * lac
    return tot / norm


def ridge(P, freq=1.0, octaves=3, seed=0):
    return 1.0 - np.abs(2.0 * fbm(P, freq, octaves, seed) - 1.0)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------ shader node helper (same as relic_model3._nh)
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


# ------------------------------------------------------------------ small helpers
def get_coll(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def wipe_collection(name):
    c = bpy.data.collections.get(name)
    if not c:
        return
    for ch in list(c.children):
        wipe_collection(ch.name)
    for o in list(c.objects):
        d = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if d is not None and getattr(d, "users", 1) == 0:
            for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.lights, bpy.data.cameras):
                if d.name in coll and coll[d.name] == d:
                    coll.remove(d)
                    break
    bpy.data.collections.remove(c)


def mesh_from_arrays(name, V, F):
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int32)
    me = bpy.data.meshes.new(name)
    try:
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.ravel())
        me.loops.add(F.shape[0] * F.shape[1])
        me.polygons.add(F.shape[0])
        me.polygons.foreach_set("loop_start", np.arange(0, F.shape[0] * F.shape[1], F.shape[1], dtype=np.int32))
        me.loops.foreach_set("vertex_index", F.ravel())
        me.update(calc_edges=True)
        if len(me.polygons) != F.shape[0] or len(me.polygons[0].vertices) != F.shape[1]:
            raise RuntimeError("polygon layout mismatch")
    except Exception:
        bpy.data.meshes.remove(me)
        me = bpy.data.meshes.new(name)
        me.from_pydata(V.tolist(), [], F.tolist())
        me.update()
    return me


def grid_faces(nx, nz, flip=False):
    idx = np.arange(nx * nz).reshape(nx, nz)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    F = np.stack([a, b, c, d], axis=1)
    return F[:, ::-1] if flip else F


def set_smooth(me, angle_deg=None):
    for p in me.polygons:
        p.use_smooth = True


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


# ------------------------------------------------------------------ stage 1: file, scene, world, render settings
def cave_setup(save=True):
    sc = bpy.context.scene
    if sc.name != "Cave":
        sc.name = "Cave"
    if save and not bpy.data.filepath.endswith("relic_v06_cave.blend"):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "relic_v06_cave.blend"))
    for n in ("Relic_Floor", "Key", "Fill", "Rim", "Rim2"):
        o = bpy.data.objects.get(n)
        if o and o.name in sc.objects:
            o.hide_viewport = True
            o.hide_render = True
    w = bpy.data.worlds.get("Cave_World") or bpy.data.worlds.new("Cave_World")
    sc.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    out = N.new('ShaderNodeOutputWorld')
    bg = N.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.006, 0.0075, 0.010, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    L.new(bg.outputs['Background'], out.inputs['Surface'])
    # no world volume: a sun shining through an infinite volume is extinguished completely; dust lives in Cave_Volume
    ee = sc.eevee
    ee.use_raytracing = True
    ee.ray_tracing_method = 'SCREEN'
    ee.use_volumetric_shadows = True
    ee.volumetric_samples = 96
    ee.volumetric_tile_size = '4'
    ee.volumetric_start = 0.1
    ee.volumetric_end = 60.0
    ee.shadow_ray_count = 2
    ee.shadow_step_count = 8
    ee.taa_render_samples = 48
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = 0.4
    return {"file": bpy.data.filepath, "scene": sc.name}


# ------------------------------------------------------------------ stage 2: the buried pose
def _aim(rig, name, d, twist=0.0):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    m = pb.matrix.copy()
    y = m.col[1].to_3d().normalized()
    dv = Vector(d).normalized()
    q = y.rotation_difference(dv)
    R = q.to_matrix().to_4x4()
    new = Matrix.Translation(m.translation) @ R @ m.to_3x3().to_4x4()
    if twist:
        new = Matrix.Translation(new.translation) @ Matrix.Rotation(math.radians(twist), 4, dv) @ new.to_3x3().to_4x4()
    pb.matrix = new
    bpy.context.view_layer.update()


def _place(rig, name, pos):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    m = pb.matrix.copy()
    m.translation = Vector(pos)
    pb.matrix = m
    bpy.context.view_layer.update()


def pose_reset(rig):
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


POSE = {
    "spine": ((0.06, 0.00, 1.0), 0),
    "chest": ((0.10, 0.02, 1.0), 0),
    "neck": ((0.16, 0.03, 1.0), 0),
    "head": ((0.30, 0.04, 1.0), 18),
    "clavicle.L": ((0.30, 0.05, -0.16), 0),
    "clavicle.R": ((-0.30, 0.05, -0.16), 0),
    "upperarm.L": ((0.20, 0.16, -0.96), 0),
    "forearm.L": ((0.10, 0.12, -0.99), 0),
    "hand.L": ((0.06, 0.02, -1.0), 0),
    "upperarm.R": ((-0.80, 0.20, -0.55), 0),
    "forearm.R": ((-0.70, 0.26, -0.66), 0),
    "hand.R": ((-0.55, 0.20, -0.80), 0),
}
FEET = {"ik_foot.L": (0.42, 0.10, 0.18), "ik_foot.R": (-0.36, 0.06, 0.27)}


def cave_pose(tilt=True):
    rig = bpy.data.objects[RIG]
    rig.rotation_mode = 'XYZ'
    rig.rotation_euler = (0, 0, 0)
    rig.location = (0, 0, 0)
    pose_reset(rig)
    for name in ("spine", "chest", "neck", "head", "clavicle.L", "upperarm.L", "forearm.L", "hand.L",
                 "clavicle.R", "upperarm.R", "forearm.R", "hand.R"):
        d, tw = POSE[name]
        _aim(rig, name, d, tw)
    for n, p in FEET.items():
        _place(rig, n, p)
    if tilt:
        rig.rotation_euler = (TILT, 0, 0)
    bpy.context.view_layer.update()
    return "ok"


def ai_power(on=True):
    for name, st in (("Relic_RingLight", 3.0), ("Relic_Glyph", 1.7)):
        m = bpy.data.materials[name]
        b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        b.inputs['Emission Strength'].default_value = st if on else 0.0


# ------------------------------------------------------------------ render helper
def shot(name, loc, target, lens=35.0, res=(1280, 720), samples=48, cam_name="Cave_Cam", fstop=None, focus=None):
    sc = bpy.context.scene
    cam = bpy.data.objects.get(cam_name)
    if cam is None:
        cd = bpy.data.cameras.new(cam_name)
        cam = bpy.data.objects.new(cam_name, cd)
        sc.collection.objects.link(cam)
    cam.data.lens = lens
    cam.data.clip_start = 0.05
    cam.data.clip_end = 200.0
    cam.location = Vector(loc)
    look_at(cam, target)
    if fstop:
        cam.data.dof.use_dof = True
        cam.data.dof.aperture_fstop = fstop
        cam.data.dof.focus_distance = (Vector(focus if focus else target) - cam.location).length
    else:
        cam.data.dof.use_dof = False
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.eevee.taa_render_samples = samples
    sc.render.image_settings.file_format = 'PNG'
    os.makedirs(os.path.join(OUT_DIR, "renders"), exist_ok=True)
    path = os.path.join(OUT_DIR, "renders", name + ".png")
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
