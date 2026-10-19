# Motion Assist pass on the cave opening (cut 4). Helpers for checking and fixing Mender's animation with the installed
# extension. Usage in one execute_blender_code call:  exec(open(r"C:\ai\unedited\ma_cave_pass.py").read())
import importlib
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

MA = "bl_ext.user_default.motion_assist"
SCENE = "Cave"
RIG = "MenderC_Rig"


def mod(name):
    return importlib.import_module(MA + "." + name)


def setup(mass=75.0, ignore="Pick,Puff,Pebble"):
    sc = bpy.data.scenes[SCENE]
    rig = bpy.data.objects[RIG]
    s = sc.motion_assist
    s.target = rig
    s.total_mass = mass
    s.ignore = ignore
    s.use_scene_range = True
    return sc, rig, s


def joint_state(rig, b):
    """where a limited joint is now: excess over its limit, twist, and the swing split along its two anatomical axes"""
    L = mod("limits")
    lim = L.rig_limits(rig)[b]
    B, P = rig.data.bones, rig.pose.bones
    rel_pose = P[lim.parent].matrix.to_3x3().normalized().inverted() @ P[b].matrix.to_3x3().normalized()
    rel_rest = B[lim.parent].matrix_local.to_3x3().inverted() @ B[b].matrix_local.to_3x3()
    q0 = (rel_rest.inverted() @ rel_pose).to_quaternion()
    out = {"excess": round(L.excess(q0, lim), 1)}
    q = lim.ref.inverted() @ q0 if lim.ref is not None else q0
    sw, tw = L.split(q)
    out["twist"] = round(math.degrees(L.twist_angle(tw)), 1)
    y2 = sw @ L.Y
    if lim.kind == "hinge":
        out["bend"] = round(math.degrees(L.hinge_angle(y2, lim.hinge)), 1)
        p = y2 - lim.hinge * y2.dot(lim.hinge)
        out["off_plane"] = round(math.degrees(math.acos(max(-1.0, min(1.0, p.length)))), 1) if p.length < 1.0 else 0.0
    else:
        theta = math.acos(max(-1.0, min(1.0, L.Y.dot(y2))))
        d = y2 - L.Y * y2.dot(L.Y)
        if d.length > 1e-9:
            d.normalize()
            out["a1"] = round(math.degrees(theta) * d.dot(lim.a1), 1)
            out["a2"] = round(math.degrees(theta) * d.dot(lim.a2), 1)
        else:
            out["a1"] = out["a2"] = 0.0
    return out


def table(rig, scene, frames, bones):
    cur = scene.frame_current
    rows = {}
    try:
        for f in frames:
            scene.frame_set(f)
            rows[f] = {b: joint_state(rig, b) for b in bones}
    finally:
        scene.frame_set(cur)
    return rows


def ranges(rig, bones):
    L = mod("limits")
    lims = L.rig_limits(rig)
    return {b: {"kind": lims[b].kind, "ranges": lims[b].ranges} for b in bones}


# ---------------------------------------------------------------------------------------------------- quick looks
def peek(frames, views, path, size=(560, 420), samples=24, only_mender=True, floor_z=None, light=3.0):
    """render the character at some frames from some viewpoints into one contact sheet (a throwaway scene: the Cave scene's
    settings stay as they are). views: [(location, target, lens)] in world space."""
    sc = bpy.data.scenes[SCENE]
    tmp = bpy.data.scenes.new("MA_peek")
    tmp.render.engine = sc.render.engine
    try:
        tmp.eevee.taa_render_samples = samples
    except Exception:
        pass
    tmp.render.resolution_x, tmp.render.resolution_y = size
    tmp.render.resolution_percentage = 100
    tmp.render.image_settings.file_format = 'PNG'
    tmp.render.use_motion_blur = False
    tmp.view_settings.view_transform = 'Standard'
    w = bpy.data.worlds.new("MA_peek_world")
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.35, 0.37, 0.4, 1.0)
    bg.inputs[1].default_value = 1.0
    tmp.world = w
    keep = {"Mender_Cave", "Relic_AI"} if only_mender else None
    for c in sc.collection.children:
        if keep is None or c.name in keep:
            tmp.collection.children.link(c)
    cam_data = bpy.data.cameras.new("MA_peek_cam")
    cam = bpy.data.objects.new("MA_peek_cam", cam_data)
    tmp.collection.objects.link(cam)
    tmp.camera = cam
    sun_d = bpy.data.lights.new("MA_peek_sun", 'SUN')
    sun_d.energy = light
    sun = bpy.data.objects.new("MA_peek_sun", sun_d)
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(30))
    tmp.collection.objects.link(sun)
    made = []
    if floor_z is not None:
        me = bpy.data.meshes.new("MA_peek_floor")
        me.from_pydata([(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)], [], [(0, 1, 2, 3)])
        fl = bpy.data.objects.new("MA_peek_floor", me)
        fl.location = (1.0, 0.85, floor_z)
        tmp.collection.objects.link(fl)
        made.append(fl)
    tiles = []
    import tempfile, os
    td = tempfile.mkdtemp()
    for f in frames:
        for vi, (loc, tgt, lens) in enumerate(views):
            tmp.frame_set(f)
            cam.location = loc
            d = Vector(tgt) - Vector(loc)
            cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
            cam_data.lens = lens
            cam_data.clip_start = 0.02
            p = os.path.join(td, "t_%d_%d.png" % (f, vi))
            tmp.render.filepath = p
            bpy.ops.render.render(write_still=True, scene=tmp.name)
            tiles.append(p)
    # tile into one image
    imgs = [bpy.data.images.load(p) for p in tiles]
    w_, h_ = size
    cols = len(views)
    rows_n = len(frames)
    out = np.zeros((h_ * rows_n, w_ * cols, 4), dtype=np.float32)
    for k, im in enumerate(imgs):
        a = np.array(im.pixels[:], dtype=np.float32).reshape(h_, w_, 4)
        r, c = divmod(k, cols)
        out[(rows_n - 1 - r) * h_:(rows_n - r) * h_, c * w_:(c + 1) * w_] = a
    sheet = bpy.data.images.new("MA_peek_sheet", w_ * cols, h_ * rows_n, alpha=True)
    sheet.pixels = out.ravel().tolist()
    sheet.filepath_raw = path
    sheet.file_format = 'PNG'
    sheet.save()
    for im in imgs:
        bpy.data.images.remove(im)
    bpy.data.images.remove(sheet)
    # clean up
    for c in list(tmp.collection.children):
        tmp.collection.children.unlink(c)
    for ob in [cam, sun] + made:
        bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.scenes.remove(tmp)
    bpy.data.worlds.remove(w)
    bpy.data.cameras.remove(cam_data)
    bpy.data.lights.remove(sun_d)
    for me in [m for m in bpy.data.meshes if m.name.startswith("MA_peek")]:
        bpy.data.meshes.remove(me)
    return path


def quick_frames(frames, outdir, res=(640, 360), samples=16, motion_blur=False):
    """render some frames of the Cave scene with its own cameras at a low resolution; the scene's settings are restored"""
    import os
    sc = bpy.data.scenes[SCENE]
    os.makedirs(outdir, exist_ok=True)
    keep = (sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage, sc.eevee.taa_render_samples,
            sc.render.filepath, sc.render.use_motion_blur, sc.frame_current, sc.render.image_settings.file_format)
    out = []
    try:
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = res[0], res[1], 100
        sc.eevee.taa_render_samples = samples
        sc.render.use_motion_blur = motion_blur
        sc.render.image_settings.file_format = 'PNG'
        for f in frames:
            sc.frame_set(f)
            p = os.path.join(outdir, "q_%04d.png" % f)
            sc.render.filepath = p
            bpy.ops.render.render(write_still=True, scene=sc.name)
            out.append(p)
    finally:
        (sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage, sc.eevee.taa_render_samples,
         sc.render.filepath, sc.render.use_motion_blur, cur, sc.render.image_settings.file_format) = keep
        sc.frame_set(cur)
    return out


def sheet(paths, out_path, cols=4, tile=(480, 270)):
    """tile images into one contact sheet"""
    imgs = [bpy.data.images.load(p) for p in paths]
    rows_n = (len(imgs) + cols - 1) // cols
    W, H = tile
    big = np.zeros((H * rows_n, W * cols, 4), dtype=np.float32)
    for k, im in enumerate(imgs):
        w, h = im.size
        a = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)
        # nearest-neighbour resize to the tile
        yi = (np.arange(H) * h / H).astype(int); xi = (np.arange(W) * w / W).astype(int)
        a = a[yi][:, xi]
        r, c = divmod(k, cols)
        big[(rows_n - 1 - r) * H:(rows_n - r) * H, c * W:(c + 1) * W] = a
    out = bpy.data.images.new("MA_sheet", W * cols, H * rows_n, alpha=True)
    out.pixels = big.ravel().tolist()
    out.filepath_raw = out_path
    out.file_format = 'PNG'
    out.save()
    for im in imgs:
        bpy.data.images.remove(im)
    bpy.data.images.remove(out)
    return out_path
