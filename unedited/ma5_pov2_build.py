# First-person cut of the opening, version 2: the camera follows Relic's head, then his head turns to keep Mender in view after the head lift;
# an Iron Man style HUD with text (boot readouts, tracking brackets and labels on Mender, status lines that react to the story).
#   blender.exe -b relic_v15_opening_cut6_pass.blend --python ma5_pov2_build.py   -> relic_v17_pov_hud.blend
import math
import time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Quaternion, Vector

T0 = time.time()
sc = bpy.data.scenes["Cave"]
rig = bpy.data.objects["Relic_AI_Rig"]
men = bpy.data.objects["MenderC_Rig"]
pb = rig.pose.bones["head"]
Mw = rig.matrix_world
F1 = 360
C1, C2, C3 = 42, 91, 146
SLOW = 1.35


def G(f_old):
    return C3 + (f_old - 58.0) * SLOW


f_eyes = int(round(G(99)))          # first eye flicker
f_on = f_eyes + 7                   # vision fully on
f_lift = int(round(G(104)))         # the head lift starts
f_drop = int(round(G(106)))         # Mender drops the pick

R = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_4x4()
P = Matrix.Translation(Vector((0.0, -0.274, 1.862)))

# ---------------------------------------------------------------- camera
cd = bpy.data.cameras.new("POV")
cd.lens = 22.0
cd.clip_start = 0.005
cd.clip_end = 60.0
cam = bpy.data.objects.new("POV", cd)
sc.collection.objects.link(cam)
sc.timeline_markers.clear()
sc.camera = cam
cam.rotation_mode = "QUATERNION"


def sod_run(fn, hz, z, n):
    k1, k2 = z / (math.pi * hz), 1.0 / ((2 * math.pi * hz) ** 2)
    y, yd, xp = fn(1.0), 0.0, fn(1.0)
    out = []
    sub = 8
    T = 1.0 / 24.0 / sub
    for i in range(n):
        f = 1 + i
        if i > 0:
            for s in range(sub):
                x = fn(f - 1 + (s + 1) / sub)
                y = y + T * yd
                yd = yd + T * (x - y - k1 * yd) / k2
        out.append(y)
    return np.array(out)


w_look = sod_run(lambda f: 0.92 if f >= f_lift else 0.0, 1.9, 0.5, F1)      # how much of the turn toward Mender the head has made
rng = np.random.default_rng(5)
impacts = [(C1, 0.55), (C2, 0.65), (C3, 2.0), (f_drop, 0.5)]
ph = rng.uniform(0, 6.28, (len(impacts), 3))


def shake_rot(f):
    e = np.zeros(3)
    for k, (fk, amp) in enumerate(impacts):
        if f >= fk:
            t = (f - fk) / 24.0
            dec = math.exp(-t / 0.28)
            for ax in range(3):
                e[ax] += amp * dec * math.sin(2 * math.pi * (11.0 + 3 * ax) * t + ph[k, ax])
    t = (f - 1) / 24.0
    e += 0.12 * np.array([math.sin(2 * math.pi * 0.23 * t), math.sin(2 * math.pi * 0.31 * t + 1.0), math.sin(2 * math.pi * 0.17 * t + 2.0)])
    return Euler([math.radians(x) for x in e], "XYZ").to_matrix().to_4x4()


tgt_s = None
cam_mats = {}
for f in range(1, F1 + 1):
    sc.frame_set(f)
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    M = Mw @ delta @ P @ R
    mp = men.matrix_world @ men.pose.bones["head"].head
    tgt_s = mp.copy() if tgt_s is None else tgt_s.lerp(mp, 0.22)            # his gaze trails Mender a little
    d = (tgt_s - M.translation).normalized()
    fwd = (M.to_3x3() @ Vector((0, 0, -1))).normalized()
    qd = fwd.rotation_difference(d)
    qw = Quaternion((1, 0, 0, 0)).slerp(qd, min(max(float(w_look[f - 1]), 0.0), 1.0))
    M2 = Matrix.Translation(M.translation) @ qw.to_matrix().to_4x4() @ M.to_3x3().to_4x4()
    M2 = M2 @ shake_rot(f)
    cam_mats[f] = M2
    loc, q, s = M2.decompose()
    cam.location, cam.rotation_quaternion = loc, q
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
for fc in cam.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# ---------------------------------------------------------------- vision power
power = [(1, 0.035), (C1 - 1, 0.035), (C1, 0.10), (C1 + 3, 0.035), (C2 - 1, 0.035), (C2, 0.12), (C2 + 3, 0.04),
         (C3 - 1, 0.04), (C3, 0.5), (C3 + 2, 0.06), (C3 + 8, 0.05),
         (int(round(G(74))), 0.2), (int(round(G(76))), 0.04), (int(round(G(78))), 0.25), (int(round(G(79))), 0.03), (int(round(G(82))), 0.3),
         (int(round(G(85))), 0.08), (int(round(G(87))), 0.12),
         (f_eyes - 1, 0.12), (f_eyes, 0.7), (f_eyes + 2, 0.02), (f_eyes + 4, 0.9), (f_eyes + 5, 0.15), (f_on, 1.0), (F1, 1.0)]
last = 0
for f, p in power:
    f = max(f, last + 1)
    last = f
    sc.view_settings.exposure = math.log2(max(p, 0.004))
    sc.view_settings.keyframe_insert("exposure", frame=f)
for fc in sc.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
    if fc.data_path == "view_settings.exposure":
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if C1 - 2 < kp.co[0] < f_eyes - 1 else "LINEAR"

# ---------------------------------------------------------------- HUD
mat = bpy.data.materials.new("POV_HUD")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1.0, 0.5, 0.1, 1.0)
em.inputs["Strength"].default_value = 3.2
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0], out.inputs[0])
Z = -0.05          # right in front of the lens so no rock can cover the HUD; everything under HUD_root is scaled up from 0.4 m units
HS = 0.125
HW, HH = 0.327, 0.184          # half width and height of the view at the HUD plane (22 mm lens, 36 mm sensor, 16:9)
hud_all = []


def blink(ob, frames_vis):
    """frames_vis: [(frame, visible)] stepped"""
    for f, v in frames_vis:
        ob.hide_render = not v
        ob.keyframe_insert("hide_render", frame=f)
    for fc in ob.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def rect(name, cx, cy, w, h, parent):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(cx - w / 2, cy - h / 2, 0), (cx + w / 2, cy - h / 2, 0), (cx + w / 2, cy + h / 2, 0), (cx - w / 2, cy + h / 2, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    ob.parent = parent
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob


def text(body, x, y, parent, size=0.0125, align="LEFT"):
    cu = bpy.data.curves.new("HUDT", "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.extrude = 0.0
    ob = bpy.data.objects.new("HUDT_" + body[:10], cu)
    sc.collection.objects.link(ob)
    ob.parent = parent
    ob.location = (x, y, 0.0)
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob


root = bpy.data.objects.new("HUD_root", None)
sc.collection.objects.link(root)
root.parent = cam
root.location = (0.0, 0.0, Z)
root.scale = (HS, HS, HS)

# frame brackets and a centre reticle
X, Y, L, T = 0.27, 0.145, 0.05, 0.0012
for sx in (-1, 1):
    for sy in (-1, 1):
        hud_all.append(rect("HUD_h", sx * (X - L / 2 + T / 2), sy * Y, L, T, root))
        hud_all.append(rect("HUD_v", sx * X, sy * (Y - L / 2 + T / 2), T, L, root))
for k in range(48):
    am = 2 * math.pi * (k + 0.27) / 48
    ob = rect("HUD_r", 0.0, 0.0, 0.0026, T, root)
    ob.location = (0.018 * math.cos(am), 0.018 * math.sin(am), 0.0)
    ob.rotation_euler = (0, 0, am)
    hud_all.append(ob)
for ob in hud_all:
    blink(ob, [(1, False), (f_on, True)])

# boot readouts, top left, typed on one line at a time with a flicker
lines = [("OPTICS", "ONLINE"), ("NEURAL LINK", "UNSTABLE"), ("MOTOR CONTROL", "4 %"), ("POWER", "11 %"), ("LOCATION", "UNKNOWN"), ("DATE", "NO DATA")]
y0 = 0.128
for k, (a, b) in enumerate(lines):
    fk = f_on + 3 + 5 * k
    ob = text("%s  :  %s" % (a, b), -0.24, y0 - 0.0155 * k, root)
    blink(ob, [(1, False), (fk, True), (fk + 1, False), (fk + 2, True)])
    hud_all.append(ob)
# motor control climbs as he wakes up
for (fa, fb, s) in ((f_on + 28, f_on + 85, "MOTOR CONTROL  :  27 %"), (f_on + 85, F1 + 1, "MOTOR CONTROL  :  61 %")):
    ob = text(s, -0.24, y0 - 0.0155 * 2, root)
    blink(ob, [(1, False), (fa, True), (fb, False)])
f_m1 = f_on + 28
old = [o for o in bpy.data.objects if o.name.startswith("HUDT_MOTOR CON") and o.data.body.endswith("4 %")]
for o in old:
    blink(o, [(1, False), (f_on + 3 + 10, True), (f_on + 3 + 11, False), (f_on + 3 + 12, True), (f_m1, False)])

# a status line, bottom left, that reacts to the story
status = [(f_on + 6, f_on + 40, "TARGET  :  UNKNOWN ORGANIC"), (f_on + 40, f_on + 92, "TARGET  :  TOOL DROPPED"), (f_on + 92, F1 + 1, "TARGET  :  NO THREAT DETECTED")]
for fa, fb, s in status:
    ob = text(s, -0.24, -0.13, root)
    blink(ob, [(1, False), (fa, True), (fb, False)])

# tracking brackets and labels that follow Mender
track = bpy.data.objects.new("HUD_track", None)
sc.collection.objects.link(track)
track.parent = root
track_items = []
bw, bh, bl, bt = 0.07, 0.10, 0.02, 0.0012
for sx in (-1, 1):
    for sy in (-1, 1):
        rect("TRK_h", sx * (bw / 2 - bl / 2), sy * bh / 2, bl, bt, track)
        rect("TRK_v", sx * bw / 2, sy * (bh / 2 - bl / 2), bt, bl, track)
labels = [(f_on + 8, f_lift + 30, "UNIDENTIFIED"), (f_lift + 30, F1 + 1, "HUMAN")]
for fa, fb, s in labels:
    ob = text(s, bw / 2 + 0.006, bh / 2 - 0.009, track, size=0.0115)
    track_items.append((ob, fa, fb))
for fa, fb, s in ((f_lift + 10, F1 + 1, "PULSE  134"), ):
    ob = text(s, bw / 2 + 0.006, bh / 2 - 0.025, track, size=0.0105)
    track_items.append((ob, fa, fb))
for fa, fb, s in ((f_on + 14, F1 + 1, "HANDS  :  EMPTY"), ):
    ob = text(s, bw / 2 + 0.006, bh / 2 - 0.041, track, size=0.0105)
    track_items.append((ob, fa, fb))
# the tracker's position and size per frame from Mender's head and hips
vis = []
for f in range(1, F1 + 1):
    sc.frame_set(f)
    bpy.context.view_layer.update()
    ph_ = men.matrix_world @ men.pose.bones["head"].head
    pp_ = men.matrix_world @ men.pose.bones["pelvis"].head
    a = world_to_camera_view(sc, cam, ph_)
    b = world_to_camera_view(sc, cam, pp_)
    cx, cy = (a.x + b.x) / 2, (a.y + b.y) / 2
    on = f >= f_on + 8 and 0.05 < a.x < 0.95 and 0.08 < a.y < 0.95 and a.z > 0.2
    size_y = max(abs(a.y - b.y) * 1.5, 0.28)
    track.location = ((cx - 0.5) * 2 * HW, (cy - 0.5) * 2 * HH, 0.0)
    s = min(max(size_y * 2 * HH / bh, 0.5), 3.0)
    track.scale = (s, s, 1.0)
    track.hide_render = not on
    track.keyframe_insert("location", frame=f)
    track.keyframe_insert("scale", frame=f)
    track.keyframe_insert("hide_render", frame=f)
    for ch in [o for o in track.children if o.name.startswith("TRK_")]:
        ch.hide_render = not on
        ch.keyframe_insert("hide_render", frame=f)
    for ob, fa, fb in track_items:
        ob.hide_render = not (on and fa <= f < fb)
        ob.keyframe_insert("hide_render", frame=f)
    vis.append(on)
for fc in track.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "CONSTANT" if fc.data_path == "hide_render" else "LINEAR"

for ch in track.children:
    if ch.animation_data and ch.animation_data.action:
        for fc in ch.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
print("tracker on frames", [i + 1 for i, v in enumerate(vis) if v][:3], "off after f_on+8:", [i + 1 for i, v in enumerate(vis) if (not v) and i + 1 >= f_on + 8][:40])
sc.frame_set(1)
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.resolution_percentage = 100
sc.eevee.taa_render_samples = 16
sc.render.filepath = r"C:\ai\unedited\renders\pov2_anim\f_"
sc.frame_start, sc.frame_end = 1, F1
bpy.ops.wm.save_as_mainfile(filepath=r"C:\ai\unedited\relic_v17_pov_hud.blend")
print("POV2 built", round(time.time() - T0, 1), "s", "eyes", f_eyes, "on", f_on, "lift", f_lift, "drop", f_drop)
