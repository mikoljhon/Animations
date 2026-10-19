# First-person cut of the opening (test): camera at Relic's eyes, vision powers up when his eyes light, shake on the strikes, a light HUD.
#   blender.exe -b relic_v15_opening_cut6_pass.blend --python ma5_pov_build.py   -> relic_v16_pov_test.blend
import math
import time

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

T0 = time.time()
sc = bpy.data.scenes["Cave"]
rig = bpy.data.objects["Relic_AI_Rig"]
pb = rig.pose.bones["head"]
Mw = rig.matrix_world
F1 = 360
C1, C2, C3 = 42, 91, 146
SLOW = 1.35


def G(f_old):
    return C3 + (f_old - 58.0) * SLOW


R = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_4x4()
P = Matrix.Translation(Vector((0.0, -0.274, 1.862)))

cd = bpy.data.cameras.new("POV")
cd.lens = 22.0
cd.clip_start = 0.005
cd.clip_end = 60.0
cam = bpy.data.objects.new("POV", cd)
sc.collection.objects.link(cam)
sc.timeline_markers.clear()
sc.camera = cam
cam.rotation_mode = "QUATERNION"

# --- shake: a decaying tremor after each strike plus a slow breathing sway
rng = np.random.default_rng(5)
impacts = [(C1, 0.55), (C2, 0.65), (C3, 2.0), (int(G(106)), 0.5)]
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


for f in range(1, F1 + 1):
    sc.frame_set(f)
    delta = pb.matrix @ pb.bone.matrix_local.inverted()
    M = Mw @ delta @ P @ R
    M = M @ shake_rot(f)
    loc, q, s = M.decompose()
    cam.location, cam.rotation_quaternion = loc, q
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
for fc in cam.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# --- vision power through the exposure: dormant and dim, a shock flash at the big strike, the ring flicker, then the eyes light and it comes on
f_eyes = int(round(G(99)))           # the eyes' first flicker in the animation (193)
power = [(1, 0.035), (C1 - 1, 0.035), (C1, 0.10), (C1 + 3, 0.035), (C2 - 1, 0.035), (C2, 0.12), (C2 + 3, 0.04),
         (C3 - 1, 0.04), (C3, 0.5), (C3 + 2, 0.06), (C3 + 8, 0.05),
         (int(round(G(74))), 0.2), (int(round(G(76))), 0.04), (int(round(G(78))), 0.25), (int(round(G(79))), 0.03), (int(round(G(82))), 0.3),
         (int(round(G(85))), 0.08), (int(round(G(87))), 0.12),
         (f_eyes - 1, 0.12), (f_eyes, 0.7), (f_eyes + 2, 0.02), (f_eyes + 4, 0.9), (f_eyes + 5, 0.15), (f_eyes + 7, 1.0), (F1, 1.0)]
sc.view_settings.exposure = 0.0
last = 0
for f, p in power:
    f = max(f, last + 1)
    last = f
    sc.view_settings.exposure = math.log2(max(p, 0.004))
    sc.view_settings.keyframe_insert("exposure", frame=f)
    ac = sc.animation_data.action
for fc in sc.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
    if fc.data_path == "view_settings.exposure":
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if kp.co[0] < f_eyes - 1 and kp.co[0] > C1 - 2 else "LINEAR"

# --- HUD: four corner brackets, a centre reticle and a small bar, orange like the eye glyphs, parented to the camera, on from the moment the vision comes up
mat = bpy.data.materials.new("POV_HUD")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1.0, 0.45, 0.08, 1.0)
em.inputs["Strength"].default_value = 2.2
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0], out.inputs[0])


def rect(name, cx, cy, w, h, z=-0.4):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(cx - w / 2, cy - h / 2, z), (cx + w / 2, cy - h / 2, z), (cx + w / 2, cy + h / 2, z), (cx - w / 2, cy + h / 2, z)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    ob.parent = cam
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob


hud = []
X, Y, L, T = 0.27, 0.145, 0.05, 0.0012
for sx in (-1, 1):
    for sy in (-1, 1):
        hud.append(rect("HUD_h", sx * (X - L / 2 + T / 2), sy * Y, L, T))
        hud.append(rect("HUD_v", sx * X, sy * (Y - L / 2 + T / 2), T, L))
ring_pts = []
for k in range(48):
    a0, a1 = 2 * math.pi * k / 48, 2 * math.pi * (k + 0.55) / 48
    ring_pts.append((a0, a1))
for a0, a1 in ring_pts:
    am = (a0 + a1) / 2
    ob = rect("HUD_r", 0.018 * math.cos(am), 0.018 * math.sin(am), 0.0026, T)
    ob.rotation_euler = (0, 0, am)
    ob.location = (0, 0, 0)
    ob.rotation_euler = (0, 0, 0)
    hud.append(ob)
hud.append(rect("HUD_bar", -0.27, 0.0, T, 0.08))
hud.append(rect("HUD_bar2", 0.27, 0.0, T, 0.08))
for ob in hud:
    ob.hide_render = True
    ob.keyframe_insert("hide_render", frame=1)
    ob.hide_render = False
    ob.keyframe_insert("hide_render", frame=f_eyes + 7)
    for fc in ob.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"

sc.frame_set(1)
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.resolution_percentage = 100
sc.eevee.taa_render_samples = 16
sc.render.filepath = r"C:\ai\unedited\renders\pov_anim\f_"
sc.frame_start, sc.frame_end = 1, F1
bpy.ops.wm.save_as_mainfile(filepath=r"C:\ai\unedited\relic_v16_pov_test.blend")
print("POV built", round(time.time() - T0, 1), "s, eyes at", f_eyes)
