# ma_demo_autopose.py : AutoPose test poses for Mender (Mender_Studio scene). Each pose is given only by goals for the
# hips, wrists and ankles (the network decides the spine, neck, head, shoulders, elbows, knees and feet). Rendered twice:
# plain IK (hands and feet reach, the rest stays stiff) and AutoPose. Exec after ma_demo_jump.py; call render_all().

import importlib
import os

import bmesh
import bpy
from mathutils import Vector


def _motion_assist(module):
    """the installed Motion Assist extension (or a development copy on sys.path)"""
    for base in ("bl_ext.user_default.motion_assist", "motion_assist"):
        try:
            return importlib.import_module(base + "." + module)
        except ImportError:
            pass
    raise ImportError("Motion Assist is not installed")

POSES = {
    "reach up": {"hips": (0.0, 0.04, 0.80), "wrist_L": (0.20, -0.22, 1.60), "wrist_R": (-0.18, -0.25, 1.62),
                 "ankle_L": (0.13, 0.02, 0.10), "ankle_R": (-0.13, -0.05, 0.10)},
    "lift a box": {"hips": (0.0, 0.20, 0.50), "wrist_L": (0.19, -0.34, 0.30), "wrist_R": (-0.19, -0.34, 0.30),
                   "ankle_L": (0.17, 0.02, 0.10), "ankle_R": (-0.17, 0.02, 0.10)},
    "lunge punch": {"hips": (0.0, -0.10, 0.70), "wrist_L": (0.13, -0.24, 1.15), "wrist_R": (-0.08, -0.80, 1.15),
                    "ankle_L": (0.13, -0.45, 0.10), "ankle_R": (-0.13, 0.42, 0.10)},
    "climb": {"hips": (0.0, 0.06, 0.86), "wrist_L": (0.24, -0.30, 1.66), "wrist_R": (-0.22, -0.30, 1.30),
              "ankle_L": (0.15, -0.26, 0.48), "ankle_R": (-0.12, 0.06, 0.10)},
}
OUT = r"C:\ai\unedited\renders\ma_autopose"


SWEEP_KEYS = [  # frame: right wrist goal, hips goal (the left hand, neck and feet are left to the network or planted)
    (1, (-0.37, -0.02, 0.78), (0.0, 0.0, 0.835)),
    (24, (-0.28, -0.58, 1.40), (0.0, -0.03, 0.82)),
    (44, (-0.12, -0.18, 1.70), (0.0, 0.0, 0.84)),
    (64, (-0.78, -0.12, 1.12), (-0.06, 0.0, 0.80)),
    (88, (-0.22, -0.58, 0.26), (0.0, 0.18, 0.52)),
    (112, (-0.37, -0.02, 0.78), (0.0, 0.0, 0.835)),
]
SWEEP_END = 120
FEET = {"ankle_L": (0.11, 0.0135, 0.099), "ankle_R": (-0.11, 0.0135, 0.099)}


def _sweep_at(f):
    """smoothly interpolated goals for frame f (Catmull-Rom through the keys)"""
    ks = SWEEP_KEYS
    if f <= ks[0][0]:
        return Vector(ks[0][1]), Vector(ks[0][2])
    if f >= ks[-1][0]:
        return Vector(ks[-1][1]), Vector(ks[-1][2])
    i = max(k for k in range(len(ks) - 1) if ks[k][0] <= f)
    t = (f - ks[i][0]) / (ks[i + 1][0] - ks[i][0])
    t = t * t * (3 - 2 * t)
    out = []
    for c in (1, 2):
        p0 = Vector(ks[max(i - 1, 0)][c])
        p1, p2 = Vector(ks[i][c]), Vector(ks[i + 1][c])
        p3 = Vector(ks[min(i + 2, len(ks) - 1)][c])
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return out[0], out[1]


def sweep_action(name, use_net, auto_hips=False):
    """AutoPose (or plain IK) on every frame while the right-hand goal sweeps around; keyed into a new action.
    auto_hips: the feet-anchored network places the hips itself (no hips goal)"""
    autopose = _motion_assist("autopose")
    sc = bpy.data.scenes["Mender_Studio"]
    rig = bpy.data.objects["Mender_Rig"]
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = None
    clear_pose(rig)
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    rig.animation_data.action = act
    for f in range(1, SWEEP_END + 1):
        wr, hips = _sweep_at(f)
        goals = {"wrist_R": wr, **{k: Vector(v) for k, v in FEET.items()}}
        if not auto_hips:
            goals["hips"] = hips
        sc.frame_set(f)
        autopose.solve(rig, goals, keep=(), use_net=use_net, key=True, frame=f, auto_hips=auto_hips)
    if rig.animation_data.action_slot is None and act.slots:
        rig.animation_data.action_slot = act.slots[0]
    return act.name


def sweep_markers():
    """a blue ball that follows the hand goal, and the planted feet goals"""
    goal_markers({"wrist_R": (0, 0, 0), **FEET})
    ob = bpy.data.objects["MA_Test_wrist_R"]
    for f in range(1, SWEEP_END + 1):
        ob.location = _sweep_at(f)[0]
        ob.keyframe_insert("location", frame=f)


def goal_markers(goals, color=(0.1, 0.6, 1.0)):
    sc = bpy.context.scene
    mat = bpy.data.materials.get("MA_TestGoal")
    if mat is None:
        mat = bpy.data.materials.new("MA_TestGoal")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Strength"].default_value = 2.0
    for ob in [o for o in bpy.data.objects if o.name.startswith("MA_Test_")]:
        bpy.data.objects.remove(ob, do_unlink=True)
    for j, p in goals.items():
        me = bpy.data.meshes.new("MA_Test_" + j)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.042)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        ob = bpy.data.objects.new("MA_Test_" + j, me)
        sc.collection.objects.link(ob)
        ob.location = p


def render_all(cam_loc=(2.5, -3.3, 1.35), target=(0.0, -0.12, 0.92)):
    autopose = _motion_assist("autopose")
    sc = bpy.data.scenes["Mender_Studio"]
    bpy.context.window_manager.windows[0].scene = sc
    rig = bpy.data.objects["Mender_Rig"]
    if rig.animation_data:
        rig.animation_data.action = None
    side_camera("MA_FrontCam", loc=cam_loc, target=target, lens=40.0)
    for n in ("MA_Mender_Rig_COM_before", "MA_Mender_Rig_COM_path", "MA_Mender_Rig_COM_path_Ball"):
        if bpy.data.objects.get(n):
            bpy.data.objects[n].hide_render = True
    os.makedirs(OUT, exist_ok=True)
    sc.render.resolution_x, sc.render.resolution_y = 960, 1080
    files = []
    for k, (name, goals) in enumerate(POSES.items()):
        goal_markers(goals)
        for mode, use_net in (("ik", False), ("ai", True)):
            clear_pose(rig)
            bpy.context.view_layer.update()
            autopose.solve(rig, {j: Vector(p) for j, p in goals.items()}, keep=(), use_net=use_net)
            bpy.context.view_layer.update()
            sc.render.filepath = os.path.join(OUT, "%d_%s.png" % (k, mode))
            bpy.ops.render.render(write_still=True)
            files.append(sc.render.filepath)
    clear_pose(rig)
    return files
