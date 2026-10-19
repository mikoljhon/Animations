# ma_demo_jump.py : a quick, deliberately naive forward hop for Mender (Mender_Studio scene), keyed the way a
# beginner would (a few poses, default Bezier), to test the Motion Assist add-on on. Exec in Blender; call build_jump().

import math

import bpy
from mathutils import Quaternion, Vector

RIGN = "Mender_Rig"
XAX = Vector((1.0, 0.0, 0.0))
F_END = 56

STAND = dict(hips=(0, 0, -0.01), tilt=0, spine=0, chest=0, neck=0, head=0, arm=4, fore=-12, feet=(0, 0, 0), heel=0)
KEYS = [
    (1, STAND),
    (6, STAND),
    (12, dict(hips=(0, 0.04, -0.22), tilt=10, spine=22, chest=10, neck=-12, head=-10, arm=50, fore=-18, feet=(0, 0, 0), heel=0)),
    (16, dict(hips=(0, -0.10, 0.035), tilt=6, spine=12, chest=4, neck=-6, head=-6, arm=-75, fore=-22, feet=(0, 0, 0), heel=35)),
    (21, dict(hips=(0, -0.46, 0.36), tilt=0, spine=6, chest=2, neck=0, head=-4, arm=-115, fore=-28, feet=(0, -0.38, 0.45), heel=10)),
    (26, dict(hips=(0, -0.87, -0.02), tilt=4, spine=10, chest=4, neck=-4, head=-4, arm=-60, fore=-22, feet=(0, -0.90, 0), heel=0)),
    (30, dict(hips=(0, -0.94, -0.21), tilt=12, spine=24, chest=10, neck=-12, head=-10, arm=-35, fore=-30, feet=(0, -0.90, 0), heel=0)),
    (38, dict(hips=(0, -0.91, -0.03), tilt=2, spine=6, chest=2, neck=-2, head=-2, arm=-8, fore=-14, feet=(0, -0.90, 0), heel=0)),
    (50, dict(STAND, hips=(0, -0.90, -0.01), feet=(0, -0.90, 0))),
]


def clear_rig(rig):
    rig.animation_data_clear()
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)


def bend(pb, deg, axis=XAX):
    """rotate a bone about a world axis (its parents only bend about the same axis, so the local axis is fixed)"""
    R = pb.bone.matrix_local.to_3x3()
    ax = (R.inverted() @ axis).normalized()
    pb.rotation_euler = Quaternion(ax, math.radians(deg)).to_euler(pb.rotation_mode)


def offset(pb, off):
    pb.location = pb.bone.matrix_local.to_3x3().inverted() @ Vector(off)


def foot(pb, off, heel):
    """IK foot: move by `off` and lift the heel by `heel` degrees, pivoting on the ball of the foot"""
    b = pb.bone
    R = b.matrix_local.to_3x3()
    q = Quaternion(XAX, math.radians(heel))
    new_head = b.tail_local + q @ (b.head_local - b.tail_local)
    pb.location = R.inverted() @ (new_head - b.head_local + Vector(off))
    pb.rotation_euler = Quaternion((R.inverted() @ XAX).normalized(), math.radians(heel)).to_euler(pb.rotation_mode)


def set_pose(rig, k):
    P = rig.pose.bones
    offset(P["pelvis"], k["hips"])
    bend(P["pelvis"], k["tilt"])
    for n in ("spine", "chest", "neck", "head"):
        bend(P[n], k[n])
    for s in ("L", "R"):
        bend(P["upperarm." + s], k["arm"])
        bend(P["forearm." + s], k["fore"])
        foot(P["ik_foot." + s], k["feet"], k["heel"])
        bend(P["toe." + s], -k["heel"])


KEYED = ["pelvis", "spine", "chest", "neck", "head", "upperarm.L", "upperarm.R", "forearm.L", "forearm.R",
         "ik_foot.L", "ik_foot.R", "toe.L", "toe.R"]


def build_jump():
    sc = bpy.data.scenes["Mender_Studio"]
    bpy.context.window_manager.windows[0].scene = sc
    rig = bpy.data.objects[RIGN]
    clear_rig(rig)
    sc.frame_start, sc.frame_end = 1, F_END
    for f, k in KEYS:
        set_pose(rig, k)
        for n in KEYED:
            pb = rig.pose.bones[n]
            pb.keyframe_insert("location", frame=f, group=n)
            pb.keyframe_insert("rotation_euler", frame=f, group=n)
    sc.frame_set(1)
    return {"keys": [f for f, _k in KEYS], "end": F_END}


def assign(rig, action):
    """switch the rig to `action`; bones the action does not key would keep stale values, so reset the pose first"""
    clear_pose(rig)
    rig.animation_data.action = action
    if rig.animation_data.action_slot is None and action.slots:
        rig.animation_data.action_slot = action.slots[0]


def clear_pose(rig):
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)


def side_camera(name="MA_SideCam", loc=(5.6, -0.45, 1.25), target=(0.0, -0.45, 0.82), lens=42.0):
    sc = bpy.data.scenes["Mender_Studio"]
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        sc.collection.objects.link(cam)
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = d.to_track_quat('-Z', 'Y')
    cam.data.lens = lens
    sc.camera = cam
    return cam.name
