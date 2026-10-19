# relic_fists.py : give Relic's open hands a two-segment curl ("fist1/fist2") so he can make fists.   exec'd (needs bpy, math, Quaternion)
# the hand mesh is one rigid blob per hand; its vertices past the knuckle line are re-weighted from hand.X to fist1.X then fist2.X
import bpy, math
from mathutils import Vector, Quaternion


def _sm(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


KN1, KN2 = 0.085, 0.13           # knuckle and mid-finger joint lines along the hand bone (metres from the wrist; measured from the mesh:
                                 # the fingers start at 8.5-9.5 cm, the tips at 17.5 cm)
Z1, Z2 = (0.05, 0.07), (0.105, 0.06)       # blend bands (start, width). Measured with fist_spike.py on a closed fist: these give 0 stretched edges
                                           # (narrow 2 cm bands: 104-147, because the AI hand is not cut into finger segments)


def add_fists(rig, mesh_names=("Relic_AI", "Relic_AI_Crust")):
    if 'fist1.L' in rig.data.bones:
        return False
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    for s in 'LR':
        h = eb['hand.' + s]
        y = (h.tail - h.head).normalized()
        b1 = eb.new('fist1.' + s)
        b1.head, b1.tail, b1.roll, b1.parent = h.head + y * KN1, h.head + y * KN2, h.roll, h
        b2 = eb.new('fist2.' + s)
        b2.head, b2.tail, b2.roll, b2.parent = h.head + y * KN2, h.head + y * 0.18, h.roll, b1
    bpy.ops.object.mode_set(mode='OBJECT')
    for nm in mesh_names:
        o = bpy.data.objects.get(nm)
        if o is None:
            continue
        gi = {g.name: g.index for g in o.vertex_groups}
        for s in 'LR':
            inv = rig.data.bones['hand.' + s].matrix_local.inverted()
            gh = o.vertex_groups['hand.' + s]
            g1 = o.vertex_groups.get('fist1.' + s) or o.vertex_groups.new(name='fist1.' + s)
            g2 = o.vertex_groups.get('fist2.' + s) or o.vertex_groups.new(name='fist2.' + s)
            for v in o.data.vertices:
                w = 0.0
                for g in v.groups:
                    if g.group == gh.index:
                        w = g.weight
                        break
                if w <= 0.0:
                    continue
                ly = (inv @ v.co).y
                a1 = _sm((ly - Z1[0]) / Z1[1])
                if a1 <= 0.0:
                    continue
                a2 = _sm((ly - Z2[0]) / Z2[1])
                gh.add([v.index], w * (1.0 - a1), 'REPLACE')
                g1.add([v.index], w * a1 * (1.0 - a2), 'REPLACE')
                if a2 > 0.0:
                    g2.add([v.index], w * a1 * a2, 'REPLACE')
    return True


# the fingers' flexion axis. Relic's fingers lie side by side along the hand bone's local Z, so they fold about local Z, toward the palm:
# +X on the right hand, -X on the (mirrored) left hand. (Until 2026-10-12 this curled about local X, which swung the fingers SIDEWAYS
# into each other: a smeared half-fist.)
FLEX = {'L': (Vector((0.0, 0.0, 1.0)), 1.0), 'R': (Vector((0.0, 0.0, 1.0)), -1.0)}


def set_fists(rig, curl, key=False, frame=None, c1=1.3, c2=1.45):
    """curl = (left, right) 0 open .. 1 closed"""
    for s, c in zip('LR', curl):
        axis, sgn = FLEX[s]
        for n, a in (('fist1.' + s, c1 * c), ('fist2.' + s, c2 * c)):
            pb = rig.pose.bones.get(n)
            if pb is None:
                continue
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion(axis, sgn * a)
            if key:
                pb.keyframe_insert('rotation_quaternion', frame=frame)
