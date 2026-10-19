# Motion Assist: real objects that show the centre of mass in renders (a tube along its path and a ball riding it).

import bmesh
import bpy

COLL = "Motion Assist"


def _collection(scene):
    coll = bpy.data.collections.get(COLL)
    if coll is None:
        coll = bpy.data.collections.new(COLL)
    if scene.collection.children.get(coll.name) is None:
        scene.collection.children.link(coll)
    return coll


def _material(name, color, strength=2.5):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (*color, 1.0)
    return mat


def _remove(name):
    ob = bpy.data.objects.get(name)
    if ob is not None:
        data = ob.data
        act = ob.animation_data.action if ob.animation_data else None
        bpy.data.objects.remove(ob, do_unlink=True)
        if act is not None and act.users == 0:
            bpy.data.actions.remove(act)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)
            elif isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)


def make_trail(scene, rig, com, f0, label="path", color=(1.0, 0.55, 0.1), radius=0.006, ball=True):
    base = "MA_%s_COM_%s" % (rig.name, label)
    _remove(base)
    _remove(base + "_Ball")
    coll = _collection(scene)
    mat = _material(base, color)
    cu = bpy.data.curves.new(base, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    sp = cu.splines.new('POLY')
    sp.points.add(len(com) - 1)
    for i, p in enumerate(com):
        sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
    cu.materials.append(mat)
    ob = bpy.data.objects.new(base, cu)
    coll.objects.link(ob)
    out = [ob.name]
    if ball:
        me = bpy.data.meshes.new(base + "_Ball")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=radius * 5)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        bo = bpy.data.objects.new(base + "_Ball", me)
        coll.objects.link(bo)
        for i, p in enumerate(com):
            bo.location = tuple(float(x) for x in p)
            bo.keyframe_insert("location", frame=f0 + i)
        out.append(bo.name)
    return out
