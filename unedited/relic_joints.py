# relic_joints.py : cover the gaps between Relic's armor plates at the limb joints.
#   1. dark mechanical joint pieces (a sphere at each ball joint, a short cylinder across each hinge) parented to the bone, sized from the mesh
#   2. the inside of every plate renders dark (a gap reads as a crevice, never as see-through)
#   blender.exe -b fight_v04.blend --python relic_joints.py    -> fight_v05.blend
#   (Relic's mesh is a hollow shell of plates: when a thigh turns 60-80 deg its top plate swings away from the waist plate and the sky
#    showed through the body - "the leg disconnects from the body". The joint balls fill that opening with a mechanical joint.)
import bpy, bmesh, math
import numpy as np
from mathutils import Vector, Matrix
OUT = r"C:\ai\unedited"
rig = bpy.data.objects["Relic_AI_Rig"]
body = bpy.data.objects["Relic_AI"]
B = rig.data.bones
# joint material: dark gunmetal, a little worn
jm = bpy.data.materials.get("RelicJoint") or bpy.data.materials.new("RelicJoint")
jm.use_nodes = True
bsdf = jm.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.045, 0.043, 0.04, 1.0)
bsdf.inputs["Metallic"].default_value = 0.85
bsdf.inputs["Roughness"].default_value = 0.45
# how thick is the limb at each joint: distance from the joint to the nearest mesh points around it
co = np.array([body.matrix_world @ v.co for v in body.data.vertices]) if False else None
me = body.data
pts = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", pts); pts = pts.reshape(-1, 3)   # armature space (body parented with identity)
gidx = {g.name: g.index for g in body.vertex_groups}
vg_of = np.full(len(me.vertices), -1)
vg_w = np.zeros(len(me.vertices))
for v in me.vertices:
    for g in v.groups:
        if g.weight > vg_w[v.index]:
            vg_w[v.index] = g.weight; vg_of[v.index] = g.group
def radius_at(p, along, groups=None):
    """how big a ball fits INSIDE the armour shell around joint p: the closest surrounding plates (8th percentile of the distances of
    the vertices in a slab across the joint), so the ball never pokes out through the armour"""
    sel = np.ones(len(pts), dtype=bool) if not groups else np.isin(vg_of, [gidx[g] for g in groups if g in gidx])
    d = pts[sel] - np.array(p)
    ax = np.array(along) / (np.linalg.norm(along) + 1e-9)
    t = d @ ax
    near = np.abs(t) < 0.03
    if near.sum() < 20:
        return 0.05
    r = np.linalg.norm(d[near] - np.outer(t[near], ax), axis=1)
    return float(np.percentile(r, 8))
coll = bpy.data.collections.get("Relic_Joints") or bpy.data.collections.new("Relic_Joints")
if coll.name not in [c.name for c in bpy.context.scene.collection.children]:
    try:
        rig.users_collection[0].children.link(coll)
    except Exception:
        bpy.context.scene.collection.children.link(coll)
for o in list(coll.objects):
    bpy.data.objects.remove(o, do_unlink=True)
JOINTS = [("upperarm", "ball"), ("forearm", "hinge"), ("hand", "ball"), ("thigh", "ball"), ("shin", "hinge"), ("foot", "ball")]
RMAX = {"upperarm": 0.10, "thigh": 0.11}          # the big ball joints; everything else up to 9 cm (a 13 cm hip ball bulged out of the armour)
made = []
for base, kind in JOINTS:
    for sd in "LR":
        bn = "%s.%s" % (base, sd)
        b = B[bn]
        head = b.head_local
        along = (b.tail_local - b.head_local).normalized()
        par = b.parent.name if b.parent else bn
        r = max(0.03, min(RMAX.get(base, 0.09), radius_at(head, along, [bn, par]) * 0.92))
        bm = bmesh.new()
        if kind == "ball":
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r)
        else:
            # a hinge drum lying across the bend (along the bone's X axis), a bit narrower than the limb
            bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r * 0.95, radius2=r * 0.95, depth=r * 1.5)
            bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'Y'), verts=bm.verts)
        m = bpy.data.meshes.new("RelicJoint_" + bn)
        bm.to_mesh(m); bm.free()
        for poly in m.polygons:
            poly.use_smooth = True
        o = bpy.data.objects.new("RelicJoint_" + bn, m)
        coll.objects.link(o)
        o.data.materials.append(jm)
        o.parent = rig
        o.parent_type = 'BONE'
        o.parent_bone = bn
        # bone parenting puts the child at the bone's TAIL: move it back to the head, in the bone's own frame
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.location = (0.0, -b.length, 0.0)
        made.append((bn, round(r, 3)))
print("JOINTS", made)
DARK_BACKFACES = False      # v6: off - it also blackened outside faces whose normals the AI mesh has flipped (black flags at the hips and hands)
# 2. dark insides: in every material of the body and the crust, mix to near-black where the surface faces away from the camera
for ob in ((body, bpy.data.objects.get("Relic_AI_Crust")) if DARK_BACKFACES else ()):
    if ob is None:
        continue
    for slot in ob.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes or mat.get("dark_backfaces"):
            continue
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if out is None or not out.inputs["Surface"].links:
            continue
        src = out.inputs["Surface"].links[0].from_socket
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dark = nt.nodes.new("ShaderNodeBsdfDiffuse"); dark.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
        nt.links.new(src, mix.inputs[1])
        nt.links.new(dark.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        mat["dark_backfaces"] = True
        if hasattr(mat, "use_backface_culling"):
            mat.use_backface_culling = False
        print("BACKFACE dark:", mat.name)
bpy.ops.wm.save_as_mainfile(filepath=OUT + "/fight_v06.blend")
print("JOINTS SAVED")
