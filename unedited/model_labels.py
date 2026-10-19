# model_labels.py : look at Relic's mesh the way the skinning sees it. Colours every vertex by the bone that owns it (its largest weight) and
#   renders the rest pose from 4 sides, plus the plain model from the same 4 sides (to see where it is already see-through).
#   blender.exe -b <file.blend> --python model_labels.py -- <tag>      -> renders\labels_<tag>_<view>.png, renders\plain_<tag>_<view>.png
import bpy, sys, colorsys
from mathutils import Vector
tag = sys.argv[sys.argv.index("--") + 1]
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 520, 760
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["Relic_AI_Rig"]
rig.animation_data_clear()
for b in rig.pose.bones:
    b.rotation_mode = 'QUATERNION'; b.rotation_quaternion = (1, 0, 0, 0); b.location = (0, 0, 0)
ob = bpy.data.objects["Relic_AI"]
for o in bpy.data.objects:
    if o.type == 'MESH' and o != ob and (o.parent == rig or "Crust" in o.name or o.name.startswith("ref")):
        o.hide_render = True
me = ob.data
gi = {g.index: g.name for g in ob.vertex_groups}
names = sorted({gi[g.group] for v in me.vertices for g in v.groups})
pal = {}
for k, n in enumerate(names):
    h = (k * 0.61803) % 1.0
    pal[n] = colorsys.hsv_to_rgb(h, 0.75, 0.95)
ca = me.color_attributes.get("bone_label") or me.color_attributes.new("bone_label", 'FLOAT_COLOR', 'POINT')
for v in me.vertices:
    if v.groups:
        g = max(v.groups, key=lambda g: g.weight)
        c = pal[gi[g.group]]
    else:
        c = (0, 0, 0)
    ca.data[v.index].color = (c[0], c[1], c[2], 1.0)
lm = bpy.data.materials.new("LabelMat"); lm.use_nodes = True
nt = lm.node_tree
em = nt.nodes.new("ShaderNodeEmission"); at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "bone_label"
nt.links.new(at.outputs["Color"], em.inputs["Color"])
out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
orig = [s.material for s in ob.material_slots]
cam = bpy.data.objects.new("ML", bpy.data.cameras.new("ML")); sc.collection.objects.link(cam)
cam.data.lens = 50; cam.data.clip_end = 100; sc.camera = cam
c = rig.matrix_world @ rig.pose.bones['chest'].head
c = Vector((c.x, c.y, c.z - 0.35))
fw = rig.matrix_world.to_3x3() @ Vector((0, -1, 0)); fw.z = 0; fw.normalize()
lf = Vector((-fw.y, fw.x, 0.0))
views = (("front", fw), ("left", lf), ("back", -fw), ("right", -lf))
for mode in ("labels", "plain"):
    for i, s in enumerate(ob.material_slots):
        s.material = lm if mode == "labels" else orig[i]
    for nm, d in views:
        cam.location = c + d * 5.2 + Vector((0, 0, 0.3))
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = r"C:\ai\unedited\renders\%s_%s_%s.png" % (mode, tag, nm)
        bpy.ops.render.render(write_still=True)
print("LABELS", tag, len(names), "bones:", names)
