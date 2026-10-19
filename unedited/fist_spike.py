# fist_spike.py : stretch test of a closed fist on its own: for each curl, count mesh edges stretched >3x and >6 cm (the QA spike rule)
#   and the worst stretch in the hands.   blender.exe -b fight_v06.blend --python fist_spike.py -- c1,c2 ...
import bpy, sys, numpy as np
OUT = r"C:\ai\unedited"
exec(open(OUT + r"\relic_fists.py", encoding="utf-8").read())
import os
if os.environ.get("FIST_CFG"):                  # "kn1,kn2,z1s,z1w,z2s,z2w,dqs"
    _c = [float(x) for x in os.environ["FIST_CFG"].split(",")]
    KN1, KN2, Z1, Z2 = _c[0], _c[1], (_c[2], _c[3]), (_c[4], _c[5])
    for _o in bpy.data.objects:
        for _m in _o.modifiers:
            if _m.type == 'ARMATURE':
                _m.use_deform_preserve_volume = bool(_c[6])
    print("FISTCFG", KN1, KN2, Z1, Z2, "dqs", bool(_c[6]))
rig = bpy.data.objects["Relic_AI_Rig"]
rig.animation_data_clear()
ob = bpy.data.objects["Relic_AI"]
add_fists(rig)
for b in rig.pose.bones:
    b.rotation_mode = 'QUATERNION'
    b.rotation_quaternion = (1, 0, 0, 0)
me0 = ob.data
rest = np.zeros(len(me0.vertices) * 3); me0.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
edges = np.zeros(len(me0.edges) * 2, dtype=int); me0.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
rl = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
a = sys.argv.index("--") + 1
for s in sys.argv[a:]:
    c1, c2 = (float(x) for x in s.split(","))
    set_fists(rig, (1.0, 1.0), c1=c1, c2=c2)
    bpy.context.view_layer.update()
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    m = ev.to_mesh()
    co = np.zeros(len(m.vertices) * 3); m.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    ev.to_mesh_clear()
    L = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    r = L / np.maximum(rl, 1e-6)
    bad = (r > 3) & (L > 0.06)
    print("FISTSPIKE curl %.2f,%.2f : spike edges %d, edges >2x and >3cm %d, worst x%.1f (%.3f m)" % (
        c1, c2, int(bad.sum()), int(((r > 2) & (L > 0.03)).sum()), float(r[L > 0.02].max()), float(L[np.argmax(np.where(L > 0.02, r, 0))])))
