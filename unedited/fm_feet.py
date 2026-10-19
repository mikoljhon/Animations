import bpy, numpy as np
sc = bpy.data.scenes["Street"]
ob = bpy.data.objects["Relic_AI"]
rig = bpy.data.objects["Relic_AI_Rig"]
dg = bpy.context.evaluated_depsgraph_get()
for f in (4, 13, 40, 62, 90, 120, 150, 180, 220, 260):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.array([ (ob.matrix_world @ v.co)[:] for v in me.vertices ])
    low = co[:,2].min()
    # lowest points left/right
    L = co[co[:,0] < -0.0]; R = co[co[:,0] > 0.0]
    ft = {}
    for s in "LR":
        ft[s] = [tuple(round(x,2) for x in (rig.matrix_world @ rig.pose.bones[n].head)) for n in ("foot."+s,"toe."+s)] + [tuple(round(x,2) for x in (rig.matrix_world @ rig.pose.bones["toe."+s].tail))]
    print("FEET", f, "mesh min z", round(float(low),3), "ankle/toe", ft)
    ev.to_mesh_clear()
