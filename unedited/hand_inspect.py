import bpy, numpy as np
rig = bpy.data.objects["Relic_AI_Rig"]
for nm in ("Relic_AI", "Relic_AI_Crust"):
    o = bpy.data.objects[nm]
    print("OBJ", nm, len(o.data.vertices), "mods", [(m.type, m.name) for m in o.modifiers], "vgs", len(o.vertex_groups), "parent", o.parent.name if o.parent else None)
o = bpy.data.objects["Relic_AI"]
gi = {g.name: g.index for g in o.vertex_groups}
for side in "LR":
    hb = rig.data.bones["hand." + side]
    inv = hb.matrix_local.inverted()
    pts = []
    for v in o.data.vertices:
        for g in v.groups:
            if g.group == gi["hand." + side] and g.weight > 0.5:
                pts.append((inv @ v.co)[:])
                break
    pts = np.array(pts)
    print("HAND", side, len(pts), "bounds", np.round(pts.min(0), 3), np.round(pts.max(0), 3), "hand bone length", round(hb.length, 3))
    # histogram along y
    h, e = np.histogram(pts[:, 1], bins=10)
    print("YHIST", side, list(h), np.round(e, 2).tolist())
print("GROUPS", sorted(gi.keys()))
for side in "LR":
    hb = rig.data.bones["hand." + side]; inv = hb.matrix_local.inverted()
    P=[]
    for v in o.data.vertices:
        for g in v.groups:
            if g.group == gi["hand." + side] and g.weight > 0.5:
                P.append((inv @ v.co)[:]); break
    P=np.array(P)
    tip=P[P[:,1]>0.14]; palm=P[(P[:,1]>0.02)&(P[:,1]<0.08)]
    print("CURL", side, "tip mean z", round(tip[:,2].mean(),3), "palm mean z", round(palm[:,2].mean(),3), "tip mean x", round(tip[:,0].mean(),3), "palm x", round(palm[:,0].mean(),3), "x range tip", np.round([tip[:,0].min(), tip[:,0].max()],3), "z range tip", np.round([tip[:,2].min(), tip[:,2].max()],3))
