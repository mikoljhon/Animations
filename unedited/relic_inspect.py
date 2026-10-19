import bpy
for c in bpy.data.collections:
    if c.name.startswith("Relic"):
        print("COLL", c.name, len(c.objects), [o.name for o in c.objects][:14])
rig = bpy.data.objects.get("Relic_AI_Rig")
print("RIG", rig is not None, len(rig.data.bones) if rig else 0)
if rig:
    print("BONES", [b.name for b in rig.data.bones])
    print("IKCON", [(pb.name, [c.type for c in pb.constraints]) for pb in rig.pose.bones if pb.constraints][:8])
    print("DIMS", tuple(round(x, 2) for x in rig.dimensions), "loc", tuple(round(x, 2) for x in rig.location), "rot", tuple(round(x, 3) for x in rig.rotation_euler), "scale", tuple(round(x, 3) for x in rig.scale))
    print("PARENT", rig.parent)
for o in bpy.data.objects:
    if o.name.startswith("Relic_AI") and o.type == 'MESH':
        print("MESH", o.name, len(o.data.vertices), [m.type for m in o.modifiers], o.parent.name if o.parent else None)
