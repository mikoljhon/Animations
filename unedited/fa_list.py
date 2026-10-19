import bpy
for o in bpy.data.objects:
    if abs(o.location.x) < 6 and 1 < o.location.y < 20 and o.type=='MESH' and not o.name.startswith(("KillDroid","Relic")):
        print("OBJ", o.name, tuple(round(v,1) for v in o.location), tuple(round(v,1) for v in o.dimensions), o.parent.name if o.parent else None)
