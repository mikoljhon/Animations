import bpy
kr = bpy.data.objects["KillDroid_Rig"]
for b in kr.data.bones:
    m=b.matrix_local
    print("BONE", b.name, "par", b.parent.name if b.parent else None, "conn", b.use_connect, "head", tuple(round(x,2) for x in b.head_local), "tail", tuple(round(x,2) for x in b.tail_local), "len", round(b.length,2))
for pb in kr.pose.bones:
    for c in pb.constraints:
        print("CON", pb.name, c.type, getattr(c,"subtarget",None), getattr(c,"pole_subtarget",None), getattr(c,"chain_count",None))
print("ARM", kr.rotation_euler, kr.location, kr.scale)
