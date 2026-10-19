import bpy
for n in ("KillDroid_Rig","Relic_AI_Rig"):
    r=bpy.data.objects[n]
    for b in ("foot.L","toe.L","head","hand.L","hand.R","upperarm.L","clavicle.L"):
        pb=r.pose.bones[b]
        print("FACE",n,b,tuple(round(x,2) for x in (r.matrix_world@pb.head)),tuple(round(x,2) for x in (r.matrix_world@pb.tail)))
