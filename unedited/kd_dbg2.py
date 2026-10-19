import bpy
sc = bpy.data.scenes["Street"]
kr = bpy.data.objects["KillDroid_Rig"]
for f in range(380, 394, 1):
    sc.frame_set(f)
    W = lambda n, t=False: kr.matrix_world @ (kr.pose.bones[n].tail if t else kr.pose.bones[n].head)
    sh = W("upperarm.R"); el = W("forearm.R"); wr = W("forearm.R", True); tg = W("ik_hand.R"); pole = W("ik_elbow.R")
    print("DBG", f, "sh", tuple(round(x,2) for x in sh), "el", tuple(round(x,2) for x in el), "wr", tuple(round(x,2) for x in wr), "tgt", tuple(round(x,2) for x in tg), "pole", tuple(round(x,2) for x in pole), "dist", round((tg-sh).length,2))
