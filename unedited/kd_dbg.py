import bpy
sc = bpy.data.scenes["Street"]
kr = bpy.data.objects["KillDroid_Rig"]
for f in (1, 100, 340):
    sc.frame_set(f)
    W = lambda n, t=False: tuple(round(x,2) for x in (kr.matrix_world @ (kr.pose.bones[n].tail if t else kr.pose.bones[n].head)))
    print("DBG", f, "pelvis", W("pelvis"), "thigh.L", W("thigh.L"), "shin.L", W("shin.L"), "ankle", W("shin.L", True), "ikfoot", W("ik_foot.L"), "ikknee", W("ik_knee.L"), "ikhand.L", W("ik_hand.L"), "wrist", W("forearm.L", True), "elbow", W("forearm.L"), "ikelb", W("ik_elbow.L"), "head", W("head"))
