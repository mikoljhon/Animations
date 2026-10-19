import bpy, json
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
sp = json.load(open(r"C:\ai\unedited\anim_cache5\fm_spans_relic.json"))
W = lambda n, t=False: tuple(round(x,2) for x in (rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)))
for nm in ("cross", "jab", "hook"):
    s, d, h = sp[nm]
    for off in (0, h - 3, h):
        sc.frame_set(s + off)
        print("DBG", nm, off, "shoulderR", W("upperarm.R"), "elbowR", W("forearm.R"), "wristR", W("forearm.R", True), "ikhandR", W("ik_hand.R"), "|| shoulderL", W("upperarm.L"), "wristL", W("forearm.L", True), "ikhandL", W("ik_hand.L"))
