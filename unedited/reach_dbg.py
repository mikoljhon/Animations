import bpy, numpy as np
sc = bpy.data.scenes["Street"]; rig = bpy.data.objects["Relic_AI_Rig"]
d = {"L": [], "R": []}; refd = {"L": [], "R": []}
for f in range(5, 381):
    sc.frame_set(f)
    for s in "LR":
        wr = rig.matrix_world @ rig.pose.bones["forearm." + s].tail
        tg = rig.matrix_world @ rig.pose.bones["ik_hand." + s].head
        d[s].append((wr - tg).length)
        sh = rig.matrix_world @ rig.pose.bones["upperarm." + s].head
        refd[s].append((tg - sh).length)
for s in "LR":
    print("REACH", s, "hand misses its target by mean %.1f cm, max %.1f cm; target distance from shoulder mean %.2f m (arm length %.2f)" % (100*np.mean(d[s]), 100*np.max(d[s]), np.mean(refd[s]), rig.data.bones["upperarm."+s].length + rig.data.bones["forearm."+s].length))
