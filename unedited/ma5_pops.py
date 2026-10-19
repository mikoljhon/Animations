import bpy, numpy as np
sc = bpy.data.scenes["Cave"]; rig = bpy.data.objects["MenderC_Rig"]
ad = rig.animation_data
cur, slot = ad.action, ad.action_slot
P, Mw = rig.pose.bones, rig.matrix_world
bones = ["upperarm.L", "forearm.L", "hand.L", "upperarm.R", "forearm.R", "hand.R", "thigh.L", "shin.L", "thigh.R", "shin.R", "spine", "head"]
res = {}
for tag, an in (("orig", "MenderC_cut5_original"), ("new", cur.name)):
    a = bpy.data.actions[an]
    ad.action, ad.action_slot = a, a.slots[0]
    ax = {b: [] for b in bones}
    for f in range(1, 361):
        sc.frame_set(f)
        for b in bones:
            ax[b].append(np.array((Mw @ P[b].matrix).to_3x3().col[1].normalized()))
    for b in bones:
        A = np.array(ax[b])
        d = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", A[1:], A[:-1]), -1, 1)))
        j = np.abs(d[1:] - d[:-1])
        res.setdefault(b, {})[tag] = (round(float(j.max()), 1), int(j.argmax()) + 2, round(float(d.max()), 1))
for b, v in res.items():
    print("POP", b, v)
