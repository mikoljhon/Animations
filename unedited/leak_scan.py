# leak_scan.py : list every mesh piece whose skin weights LEAK between body regions that never move together
#   (an arm piece also weighted to a leg, a left limb weighted to the right one, the torso weighted to a forearm/hand/shin/foot).
#   blender.exe -b fight_v03.blend --python leak_scan.py
import bpy, numpy as np
exec(open(r"C:\ai\unedited\skin_rules.py", encoding="utf-8").read())
rig = bpy.data.objects["Relic_AI_Rig"]
for nm in ("Relic_AI", "Relic_AI_Crust"):
    ob = bpy.data.objects.get(nm)
    if ob is None:
        continue
    rep = leak_report(ob)
    print("LEAKS", nm, "pieces with leaks:", len(rep), " vertices:", sum(r['verts'] for r in rep), " total leaked weight: %.1f" % sum(r['weight'] for r in rep))
    for r in rep[:25]:
        print("   piece %(piece)d  region %(region)s  verts %(verts)d  leaked weight %(weight).2f  max %(max).2f  bones %(bones)s" % r)
