import bpy, math
from mathutils import Vector
sc = bpy.data.scenes["Street"]
kr = bpy.data.objects["KillDroid_Rig"]
prev = {}
rows = []
for f in range(1, 600):
    sc.frame_set(f)
    cur = {}
    for pb in kr.pose.bones:
        cur[pb.name] = (kr.matrix_world @ pb.matrix).to_quaternion()
    pos = {n: kr.matrix_world @ kr.pose.bones[n].head for n in ("foot.L","foot.R","toe.L","toe.R","hand.L","hand.R","head","pelvis","forearm.R","forearm.L")}
    if prev:
        for n, q in cur.items():
            a = math.degrees(q.rotation_difference(prev[n]).angle); a = min(a, 360.0 - a)
            if a > 9: rows.append((f, n, round(a,1)))
    prev = cur
    if f % 6 == 0 or f in (1,):
        pass
print("SPIKES", len(rows))
for r in rows: print("SPIKE", r)
# feet: height and slide
last = {}
for f in range(1, 600):
    sc.frame_set(f)
    for s in "LR":
        ft = kr.matrix_world @ kr.pose.bones["foot."+s].head
        tt = kr.matrix_world @ kr.pose.bones["toe."+s].tail
        low = min(ft.z, tt.z)
        if f % 8 == 0: print("FOOT", f, s, round(ft.x,2), round(ft.y,2), round(ft.z,2), "toe z", round(tt.z,2))
