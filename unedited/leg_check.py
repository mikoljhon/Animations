import bpy, math, json
from mathutils import Vector
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
sp = json.load(open(r"C:\ai\unedited\anim_cache5\fm_spans_relic.json"))
def seg(f):
    for k,(s,d,h) in sp.items():
        if s-1 <= f <= s+d+1: return k
    return "gap"
prev = {}
rows = []
kneemax = {}
for f in range(1, 304):
    sc.frame_set(f)
    W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
    cur = {}
    for s in "LR":
        hip, kn, an = W("thigh."+s), W("shin."+s), W("shin."+s, True)
        a = (hip-kn).normalized(); b = (an-kn).normalized()
        flex = 180.0 - math.degrees(a.angle(b))      # 0 = straight leg, larger = bent
        tgt = W("ik_foot."+s)
        err = (tgt-an).length
        # knee direction relative to the line hip-ankle and body forward
        cur[s] = (flex, err)
        q = [ (rig.matrix_world @ rig.pose.bones[n].matrix).to_quaternion() for n in ("thigh."+s, "shin."+s, "foot."+s)]
        if s in prev:
            for n, qq, pq in zip(("thigh","shin","foot"), q, prev[s]):
                ang = math.degrees(qq.rotation_difference(pq).angle); ang = min(ang, 360-ang)
                if ang > 14: rows.append((f, seg(f), s, n, round(ang,1)))
        prev[s] = q
    m = seg(f)
    kneemax.setdefault(m, [0,0,0,0])
    for i,s in enumerate("LR"):
        kneemax[m][i] = max(kneemax[m][i], cur[s][0]); kneemax[m][2+i] = max(kneemax[m][2+i], cur[s][1])
print("LEG spikes", len(rows))
for r in rows[:40]: print("LEGSPIKE", r)
for m, v in kneemax.items(): print("LEGMAX", m, "max knee flex L/R", round(v[0]), round(v[1]), "max IK foot error L/R (m)", round(v[2],3), round(v[3],3))
