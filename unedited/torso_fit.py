# torso_fit.py : measure how far Relic's torso armour reaches (front, back, left, right) at each height, and his helmet, in the rest pose,
#   so hands can be kept OUTSIDE his body (the capture comes from a much slimmer person).
#   blender.exe -b fight_v05.blend --python torso_fit.py      -> anim_cache5\relic_torso.json
#   Frame: origin = pelvis head, axis = pelvis head -> neck head (t = 0..1 along it), left = right shoulder -> left shoulder, fwd = left x up.
import bpy, json, numpy as np
from mathutils import Vector
rig = bpy.data.objects["Relic_AI_Rig"]
ob = bpy.data.objects["Relic_AI"]
B = rig.data.bones
P0, P1 = B['pelvis'].head_local, B['neck'].head_local
up = (P1 - P0); Lax = up.length; up.normalize()
left = B['upperarm.L'].head_local - B['upperarm.R'].head_local
left = (left - up * left.dot(up)).normalized()
fwd = left.cross(up).normalized()
me = ob.data
n = len(me.vertices)
co = np.zeros(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
gi = {g.index: g.name for g in ob.vertex_groups}
TORSO = ("pelvis", "spine", "chest", "neck")
dom = np.zeros(n, bool)
headv = np.zeros(n, bool)
for v in me.vertices:
    if not v.groups:
        continue
    g = max(v.groups, key=lambda g: g.weight)
    nm = gi[g.group]
    dom[v.index] = nm in TORSO
    headv[v.index] = nm == "head"
d = co - np.array(P0)
t = d @ np.array(up) / Lax
f = d @ np.array(fwd)
l = d @ np.array(left)
rows = []
for tc in np.arange(-0.25, 1.21, 0.05):
    s = dom & (np.abs(t - tc) < 0.04)
    if s.sum() < 30:
        continue
    pf = f[s]; pl = l[s]
    rows.append(dict(t=round(float(tc), 3), front=float(np.percentile(np.clip(pf, 0, None), 97)), back=float(np.percentile(np.clip(-pf, 0, None), 97)),
                     left=float(np.percentile(np.clip(pl, 0, None), 97)), right=float(np.percentile(np.clip(-pl, 0, None), 97)), n=int(s.sum())))
hc = co[headv]
center = hc.mean(0)
rad = float(np.percentile(np.linalg.norm(hc - center, axis=1), 90))
hcl = center - np.array(P0)
out = dict(axis_len=Lax, rows=rows, head=dict(center_t=float(hcl @ np.array(up) / Lax), center_f=float(hcl @ np.array(fwd)), center_l=float(hcl @ np.array(left)), radius=rad),
           shoulder_half=float((B['upperarm.L'].head_local - B['upperarm.R'].head_local).length / 2))
json.dump(out, open(r"C:\ai\unedited\anim_cache5\relic_torso.json", "w"), indent=1)
for r in rows:
    print("TORSO t %5.2f  front %.3f back %.3f left %.3f right %.3f  (%d verts)" % (r['t'], r['front'], r['back'], r['left'], r['right'], r['n']))
print("HEAD", out['head'], "axis %.3f shoulder half %.3f" % (Lax, out['shoulder_half']))
