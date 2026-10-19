# kd_clearance.py : does the KillDroid's crusher (or its left fist) ever cut into its own hull (engine hump, turret, chest, pelvis)?
#   Rigid parts on bones cannot stretch, so the skin checks of the QA gate never see this; the hump in particular blocks big overhead swings.
#   blender.exe -b pipe_<tag>.blend --python kd_clearance.py -- <tag> [every_n_frames]   -> anim_cache5\clear_<tag>.json, printed summary
#   Per frame: polygon pairs where the arm group overlaps the hull group (BVH overlap, world space), and the crusher's height.
import bpy, sys, json
from mathutils import Vector
from mathutils.bvhtree import BVHTree
OUT = r"C:\ai\unedited"
a = sys.argv.index("--") + 1
tag = sys.argv[a]
step = int(sys.argv[a + 1]) if len(sys.argv) > a + 1 else 1
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["KillDroid_Rig"]
HULL = ("chest", "spine", "neck", "head", "pelvis")
ARMS = {"crusher": ("forearm.R", "hand.R", "claw1a.R", "claw2a.R"), "left fist": ("forearm.L", "hand.L")}
parts = [o for o in rig.children if o.type == 'MESH' and o.parent_type == 'BONE']
hull = [o for o in parts if o.parent_bone in HULL]
groups = {k: [o for o in parts if o.parent_bone in bones] for k, bones in ARMS.items()}


def bvh(objs):
    V, P = [], []
    for o in objs:
        me = o.data
        M = o.matrix_world
        off = len(V)
        V += [M @ v.co for v in me.vertices]
        P += [tuple(i + off for i in p.vertices) for p in me.polygons]
    return BVHTree.FromPolygons(V, P)


rows = []
for f in range(sc.frame_start, sc.frame_end + 1, step):
    sc.frame_set(f)
    th = bvh(hull)
    row = dict(f=f)
    for k, objs in groups.items():
        row[k] = len(th.overlap(bvh(objs)))
    row["crusher_z"] = round(max((o.matrix_world @ Vector(c)).z for o in groups["crusher"] for c in o.bound_box), 2)
    row["wrist_R_z"] = round((rig.matrix_world @ rig.pose.bones['forearm.R'].tail).z, 2)
    rows.append(row)
bad = {k: [(r["f"], r[k]) for r in rows if r[k] > 0] for k in ARMS}
print("CLEARANCE", tag, {k: "%d frames, worst %s" % (len(v), max(v, key=lambda x: x[1]) if v else None) for k, v in bad.items()})
json.dump(dict(rows=rows, bad=bad), open(OUT + r"\anim_cache5\clear_%s.json" % tag, "w"))
