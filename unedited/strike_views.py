# strike_views.py : close-up check of every strike in a pipe_<tag>.blend, Relic (or the KillDroid) next to the real capture, from two sides.
#   For each clip: the chamber (most bent before the strike), the strike (most extended) and the recoil (most bent after).
#   blender.exe -b pipe_<tag>.blend --python strike_views.py -- <tag> arm|leg [relic|kd]
#   -> renders\sv_<tag>\<clip>_<frame>_<relic|kd|real>_<front|side>.png  and  sv_<tag>\frames.json (composed by strike_sheet.py)
import bpy, sys, math, json, os
from mathutils import Vector
OUT = r"C:\ai\unedited"
a = sys.argv.index("--") + 1
tag, kind = sys.argv[a], sys.argv[a + 1]
who = sys.argv[a + 2] if len(sys.argv) > a + 2 else "relic"
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 520, 620
sc.eevee.taa_render_samples = 8
rig = bpy.data.objects["KillDroid_Rig" if who == "kd" else "Relic_AI_Rig"]
_leg = lambda r: r.data.bones['thigh.L'].length + r.data.bones['shin.L'].length
KS = _leg(rig) / _leg(bpy.data.objects["Relic_AI_Rig"])      # the cameras step back for a bigger body
spans = json.load(open(OUT + r"\anim_cache5\spans_%s.json" % tag))
odir = OUT + r"\renders\sv_%s" % tag
os.makedirs(odir, exist_ok=True)
for o in bpy.data.objects:
    if "Crust" in o.name:
        o.hide_render = True
relic_meshes = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render and
                (any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers) or o.parent == rig)]     # skin + the joint pieces
ref_objs = [o for o in bpy.data.objects if o.name.startswith("ref")]
OFF = Vector((0.0, -1.6, 0.0))      # the mannequin stands this far from Relic (pipe_test.py)
cam = bpy.data.objects.new("SV", bpy.data.cameras.new("SV")); sc.collection.objects.link(cam)
cam.data.lens = 45; cam.data.clip_end = 200
sc.camera = cam
pb = rig.pose.bones
M = lambda: rig.matrix_world


def W(n, tail=False):
    return M() @ (pb[n].tail if tail else pb[n].head)


if kind == "arm":
    chain = ("upperarm", "forearm")
else:
    chain = ("thigh", "shin")


def ext_bend(f, sd):
    sc.frame_set(f)
    r, m, e = W(chain[0] + "." + sd), W(chain[1] + "." + sd), W(chain[1] + "." + sd, True)
    fw = (M().to_3x3() @ Vector((0, -1, 0))); fw.z = 0; fw.normalize()
    ln = (m - r).length + (e - m).length
    reach = ((e - r).dot(fw) + 0.5 * max(0.0, (e - r).z)) / ln      # how far the fist is driven forward (and up, for uppercuts)
    bend = 180.0 - math.degrees((r - m).angle(e - m))
    height = e.z
    return reach, bend, height


picked = {}
for clip, (f0, n, mid) in spans.items():
    rows = []
    for f in range(f0, f0 + n):
        for sd in "LR":
            reach, bend, h = ext_bend(f, sd)
            rows.append((f, sd, reach, bend, h))
    if kind == "arm":
        pk = max(rows, key=lambda r: r[2])
    else:
        pk = max(rows, key=lambda r: r[4])
    f_pk, sd_pk = pk[0], pk[1]
    before = [r for r in rows if r[1] == sd_pk and f_pk - 14 <= r[0] < f_pk - 2]
    after = [r for r in rows if r[1] == sd_pk and f_pk + 2 < r[0] <= f_pk + 14]
    fr = []
    if before:
        fr.append(("chamber", max(before, key=lambda r: r[3])[0]))
    fr.append(("strike", f_pk))
    if after:
        fr.append(("recoil", max(after, key=lambda r: r[3])[0]))
    picked[clip] = dict(side=sd_pk, frames=fr, who=who)
    print("PICK", clip, sd_pk, fr)


def shoot(path, center, fw, lf, view, dist):
    if view == "front":
        d = fw * dist + lf * dist * 0.4
    else:
        d = -lf * dist + fw * dist * 0.15
    cam.location = center + d + Vector((0, 0, 0.25))
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


dist = (2.7 if kind == "arm" else 3.7) * KS
for clip, info in picked.items():
    for lab, f in info["frames"]:
        sc.frame_set(f)
        fw = (M().to_3x3() @ Vector((0, -1, 0))); fw.z = 0; fw.normalize()
        lf = Vector((-fw.y, fw.x, 0.0))
        c_rel = W("chest") if kind == "arm" else (W("pelvis") + Vector((0, 0, -0.1)))
        ref_e = bpy.data.objects.get("ref_spine" if kind == "arm" else "ref_hips")
        c_ref = ref_e.matrix_world.translation.copy() if ref_e else c_rel + OFF
        if kind == "arm":
            c_ref = c_ref + Vector((0, 0, 0.25 * KS))
        for shot_of, center, show_rel in ((who, c_rel, True), ("real", c_ref, False)):
            for o in relic_meshes:
                o.hide_render = not show_rel
            for o in ref_objs:
                o.hide_render = show_rel
            for view in ("front", "side"):
                shoot(odir + r"\%s_%04d_%s_%s.png" % (clip, f, shot_of, view), center, fw, lf, view, dist)
for o in relic_meshes:
    o.hide_render = False
json.dump(picked, open(odir + r"\frames.json", "w"), indent=1)
print("SV DONE", tag)
