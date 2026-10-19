# reel_vid.py : the comparison reel for a pipe_<tag>.blend (made by pipe_test.py or kd_beats.py)
#   A = side view, Relic (or the KillDroid) next to the real fighter (white figure, the raw motion capture), camera tracks both
#   B = close three-quarter front view of Relic alone, camera tracks his chest
#   blender.exe -b pipe_<tag>.blend --python reel_vid.py -- <tag> [kd] [frames]   -> renders\reel_<tag>_A|B\f_####.jpg   (then reel_encode.py)
import bpy, sys, os
from mathutils import Vector
sc = bpy.data.scenes["Street"]
_a = sys.argv[sys.argv.index("--") + 1:]
tag = _a[0]
who = "kd" if "kd" in _a[1:] else "relic"
only = [int(x) for x in _a[1:] if x.isdigit()]          # optional: just these frames (framing test)
sc.render.resolution_x = sc.render.resolution_y = 720
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.quality = 92
sc.eevee.taa_render_samples = 6
for o in bpy.data.objects:
    if "Crust" in o.name:
        o.hide_render = True
rig = bpy.data.objects["KillDroid_Rig" if who == "kd" else "Relic_AI_Rig"]
_leg = lambda r: r.data.bones['thigh.L'].length + r.data.bones['shin.L'].length
KS = _leg(rig) / _leg(bpy.data.objects["Relic_AI_Rig"])      # the cameras step back (and up) for a bigger body
ref = [o for o in bpy.data.objects if o.name.startswith("ref")]
hips = bpy.data.objects["ref_hips"]


def cam(name, lens):
    c = bpy.data.objects.new(name, bpy.data.cameras.new(name)); sc.collection.objects.link(c)
    c.data.lens = lens; c.data.clip_end = 500.0
    return c


cA, cB = cam("REEL_A", 50), cam("REEL_B", 45)
fw = rig.matrix_world.to_3x3() @ Vector((0, -1, 0)); fw.z = 0; fw.normalize()      # where Relic faces
lf = Vector((-fw.y, fw.x, 0.0))
frames = range(sc.frame_start, sc.frame_end + 1)
pa, pb = {}, {}
for f in frames:
    sc.frame_set(f)
    p = rig.matrix_world @ rig.pose.bones['pelvis'].head
    pa[f] = (p + hips.matrix_world.translation) * 0.5
    pb[f] = rig.matrix_world @ rig.pose.bones['chest'].head


def smooth(d, k=10):
    out = {}
    for f in d:
        vs = [d[g] for g in range(f - k, f + k + 1) if g in d]
        out[f] = sum(vs, Vector()) / len(vs)
    return out


pa, pb = smooth(pa), smooth(pb)
for d in ("A", "B"):
    os.makedirs(r"C:\ai\unedited\renders\reel_%s_%s" % (tag, d), exist_ok=True)
for f in (only or frames):
    sc.frame_set(f)
    ta = Vector((pa[f].x, pa[f].y, 1.15 * KS))
    cA.location = ta - lf * 7.2 * KS + Vector((0, 0, 0.3 * KS))
    cA.rotation_euler = (ta - cA.location).to_track_quat('-Z', 'Y').to_euler()
    tb = Vector((pb[f].x, pb[f].y, pb[f].z - 0.25 * KS))
    cB.location = tb + (fw * 3.9 + lf * 1.7 + Vector((0, 0, 0.35))) * KS
    cB.rotation_euler = (tb - cB.location).to_track_quat('-Z', 'Y').to_euler()
    for c, d, show_ref in ((cA, "A", True), (cB, "B", False)):
        for o in ref:
            o.hide_render = not show_ref
        sc.camera = c
        sc.render.filepath = r"C:\ai\unedited\renders\reel_%s_%s\f_%04d.jpg" % (tag, d, f)
        bpy.ops.render.render(write_still=True)
print("REEL DONE", tag)
