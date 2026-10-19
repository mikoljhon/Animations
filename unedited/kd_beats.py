# kd_beats.py : the KillDroid's beats of episode 1 (v02 "Breaker") through the fight pipeline (FIGHT_ANIMATION_PROCESS.md): approved capture
#   clips only (mocap_ref/approved_kd.json) -> fit -> solve (spine fit, arms kept out of its own hull) -> blend -> QA gate, with the crusher jaws
#   (claw1a.R / claw2a.R, opened by turning about local X) and the turret lamps (KillDroid_Rig["lamp_power"], drives the six spot lights and
#   the lenses) keyed on top.
#   blender.exe -b fight_v08.blend --python kd_beats.py -- TAG   -> pipe_TAG.blend (+ the white reference figure and cameras for strike_views.py /
#                                                                  reel_vid.py), anim_cache5\qa_TAG.json, spans_TAG.json, beats_TAG.json
#   Run it through run_gated.ps1.  Relic is moved out of the way (as in pipe_test.py): his side of the beats is his own take.
#   beat 1  the miss: a crusher swing from the side and low (a body hook, 14_02, slowed). The engine hump blocks a big overhead wind-up and a
#           head-high hook (14_01) swept the crusher arm through its own turret, so this swing stays at body height. The jaws open on the
#           wind-up and snap shut on the follow-through.
#   beat 2  the hit: the mechanical LEFT fist (a straight right of 02_05, mirrored), thrown with the crusher held low in front: a boxer's
#           guard (14_03) held the crusher up beside the turret, pointing at the sky, and the KillDroid has no guard
#   beat 4  the hold: it bends, takes Relic off the street with the crusher and lifts him to its turret (137_40 Strong Man Pick Up); the jaws
#           open on the reach, clamp on the grab (not fully: Relic's torso is between them) and hold
#   beat 5  the finish: it holds him up; the rail hits the turret, the six lamps flicker and die (lamp_power 1 -> 0), its hydraulics lock
#           where they are and the jaws go slack and drop Relic. (Every captured kneel / collapse brings the hands in against its cab and fails
#           the gate, so it does not sink here; its fall is a physics job, not a capture.)
#   Beat 3 (the backhand and the boot stomp) has no clean capture in the library yet (searched: no backhand, no stomp), so it is not in this take.
import bpy, sys, json, math
from mathutils import Vector, Quaternion
OUT = r"C:\ai\unedited"
exec(open(OUT + r"\fight_pipeline.py", encoding="utf-8").read())
tag = sys.argv[sys.argv.index("--") + 1]
sc = bpy.data.scenes["Street"]
kr = bpy.data.objects["KillDroid_Rig"]
rr = bpy.data.objects["Relic_AI_Rig"]
rr.location = (30.0, 30.0, 0.0)
kr.animation_data_clear()
# the fight lane is cleared of boulders as in the fight (fight_anim.py)
for nm, nx in (("Boulder19", -6.8), ("Boulder22", 6.4), ("Boulder28", -7.4)):
    if nm in bpy.data.objects:
        bpy.data.objects[nm].location.x = nx
approved = json.load(open(OUT + r"\mocap_ref\approved_kd.json"))["approved"]
BASE = (0.0, 4.6)                      # where it starts, facing -Y; Relic would stand about 2.4 m in front of it
RELIC_Y = BASE[1] - 2.4
# (beat, clip, slow, gap before, look target (x, y, z) in world): slower than the capture, the user's rule for heavy bodies
BEATS = [("beat 1  crusher swing, low from the side (the miss)", "kd_swing_14_02", 2.0, 0, (0.0, RELIC_Y, 1.95)),
         ("beat 2  left fist (the hit)", "kd_punch_L_02_05", 1.6, 16, (0.0, RELIC_Y, 1.95)),
         ("beat 4  crusher grab and lift (the hold)", "kd_lift_137_40", 1.2, 18, (0.0, RELIC_Y + 0.6, 1.2))]
HOLD = 96                              # beat 5: frames it holds Relic up after the lift; the rail lands HIT_AT frames into it
HIT_AT = 24
for b in BEATS:
    if b[1] not in approved:
        raise SystemExit("clip %s is not in approved_kd.json: run qa_clips.sh kd first" % b[1])
pipe = FightPipeline(kr, rr)
f = 4
spans, beats = {}, []
base = BASE
for label, clip, slow, gap, look in BEATS:
    f += gap
    end = pipe.add_clip(clip, f, base=base, yaw=0.0, look=Vector(look), slow=slow, blend_in=10)
    spans[clip] = (f, end - f, (end - f) // 2)
    beats.append(dict(label=label, clip=clip, start=f, end=end, slow=slow))
    base = tuple(pipe.frames[end]['hips'].xy)        # the next clip starts where this one left the hips
    f = end + 1
pipe.bake()
PB = kr.pose.bones
W = lambda n, t=False: kr.matrix_world @ (PB[n].tail if t else PB[n].head)


# ------------------------------------------------------------------------------------------------ the crusher jaws and the lamps
def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ramp(keys, x):
    """piecewise smoothstep through (frame, value) keys"""
    if x <= keys[0][0]:
        return keys[0][1]
    for (x0, v0), (x1, v1) in zip(keys, keys[1:]):
        if x <= x1:
            return v0 + (v1 - v0) * ease((x - x0) / max(1e-6, x1 - x0))
    return keys[-1][1]


def wrist_track(bt, side='R'):
    out = {}
    for fr in range(bt['start'], bt['end'] + 1):
        sc.frame_set(fr)
        out[fr] = W('forearm.' + side, True)
    return out


SHUT, GRIP, OPEN, SLACK = 0.04, 0.24, 0.58, 0.50          # radians about each jaw's local X (0 = closed at rest)
b1, b2, b4 = beats
# beat 1: the strike = the fastest frame of the crusher wrist
tr = wrist_track(b1)
fs = max(range(b1['start'] + 20, b1['end'] - 5), key=lambda q: (tr[q] - tr[q - 1]).length)
# beat 4: the grab = the lowest frame of the crusher wrist
tr4 = wrist_track(b4)
fg = min(range(b4['start'] + 18, b4['end'] - 20), key=lambda q: tr4[q].z)
# beat 5: the rail hits the turret while it holds him up; then the lamps die and the jaws let go
f_hit = pipe.last + HIT_AT
f_end = pipe.last + HOLD
beats.append(dict(label="beat 5  the rail hits the turret: lamps die, jaws let go (the finish)", clip="hold", start=pipe.last + 1, end=f_end, slow=1.0))
JAW = [(b1['start'], SHUT), (b1['start'] + 6, SHUT), (fs - 12, OPEN), (fs + 1, OPEN), (fs + 5, SHUT), (b1['end'] + 6, 0.08),
       (b4['start'], 0.08), (b4['start'] + 6, 0.10), (fg - 10, OPEN + 0.04), (fg, OPEN + 0.04), (fg + 4, GRIP),
       (f_hit + 2, GRIP), (f_hit + 26, SLACK)]
FLICKER = [0.15, 0.9, 1.0, 0.2, 0.05, 0.7, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.12, 0.0]
for fr in range(pipe.first, f_end + 1):
    a = ramp(JAW, fr)
    PB['claw1a.R'].rotation_mode = PB['claw2a.R'].rotation_mode = 'QUATERNION'
    PB['claw1a.R'].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), -a)          # outer jaw
    PB['claw2a.R'].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), a)           # inner jaw
    PB['claw1a.R'].keyframe_insert('rotation_quaternion', frame=fr)
    PB['claw2a.R'].keyframe_insert('rotation_quaternion', frame=fr)
    k = fr - f_hit
    kr["lamp_power"] = 1.0 if k < 0 else (FLICKER[k] if k < len(FLICKER) else 0.0)
    kr.keyframe_insert('["lamp_power"]', frame=fr)
kr.update_tag()
lamps = [o for o in bpy.data.objects if o.type == 'LIGHT' and o.name.startswith("KillDroid_Lamp")]
chk = {}
for fr in (pipe.first, f_hit + 1, f_hit + 2, f_end):
    sc.frame_set(fr)
    dg = bpy.context.evaluated_depsgraph_get()
    chk[fr] = round(sum(o.evaluated_get(dg).data.energy for o in lamps), 1)
print("LAMPS (frame: total spot energy)", chk, "| strike", fs, "grab", fg, "rail hit", f_hit)

# ------------------------------------------------------------------------------------------------ QA gate on the finished take (jaws included)
pipe.last = f_end                      # the hold too: the slack jaws must not cut into the turret either
rep = pipe.qa()
print(pipe.qa_text(rep))
json.dump(rep, open(OUT + r"\anim_cache5\qa_%s.json" % tag, "w"), default=str, indent=1)
json.dump({k: list(v) for k, v in spans.items()}, open(OUT + r"\anim_cache5\spans_%s.json" % tag, "w"))
json.dump(dict(beats=beats, strike=fs, grab=fg, rail_hit=f_hit, jaw_keys=JAW, lamps=chk, who="kd"), open(OUT + r"\anim_cache5\beats_%s.json" % tag, "w"), indent=1)

# ------------------------------------------------------------------------------------------------ reference figure, a side camera, save
fw = kr.matrix_world.to_3x3() @ Vector((0.0, -1.0, 0.0))
pipe.ref_figure(-fw * 2.6)                                   # the white figure stands behind it, clear of the swing
for o in bpy.data.objects:
    if "Crust" in o.name:
        o.hide_render = True
dbg = bpy.data.objects.new("DBG", bpy.data.cameras.new("DBG"))
sc.collection.objects.link(dbg)
dbg.data.lens = 30
dbg.data.clip_end = 1500.0
mid = Vector((BASE[0], BASE[1] + 1.0, 1.6))
dbg.location = (9.5, mid.y, 1.9)
dbg.rotation_euler = (mid - dbg.location).to_track_quat('-Z', 'Y').to_euler()
sc.camera = dbg
sc.frame_start, sc.frame_end = pipe.first, f_end
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\pipe_%s.blend" % tag)
print("PIPE SAVED", tag, "QA", "PASS" if rep['pass'] else "FAIL", "frames", pipe.first, f_end)
