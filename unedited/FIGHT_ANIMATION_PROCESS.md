# UNEDITED - fight animation process

Fight motion gets onto Relic or the KillDroid this way and no other. Every step is a script. A take is used only if it passes the QA gate **and** the close-up check.
All files are in `C:\ai\unedited`. Base scene: **`fight_v08.blend`** (2026-10-17) = fight_v07.blend with the KillDroid replaced by v02 "Breaker" (`blender.exe -b fight_v07.blend --python fight_build.py -- swap fight_v08.blend`; Relic and the street untouched; fight_build.py never overwrites an existing base). Run every Blender job through `run_gated.ps1` (one at a time, only while the PC is free).

## Step 0: the model itself must be fit for animation (run after ANY change to Relic's mesh, weights or rig)

`blender.exe -b <file> --python pose_test.py -- <tag>` then `python pose_sheet.py <old> <new>` (and `... close` for joint close-ups): 8 extreme poses (arms out, arms overhead, elbows fully bent, punch reach, high front kick, side kick, deep crouch, twist + lean + head turn), 4 views each, scenery hidden so any hole shows the bright sky. No sky through the body, no stretched or torn plates.

How fight_v07 was made (`relic_model_v7.py` on fight_v06): every face of an arm or leg belongs to exactly one bone and the mesh is CUT where bones meet (clean planar cuts through the hip, elbow, knee, ankle and toe joints; edges split), so limb plates are rigid and cannot stretch; torso plates follow torso bones only; the shoulders keep their smooth weights (a cut there sliced the shoulder pads in half); a dark watertight inner body (torso tube + a tube per limb bone, overlapping at the joints, just inside the armour) fills every gap; the original custom normals are copied back so the surface shades as before. The mud crust copy (`Relic_AI_Crust`) is NOT rebuilt: it is hidden in all fight scenes.
Lessons from building it (all handled in the script now): which limb segment a face belongs to is decided by POSITION only, by the joint planes along the whole limb (the original weights had big parts of both forearms on the upper arm, 1,000+ faces each); a small loose piece follows the armour it touches, and a cap sitting at a hinge (elbow point, knee cap, heel spur) follows the lower bone; small mislabelled patches are merged before cutting (each would have flown off as a fleck); blends near the shoulders fade in smooth bands, never at a hard radius (a piece was torn 56x at one). The QA gate checks for **floating pieces** (a loose piece more than 5 cm from every other piece).

## The rules the gate enforces

| Check | Limit |
|---|---|
| **Anatomy, on its own terms:** no elbow or knee ever points the way a human joint cannot (checked on EVERY frame, transitions included, on Relic AND on the reference capture) | never (0 frames) |
| **Fists never inside Relic's own body** (torso armour or helmet, every frame) | never closer than 1 cm |
| **KillDroid: its crusher and fist never cut into its own hull** (forearm / hand / jaw parts against the parts on its pelvis, spine, chest, neck and head: cab, engine hump, turret; polygon against polygon, every 2nd frame, transitions and the jaw keys included) | never (0 frames) |
| Knees bend as much as the real knees | 95% of frames within 10 deg, never more than 25 deg |
| Elbows bend as much as the real elbows (looser: fists are pushed out of Relic's bulkier body on purpose) | 95% within 20 deg, never more than 45 deg |
| The body faces, twists and leans like the real person (shoulder line, hip line, spine) | 95% within 12 deg, never more than 25 deg |
| Elbows point the way the real person's elbows did (frames where a fist was pushed out of the armour are judged by the clearance rule instead) | 95% within 15 deg, never more than 35 deg |
| Knees point the way the real person's knees did | 95% within 15 deg, never more than 35 deg |
| Hands go where the real hands went (same exception) | 95% within 12 deg |
| An elbow or knee bends backwards past its hinge | never (0 frames) |
| A joint turns faster than a human joint can | never (per-joint caps: spine 30, upper arm 55, forearm 65, thigh 55, shin 70 deg per frame at 24 fps) |
| A joint outside its anatomical range | never more than 3 deg |
| Skin weights leak between body parts (a vertex pulled by two bones more than two joints apart, e.g. hand + thigh) | never (0 pieces) |
| The mesh stretches into a spike (an edge more than 3x its rest length and longer than 6 cm), checked every 2nd frame | never (0 edges) |
| A loose armour piece floats away (more than 5 cm from every other piece; on the KillDroid its rigid parts count as pieces too) | never |
| A foot sinks into the floor | never more than 1.5 cm |

When the elbow or knee bend check fails, the gate now lists the frames (`worst elbow bend frames`, `worst knee bend frames`).

## The steps

1. **Source: real motion capture only** (CMU library, `C:\ai\mocap\data\seq30.npz`). Find candidates with `mocap_ref\catalog.py` / `catalog_kicks.py`. Cut a clip with `mocap_ref\cut_segments.py` (add a line to `mocap_ref\jobs.json`). The cutter repairs flipped arms automatically (`mocap_ref\anatomy.py`) and prints which arms it repaired.
2. **Look at the raw clip first:** `python mocap_ref\raw_strip.py <clip> 2 0 14` draws the capture straight from the file (no Blender). Elbows must bend forward/up, never with the forearm pointing back.
3. **Approve the clip.** `bash qa_clips.sh relic <clip>` -> PASS/FAIL. Only clips in `mocap_ref\approved_relic.json` may be used. Do not force a failing clip: pick another.
4. **Fit** (`mocap_keys.py`): pelvis path and height scale with the leg; hands from Relic's TRUE shoulders by the arm length; knees from the real hip joints; a planted foot keeps the widened stance + foot lock, a RAISED foot is placed from Relic's own hip joint (so a kicking leg bends exactly like the fighter's); how far each fist was outside the fighter's (slim) body is measured (`body_clear.py`).
5. **Solve** (`rig_guard.py` + `limb_solver.py`): the pelvis is lowered until each planted leg has the fighter's hip-ankle distance (Relic's wider stance would otherwise straighten his knees); each fist is kept at least as far outside Relic's armour as the fighter's was outside theirs (only ever pushed outward; armour profile `anim_cache5\relic_torso.json` from `torso_fit.py`); hand/foot exact; elbow/knee follow the real direction but never the impossible way or outside the shoulder/hip cone; joint ranges and speed caps; keyed as FK.
6. **Blend** (`fight_pipeline.py`) and the **QA gate** (`qa()`), all automatic in `pipe_test.py`.
7. **Close-up check:** `strike_views.py` + `strike_sheet.py` (Relic vs the capture, front and side, chamber/impact/recoil). Look at it before anyone sees a reel.
8. **Reel:** `reel_vid.py` + `reel_encode.py` (low priority, one Blender at a time).

## Commands

```
cd C:\ai\unedited
powershell -File run_gated.ps1 -blend fight_v08.blend -script pipe_test.py -a "relic MYTAKE clip1 clip2" -log anim_cache5\pipe_MYTAKE.txt
powershell -File run_gated.ps1 -blend pipe_MYTAKE.blend -script strike_views.py -a "MYTAKE arm" -log anim_cache5\sv_MYTAKE.txt     (leg for kicks; add kd for the KillDroid)
python strike_sheet.py MYTAKE
powershell -File run_gated.ps1 -blend pipe_MYTAKE.blend -script reel_vid.py -a "MYTAKE" -log anim_cache5\reel_MYTAKE.txt          (add kd for the KillDroid)
python reel_encode.py MYTAKE renders\MYTAKE.mp4                                                                                   (add kd for the KillDroid)
GATED=1 bash qa_clips.sh relic          (or kd; BASE=... for another base, default fight_v08.blend)
```

## The KillDroid (v02 "Breaker", 2026-10-17)

Same pipeline, same gate, its own clip library and take script:
- **Clips:** `GATED=1 bash qa_clips.sh kd <clips>` -> `mocap_ref\approved_kd.json`. Its clips are named `kd_*` in `jobs.json`.
- **Take:** `kd_beats.py` = its episode 1 beats (`powershell -File run_gated.ps1 -blend fight_v08.blend -script kd_beats.py -a "TAG" -log anim_cache5\pipe_TAG.txt`) -> `pipe_TAG.blend` + `beats_TAG.json`. On top of the pipeline it keys the crusher jaws (`claw1a.R` outer / `claw2a.R` inner, a turn about each bone's local X: 0 closed, about 0.6 rad open) and the turret lamps (`KillDroid_Rig["lamp_power"]` 1 -> 0 drives the six spot lights and the lenses), then runs the gate on the finished take (jaws, transitions and the final hold included). `kd_clearance.py` re-checks the hull on every frame.
- **Spine fit** (`kd_ctrl.kd_apply`, `P['spine_fit']`, on by default for the KillDroid, off for Relic until his clips are re-approved with it): the chest copies the real shoulder line and is tilted about it until the pelvis -> neck line is the real one. The old 0.85-of-the-lean torso fell 15-20 deg short in any deep bend (a pick-up, a kneel) and a side lean applied after a deep forward lean tipped the shoulder line; with the fit, the pick-up went from 20 to 9 deg torso error and its elbows from 39 to 0.
- **Its arms are kept out of its own hull** (`rig_guard.kd_clear_arm`): it is far bulkier than Relic (cab and hips almost as wide as its shoulders, forearms as thick as a man's waist, a half-metre crusher). After each arm solve the forearm / hand / jaw parts are tested polygon against polygon with the hull as posed; if they meet, the wrist target moves along one outward direction by the least distance that clears it (doubling, then bisection, +2 cm spare) and the arm is solved again. Its legs are solved first, because the rams on its hips aim at its thighs and are part of the hull; transitions get the same treatment (`_blend_hull_clear`). The merged parts have no usable inside / outside (a nearest-surface sign test called 95% of the arm "inside" at rest), so only the overlap test is used.
- **What fails on it, and why** (so the next clip search starts in the right place): a boxer's guard puts the crusher beside the turret pointing at the sky (pushed out of the hull, but wrong for a brute with no guard); a head-high hook sweeps the crusher arm through its own turret (14_01); every captured kneel, squat or collapse brings the hands in against the cab, and pushing them out costs more elbow fidelity than the gate allows (23_03, 115_07, 115_09, 26_10). Body-level swings, a straight punch thrown with the other hand held low, and the Strong Man pick-up pass.
- The gate does not yet test arm against leg (the crusher passing through its own thighs); the close-ups must.

## The base scene (fight_v06.blend)

`fight_v01.blend` -> `relic_reskin.py` v6 -> `fight_v06s.blend` -> `relic_joints.py` -> `fight_v06.blend`:
1. Leak cleaning (`skin_rules.py`): a vertex may only follow bones at most two joints apart.
2. Piece by piece: small or one-bone armour pieces move rigidly, pieces wrapping a joint bend inside one band, the torso skin keeps its weights blurred over 4 cm.
3. **Hip pieces by position (v6):** every armour piece of the pelvis / upper thighs that reaches the hip region follows the pelvis above the joint and the thigh below it, blending across 7 cm, on ALL its vertices (a crotch piece is shared smoothly between both thighs). Neighbouring plates move together, so the leg stays attached on high kicks.
4. Plain linear skinning (dual-quaternion "preserve volume" was tried: it stretched the fists more).
5. Dark mechanical ball joints / hinge drums inside the armour at every limb joint.
6. Fists (`relic_fists.py`): the fingers fold about the hand's local Z (the line across the knuckles), toward the palm; knuckle line 8.5 cm and mid-finger line 13 cm from the wrist with wide blend bands, curl 1.3 / 1.45 rad. `fist_spike.py` measures a closed fist on its own: 0 stretched edges (narrow 2 cm joint bands gave 104-147, because the AI-made hand is not cut into finger segments).

## What went wrong before (so it does not happen again)

- **The backwards elbows were in the SOURCE DATA.** The CMU "Daz-friendly" BVH conversion has a whole arm flipped 180 deg in some takes (84 of 2,543 clips; in our fight set punch_143_23_a both arms, punch_13_17_a left, punch_14_03_a right, kick_front_a left, kick_front_low left). The white reference figure showed it too, Relic copied it faithfully, and every "matches the real capture" check passed. Fixed in the cutter (`anatomy.py`), plus the anatomy check and the solver limit. Lesson: **never trust the reference alone; check it against anatomy.**
- **Relic is much bulkier than the captured fighter** (shoulders 72 cm apart vs 23 cm; chest armour 22 cm in front of the spine): a guard that sat in front of the fighter's chin landed inside Relic's chest. Fixed by `body_clear.py` (fists keep the fighter's clearance from Relic's own armour) and the stance fix (his wider stance had straightened his knees by up to 20 deg).
- **The raised kicking leg was straighter than the fighter's:** the stance widening was applied to raised feet too. A raised foot is now placed from Relic's hip joint.
- **Parts came apart at the hip on high kicks:** waist plates and thigh plates were separate rigid blocks. Fixed by the whole-piece hip rule (a first try cut pieces at a sphere's edge and tore them 4.6x; the spike check caught it).
- **"Dark insides of the plates" made black flags:** the AI mesh has faces with flipped normals, so outside faces also turned black. Removed.
- **The fist curled sideways:** the fist bones turned about local X (the palm normal), swinging the fingers into each other. Now local Z.
- The finger pieces were skinned partly to the thigh (spike toward the hip on punches): `skin_rules.py`. A waist plate tore on high kicks (weights jumped 62% -> 6%): torso weights blurred.
- Older: arm offsets used the last frame's heading (bug, fixed); an elbow-direction stabilizer lagged onto the wrong side (removed); a generic limit table pushed elbows away from the real ones (replaced by the exact solver); knees measured from the human's narrow hips (now from the real hip joints); ankle range measured from Relic's own heel-up rest foot.
- **The KillDroid passed the old gate while its crusher cut through its own turret** (2026-10-17): rigid parts on bones cannot stretch, so none of the skin checks could see it. The hull check and the hull clearance were added; the first KillDroid take (boxing hook, boxing guard, kneel) is gone.
- **A first-frame-only cut:** the hip rams aim themselves at the thighs, and the hull was measured before the legs were solved, so on a clip's first frame the ram still pointed at the old thigh. Its legs are now solved first.
- **The floating-piece check failed on the KillDroid's leg cores:** they part from its pelvis core when the legs spread but stay 3 mm inside the thigh armour. The check now counts its rigid parts as pieces.
