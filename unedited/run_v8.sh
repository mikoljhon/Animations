#!/usr/bin/env bash
# v8: fists kept outside Relic's body, knees bend like the fighter's (stance), raised feet from the hip, whole-piece hip skin, real fist curl
cd /c/ai/unedited
P="powershell -NoProfile -ExecutionPolicy Bypass -File C:/ai/unedited/run_py.ps1"
$P -blend fight_v06.blend -script pipe_test.py -log anim_cache5/pipe_punches8.txt -a "relic punches8 punch_144_20_a punch_144_13_a punch_143_23_a punch_14_03_a punch_13_17_a guard_76_04"
grep -E "^QA|\[XX\]" anim_cache5/pipe_punches8.txt
[ -f pipe_kicks8.blend ] || $P -blend fight_v06.blend -script pipe_test.py -log anim_cache5/pipe_kicks8.txt -a "relic kicks8 kick_front_a kick_front_low kick_low_round kick_round_b kick_round_c"
grep -E "^QA|\[XX\]" anim_cache5/pipe_kicks8.txt
echo TAKES_DONE
for t in "punches8 arm" "kicks8 leg"; do tag=${t%% *}
  $P -blend pipe_$tag.blend -script strike_views.py -log anim_cache5/sv_$tag.txt -a "$t"; python strike_sheet.py $tag
done
echo STRIKES_DONE
for tag in punches8 kicks8; do
  $P -blend pipe_$tag.blend -script reel_vid.py -log anim_cache5/reel_$tag.txt -a "$tag"; grep -E "REEL DONE|Error" anim_cache5/reel_$tag.txt
done
python reel_encode.py punches8 renders/UNEDITED_fight_punches_v8.mp4
python reel_encode.py kicks8 renders/UNEDITED_fight_kicks_v8.mp4
echo REELS8_DONE
bash qa_clips.sh relic > anim_cache5/qa_clips_run8.log 2>&1
tail -3 anim_cache5/qa_clips_run8.log
echo ALL8_DONE
