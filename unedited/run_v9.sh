#!/usr/bin/env bash
# v9: the rebuilt model (fight_v07.blend: rigid cut armour at hips/elbows/knees/ankles, smooth shoulders, inner body)
cd /c/ai/unedited
P="powershell -NoProfile -ExecutionPolicy Bypass -File C:/ai/unedited/run_py.ps1"
[ -f pipe_punches9.blend ] && [ pipe_punches9.blend -nt fight_v07.blend ] || $P -blend fight_v07.blend -script pipe_test.py -log anim_cache5/pipe_punches9.txt -a "relic punches9 punch_144_20_a punch_144_13_a punch_143_23_a punch_14_03_a punch_13_17_a guard_76_04"
grep -E "^QA|\[XX\]" anim_cache5/pipe_punches9.txt
$P -blend fight_v07.blend -script pipe_test.py -log anim_cache5/pipe_kicks9.txt -a "relic kicks9 kick_front_a kick_front_low kick_low_round kick_round_b kick_round_c"
grep -E "^QA|\[XX\]" anim_cache5/pipe_kicks9.txt
$P -blend fight_v07.blend -script pose_test.py -log anim_cache5/posetest_v07.txt -a "v07"; python pose_sheet.py v06 v07; python pose_sheet.py v06 v07 close
echo TAKES_DONE
for t in "punches9 arm" "kicks9 leg"; do tag=${t%% *}
  $P -blend pipe_$tag.blend -script strike_views.py -log anim_cache5/sv_$tag.txt -a "$t"; python strike_sheet.py $tag
done
echo STRIKES_DONE
for tag in punches9 kicks9; do
  $P -blend pipe_$tag.blend -script reel_vid.py -log anim_cache5/reel_$tag.txt -a "$tag"; grep -E "REEL DONE|Error" anim_cache5/reel_$tag.txt
done
python reel_encode.py punches9 renders/UNEDITED_fight_punches_v9.mp4
python reel_encode.py kicks9 renders/UNEDITED_fight_kicks_v9.mp4
echo REELS9_DONE
