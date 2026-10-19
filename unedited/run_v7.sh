#!/usr/bin/env bash
# rebuild + verify + render the v7 takes (repaired capture, anatomy-limited solver, joints), one Blender at a time, low priority
cd /c/ai/unedited
P="powershell -NoProfile -ExecutionPolicy Bypass -File C:/ai/unedited/run_py.ps1"
$P -blend fight_v05.blend -script pipe_test.py -log anim_cache5/pipe_kicks7.txt -a "relic kicks7 kick_front_a kick_front_low kick_low_round kick_round_b kick_round_c"
grep -E "^QA|\[XX\]" anim_cache5/pipe_kicks7.txt
for t in "punches7 arm" "kicks7 leg"; do tag=${t%% *}
  $P -blend pipe_$tag.blend -script strike_views.py -log anim_cache5/sv_$tag.txt -a "$t"; python strike_sheet.py $tag
done
echo STRIKES_DONE
for tag in punches7 kicks7; do
  $P -blend pipe_$tag.blend -script reel_vid.py -log anim_cache5/reel_$tag.txt -a "$tag"; grep -E "REEL DONE|Error" anim_cache5/reel_$tag.txt
done
python reel_encode.py punches7 renders/UNEDITED_fight_punches_v7.mp4
python reel_encode.py kicks7 renders/UNEDITED_fight_kicks_v7.mp4
echo REELS7_DONE
bash qa_clips.sh relic > anim_cache5/qa_clips_run7.log 2>&1
tail -3 anim_cache5/qa_clips_run7.log
echo ALL7_DONE
