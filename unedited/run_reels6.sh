#!/usr/bin/env bash
# render + encode the two comparison reels (one Blender at a time, low priority)
cd /c/ai/unedited
for tag in punches6 kicks6; do
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:/ai/unedited/run_py.ps1" -blend pipe_$tag.blend -script reel_vid.py -log anim_cache5/reel_$tag.txt -a "$tag"
  grep -E "REEL DONE|Error|Trace" anim_cache5/reel_$tag.txt
done
python reel_encode.py punches6 renders/UNEDITED_fight_punches_v6.mp4
python reel_encode.py kicks6 renders/UNEDITED_fight_kicks_v6.mp4
echo REELS6_DONE
