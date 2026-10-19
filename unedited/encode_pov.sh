#!/bin/bash
FF="/c/Users/mike/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
DIR="/c/ai/unedited/renders/pov_anim"
OUT="/c/ai/unedited/renders/UNEDITED_ep1_opening_POV_test.mp4"
AUDIO="/c/ai/unedited/audio/ep1_opening_sfx_v03.wav"
until grep -q "Blender quit" "$DIR/render_log.txt" 2>/dev/null; do sleep 5; done
n=$(ls "$DIR" | grep -c '^f_')
echo "render finished: $n frames"
"$FF" -y -v error -framerate 24 -i "$DIR/f_%04d.png" -i "$AUDIO" -af "lowpass=f=450:enable='lt(t,8.6)',afade=t=out:st=15.0:d=0.6" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium -c:a aac -b:a 192k -shortest "$OUT" && echo "encoded: $OUT"
ls -la "$OUT"
