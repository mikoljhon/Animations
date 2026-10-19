#!/bin/bash
# waits for the background Blender render of relic_v14_opening_cut6_pass.blend to finish, then encodes the MP4 (with the sound sketch) and a contact sheet
FF="/c/Users/mike/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
DIR="/c/ai/unedited/renders/anim_open6"
OUT="/c/ai/unedited/renders/UNEDITED_ep1_opening_animatic_cut6.mp4"
SHEET="/c/ai/unedited/renders/UNEDITED_ep1_opening_animatic_cut6_sheet.png"
AUDIO="/c/ai/unedited/audio/ep1_opening_sfx_v03.wav"
until grep -q "Blender quit" "$DIR/render_log2.txt" 2>/dev/null; do sleep 5; done
n=$(ls "$DIR" | grep -c '^f_')
echo "render finished: $n frames"
"$FF" -y -v error -framerate 24 -i "$DIR/f_%04d.png" -i "$AUDIO" -af "afade=t=out:st=15.0:d=0.6" -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium -c:a aac -b:a 192k -shortest "$OUT" && echo "encoded: $OUT"
"$FF" -y -v error -i "$DIR/f_%04d.png" -vf "select='not(mod(n\,20))',scale=426:240,tile=6x3" -frames:v 1 "$SHEET" && echo "sheet: $SHEET"
ls -la "$OUT" "$SHEET"
