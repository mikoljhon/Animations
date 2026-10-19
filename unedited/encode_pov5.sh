#!/bin/bash
# waits for both renders (main frames 1..360 in pov3_anim, tail frames 349..422 with the typed log line in pov4_tail), then encodes the first-person opening with the voices
FF="/c/Users/mike/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
A="/c/ai/unedited/renders/pov3_anim"
B="/c/ai/unedited/renders/pov4_tail"
S="/c/ai/unedited/renders/pov_seq"
OUT="/c/ai/unedited/renders/UNEDITED_ep1_opening_POV_v6_elevenlabs.mp4"
until grep -q "Blender quit" "$A/render_log.txt" 2>/dev/null && grep -q "Blender quit" "$B/render_log.txt" 2>/dev/null; do sleep 10; done
rm -rf "$S"; mkdir -p "$S"
for i in $(seq 1 348); do n=$(printf "%04d" $i); cp "$A/f_$n.png" "$S/f_$n.png"; done
for i in $(seq 349 422); do n=$(printf "%04d" $i); cp "$B/f_$n.png" "$S/f_$n.png"; done
echo "frames: $(ls $S | wc -l)"
"$FF" -y -v error -framerate 24 -i "$S/f_%04d.png" -i /c/ai/unedited/audio/ep1_opening_mix_v07.wav -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium -c:a aac -b:a 192k -shortest "$OUT" && echo "encoded: $OUT"
"$FF" -y -v error -i "$S/f_%04d.png" -vf "select='not(mod(n\,24))',scale=320:180,tile=6x3" -frames:v 1 /c/ai/unedited/renders/UNEDITED_ep1_opening_POV_v6_sheet.png && echo sheet
ls -la "$OUT"
