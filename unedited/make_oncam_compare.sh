#!/bin/bash
# on-camera before/after crops: the dig close-up (f58) and the scramble (f150), cut 3 on the left, cut 4 on the right
FF="/c/Users/mike/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
R=/c/ai/unedited/renders
FONT="C\:/Windows/Fonts/arialbd.ttf"
"$FF" -y -v error -i $R/anim_open3/f_0058.png -i $R/anim_open4/f_0058.png -i $R/anim_open3/f_0150.png -i $R/anim_open4/f_0150.png -filter_complex \
"[0:v]crop=640:480:520:0,drawtext=fontfile='$FONT':text='cut 3':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=6[a];\
[1:v]crop=640:480:520:0,drawtext=fontfile='$FONT':text='cut 4':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=0x1b6e3c@0.85:boxborderw=6[b];\
[2:v]crop=640:480:640:170,drawtext=fontfile='$FONT':text='cut 3':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=6[c];\
[3:v]crop=640:480:640:170,drawtext=fontfile='$FONT':text='cut 4':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=0x1b6e3c@0.85:boxborderw=6[d];\
[a][b]hstack[t];[c][d]hstack[u];[t][u]vstack" -frames:v 1 $R/UNEDITED_cave_pass_on_camera.png && ls -la $R/UNEDITED_cave_pass_on_camera.png
