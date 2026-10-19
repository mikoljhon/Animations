#!/usr/bin/env bash
# Side-by-side AutoPose demo: plain IK (left) vs AutoPose (right), from renders\ma_sweep_ik and renders\ma_sweep_ai,
# plus the four-pose comparison grid from renders\ma_autopose.
cd /c/ai/unedited/renders
FF="/c/Users/mike/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe"
B="C\\:/Windows/Fonts/arialbd.ttf"
R="C\\:/Windows/Fonts/arial.ttf"
"$FF" -y -loglevel error -framerate 24 -i ma_sweep_ik/f_%04d.png -framerate 24 -i ma_sweep_ai/f_%04d.png -filter_complex "\
[0:v]drawtext=fontfile='$B':text='PLAIN IK':x=28:y=24:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=10,\
drawtext=fontfile='$R':text='hand and feet only, the body stays stiff':x=28:y=82:fontsize=22:fontcolor=0xd0d4dc:box=1:boxcolor=black@0.35:boxborderw=8[l];\
[1:v]drawtext=fontfile='$B':text='AUTOPOSE (AI)':x=28:y=24:fontsize=34:fontcolor=0x7cc8ff:box=1:boxcolor=black@0.45:boxborderw=10,\
drawtext=fontfile='$R':text='same goals, the network poses the whole body':x=28:y=82:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=8[r];\
[l][r]hstack=inputs=2,format=yuv420p[v]" -map "[v]" -c:v libx264 -crf 18 -preset slow -movflags +faststart UNEDITED_autopose_sweep_demo.mp4
cd ma_autopose
"$FF" -y -loglevel error -i 0_ik.png -i 1_ik.png -i 2_ik.png -i 3_ik.png -i 0_ai.png -i 1_ai.png -i 2_ai.png -i 3_ai.png -filter_complex "\
[0][1][2][3]hstack=inputs=4[top];[4][5][6][7]hstack=inputs=4[bot];[top][bot]vstack,scale=1920:-1,\
drawtext=fontfile='$B':text='PLAIN IK':x=16:y=14:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=8,\
drawtext=fontfile='$B':text='AUTOPOSE (AI)':x=16:y=554:fontsize=28:fontcolor=0x7cc8ff:box=1:boxcolor=black@0.45:boxborderw=8,\
drawtext=fontfile='$R':text='reach up':x=16:y=500:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.4:boxborderw=6,\
drawtext=fontfile='$R':text='lift a box':x=496:y=500:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.4:boxborderw=6,\
drawtext=fontfile='$R':text='lunge punch':x=976:y=500:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.4:boxborderw=6,\
drawtext=fontfile='$R':text='climb':x=1456:y=500:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.4:boxborderw=6,\
drawtext=fontfile='$R':text='blue balls = the only goals given (hips, hands, feet)':x=(w-tw)/2:y=h-40:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=8" ../UNEDITED_autopose_compare.png
ls -la ../UNEDITED_autopose_sweep_demo.mp4 ../UNEDITED_autopose_compare.png
