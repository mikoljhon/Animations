#!/usr/bin/env bash
# Copy the trained AutoPose models into the add-on and build the installable extension zip into dist/.
cd "$(dirname "$0")"
mkdir -p motion_assist/models dist
for f in autopose.npz autopose_feet.npz; do
  [ -f /c/ai/mocap/models/$f ] && cp /c/ai/mocap/models/$f motion_assist/models/
done
rm -rf motion_assist/__pycache__
"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" --factory-startup --command extension build \
  --source-dir "$PWD/motion_assist" --output-dir "$PWD/dist" 2>&1 | tail -n 2
