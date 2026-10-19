#!/usr/bin/env bash
# Copy the add-on source over the installed extension (development): then dev_reload() inside Blender.
SRC="$(dirname "$0")/motion_assist"
DST="/c/Users/mike/AppData/Roaming/Blender Foundation/Blender/5.2/extensions/user_default/motion_assist"
cp "$SRC"/*.py "$DST"/
echo synced
