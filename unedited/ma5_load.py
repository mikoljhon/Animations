# Loads the whole cave / Mender / animation helper chain (definitions only) into the caller's globals.
# Usage in one execute_blender_code call:   exec(open(r"C:\ai\unedited\ma5_load.py").read())
import math, os, json
import numpy as np
import bpy
from mathutils import Matrix, Quaternion, Vector

F_TOTAL_5, DROP_5 = 360, 211
_CHAIN = ["relic_cave.py", "relic_cave2.py", "relic_cave3.py", "relic_cave4.py", "relic_cave5.py", "mender_build.py", "mender_scene.py",
          "mender_pose.py", "mender_cave.py", "mender_cave2.py", "mender_cave3.py", "mender_anim.py", "mender_anim2.py", "mender_anim3.py",
          "mender_anim4.py", "mender_anim5.py", "mender_anim6.py", "mender_fx.py", "mender_anim7.py"]
for _n in _CHAIN:
    exec(compile(open(os.path.join(r"C:\ai\unedited", _n), encoding="utf-8").read(), _n, "exec"))
