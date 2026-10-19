# torso_dbg.py : at given frames, compare Relic's shoulder line / hip line / chest facing with the reference mannequin's (ref_* empties)
#   blender.exe -b pipe_<tag>.blend --python torso_dbg.py -- f1 f2 ...
import bpy, sys, math
from mathutils import Vector
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
pb = rig.pose.bones
E = lambda n: bpy.data.objects["ref_" + n].matrix_world.translation.copy()
W = lambda n, t=False: rig.matrix_world @ (pb[n].tail if t else pb[n].head)
a = sys.argv.index("--") + 1
for f in [int(x) for x in sys.argv[a:]]:
    sc.frame_set(f)
    sh_r, sh_m = W('upperarm.R') - W('upperarm.L'), E('shoulder_R') - E('shoulder_L')
    hp_r, hp_m = W('thigh.R') - W('thigh.L'), E('hip_R') - E('hip_L')
    up_r, up_m = W('neck') - W('pelvis'), E('neck') - E('hips')
    yaw = lambda v: math.degrees(math.atan2(v.y, v.x))
    # chest bone axes in world
    cm = (rig.matrix_world @ pb['chest'].matrix).to_3x3()
    print("TORSO f%d  shoulders rig yaw %.0f real yaw %.0f (3D angle %.0f) | hips rig %.0f real %.0f (%.0f) | spine angle %.0f" % (
        f, yaw(sh_r), yaw(sh_m), math.degrees(sh_r.angle(sh_m)), yaw(hp_r), yaw(hp_m), math.degrees(hp_r.angle(hp_m)), math.degrees(up_r.angle(up_m))))
    print("   rig shoulder L %s R %s | real shoulder L %s R %s" % (tuple(round(x, 2) for x in W('upperarm.L')), tuple(round(x, 2) for x in W('upperarm.R')),
                                                                   tuple(round(x, 2) for x in E('shoulder_L')), tuple(round(x, 2) for x in E('shoulder_R'))))
    print("   chest axes x %s y %s z %s" % (tuple(round(x, 2) for x in cm.col[0]), tuple(round(x, 2) for x in cm.col[1]), tuple(round(x, 2) for x in cm.col[2])))
    for n in ('pelvis', 'spine', 'chest', 'neck', 'head', 'clavicle.L', 'clavicle.R', 'upperarm.L', 'upperarm.R'):
        q = pb[n].rotation_quaternion
        ax, ang = q.to_axis_angle()
        print("   %-11s local rot %4.0f deg about %s" % (n, math.degrees(ang), tuple(round(x, 2) for x in ax)))
