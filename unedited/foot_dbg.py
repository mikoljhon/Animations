import bpy, math
from mathutils import Vector
sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]
for f in (4, 30, 60):
    sc.frame_set(f)
    for s in "LR":
        W = lambda n, t=False: rig.matrix_world @ (rig.pose.bones[n].tail if t else rig.pose.bones[n].head)
        sh_dir = (W("shin."+s, True) - W("shin."+s)).normalized()
        th_dir = (W("shin."+s) - W("thigh."+s)).normalized()
        ft_dir = (W("foot."+s, True) - W("foot."+s)).normalized()
        yaw_ft = math.degrees(math.atan2(-ft_dir.x, ft_dir.y))
        print("FOOT", f, s, "shin tilt from vertical", round(math.degrees(sh_dir.angle(Vector((0,0,-1))))), "thigh tilt", round(math.degrees(th_dir.angle(Vector((0,0,-1))))), "foot axis pitch below horizontal", round(math.degrees(math.asin(-ft_dir.z))), "foot yaw world", round(yaw_ft))
        r_sh = (rig.matrix_world @ rig.pose.bones["shin."+s].matrix).to_3x3()
        r_ft = (rig.matrix_world @ rig.pose.bones["foot."+s].matrix).to_3x3()
        rel = (r_sh.inverted() @ r_ft).to_euler()
        rest = (rig.data.bones["shin."+s].matrix_local.to_3x3().inverted() @ rig.data.bones["foot."+s].matrix_local.to_3x3()).to_euler()
        print("  rel euler now", [round(math.degrees(x)) for x in rel], "rest rel", [round(math.degrees(x)) for x in rest])
