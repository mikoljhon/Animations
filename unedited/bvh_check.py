# bvh_check.py : load a raw CMU BVH with Blender's OWN importer and measure the arms, to check prep_cmu.py's forward kinematics.
#   blender.exe -b --factory-startup --python bvh_check.py -- <file.bvh> <seq30 frame> ...   (seq30 frame k = BVH data frame 1 + 4k)
import bpy, sys, math
from mathutils import Vector
a = sys.argv.index("--") + 1
path = sys.argv[a]
ks = [int(x) for x in sys.argv[a + 1:]]
bpy.ops.import_anim.bvh(filepath=path, axis_forward='-Z', axis_up='Y', global_scale=0.01, frame_start=1, use_fps_scale=False, rotate_mode='NATIVE')
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
pb = arm.pose.bones
sc = bpy.context.scene
H = lambda n: arm.matrix_world @ pb[n].head
for k in ks:
    sc.frame_set(2 + 4 * k)
    left = H('lThigh') - H('rThigh'); left.z = 0
    if left.length < 1e-4:
        left = H('lShldr') - H('rShldr'); left.z = 0
    left.normalize()
    up = Vector((0, 0, 1))
    fwd = left.cross(up) * -1.0          # facing: up x left
    fwd = up.cross(left)
    fwd.normalize()
    row = []
    for sd in "lr":
        s, e, w = H(sd + 'Shldr'), H(sd + 'ForeArm'), H(sd + 'Hand')
        u, f = e - s, w - e
        bend = math.degrees(u.angle(f))
        row.append("%s bend %3.0f forearm fwd %5.2f elbow fwd %5.2f" % (sd.upper(), bend, f.normalized().dot(fwd), u.normalized().dot(fwd)))
    eye = 0.5 * (H('leftEye') + H('rightEye')) - H('head') if 'leftEye' in pb else Vector()
    toe = (H('lFoot') - H('lShin'))
    print("BVHCHK k%d  eyes fwd %.3f | %s" % (k, eye.dot(fwd), " | ".join(row)))
print("BVHCHK DONE")
