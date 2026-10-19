# diagnostic: mean brightness of the first-person street view with and without the fog box
import bpy, math, numpy as np
from mathutils import Vector

sc = bpy.context.scene
cam = bpy.data.objects["St_Cam"]
sc.camera = cam
cam.location = (0.0, 0.6, 1.9)
cam.data.lens = 24
cam.rotation_euler = (Vector((0, 40, 7)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x, sc.render.resolution_y = 480, 270
sc.eevee.taa_render_samples = 8


def shoot(tag):
    sc.render.filepath = r"C:\ai\unedited\renders\diag_%s.png" % tag
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(sc.render.filepath, check_existing=False)
    px = np.array(img.pixels[:]).reshape(-1, 4)[:, :3]
    top = px[: len(px) // 3]
    print("DIAG", tag, "mean %.4f  top-third %.4f  max %.3f" % (px.mean(), top.mean(), px.max()))
    bpy.data.images.remove(img)


fog = bpy.data.objects.get("Street_Fog")
mat = bpy.data.materials["St_Fog"]
for d in (0.004, 0.0015, 0.0006):
    mat.node_tree.nodes["Volume Scatter"].inputs["Density"].default_value = d
    shoot("fog_%s" % str(d).replace(".", "p"))
fog.hide_render = True
shoot("nofog")
