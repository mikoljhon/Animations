import bpy
sc = bpy.data.scenes["Street"]
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.eevee.taa_render_samples = 12
sc.camera = bpy.data.objects["POV"]
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath = r"C:\ai\unedited\renders\fight_p1\f_"
