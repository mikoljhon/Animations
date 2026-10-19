# Fight 1 reference images: arena plates with StarCapital floodlight masts and camera drones (R3),
# and a neutral-grey turnaround of Relic (R1).
# Run:  blender.exe -b fight_v07.blend --python fight1_refs.py -- [shot,shot,...|all] [res%]
# fight_v07.blend is never saved; the arena with its props is saved as a copy to arena_v02.blend.
import bpy, bmesh, math, os, sys, time
from mathutils import Vector, Matrix

OUT = r"C:\ai\unedited"
REF = os.path.join(OUT, "refs", "fight1")
os.makedirs(REF, exist_ok=True)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
WANT = set((argv[0] if argv else "all").split(","))
PCT = int(argv[1]) if len(argv) > 1 else 100


def want(name):
    return "all" in WANT or name in WANT


sc = bpy.data.scenes["Street"]
rig = bpy.data.objects["Relic_AI_Rig"]


# ------------------------------------------------------------------ helpers
def mat(name, color, metallic=0.0, rough=0.5, emit=None, strength=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1.0)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    if emit is not None:
        p.inputs["Emission Color"].default_value = (*emit, 1.0)
        p.inputs["Emission Strength"].default_value = strength
    return m


def mesh_from_bm(name, bm, coll, material, loc=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    coll.objects.link(ob)
    me.materials.append(material)
    return ob


def bm_box(bm, c, s, rot=None):
    m = Matrix.Translation(Vector(c))
    if rot is not None:
        m = m @ rot
    bmesh.ops.create_cube(bm, size=1.0, matrix=m @ Matrix.Diagonal((*s, 1.0)))


def bm_cyl(bm, c, r, h, rot=None, segs=16):
    m = Matrix.Translation(Vector(c))
    if rot is not None:
        m = m @ rot
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r, depth=h, matrix=m)


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


# ------------------------------------------------------------------ Relic: rest pose, clean look
def prep_relic():
    ad = rig.animation_data
    if ad:
        ad.action = None
        for t in ad.nla_tracks:
            t.mute = True
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    crust = bpy.data.objects.get("Relic_AI_Crust")
    if crust:
        crust.hide_render = True
        crust.hide_viewport = True
    # power on, as relic_aged.ai_power(True): Tally's chest ring and the amber eyes glow
    for name, strength in (("Relic_RingLight", 3.0), ("Relic_Glyph", 2.8)):
        m = bpy.data.materials.get(name)
        if m and m.node_tree:
            b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            b.inputs['Emission Strength'].default_value = strength


def clear_ring():
    """StarCapital clears the rubble out of the fight ring: hide the boulders in front of the bridge"""
    for o in bpy.data.objects:
        if o.name.startswith("Boulder") and abs(o.location.x) < 10.0 and 0.5 < o.location.y < 22.0:
            o.hide_render = True
    # the low-poly moss blobs read as green crystals to an image model, so leave them out of the references
    mb = bpy.data.objects.get("Street_MossBlobs")
    if mb:
        mb.hide_render = True


def place_relic(loc, yaw_deg, show=True):
    if rig.animation_data:
        rig.animation_data.action = None
    rig.location = loc
    rig.rotation_mode = 'XYZ'
    rig.rotation_euler = (0, 0, math.radians(yaw_deg))
    for o in bpy.data.objects:
        if o.parent == rig or o == rig:
            if o.name != "Relic_AI_Crust":
                o.hide_render = not show


def hide_killdroid():
    for o in bpy.data.objects:
        if o.name.startswith("KillDroid") or o.name.startswith("KD_"):
            o.hide_render = True
    kc = bpy.data.collections.get("KillDroid")
    if kc:
        for o in kc.all_objects:
            o.hide_render = True


# ------------------------------------------------------------------ arena props
FLOOD_SPOTS = []
LAMP_ON = mat("Arena_LampOn", (0.9, 0.95, 1.0), 0.0, 0.2, emit=(0.86, 0.93, 1.0), strength=90.0)
LAMP_OFF = mat("Arena_LampOff", (0.03, 0.035, 0.04), 0.2, 0.15)
STEEL = mat("Arena_Steel", (0.07, 0.075, 0.08), 0.9, 0.45)
DRONE = mat("Arena_DroneBody", (0.035, 0.037, 0.04), 0.85, 0.35)
GLASS = mat("Arena_DroneLens", (0.005, 0.006, 0.008), 0.0, 0.05)
RED = mat("Arena_DroneRed", (1.0, 0.05, 0.03), 0.0, 0.3, emit=(1.0, 0.04, 0.02), strength=40.0)


def build_mast(coll, x, y, h=20.0):
    """a steel mast with a floodlight head of 3 x 5 lamps, turned toward the road centre and tilted down"""
    bm = bmesh.new()
    bm_cyl(bm, (0, 0, h / 2), 0.28, h, segs=12)
    bm_box(bm, (0, 0, 0.4), (1.4, 1.4, 0.8))                       # footing
    for k in range(10):                                              # ladder rungs on one side
        bm_box(bm, (0.3, 0, 1.5 + k * 1.5), (0.06, 0.5, 0.05))
    mast = mesh_from_bm("Arena_Mast_%d_%d" % (int(x), int(y)), bm, coll, STEEL, loc=(x, y, 0))
    # the head: a frame facing the road centre, pitched down 35 degrees
    target = Vector((0.0, y + 2.0, 0.0))
    head_loc = Vector((x, y, h + 0.6))
    d = (target - head_loc).normalized()
    rot = d.to_track_quat('Y', 'Z').to_matrix().to_4x4()            # local +Y faces the road
    bm = bmesh.new()
    bm_box(bm, (0, -0.2, 0), (4.6, 0.3, 2.8))                        # back plate
    bm_box(bm, (0, -0.45, -1.6), (0.5, 0.5, 0.6))                    # bracket onto the mast
    hb = mesh_from_bm("Arena_Head_%d_%d" % (int(x), int(y)), bm, coll, STEEL)
    hb.matrix_world = Matrix.Translation(head_loc) @ rot
    bm = bmesh.new()
    for i in range(5):
        for j in range(3):
            bm_cyl(bm, (-1.8 + i * 0.9, 0.05, -0.85 + j * 0.85), 0.36, 0.14,
                   rot=Matrix.Rotation(math.radians(90), 4, 'X'), segs=20)
    lamps = mesh_from_bm("Arena_Lamps_%d_%d" % (int(x), int(y)), bm, coll, LAMP_ON)
    lamps.matrix_world = Matrix.Translation(head_loc) @ rot
    # the light itself
    ld = bpy.data.lights.new("Arena_Flood_%d_%d" % (int(x), int(y)), 'SPOT')
    ld.energy = 60000.0
    ld.color = (0.86, 0.93, 1.0)
    ld.spot_size = math.radians(62)
    ld.spot_blend = 0.35
    ld.shadow_soft_size = 0.8
    lo = bpy.data.objects.new(ld.name, ld)
    lo.location = head_loc + d * 0.6
    lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    coll.objects.link(lo)
    FLOOD_SPOTS.append((lo, lamps))
    return mast


def build_drone(coll, name, loc, yaw=0.0, tilt=0.0):
    bm = bmesh.new()
    bm_box(bm, (0, 0, 0), (0.55, 0.42, 0.16))                         # body
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = Vector((sx * 0.42, sy * 0.36, 0.03))
            arm_rot = Matrix.Rotation(math.atan2(p.y, p.x), 4, 'Z')
            bm_box(bm, p * 0.5, (0.5, 0.05, 0.04), rot=arm_rot)       # arm
            bm_cyl(bm, p + Vector((0, 0, 0.05)), 0.05, 0.08, segs=10)  # motor
            bm_cyl(bm, p + Vector((0, 0, 0.1)), 0.21, 0.01, segs=24)   # rotor disc
    body = mesh_from_bm(name, bm, coll, DRONE)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.09,
                              matrix=Matrix.Translation((0.0, -0.18, -0.1)))
    lens = mesh_from_bm(name + "_Lens", bm, coll, GLASS)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=0.035,
                              matrix=Matrix.Translation((0.12, -0.2, -0.06)))
    light = mesh_from_bm(name + "_Red", bm, coll, RED)
    for o in (lens, light):
        o.parent = body
    body.location = loc
    body.rotation_euler = (math.radians(tilt), 0, math.radians(yaw))
    return body


def add_arena_props():
    coll = bpy.data.collections.get("Arena_Props") or bpy.data.collections.new("Arena_Props")
    if coll.name not in [c.name for c in sc.collection.children]:
        sc.collection.children.link(coll)
    for x in (-13.5, 13.5):
        for y in (3.0, 15.0, 38.0, 52.0):        # the bridge deck crosses y 22 to 30, so no masts there
            build_mast(coll, x, y)
    dcoll = bpy.data.collections.get("Arena_Drones") or bpy.data.collections.new("Arena_Drones")
    if dcoll.name not in [c.name for c in sc.collection.children]:
        sc.collection.children.link(dcoll)
    for k, (loc, yaw, tilt) in enumerate([((-4.0, 19.0, 9.0), 20, 6), ((3.0, 23.5, 11.0), -35, -4),
                                           ((8.0, 17.0, 7.5), 140, 8), ((-9.0, 27.5, 9.8), 75, -6),
                                           ((0.5, 14.0, 12.0), 190, 5), ((5.5, 31.0, 8.6), -110, 7)]):
        build_drone(dcoll, "Arena_Drone_%d" % k, loc, yaw, tilt)
    return coll, dcoll


def set_floods(on):
    for lo, lamps in FLOOD_SPOTS:
        lo.hide_render = not on
        lamps.data.materials[0] = LAMP_ON if on else LAMP_OFF


def set_drones(show):
    for o in bpy.data.collections["Arena_Drones"].all_objects:
        o.hide_render = not show


def render(scene, path, res, samples=32):
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = PCT
    try:
        scene.eevee.taa_render_samples = samples
    except Exception:
        pass
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True, scene=scene.name)
    print("RENDERED %s in %.1f s" % (path, time.time() - t))


# ------------------------------------------------------------------ arena plates (R3)
def arena_plates():
    cam = bpy.data.objects["St_Cam"]
    sc.camera = cam
    cam.data.clip_end = 1500.0
    shots = [
        # name, floods, drones, relic (loc, yaw) or None, cam loc, target, lens
        ("R3_arena_wide_high", False, False, None, (-6.0, -16.0, 15.0), (2.0, 30.0, 4.0), 24),
        ("R3_arena_lane", True, True, None, (0.5, 3.0, 1.6), (0.5, 20.0, 3.0), 28),
        ("R3_arena_lane_relic", True, True, ((-4.5, 13.0, 0.0), 90), (0.5, 3.0, 1.6), (0.5, 20.0, 3.0), 28),
        ("R3_arena_bridge_drones", True, True, None, (3.0, 16.0, 0.8), (0.0, 26.0, 13.0), 20),
        ("R3_arena_crack_lip_relic", True, True, ((0.0, 1.2, 0.0), 0), (1.0, -3.5, 1.3), (0.0, 3.0, 1.7), 30),
        ("R3_arena_crack_side_relic", True, True, ((0.0, 1.2, 0.0), 180), (7.0, 0.8, 1.2), (-3.0, 3.0, 1.5), 28),
    ]
    for name, floods, drones, relic, loc, tgt, lens in shots:
        if not want(name):
            continue
        set_floods(floods)
        set_drones(drones)
        if relic:
            place_relic(relic[0], relic[1], show=True)
        else:
            place_relic((0.0, 0.8, 0.0), 180, show=False)
        cam.location = loc
        cam.data.lens = lens
        look_at(cam, tgt)
        render(sc, os.path.join(REF, name + ".png"), (1920, 1080), samples=32)


# ------------------------------------------------------------------ Relic turnaround (R1)
def relic_studio():
    st = bpy.data.scenes.get("RefStudio") or bpy.data.scenes.new("RefStudio")
    for cname in ("Relic_AI", "Relic_Joints"):
        c = bpy.data.collections.get(cname)
        if c and c.name not in [x.name for x in st.collection.children]:
            st.collection.children.link(c)
    st.render.engine = 'BLENDER_EEVEE'
    st.view_settings.view_transform = 'AgX'
    st.view_settings.look = 'None'
    st.view_settings.exposure = 0.0
    try:
        st.eevee.use_raytracing = True
    except Exception:
        pass
    w = bpy.data.worlds.get("Ref_World") or bpy.data.worlds.new("Ref_World")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.2, 0.205, 0.21, 1.0)
    bg.inputs['Strength'].default_value = 0.6
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    st.world = w
    sc_coll = bpy.data.collections.get("Ref_Studio") or bpy.data.collections.new("Ref_Studio")
    if sc_coll.name not in [x.name for x in st.collection.children]:
        st.collection.children.link(sc_coll)
    for o in list(sc_coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bm = bmesh.new()
    s = 60.0
    bm.faces.new([bm.verts.new(v) for v in [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)]])
    mesh_from_bm("Ref_Floor", bm, sc_coll, mat("Ref_FloorMat", (0.16, 0.165, 0.17), 0.0, 0.7))

    def area(name, loc, energy, size, color):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy, ld.size, ld.color = energy, size, color
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        sc_coll.objects.link(lo)
        look_at(lo, (0, 0, 1.1))

    area("Ref_Key", (-3.0, -3.6, 3.8), 1400, 3.0, (1.0, 0.96, 0.9))
    area("Ref_Fill", (3.6, -2.8, 2.0), 500, 3.5, (0.9, 0.95, 1.0))
    area("Ref_Rim", (1.5, 3.6, 3.2), 900, 2.0, (1.0, 1.0, 1.0))
    area("Ref_Rim2", (-2.6, 3.0, 2.4), 500, 2.0, (1.0, 0.95, 0.9))
    cd = bpy.data.cameras.get("Ref_Cam") or bpy.data.cameras.new("Ref_Cam")
    cam = bpy.data.objects.new("Ref_Cam", cd)
    sc_coll.objects.link(cam)
    st.camera = cam
    place_relic((0.0, 0.0, 0.0), 0, show=True)          # rest pose faces -Y, his left is +X
    r = 4.7
    views = [
        ("R1_relic_front", (0.0, -r, 1.05), (0, 0, 1.05), 70, (1024, 1536)),
        ("R1_relic_q3_front", (r * math.sin(math.radians(35)), -r * math.cos(math.radians(35)), 1.05), (0, 0, 1.05), 70, (1024, 1536)),
        ("R1_relic_side", (r, 0.0, 1.05), (0, 0, 1.05), 70, (1024, 1536)),
        ("R1_relic_back", (0.0, r, 1.05), (0, 0, 1.05), 70, (1024, 1536)),
        ("R1_relic_q3_back", (-r * math.sin(math.radians(35)), r * math.cos(math.radians(35)), 1.05), (0, 0, 1.05), 70, (1024, 1536)),
        ("R1_relic_head", (0.75, -1.55, 1.95), (0.0, -0.05, 1.82), 85, (1024, 1024)),
        ("R1_relic_hero", (2.1, -3.0, 0.45), (0.0, 0.0, 1.2), 35, (1024, 1536)),
    ]
    for name, loc, tgt, lens, res in views:
        if not want(name):
            continue
        cam.location = loc
        cd.lens = lens
        cd.clip_end = 200.0
        look_at(cam, tgt)
        render(st, os.path.join(REF, name + ".png"), res, samples=32)


# ------------------------------------------------------------------ run
prep_relic()
hide_killdroid()
clear_ring()
add_arena_props()
arena_plates()
if want("save"):
    place_relic((-7.0, 24.0, 0.0), 90, show=True)
    set_floods(True)
    set_drones(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "arena_v02.blend"), copy=True)
    print("SAVED arena_v02.blend")
relic_studio()
print("DONE")
