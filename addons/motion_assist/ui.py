# Motion Assist: settings, operators and the sidebar panel (3D View > Sidebar > Motion).

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import analysis, autopose, fixes, grip, overlap, trail
from . import body as bodymod


def _is_armature(self, ob):
    return ob.type == 'ARMATURE'


class MA_Settings(bpy.types.PropertyGroup):
    target: PointerProperty(name="Character", type=bpy.types.Object, poll=_is_armature, description="The rig to check and fix")
    total_mass: FloatProperty(name="Mass", default=75.0, min=1.0, soft_max=500.0, description="Total mass in kilograms")
    gear_density: FloatProperty(name="Gear Density", default=0.35, min=0.0, soft_max=5.0,
                                description="Density of meshes parented to bones (gear, props), relative to the skinned body")
    ignore: StringProperty(name="Ignore", default="", description="Comma-separated parts of object names to leave out of the body and the ground")
    use_scene_range: BoolProperty(name="Scene Range", default=True)
    frame_start: IntProperty(name="Start", default=1)
    frame_end: IntProperty(name="End", default=100)
    ground: EnumProperty(name="Ground", default='SCENE', items=[
        ('SCENE', "Scene Geometry", "Contacts with any visible mesh near the character"),
        ('PLANE', "Flat Floor", "A flat floor at a fixed height")])
    ground_height: FloatProperty(name="Floor Height", default=0.0, subtype='DISTANCE', unit='LENGTH')
    contact_radius: FloatProperty(name="Contact Distance", default=0.03, min=0.0, soft_max=0.2, subtype='DISTANCE', unit='LENGTH',
                                  description="Points this close to the ground can count as contacts")
    max_speed: FloatProperty(name="Contact Speed", default=0.5, min=0.0, soft_max=3.0,
                             description="Points near the ground moving slower than this (m/s) count as contacts")
    friction: FloatProperty(name="Friction", default=0.8, min=0.0, soft_max=2.0)
    gravity: FloatProperty(name="Gravity", default=9.81, min=0.0)
    zmp_tolerance: FloatProperty(name="Balance Tolerance", default=0.02, min=0.0, subtype='DISTANCE', unit='LENGTH')
    comp_frames: IntProperty(name="Push/Land Frames", default=4, min=0, max=12,
                             description="Frames before take-off and after landing where the hips are adjusted to match the flight")
    fix_spin: BoolProperty(name="Fix Spin", default=True, description="Keep angular momentum constant in the air")
    lean_land: BoolProperty(name="Lean & Land", default=True,
                            description="Lean into the push before take-off and reach the feet ahead on landing, as far as the momentum needs")
    lock_blend: IntProperty(name="Release Frames", default=2, min=0, max=10)
    overlap_freq: FloatProperty(name="Stiffness", default=2.5, min=0.1, soft_max=10.0, description="Spring frequency in Hz (lower is floppier)")
    overlap_damp: FloatProperty(name="Damping", default=0.45, min=0.0, soft_max=2.0)
    overlap_resp: FloatProperty(name="Response", default=0.0, soft_min=-2.0, soft_max=2.0,
                                description="Above 0 anticipates, below 0 lags at the start of a move")
    overlap_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0)
    follow_amount: FloatProperty(name="Amount", default=1.0, min=0.0, max=2.0,
                                 description="How loose the follow-through is (1 = natural)")
    use_limits: BoolProperty(name="Joint Limits", default=True,
                             description="Keep joints inside human ranges (hinged knees and elbows, limited twists) in AutoPose and Overlap")
    hold_left: PointerProperty(name="Left Hand Holds", type=bpy.types.Object,
                               description="A prop the left hand holds (a pick, a tool, a rod): Fix Joints keeps the hand on it and finds "
                                           "the roll and place on the shaft that keep the wrist natural")
    hold_right: PointerProperty(name="Right Hand Holds", type=bpy.types.Object,
                                description="A prop the right hand holds: Fix Joints keeps the hand on it and finds the roll and place on "
                                            "the shaft that keep the wrist natural")
    autopose_live: BoolProperty(name="Live", default=False, description="Re-pose the body whenever a goal moves")
    autopose_pin_pelvis: BoolProperty(name="Keep Hips Rotation", default=False)
    autopose_auto_hips: BoolProperty(name="Auto Hips", default=False,
                                     description="Let the network place the hips and upper body too: only the hand and foot goals count (reach low and the body crouches)")
    autopose_key: BoolProperty(name="Key", default=False, description="Insert keys for the new pose on the current frame")
    show_overlay: BoolProperty(name="Overlay", default=True)
    show_path: BoolProperty(name="Path", default=True)
    show_support: BoolProperty(name="Support", default=True)
    show_zmp: BoolProperty(name="Balance Point", default=True)
    show_contacts: BoolProperty(name="Contacts", default=True)


def _rig(context):
    s = context.scene.motion_assist
    if s.target is None and context.object is not None and context.object.type == 'ARMATURE':
        s.target = context.object
    return s.target


class _RigOp:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.motion_assist
        return s.target is not None or (context.object is not None and context.object.type == 'ARMATURE')


class MA_OT_build_body(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.build_body"
    bl_label = "Build Body"
    bl_description = "Measure the character's meshes: mass, centre of mass and inertia of every bone, and contact points"

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        try:
            r = bodymod.build_body(rig, s.total_mass, bodymod.parse_ignore(s.ignore), s.gear_density)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "%d bones with mass, %d contact points" % (r["bones"], r["probes"]))
        return {'FINISHED'}


def _analyze(op, context):
    s = context.scene.motion_assist
    rig = _rig(context)
    if bodymod.get_body(rig) is None:
        bodymod.build_body(rig, s.total_mass, bodymod.parse_ignore(s.ignore), s.gear_density)
    return analysis.analyze(context.scene, rig, s)


class MA_OT_analyze(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.analyze"
    bl_label = "Analyze Motion"
    bl_description = "Check the animation: centre of mass, contacts, jumps, balance"
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            d = _analyze(self, context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, analysis.summary(d))
        return {'FINISHED'}


class MA_OT_ballistic(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.ballistic_fix"
    bl_label = "Fix Jumps"
    bl_description = "Make every jump obey gravity: parabolic flight, constant spin, matching push-off and landing"

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        try:
            _analyze(self, context)
            notes, _d = fixes.fix_jumps(context.scene, rig, s, lean=s.lean_land)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "; ".join(notes))
        return {'FINISHED'}


class MA_OT_foot_lock(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.foot_lock"
    bl_label = "Lock Contacts"
    bl_description = "Stop sliding: points touching the ground stay where they first touched (IK targets are keyed)"

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        try:
            d = _analyze(self, context)
            notes = fixes.foot_lock(context.scene, rig, s, d)
            analysis.analyze(context.scene, rig, s)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "; ".join(notes) or "no IK chains with contacts")
        return {'FINISHED'}


class MA_OT_overlap(bpy.types.Operator):
    bl_idname = "motion_assist.bake_overlap"
    bl_label = "Bake Overlap"
    bl_description = "Add follow-through to the selected bones (they lag, overshoot and settle); keys every frame of the range"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'POSE' and bool(context.selected_pose_bones)

    def execute(self, context):
        s = context.scene.motion_assist
        rig = context.object
        f0, f1 = analysis.frame_range(context.scene, s)
        names = [pb.name for pb in context.selected_pose_bones if pb.id_data == rig]
        try:
            r = overlap.bake_overlap(context.scene, rig, names, f0, f1, s.overlap_freq, s.overlap_damp, s.overlap_resp, s.overlap_strength,
                                     use_limits=s.use_limits)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        msg = "overlap on %d bones" % len(r["bones"])
        if r["skipped"]:
            msg += " (skipped %s: constrained)" % ", ".join(r["skipped"])
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MA_OT_follow(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.follow_through"
    bl_label = "Follow-Through"
    bl_description = ("Secondary motion everywhere: spine, shoulders, head and arms lag and settle, feet keep travelling "
                      "with the push while in the air, gear marked with the ma_gear property swings (keys every frame)")

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        try:
            d = _analyze(self, context)
            r = overlap.follow_through(context.scene, rig, d, s.follow_amount, s.use_limits)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "follow-through on %d bones, %d legs, %d gear groups" % (len(r["bones"]), len(r["feet"]), len(r["gear"])))
        return {'FINISHED'}


class MA_OT_trail(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.com_trail"
    bl_label = "Path Objects"
    bl_description = "Make objects that show the centre of mass in renders: a tube along its path and a ball riding it"

    def execute(self, context):
        rig = _rig(context)
        d = analysis.CACHE.get(rig.name)
        if d is None:
            self.report({'ERROR'}, "analyze first")
            return {'CANCELLED'}
        trail.make_trail(context.scene, rig, d["com"], d["f0"], "path")
        if d.get("com_before") is not None:
            trail.make_trail(context.scene, rig, d["com_before"], d["f0"], "before", (0.55, 0.55, 0.6), ball=False)
        return {'FINISHED'}


class MA_OT_autopose(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.autopose"
    bl_label = "Auto Pose"
    bl_description = ("Keep the hips, neck, hands and feet where they are (or at their goals) and let the network pose "
                      "the rest of the body naturally")

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        try:
            goals = autopose.read_goals(rig)
            autopose.solve(rig, goals, keep=() if goals else autopose.CORE, pin_pelvis=s.autopose_pin_pelvis, key=s.autopose_key,
                           auto_hips=s.autopose_auto_hips, use_limits=s.use_limits)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class MA_OT_add_goals(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.add_goals"
    bl_label = "Add Goals"
    bl_description = "Add an empty at the hips, neck, both hands and both feet: move them, then Auto Pose (or turn on Live)"

    def execute(self, context):
        try:
            autopose.add_goals(_rig(context))
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class MA_OT_remove_goals(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.remove_goals"
    bl_label = "Remove Goals"

    def execute(self, context):
        autopose.remove_goals(_rig(context))
        return {'FINISHED'}


class MA_OT_check_joints(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.check_joints"
    bl_label = "Check Joints"
    bl_description = "Find frames where a joint bends or twists further than a human joint can"
    bl_options = {'REGISTER'}

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        f0, f1 = analysis.frame_range(context.scene, s)
        try:
            bad = autopose.check_animation(context.scene, rig, f0, f1)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if not bad:
            rig["ma_joints"] = "all joints inside their limits"
        else:
            worst = sorted(bad.items(), key=lambda kv: -kv[1][1])[:4]
            rig["ma_joints"] = "; ".join("%s %d frames (up to %.0f deg, f%d)" % (b, n, w, f) for b, (n, w, f) in worst)
        self.report({'INFO'}, str(rig["ma_joints"]))
        return {'FINISHED'}


class MA_OT_fix_joints(_RigOp, bpy.types.Operator):
    bl_idname = "motion_assist.fix_joints"
    bl_label = "Fix Joints"
    bl_description = ("Bring every frame back inside human joint limits: knees point the way knees bend, feet stand on their "
                      "soles, hands stay where they are (keys every frame of the range). A hand set to hold a prop stays on "
                      "it, with the roll and place on the shaft that keep the wrist natural")

    def execute(self, context):
        s = context.scene.motion_assist
        rig = _rig(context)
        f0, f1 = analysis.frame_range(context.scene, s)
        props = {side: ob for side, ob in (("L", s.hold_left), ("R", s.hold_right)) if ob is not None}
        try:
            holds = leans = None
            if props:
                holds, leans = grip.plan(context.scene, rig, f0, f1, props)
            moved = autopose.fix_animation(context.scene, rig, f0, f1, holds=holds, leans=leans)
            bad = autopose.check_animation(context.scene, rig, f0, f1)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        rig["ma_joints"] = ("all joints inside their limits" if not bad else
                            "; ".join("%s %d frames (up to %.0f deg)" % (b, n, w) for b, (n, w, _f) in
                                      sorted(bad.items(), key=lambda kv: -kv[1][1])[:4]))
        self.report({'INFO'}, "fixed %d joints; %s" % (len(moved), rig["ma_joints"]))
        return {'FINISHED'}


class MA_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Motion"
    bl_label = "Motion Assist"

    def draw(self, context):
        s = context.scene.motion_assist
        lay = self.layout
        lay.prop(s, "target")
        rig = s.target
        box = lay.box()
        box.label(text="Body", icon='ARMATURE_DATA')
        row = box.row(align=True)
        row.prop(s, "total_mass")
        row.prop(s, "gear_density", text="Gear")
        box.prop(s, "ignore")
        b = bodymod.get_body(rig) if rig else None
        box.operator("motion_assist.build_body", icon='MOD_PHYSICS')
        if b is not None:
            box.label(text="%d bones with mass, %d contact points" % (len(b.names), len(b.plocal)))

        box = lay.box()
        box.label(text="Check", icon='VIEWZOOM')
        box.prop(s, "use_scene_range")
        if not s.use_scene_range:
            row = box.row(align=True)
            row.prop(s, "frame_start")
            row.prop(s, "frame_end")
        box.prop(s, "ground")
        if s.ground == 'PLANE':
            box.prop(s, "ground_height")
        row = box.row(align=True)
        row.prop(s, "contact_radius", text="Distance")
        row.prop(s, "max_speed", text="Speed")
        box.operator("motion_assist.analyze", icon='PLAY')
        if rig is not None and rig.get("ma_summary"):
            col = box.column(align=True)
            for part in str(rig["ma_summary"]).split(" | "):
                col.label(text=part)

        box = lay.box()
        box.label(text="Fix", icon='MODIFIER')
        row = box.row(align=True)
        row.prop(s, "comp_frames")
        row.prop(s, "fix_spin")
        box.prop(s, "lean_land")
        box.operator("motion_assist.ballistic_fix", icon='FORCE_FORCE')
        box.prop(s, "lock_blend")
        box.operator("motion_assist.foot_lock", icon='PINNED')

        box = lay.box()
        box.label(text="Joint Limits", icon='CON_ROTLIMIT')
        box.prop(s, "use_limits")
        col = box.column(align=True)
        col.prop(s, "hold_left", text="Left Holds")
        col.prop(s, "hold_right", text="Right Holds")
        row = box.row(align=True)
        row.operator("motion_assist.check_joints", icon='VIEWZOOM')
        row.operator("motion_assist.fix_joints", icon='CHECKMARK')
        if rig is not None and rig.get("ma_joints"):
            box.label(text=str(rig["ma_joints"])[:90])

        box = lay.box()
        box.label(text="AutoPose (AI)", icon='OUTLINER_OB_ARMATURE')
        row = box.row(align=True)
        row.operator("motion_assist.add_goals", icon='EMPTY_AXIS')
        row.operator("motion_assist.remove_goals", text="", icon='X')
        row = box.row(align=True)
        row.prop(s, "autopose_live", toggle=True)
        row.prop(s, "autopose_key", toggle=True)
        row = box.row(align=True)
        row.prop(s, "autopose_pin_pelvis")
        sub = row.row()
        sub.enabled = autopose.has_model("feet")
        sub.prop(s, "autopose_auto_hips")
        box.operator("motion_assist.autopose", icon='POSE_HLT')

        box = lay.box()
        box.label(text="Follow-Through", icon='FORCE_HARMONIC')
        row = box.row(align=True)
        row.prop(s, "follow_amount")
        row.operator("motion_assist.follow_through", text="Whole Body")
        box.label(text="Selected bones only:")
        row = box.row(align=True)
        row.prop(s, "overlap_freq")
        row.prop(s, "overlap_damp")
        row = box.row(align=True)
        row.prop(s, "overlap_resp")
        row.prop(s, "overlap_strength")
        box.operator("motion_assist.bake_overlap", icon='SHAPEKEY_DATA')

        box = lay.box()
        row = box.row(align=True)
        row.prop(s, "show_overlay")
        row.prop(s, "show_path")
        row = box.row(align=True)
        row.prop(s, "show_support")
        row.prop(s, "show_zmp")
        row.prop(s, "show_contacts")
        box.operator("motion_assist.com_trail", icon='CURVE_PATH')


classes = (MA_Settings, MA_OT_build_body, MA_OT_analyze, MA_OT_ballistic, MA_OT_foot_lock, MA_OT_overlap, MA_OT_follow, MA_OT_trail,
           MA_OT_autopose, MA_OT_add_goals, MA_OT_remove_goals, MA_OT_check_joints, MA_OT_fix_joints, MA_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.motion_assist = PointerProperty(type=MA_Settings)


def unregister():
    del bpy.types.Scene.motion_assist
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
