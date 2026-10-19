# Motion Assist (Blender add-on, v0.4)

Physics checks, joint limits and AI posing for character animation, in the spirit of Cascadeur, written from scratch for Blender 4.2+ (tested on 5.2).

## Install

Edit > Preferences > Get Extensions > the arrow menu (top right) > Install from Disk > pick `dist/motion_assist-0.4.0.zip`.
The panel is in the 3D View sidebar (press N), tab **Motion**.

## Use

1. **Character**: pick the rig. Set the total mass (body plus gear).
2. **Build Body**: measures the meshes. Every bone gets a mass, a centre of mass and an inertia, taken from the convex hulls of the skinned vertices and of meshes parented to bones. Gear counts at the "Gear" density. Give any object a custom property `ma_density` (for example 1.6 for metal parts) or `ma_ignore` to change that. Props, dust and anything else that is not ground can go in the "Ignore" box (parts of names).
3. **Analyze Motion** (scene frame range): finds the contacts with the ground and the jumps, and shows the centre of mass and its path. The path is coloured green (balanced), orange (in the air), red (off balance), yellow (would slip) or purple (pulled down). It also shows the support polygon and the balance point (magenta cross). The summary lists each jump with its keyed height and the height gravity allows for its timing.
4. **Fix Jumps**: in the air, the centre of mass follows the true gravity arc between take-off and landing, and the spin (angular momentum) stays constant. The hips get a matching push before take-off and give after landing. With **Lean & Land** on, the momentum also shapes the take-off and landing. The body leans forward over the toes so the push goes through the centre of mass, and the feet reach ahead before touchdown so the stopping force does too; the body then carries forward over them. Keys every frame of the range on the root, hips and leg IK targets.
5. **Lock Contacts**: points touching the ground stay where they touched (no foot sliding), with a smooth release. Needs IK legs.
6. **Joint Limits**:
   - Knees and elbows are hinges that bend one way only. Hips and shoulders move within a cone of natural motion (the shoulder is measured from a T-pose, so reaching overhead works), with limited twist. Wrists, ankles, spine, neck and head have human ranges too. Wrist and ankle are measured from a neutral hand and foot, and knee and elbow from straight, not from the rig's rest pose (a rest foot that points down 16 degrees would otherwise make every ankle limit wrong).
   - **Check Joints** lists the frames where a joint goes past its range.
   - **Fix Joints** re-solves every frame inside the limits. Knees track over the toes (IK knee targets are re-placed every frame), arms reach the same hands, the spine is clamped, the head keeps its facing and shares a turn it cannot make alone with the neck.
   - Feet: a planted foot keeps its toes where they are. One that no ankle could do (the boot rolled onto its side, the shin leaning over it further than an ankle bends) is turned back into range about the ball of the foot: it stands on its sole again, the heel lifts as far as the ankle needs, and the lowest point of the boot keeps its height.
   - **Left Holds / Right Holds**: set a prop (a pick, a tool, a rod) and the hand stays on it. A hand holds a shaft the same way at any roll about it and at any place along it, so Fix Joints records how each hand holds the prop, finds the roll and the place that keep the wrist natural (the real arm solver judges every candidate, smooth in time), and reaches that on every frame. Where an arm would be stretched straight the torso leans a little toward the prop. After the hand lets go, its own animation takes over again over nine frames, so bake any follow-through on the free arms before running this. Elbows can turn at most 20 degrees a frame (`autopose.ELBOW_TURN`), so an arm never flips from one frame to the next.
   - With **Joint Limits** on, AutoPose and Overlap stay inside the limits too.
7. **Follow-Through**:
   - **Whole Body** adds secondary motion everywhere. The spine, shoulders, neck, head and arms lag and settle.
   - Legs in the air carry the push into a straight leg through the toes, then fold (the heel flicks up) and reach for the ground. A spring on the knee bend that overshoots does this, and frames on the ground stay exact.
   - Gear swings like a damped pendulum from its attachment point. Mark meshes parented to bones with custom properties: `ma_gear` (group name), `ma_gear_pivot` (`top` or `bottom`), `ma_gear_freq`, `ma_gear_damp`, `ma_gear_gravity`, `ma_gear_swing` (biggest swing in degrees).
   - **Amount** sets how loose it is.
   - **Bake Overlap** (pose mode, select bones) does the same for chosen bones only. Each bone lags, overshoots and settles (spring stiffness in Hz, damping, response).
   - A good order: Fix Jumps, then Follow-Through, then Fix Jumps again (with Lean & Land off), then Fix Joints last (with the held props set, so the hands go back onto them).
8. **Path Objects**: a tube and a ball that show the centre of mass in renders.
9. **AutoPose (AI)**:
   - **Add Goals** puts six spheres at the hips, neck, hands and feet. Move them, then press **Auto Pose**, or turn on **Live** to re-pose as you drag.
   - A small neural network predicts a natural position for every other joint: the spine curve, head, shoulders, elbows, knees and how the hands and feet turn. The rig is then fitted with its own bone lengths, inside the joint limits. Elbows and knees take the most natural swivel that stays in range, and the hands and feet land exactly on their goals.
   - **Auto Hips** lets a second network place the hips too. Move a hand with the feet planted, and the body crouches or stands up to reach it.
   - Without goals, Auto Pose keeps the current hips, neck, hands and feet and re-poses the rest. **Key** keys the result on the current frame.
   - It works on rigs named like Mender's and Relic's: pelvis, spine, chest, neck, head, clavicle/upperarm/forearm/hand, thigh/shin/foot/toe, and IK legs.

   The networks are our own, trained on the CMU Graphics Lab Motion Capture Database: 2,543 motions, 928k poses, retargeted to one skeleton by cgspeed.com. They run with numpy, so they need no extra installs. The training scripts are in `C:\ai\mocap` (`prep_cmu.py`, `train_autopose.py`).

Every fix is one undo step.

## From Python

The same tools are plain functions, for scripts (each takes the scene and rig):

- `autopose.check_animation(scene, rig, f0, f1)` and `autopose.fix_animation(scene, rig, f0, f1, parts=None, holds=None, leans=None)`. `parts` picks from `("spine", "arms", "legs")`.
- `grip.plan(scene, rig, f0, f1, {"L": prop, "R": prop})` returns `(holds, leans)` for `fix_animation`. Compute it before changing the arms: it reads the hold from the animation as it is.
- `overlap.bake_overlap`, `overlap.bake_leg_follow`, `overlap.bake_gear` and `overlap.follow_through` for the secondary motion.

## Credits

Motion data: the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), created with funding from NSF EIA-0196217. Converted to BVH by Bruce Hahne (cgspeed.com). The data may be used in commercial products but must not be resold.

## Not yet

- Fixing balance on the ground. The check flags it, but a real fix needs a small physics solver (planned).
- AI inbetweening (planned, from the same CMU data saved at 30 fps in `seq30.npz`).
- Matching AutoPose to each character's proportions. Everything is scaled by leg length; the exact IK and Auto Hips' reach correction cover most of the difference.
- Contacts with moving scenery (the ground is read once per analysis).
- Fingers, and a grip that follows a prop that is not a straight shaft.
