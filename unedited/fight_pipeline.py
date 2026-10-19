# fight_pipeline.py : THE repeatable process for fight animation on Relic and the KillDroid.
#   1. SOURCE   real motion capture clips (mocap_ref/segs/*.npz, cut with mocap_ref/cut_segments.py)
#   2. FIT      mocap_keys.mocap_frames: per-frame fit onto the rig (hands from the true shoulders, knees from the real hip joints, feet kept at the
#               real leg angle on the rig's wider hips, real elbow/knee directions, foot orientation, foot lock)
#   3. SOLVE    rig_guard.apply_safe: torso from the fit, arms and legs EXACT (limb_solver: elbow/knee only bend the natural way), every joint
#               inside its anatomical range, every joint speed-limited; keyed as plain FK
#   4. BLEND    clip to clip transitions are blended joint by joint (quaternion slerp of the solved poses), never by mixing targets
#   5. QA GATE  qa(): elbow/knee direction vs the real capture, hand/foot placement, hinge never past its range (no backwards bend), joint speed,
#               joint limits, mesh stretch, feet in the floor. Everything must PASS before a take is used.
#   exec'd in Blender (it execs kd_ctrl.py, relic_fists.py, mocap_keys.py, rig_guard.py itself).  Use:
#       pipe = FightPipeline(rig, other_rig)
#       pipe.add_clip("punch_144_20_a", start=4, base=(0, 2), yaw=0.0, look=Vector(...))
#       pipe.bake(); rep = pipe.qa(); print(pipe.qa_text(rep))
import bpy, math, json
import numpy as np
from mathutils import Vector, Matrix, Quaternion

OUT = r"C:\ai\unedited"
SEGS = __import__("os").environ.get("PIPE_SEGS") or (OUT + r"\mocap_ref\segs")     # PIPE_SEGS: another clip folder (tests)
FWD_DEFAULT = {"Relic_AI_Rig": Vector((0.0, 1.0, 0.0)), "KillDroid_Rig": Vector((0.0, -1.0, 0.0))}

# ------------------------------------------------------------------------------------------------ QA thresholds (the bar every take must clear)
QA = dict(
    elbow_dir_max=35.0,       # deg: elbow direction (around shoulder-wrist) vs the real capture, worst frame, while the arm is bent > 25 deg
    elbow_dir_p95=15.0,       # deg: 95th percentile
    knee_dir_max=35.0,
    knee_dir_p95=15.0,
    hand_dir_p95=12.0,        # deg: shoulder-to-hand direction vs the real capture, 95th percentile
    hinge_backwards=0,        # frames with an elbow or knee bent past its straight limit (backwards): must be zero
    speed_over=0,             # joint turns faster than the per-joint cap (+10 deg tolerance)
    limit_excess_max=3.0,     # deg: any joint outside its range after the solve
    anatomy_bad=0,            # frames (ANY frame, transitions too) where an elbow / knee points the impossible way for a human body
                              # (limb_solver.POINT_LIMIT away from natural_point): must be zero, for Relic AND for the reference capture
    fist_inside=0,            # frames where a fist is closer than 1 cm to (or inside) Relic's own torso armour or helmet (body_clear.py)
    knee_bend_p95=10.0,       # deg: knee bend vs the real capture (the kicking leg must bend as much as the fighter's), 95th percentile
    knee_bend_max=25.0,
    elbow_bend_p95=20.0,      # deg: elbow bend vs the real capture (looser: fists are pushed out of Relic's bulkier body on purpose)
    elbow_bend_max=45.0,
    torso_dir_p95=12.0,       # deg: shoulder line, hip line and spine vs the real capture, 95th percentile
    torso_dir_max=25.0,       # deg: worst frame
    skin_leaks=0,             # mesh pieces with a vertex pulled by two bones more than two joints apart (skin_rules.py): must be zero
    spike_edges_max=0,        # mesh edges stretched to more than 3x their rest length AND longer than 6 cm, any frame (checked every 2nd frame)
    floating_pieces=0,        # loose armour pieces that end up more than 5 cm from every other piece (flying debris), checked every 6th frame
    floor_below_cm=1.5,       # an ankle below its standing height by more than this
    hull_cuts=0,              # KillDroid: frames where its crusher or fist (rigid parts on forearm/hand/jaw bones) cuts into its own hull (the parts
                              # on pelvis/spine/chest/neck/head: cab, engine hump, turret), polygon against polygon, checked every 2nd frame
)
KD_HULL = ("chest", "spine", "neck", "head", "pelvis")
KD_HANDS = {"crusher": ("forearm.R", "hand.R", "claw1a.R", "claw2a.R"), "left fist": ("forearm.L", "hand.L")}


def _bvh_of(objs):
    from mathutils.bvhtree import BVHTree
    V, P = [], []
    for o in objs:
        M = o.matrix_world
        off = len(V)
        V += [M @ v.co for v in o.data.vertices]
        P += [tuple(i + off for i in p.vertices) for p in o.data.polygons]
    return BVHTree.FromPolygons(V, P)


def _load_tools(scope):
    for f in ("kd_ctrl.py", "relic_fists.py", "mocap_keys.py", "rig_guard.py", "skin_rules.py"):
        exec(open(OUT + "\\" + f, encoding="utf-8").read(), scope)


class FightPipeline:
    def __init__(self, rig, other=None, scope=None):
        self.rig = rig
        self.other = other
        self.scope = scope if scope is not None else globals()
        if 'apply_safe' not in self.scope:
            _load_tools(self.scope)
        S = self.scope
        S['FWD'] = dict(FWD_DEFAULT)
        if rig.name == "Relic_AI_Rig" and 'fist1.L' not in rig.data.bones:
            S['add_fists'](rig)
        S['kd_add_arm_ik'](rig)
        self.clips = []
        self.frames = {}          # frame -> params
        self.real = {}            # frame -> {joint: world Vector} (the real capture placed and scaled like the rig, for QA)
        self.clip_of = {}         # frame -> clip index
        self.blend = set()        # frames that are transitions (direction checks skipped there)

    # ------------------------------------------------------------------------------------------ build
    def add_clip(self, name, start, base=(0.0, 2.0), yaw=0.0, look=None, slow=1.0, fist=(1.0, 1.0), blend_in=8, spine_fit=None):
        """spine_fit: the chest copies the real shoulder line and the pelvis -> neck line (kd_apply), so a deep bend leans as far as the real one.
        Default: on for the KillDroid, off for Relic (his approved clips were cleared without it)"""
        S = self.scope
        seg = S['load_seg'](SEGS + "\\%s.npz" % name)
        fwd = Matrix.Rotation(yaw, 3, 'Z') @ S['FWD'][self.rig.name]
        left = Matrix.Rotation(math.pi / 2, 3, 'Z') @ fwd
        fr = S['mocap_frames'](self.rig, seg, start, base, yaw, (fwd.x, fwd.y), (left.x, left.y), fist=fist if self.rig.name == "Relic_AI_Rig" else None, slow=slow)
        S['_stabilize'](fr, self.rig, 1.0, 1.0)
        sf = (self.rig.name != "Relic_AI_Rig") if spine_fit is None else spine_fit
        for P in fr.values():
            P['spine_fit'] = sf
        kref = (self.rig.data.bones['thigh.L'].length + self.rig.data.bones['shin.L'].length) / 0.80
        real = S['mocap_world_joints'](seg, start, base, (fwd.x, fwd.y), (left.x, left.y), k=kref, slow=slow)
        idx = len(self.clips)
        for f, P in fr.items():
            if look is not None:
                P['look'] = Vector(look)
            self.frames[f] = P
            self.clip_of[f] = idx
        self.real.update(real)
        self.clips.append(dict(name=name, start=start, end=max(fr), blend_in=blend_in, desc=seg['desc']))
        return max(fr)

    # ------------------------------------------------------------------------------------------ bake
    def bake(self):
        S, rig = self.scope, self.rig
        PB = rig.pose.bones
        g = S['guard_setup'](rig)
        S['set_meshes_hidden'](True)
        names = g['names']
        solved = {}
        for ci, c in enumerate(self.clips):
            # each clip starts fresh (the transition blend below provides the continuity)
            g['prev_rel'].clear(); g['twist'].clear()
            for k in list(S['LIMB_PREV']):
                if k[0] == rig.name:
                    del S['LIMB_PREV'][k]
            for f in range(c['start'], c['end'] + 1):
                S['apply_safe'](rig, f, self.frames[f])
                solved[f] = ({n: PB[n].rotation_quaternion.copy() for n in names}, PB['pelvis'].location.copy())
        # transitions: from A's last solved pose, ease into clip B as it plays (one continuous blend across the gap and B's first frames),
        # joint by joint (quaternion slerp of the solved poses); every blended frame then goes through the joint clamp as well
        for ci in range(1, len(self.clips)):
            a, b = self.clips[ci - 1], self.clips[ci]
            fa, fb, nb = a['end'], b['start'], b['blend_in']
            qa_, la = solved[fa]
            fe = min(b['end'], fb + nb)
            for f in range(fa + 1, fe + 1):
                t = (f - fa) / float(fe - fa + 1)
                t = t * t * (3 - 2 * t)
                qb_, lb = solved[max(f, fb)]
                self._key_blend(f, qa_, qb_, la, lb, t)
                self.blend.add(f)
        S['set_meshes_hidden'](False)
        self.first, self.last = min(c['start'] for c in self.clips), max(c['end'] for c in self.clips)
        bpy.context.scene.frame_start, bpy.context.scene.frame_end = self.first, self.last
        print(S['guard_report'](rig))

    def _key_blend(self, f, qa_, qb_, la, lb, t):
        PB = self.rig.pose.bones
        for n, q0 in qa_.items():
            q1 = qb_[n]
            if q0.dot(q1) < 0:
                q1 = -q1
            PB[n].rotation_quaternion = q0.slerp(q1, t)
            PB[n].keyframe_insert('rotation_quaternion', frame=f)
        PB['pelvis'].location = la.lerp(lb, t)
        bpy.context.view_layer.update()
        # a blend must never push a foot into the floor: lift the pelvis by whatever the lowest ankle is short of its standing height
        rig = self.rig
        ank0 = rig.data.bones['foot.L'].head_local.z
        low = min((rig.matrix_world @ PB['foot.' + sd].head).z for sd in 'LR')
        floor_z = (rig.matrix_world @ Vector((0.0, 0.0, ank0))).z
        if low < floor_z:
            rest_rot = rig.data.bones['pelvis'].matrix_local.to_3x3()
            up_arm = rig.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, floor_z - low))
            PB['pelvis'].location = PB['pelvis'].location + rest_rot.inverted() @ up_arm
            bpy.context.view_layer.update()
        PB['pelvis'].keyframe_insert('location', frame=f)
        if rig.name == "KillDroid_Rig":
            self._blend_hull_clear()
        self.scope['clamp_pass'](self.rig, f, key=True, rate=False)

    def _blend_hull_clear(self):
        """a blended KillDroid frame keeps its forearm / hand / jaws out of its own hull too: the blend of two clear poses can pass through it.
        Only an arm that touches the hull is solved again (from its blended shoulder, wrist and elbow) and moved out (rig_guard.kd_clear_arm)"""
        S_, rig = self.scope, self.rig
        PB = rig.pose.bones
        lims = S_['GUARD'][rig.name]['lims']
        hull_ = S_['kd_hull_now'](rig)
        (af, au), _legs = S_['body_frames'](rig)
        for sd in 'LR':
            Sh = PB['upperarm.' + sd].head.copy()
            R_u = PB['upperarm.' + sd].matrix.to_3x3().normalized()
            R_l = PB['forearm.' + sd].matrix.to_3x3().normalized()
            if not hull_[0].overlap(S_['kd_arm_now'](rig, sd, Sh, R_u, R_l)[0]):
                continue
            Eh = PB['forearm.' + sd].head.copy()
            solve_ = lambda W_, sd=sd, Sh=Sh, Eh=Eh: S_['solve_limb'](rig, 'upperarm.' + sd, 'forearm.' + sd, Sh, W_, Eh, lims['forearm.' + sd],
                                                                      natural=('elbow', af, au))[:2]
            R_u, R_l, _W, moved = S_['kd_clear_arm'](rig, hull_, sd, Sh, PB['forearm.' + sd].tail.copy(), solve_)
            S_['write_rotation'](rig, 'upperarm.' + sd, R_u, PB['clavicle.' + sd].matrix.to_3x3().normalized())
            S_['write_rotation'](rig, 'forearm.' + sd, R_l, R_u)
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------------------------------ QA gate
    def qa(self, stretch_every=2):
        S, rig = self.scope, self.rig
        sc = bpy.context.scene
        PB, B = rig.pose.bones, rig.data.bones
        g = S['GUARD'][rig.name]
        lims = g['lims']
        W = lambda n, t=False: rig.matrix_world @ (PB[n].tail if t else PB[n].head)

        def perp(v, ax):
            return v - ax * v.dot(ax)

        torso_err = []
        fist_bad, fist_min = [], []
        n_pushed = [0]
        bend_el, bend_kn = [], []
        anat_rig, anat_ref = [], []          # (frame, joint, deg) where the elbow / knee points the impossible way
        anat_worst = {'rig': 0.0, 'ref': 0.0}
        NP, PD, PL = S['natural_point'], S['point_dir'], S['POINT_LIMIT']

        def _anat(store, who, f, frames_, joints_):
            (af, au), (lf_, lu) = frames_
            for nm, (r_, m_, e_), kind, fw_, up_ in joints_:
                u_, p_, bend_ = PD(r_, m_, e_)
                if bend_ < 35.0:
                    continue
                dv = math.degrees(p_.angle(NP(kind, u_, fw_ if kind == 'knee' else af, up_ if kind == 'knee' else au)))
                anat_worst[who] = max(anat_worst[who], dv)
                if dv > PL[kind]:
                    store.append((f, nm, round(dv)))
        el_err, kn_err, hand_err = [], [], []
        hinge_back, speed_over, lim_ex = [], [], []
        worst_frames = {}
        prevq = {}
        ank0 = B['foot.L'].head_local.z
        floor = []
        S['set_meshes_hidden'](True)
        def _fr(l, u):
            l = l.normalized()
            u = (u - l * u.dot(l)).normalized()
            return l.cross(u).normalized(), u
        for f in range(self.first, self.last + 1):
            sc.frame_set(f)
            R = self.real.get(f)
            # fists outside Relic's own armour, every frame
            if rig.name == "Relic_AI_Rig":
                bfr = S['relic_body_frame'](rig)
                for sd_ in 'LR':
                    wr_ = W('forearm.' + sd_, True)
                    F_ = wr_ + (wr_ - W('forearm.' + sd_)).normalized() * S['FIST_OFF']
                    ct_, chd_ = S['fist_clearance'](rig, F_, bfr)
                    m_ = min(x for x in (ct_, chd_) if x is not None)
                    fist_min.append(m_)
                    if m_ < 0.01:
                        fist_bad.append((f, sd_, round(100 * m_)))
            # anatomy, judged on its own (not against the reference, which can be wrong): every frame, blends included
            bf = S['body_frames'](rig, world=True)
            _anat(anat_rig, 'rig', f, bf, [(sd_ + kind_, (W(a_ + '.' + sd_), W(b_ + '.' + sd_), W(b_ + '.' + sd_, True)), kind_, bf[1][0], bf[1][1])
                                          for sd_ in 'LR' for kind_, a_, b_ in (('elbow', 'upperarm', 'forearm'), ('knee', 'thigh', 'shin'))])
            if R is not None and f not in self.blend:
                rbf = (_fr(R['shoulder_L'] - R['shoulder_R'], R['neck'] - R['hips']), _fr(R['hip_L'] - R['hip_R'], R['spine'] - R['hips']))
                _anat(anat_ref, 'ref', f, rbf, [(sd_ + 'elbow', (R['shoulder_' + sd_], R['elbow_' + sd_], R['wrist_' + sd_]), 'elbow', rbf[1][0], rbf[1][1])
                                                for sd_ in 'LR'] + [(sd_ + 'knee', (R['hip_' + sd_], R['knee_' + sd_], R['ankle_' + sd_]), 'knee', rbf[1][0], rbf[1][1])
                                                                     for sd_ in 'LR'])
            if R is not None and f not in self.blend:
                # the body itself must face, twist and lean the way the real person did, or matched hands mean twisted arms
                pairs = ((W('upperarm.R') - W('upperarm.L'), R['shoulder_R'] - R['shoulder_L'], 'shoulders'),
                         (W('thigh.R') - W('thigh.L'), R['hip_R'] - R['hip_L'], 'hips'),
                         (W('neck') - W('pelvis'), R['neck'] - R['hips'], 'spine'))
                for v_rig, v_real, nm in pairs:
                    if v_rig.length > 1e-4 and v_real.length > 1e-4:
                        e = math.degrees(v_rig.angle(v_real))
                        torso_err.append(e)
                        if e > QA['torso_dir_max']:
                            worst_frames.setdefault('torso', []).append((f, nm, round(e)))
            for sd in 'LR':
                sh, el, wr = W('upperarm.' + sd), W('forearm.' + sd), W('forearm.' + sd, True)
                hp, kn, an = W('thigh.' + sd), W('shin.' + sd), W('shin.' + sd, True)
                if R is not None and f not in self.blend:
                    rsh, rel, rwr = R['shoulder_' + sd], R['elbow_' + sd], R['wrist_' + sd]
                    rhp, rkn, ran = R['hip_' + sd], R['knee_' + sd], R['ankle_' + sd]
                    ax, rax = (wr - sh).normalized(), (rwr - rsh).normalized()
                    bend = 180 - math.degrees((sh - el).angle(wr - el))
                    rbend = 180 - math.degrees((rsh - rel).angle(rwr - rel))
                    pushed = self.frames.get(f, {}).get('_push_' + sd, 0.0) > 0.02     # fist moved out of Relic's armour on purpose
                    n_pushed[0] += int(pushed)
                    bend_el.append(abs(bend - rbend))
                    if abs(bend - rbend) > QA['elbow_bend_max']:
                        worst_frames.setdefault('elbow bend', []).append((f, sd, round(abs(bend - rbend))))
                    pe, pr = perp(el - sh, ax), perp(perp(rel - rsh, rax), ax)
                    if bend > 25 and rbend > 25 and pe.length > 1e-3 and pr.length > 1e-3 and not pushed:
                        e = math.degrees(pe.angle(pr))
                        el_err.append(e)
                        if e > QA['elbow_dir_max']:
                            worst_frames.setdefault('elbow', []).append((f, sd, round(e)))
                    if not pushed:
                        hand_err.append(math.degrees((wr - sh).angle(rwr - rsh)))
                    axk, raxk = (an - hp).normalized(), (ran - rhp).normalized()
                    kb = 180 - math.degrees((hp - kn).angle(an - kn))
                    rkb = 180 - math.degrees((rhp - rkn).angle(ran - rkn))
                    bend_kn.append(abs(kb - rkb))
                    if abs(kb - rkb) > QA['knee_bend_max']:
                        worst_frames.setdefault('knee bend', []).append((f, sd, round(abs(kb - rkb))))
                    qe, qr = perp(kn - hp, axk), perp(perp(rkn - rhp, raxk), axk)
                    if kb > 20 and rkb > 20 and qe.length > 1e-3 and qr.length > 1e-3:
                        e = math.degrees(qe.angle(qr))
                        kn_err.append(e)
                        if e > QA['knee_dir_max']:
                            worst_frames.setdefault('knee', []).append((f, sd, round(e)))
                floor.append((an.z - (rig.matrix_world @ Vector((0, 0, ank0))).z, f))
            # hinge (no backwards bend), joint limits, speed
            for n in g['names']:
                lim = lims.get(n)
                pn = B[n].parent.name if B[n].parent else None
                if lim is not None and pn == lim.parent:
                    rel_pose = PB[pn].matrix.to_3x3().normalized().inverted() @ PB[n].matrix.to_3x3().normalized()
                    rel_rest = B[pn].matrix_local.to_3x3().inverted() @ B[n].matrix_local.to_3x3()
                    q = (rel_rest.inverted() @ rel_pose).to_quaternion()
                    if lim.kind == 'hinge':
                        phi = math.degrees(S['_limits'].hinge_angle(q @ S['_limits'].Y, lim.hinge))
                        if phi < lim.ranges[0] - 2.0:
                            hinge_back.append((f, n, round(phi)))
                    qc, _t = S['clamp_cont'](q, lim, None, swing_only=n.split('.')[0] in S['SWING_ONLY'])
                    e = math.degrees(q.normalized().rotation_difference(qc).angle)
                    e = min(e, 360 - e)
                    if e > QA['limit_excess_max']:
                        lim_ex.append((f, n, round(e)))
                mt = S['MAX_TURN'].get(n.split('.')[0])
                if mt is not None:
                    pnm = PB[pn].matrix.to_3x3().normalized() if pn else Matrix.Identity(3)
                    lq = (pnm.inverted() @ PB[n].matrix.to_3x3().normalized()).to_quaternion()
                    if n in prevq and f - 1 >= self.first:
                        d = math.degrees(lq.rotation_difference(prevq[n]).angle)
                        d = min(d, 360 - d)
                        if d > mt + 10:
                            speed_over.append((f, n, round(d)))
                    prevq[n] = lq
        S['set_meshes_hidden'](False)
        # the skin: no leaked weights, and no edge ever stretched into a spike (checked on every 2nd frame)
        stretch = []
        leaks = []
        floating = []
        hull_cuts = []
        ob = bpy.data.objects.get("Relic_AI") if rig.name == "Relic_AI_Rig" else bpy.data.objects.get("KillDroid_Body")
        if ob is not None:
            leaks = [(r['piece'], r['region'], r['verts'], r['bones']) for r in S['leak_report'](ob, rig)]
            me0 = ob.data
            rest = np.zeros(len(me0.vertices) * 3); me0.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
            edges = np.zeros(len(me0.edges) * 2, dtype=int); me0.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
            rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
            # loose pieces (for the flying-debris check)
            par = np.arange(len(rest))

            def _f(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            for a_, b_ in edges:
                ra, rb = _f(a_), _f(b_)
                if ra != rb:
                    par[ra] = rb
            isl = np.array([_f(i) for i in range(len(rest))])
            ids_, cnt_ = np.unique(isl, return_counts=True)
            small_ = [(k, np.where(isl == k)[0]) for k, c in zip(ids_, cnt_) if c < 600]
            from mathutils import kdtree as _kdt
            # the KillDroid (v02) wears rigid armour parts on its bones around a skinned core: a core piece is attached when it is near
            # any of them too (the leg cores part from the pelvis core when the legs spread, but stay 3 mm inside the thigh armour)
            rigid = [o for o in rig.children if o.type == 'MESH' and o is not ob] if rig.name == "KillDroid_Rig" else []
            hull = [o for o in rigid if o.parent_type == 'BONE' and o.parent_bone in KD_HULL]
            hands = {k: [o for o in rigid if o.parent_type == 'BONE' and o.parent_bone in bones] for k, bones in KD_HANDS.items()}
            for f in range(self.first, self.last + 1, stretch_every):
                sc.frame_set(f)
                if hull:
                    th_ = _bvh_of(hull)
                    for k, objs in hands.items():
                        n_ = len(th_.overlap(_bvh_of(objs)))
                        if n_ > 0:
                            hull_cuts.append((f, k, n_))
                dg = bpy.context.evaluated_depsgraph_get()
                ev = ob.evaluated_get(dg)
                me = ev.to_mesh()
                co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
                ev.to_mesh_clear()
                if len(co) != len(rest):
                    continue
                L = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
                bad = int(((L / np.maximum(rest_len, 1e-6) > 3) & (L > 0.06)).sum())
                stretch.append((f, bad))
                if (f - self.first) % (3 * stretch_every) == 0 and small_:
                    extra = []
                    if rigid:
                        Minv = ob.matrix_world.inverted()
                        for o_ in rigid:
                            Mo = Minv @ o_.matrix_world
                            vs_ = o_.data.vertices
                            extra += [Mo @ vs_[i].co for i in range(0, len(vs_), 4)]
                    kd = _kdt.KDTree(len(co) + len(extra))
                    for i in range(len(co)):
                        kd.insert(co[i], i)
                    for i, p_ in enumerate(extra):
                        kd.insert(p_, len(co) + i)
                    kd.balance()
                    for k, idx in small_:
                        attached = False
                        for i in idx[:: max(1, len(idx) // 12)]:
                            for (_p, j, dd) in kd.find_range(co[i], 0.05):
                                if j >= len(co) or isl[j] != k:
                                    attached = True
                                    break
                            if attached:
                                break
                        if not attached:
                            floating.append((f, int(k), len(idx)))
        pct = lambda a, p: float(np.percentile(a, p)) if len(a) else 0.0
        rep = dict(
            fist_inside=fist_bad, fist_min_cm=round(100 * min(fist_min), 1) if fist_min else None, fists_pushed=n_pushed[0],
            knee_bend=dict(p95=pct(bend_kn, 95), max=max(bend_kn) if bend_kn else 0),
            elbow_bend=dict(p95=pct(bend_el, 95), max=max(bend_el) if bend_el else 0),
            anatomy_rig=anat_rig, anatomy_ref=anat_ref, anatomy_worst=anat_worst,
            torso_dir=dict(n=len(torso_err), p95=pct(torso_err, 95), max=max(torso_err) if torso_err else 0),
            skin_leaks=leaks,
            elbow_dir=dict(n=len(el_err), mean=float(np.mean(el_err)) if el_err else 0, p95=pct(el_err, 95), max=max(el_err) if el_err else 0),
            knee_dir=dict(n=len(kn_err), mean=float(np.mean(kn_err)) if kn_err else 0, p95=pct(kn_err, 95), max=max(kn_err) if kn_err else 0),
            hand_dir=dict(mean=float(np.mean(hand_err)) if hand_err else 0, p95=pct(hand_err, 95), max=max(hand_err) if hand_err else 0),
            hinge_backwards=hinge_back, speed_over=speed_over, limit_excess=lim_ex,
            stretch_worst=max((b for f, b in stretch), default=0), stretch_frames=[s for s in stretch if s[1] > QA['spike_edges_max']],
            floor_low_cm=round(-100 * min((d for d, f in floor), default=0), 1), floor_frame=min(floor)[1] if floor else None,
            worst_frames=worst_frames, frames=(self.first, self.last), clips=[c['name'] for c in self.clips])
        checks = [
            ("Relic: no elbow/knee ever points the impossible way (any frame)", len(anat_rig) <= QA['anatomy_bad'],
             "%d joint-frames, worst %.0f deg off natural %s" % (len(anat_rig), anat_worst['rig'], anat_rig[:4])),
            ("the reference capture itself is anatomically possible", len(anat_ref) <= QA['anatomy_bad'],
             "%d joint-frames, worst %.0f deg off natural %s" % (len(anat_ref), anat_worst['ref'], anat_ref[:4])),
            ("fists never inside Relic's own body (any frame)", len(fist_bad) <= QA['fist_inside'],
             "%d fist-frames, closest %s cm; %d fist-frames moved out of his armour %s" % (len(fist_bad), rep['fist_min_cm'], n_pushed[0], fist_bad[:4])),
            ("knees bend as much as the real knees (p95 / worst)", rep['knee_bend']['p95'] <= QA['knee_bend_p95'] and rep['knee_bend']['max'] <= QA['knee_bend_max'],
             "%.0f / %.0f deg (limit %.0f / %.0f)" % (rep['knee_bend']['p95'], rep['knee_bend']['max'], QA['knee_bend_p95'], QA['knee_bend_max'])),
            ("elbows bend as much as the real elbows (p95 / worst)", rep['elbow_bend']['p95'] <= QA['elbow_bend_p95'] and rep['elbow_bend']['max'] <= QA['elbow_bend_max'],
             "%.0f / %.0f deg (limit %.0f / %.0f)" % (rep['elbow_bend']['p95'], rep['elbow_bend']['max'], QA['elbow_bend_p95'], QA['elbow_bend_max'])),
            ("the body faces, twists and leans like the real one (p95 / worst)", rep['torso_dir']['p95'] <= QA['torso_dir_p95'] and rep['torso_dir']['max'] <= QA['torso_dir_max'],
             "%.0f / %.0f deg (limit %.0f / %.0f)" % (rep['torso_dir']['p95'], rep['torso_dir']['max'], QA['torso_dir_p95'], QA['torso_dir_max'])),
            ("elbows point the way the real elbows did (p95 / worst)", rep['elbow_dir']['p95'] <= QA['elbow_dir_p95'] and rep['elbow_dir']['max'] <= QA['elbow_dir_max'],
             "%.0f / %.0f deg (limit %.0f / %.0f)" % (rep['elbow_dir']['p95'], rep['elbow_dir']['max'], QA['elbow_dir_p95'], QA['elbow_dir_max'])),
            ("knees point the way the real knees did (p95 / worst)", rep['knee_dir']['p95'] <= QA['knee_dir_p95'] and rep['knee_dir']['max'] <= QA['knee_dir_max'],
             "%.0f / %.0f deg (limit %.0f / %.0f)" % (rep['knee_dir']['p95'], rep['knee_dir']['max'], QA['knee_dir_p95'], QA['knee_dir_max'])),
            ("hands go where the real hands went (p95)", rep['hand_dir']['p95'] <= QA['hand_dir_p95'], "%.0f deg (limit %.0f)" % (rep['hand_dir']['p95'], QA['hand_dir_p95'])),
            ("no elbow or knee ever bends backwards", len(hinge_back) <= QA['hinge_backwards'], "%d frames" % len(hinge_back)),
            ("no joint turns faster than a human joint can", len(speed_over) <= QA['speed_over'], "%d joint-frames" % len(speed_over)),
            ("every joint inside its range", len(lim_ex) == 0, "%d joint-frames over %.0f deg" % (len(lim_ex), QA['limit_excess_max'])),
            ("no skin weight leaks between limbs (skin_rules.py)", len(leaks) <= QA['skin_leaks'], "%d pieces %s" % (len(leaks), leaks[:3])),
            ("the mesh never stretches into a spike (edge >3x and >6 cm)", rep['stretch_worst'] <= QA['spike_edges_max'],
             "worst %d edges, frames %s (limit %d)" % (rep['stretch_worst'], [s[0] for s in rep['stretch_frames']][:6], QA['spike_edges_max'])),
            ("no armour piece floats away from the body", len(floating) <= QA['floating_pieces'],
             "%d piece-frames %s" % (len(floating), floating[:4])),
            ("feet never sink into the floor", rep['floor_low_cm'] <= QA['floor_below_cm'], "%.1f cm at frame %s (limit %.1f)" % (rep['floor_low_cm'], rep['floor_frame'], QA['floor_below_cm'])),
        ]
        rep['hull_cuts'] = hull_cuts
        if rig.name == "KillDroid_Rig":
            hc_frames = sorted({f_ for f_, k_, n_ in hull_cuts})
            checks.insert(3, ("the crusher and the fist never cut into its own hull (any frame)", len(hc_frames) <= QA['hull_cuts'],
                              "%d frames %s; %d hand-frames moved out of the hull" % (len(hc_frames), hull_cuts[:4], n_pushed[0])))
        rep['checks'] = [(a, bool(b), c) for a, b, c in checks]
        rep['pass'] = all(b for a, b, c in checks)
        return rep

    @staticmethod
    def qa_text(rep):
        s = "QA %s  (%s, frames %d-%d)\n" % ("PASS" if rep['pass'] else "FAIL", ", ".join(rep['clips']), rep['frames'][0], rep['frames'][1])
        for a, ok, c in rep['checks']:
            s += "  [%s] %-58s %s\n" % ("ok" if ok else "XX", a, c)
        for k, v in rep['worst_frames'].items():
            s += "  worst %s frames: %s\n" % (k, v[:8])
        if rep['hinge_backwards']:
            s += "  backwards bends: %s\n" % rep['hinge_backwards'][:8]
        if rep['speed_over']:
            s += "  too fast: %s\n" % rep['speed_over'][:8]
        if rep['limit_excess']:
            s += "  outside range: %s\n" % rep['limit_excess'][:8]
        return s

    # ------------------------------------------------------------------------------------------ reference figure (as pipe_test.py builds it)
    def ref_figure(self, off):
        """the raw real capture as a white stick figure (blue = left, red = right) placed `off` (world Vector) from the rig, keyed on every
        captured frame: objects ref_* in a collection RefFigure, for strike_views.py and reel_vid.py"""
        import bmesh
        sc = bpy.context.scene

        def _mat(name, col):
            m = bpy.data.materials.new(name)
            m.use_nodes = True
            m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = col
            return m
        MATS = {"L": _mat("refL", (0.15, 0.35, 0.95, 1)), "R": _mat("refR", (0.95, 0.2, 0.15, 1)), "C": _mat("refC", (0.9, 0.9, 0.9, 1))}
        BONES = [("hips", "spine", "C"), ("spine", "chest", "C"), ("chest", "neck", "C"), ("neck", "head", "C"), ("neck", "shoulder_L", "L"),
                 ("shoulder_L", "elbow_L", "L"), ("elbow_L", "wrist_L", "L"), ("neck", "shoulder_R", "R"), ("shoulder_R", "elbow_R", "R"),
                 ("elbow_R", "wrist_R", "R"), ("hips", "hip_L", "L"), ("hip_L", "knee_L", "L"), ("knee_L", "ankle_L", "L"), ("ankle_L", "toe_L", "L"),
                 ("hips", "hip_R", "R"), ("hip_R", "knee_R", "R"), ("knee_R", "ankle_R", "R"), ("ankle_R", "toe_R", "R")]
        refc = bpy.data.collections.new("RefFigure")
        sc.collection.children.link(refc)
        used = sorted({b for pr in BONES for b in pr[:2]})
        empt = {}
        for jn in used:
            e = bpy.data.objects.new("ref_" + jn, None)
            refc.objects.link(e)
            empt[jn] = e
        for fr in sorted(self.real):
            for jn in used:
                empt[jn].location = self.real[fr][jn] + off
                empt[jn].keyframe_insert("location", frame=fr)
        k = (self.rig.data.bones['thigh.L'].length + self.rig.data.bones['shin.L'].length) / 0.80 / 1.05     # thicker sticks for a bigger figure
        cyl = bpy.data.meshes.new("refcyl")
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.03 * k, radius2=0.03 * k, depth=1.0)
        bmesh.ops.translate(bm, vec=(0, 0, 0.5), verts=bm.verts)
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(-math.pi / 2, 3, 'X'), verts=bm.verts)
        bm.to_mesh(cyl)
        bm.free()
        for a_, b_, side in BONES:
            o = bpy.data.objects.new("refbone_%s_%s" % (a_, b_), cyl)
            refc.objects.link(o)
            o.data.materials.append(MATS[side])
            c1 = o.constraints.new('COPY_LOCATION')
            c1.target = empt[a_]
            c2 = o.constraints.new('STRETCH_TO')
            c2.target = empt[b_]
            c2.rest_length = 1.0
            c2.volume = 'NO_VOLUME'
        for jn in used:
            sp_ = bpy.data.objects.new("refj_" + jn, bpy.data.meshes.new("refj"))
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=(0.05 if jn == "head" else 0.04) * k)
            bm.to_mesh(sp_.data)
            bm.free()
            refc.objects.link(sp_)
            sp_.data.materials.append(MATS["C"])
            cc = sp_.constraints.new('COPY_LOCATION')
            cc.target = empt[jn]
        return refc
