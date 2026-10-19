# relic_model_v7.py : make Relic's AI-made mesh fit for animation (user, 2026-10-12: "the model keeps ripping and you can see through parts of
#   his body when he moves").
#   1. RIGID LIMB ARMOUR: every face of an arm or leg belongs to exactly one bone; the mesh is CUT (edges split) where two bones meet, with a
#      clean planar cut through the joint at the elbows, knees, ankles and toes. A rigid plate cannot stretch, so nothing rips.
#      (the torso keeps its smooth weights - it only bends gently; the hands keep theirs - the fists curl at run time)
#   2. INNER BODY: a dark watertight undersuit (torso tube + a tube per limb bone, overlapping at the joints) just inside the armour, so a gap
#      between plates shows mechanism, never the sky.
#   3. the original custom normals are copied back so the surface shades exactly as before.
#   blender.exe -b fight_v06.blend --python relic_model_v7.py      -> fight_v07.blend
import bpy, bmesh, math, json
import numpy as np
from mathutils import Vector, Matrix
OUT = r"C:\ai\unedited"
rig = bpy.data.objects["Relic_AI_Rig"]
body = bpy.data.objects["Relic_AI"]
B = rig.data.bones
TORSO = {"pelvis", "spine", "chest", "neck", "head", "clavicle.L", "clavicle.R"}
RIGID = {"upperarm", "thigh", "shin", "foot", "toe"}           # forearm + hand move together (the wrist never turns): one segment "forearm"
for b in B:
    if b.name == "root" or b.name.startswith("ik_"):
        b.use_deform = False                                    # control bones never pull on the skin

# ---------------------------------------------------------------- 0. pristine copy (for the custom-normal transfer at the end)
ref = body.copy(); ref.data = body.data.copy(); ref.name = "Relic_AI_normals_ref"
bpy.context.scene.collection.objects.link(ref)
for m in list(ref.modifiers):
    ref.modifiers.remove(m)
ref.parent = None
ref.matrix_world = body.matrix_world.copy()

# ---------------------------------------------------------------- 1. labels
bm = bmesh.new()
bm.from_mesh(body.data)
dl = bm.verts.layers.deform.active
gname = {g.index: g.name for g in body.vertex_groups}
gidx = {g.name: g.index for g in body.vertex_groups}


def vlabel(v):
    d = v[dl]
    if not d:
        return "none"
    return gname[max(d.items(), key=lambda kv: kv[1])[0]]


def seg_of(name):
    """the rigid segment (or 'torso' / 'hand.X') a bone's faces belong to"""
    if name in TORSO or name == "none":
        return "torso"
    base, sd = name.split(".")
    if base in ("forearm", "hand", "fist1", "fist2"):
        return ("forearm." + sd) if base == "forearm" else ("hand." + sd)
    return name


bm.faces.ensure_lookup_table()
flab = {}
for f in bm.faces:
    cnt = {}
    for v in f.verts:
        s = seg_of(vlabel(v))
        cnt[s] = cnt.get(s, 0) + 1
    flab[f] = max(cnt, key=cnt.get)
# smooth: a face whose neighbours mostly carry another label takes that label (removes speckle)
for _it in range(3):
    ch = 0
    new = {}
    for f in bm.faces:
        cnt = {}
        for e in f.edges:
            for g in e.link_faces:
                if g is not f:
                    cnt[flab[g]] = cnt.get(flab[g], 0) + 1
        if cnt:
            best = max(cnt, key=cnt.get)
            if best != flab[f] and cnt[best] > 0.6 * sum(cnt.values()):
                new[f] = best
    for f, l in new.items():
        flab[f] = l
    print("SMOOTH pass", _it, "relabelled", len(new))


# ---------------------------------------------------------------- 2. clean planar cuts through the hinge joints
def bisect_joint(upper, lower, joint_bone, radius):
    """cut the faces of segments upper|lower near joint_bone's head with the plane through the joint, normal = bisector of the two bones"""
    J = B[joint_bone].head_local
    du = (B[upper].tail_local - B[upper].head_local).normalized()
    dlw = (B[lower].tail_local - B[lower].head_local).normalized()
    n = (du + dlw).normalized()
    faces = [f for f in bm.faces if flab[f] in (seg_of(upper), seg_of(lower)) and (f.calc_center_median() - J).length < radius]
    if not faces:
        return 0
    lab_before = {f: flab[f] for f in faces}
    geom = list({e for f in faces for e in f.edges}) + faces + list({v for f in faces for v in f.verts})
    res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=J, plane_no=n, clear_inner=False, clear_outer=False)
    bm.faces.ensure_lookup_table()
    # every face of the two segments near the joint is now on one side: label it by the side
    k = 0
    for f in bm.faces:
        if f not in flab:
            # a new face from the cut: inherit from a neighbour of the same segments
            flab[f] = seg_of(upper)
        if flab[f] in (seg_of(upper), seg_of(lower)) and (f.calc_center_median() - J).length < radius:
            side = (f.calc_center_median() - J).dot(n)
            flab[f] = seg_of(lower) if side > 0 else seg_of(upper)
            k += 1
    return k


def bisect_root(limb, up_reach, down_reach, radius):
    """the joint where a limb meets the body (hip / shoulder): cut every torso or limb face inside a capsule around the top of the limb
    with the plane through the joint, normal = the limb's rest direction. Above the plane -> body, below -> limb. (Plates that reached
    above the hip on the thigh stuck up out of the pelvis in a crouch.)"""
    b = B[limb]
    J = b.head_local
    n = (b.tail_local - b.head_local).normalized()
    other = B[limb[:-1] + ("R" if limb.endswith("L") else "L")].head_local

    def in_region(f):
        c = f.calc_center_median()
        s = (c - J).dot(n)
        if s < -up_reach or s > down_reach:
            return False
        if ((c - J) - n * s).length > radius:
            return False
        return (c - J).length <= (c - other).length            # this side's joint, not the other one's
    labs = ("torso", seg_of(limb))
    faces = [f for f in bm.faces if flab[f] in labs and in_region(f)]
    if not faces:
        return 0
    geom = list({e for f in faces for e in f.edges}) + faces + list({v for f in faces for v in f.verts})
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=J, plane_no=n, clear_inner=False, clear_outer=False)
    bm.faces.ensure_lookup_table()
    k = 0
    for f in bm.faces:
        if f not in flab:
            flab[f] = "torso"
        if flab[f] in labs and in_region(f):
            flab[f] = seg_of(limb) if (f.calc_center_median() - J).dot(n) > 0 else "torso"
            k += 1
    return k


for sd in "LR":
    print("CUT hip", sd, bisect_root("thigh." + sd, 0.15, 0.30, 0.24))
    # NO cut at the shoulders: Relic's shoulder pads sit across the joint and a cut sliced them in half; the shoulders keep the smooth
    # v6 weights (they looked right in every stress pose) and the inner body fills the armpit
for sd in "LR":
    print("CUT elbow", sd, bisect_joint("upperarm." + sd, "forearm." + sd, "forearm." + sd, 0.22))
    print("CUT knee", sd, bisect_joint("thigh." + sd, "shin." + sd, "shin." + sd, 0.28))
    print("CUT ankle", sd, bisect_joint("shin." + sd, "foot." + sd, "foot." + sd, 0.22))
    print("CUT toe", sd, bisect_joint("foot." + sd, "toe." + sd, "toe." + sd, 0.14))


def chain_by_planes(chain):
    """which segment of a limb a face belongs to is decided by POSITION only (the joint planes along the limb), for the WHOLE limb - the
    original weights had a big forearm plate partly on the upper arm (25 cm below the elbow), and its loose cap floated in the air"""
    planes = []
    for a_, b_ in zip(chain[:-1], chain[1:]):
        J = B[b_].head_local
        n = ((B[a_].tail_local - B[a_].head_local).normalized() + (B[b_].tail_local - B[b_].head_local).normalized()).normalized()
        planes.append((J, n))
    region = {seg_of(c) for c in chain}
    k = 0
    for f in bm.faces:
        if flab[f] not in region:
            continue
        c = f.calc_center_median()
        seg = chain[0]
        for (J, n), nxt in zip(planes, chain[1:]):
            if (c - J).dot(n) > 0:
                seg = nxt
            else:
                break
        nl = seg_of(seg)
        if nl != flab[f]:
            flab[f] = nl
            k += 1
    return k


for sd in "LR":
    print("CHAIN arm", sd, "relabelled", chain_by_planes(["upperarm." + sd, "forearm." + sd, "hand." + sd]))
    print("CHAIN leg", sd, "relabelled", chain_by_planes(["thigh." + sd, "shin." + sd, "foot." + sd, "toe." + sd]))

# ---------------------------------------------------------------- 2a. small loose pieces follow the bone they SIT next to (not the noisy
# original weights): an 86-vertex piece by the left wrist was weighted 100% to the upper arm and floated in the air when the elbow bent.
SMALL_PIECE = 600
DEFORM = [b for b in B if b.use_deform and not b.name.startswith("ik_") and b.name != "root"]


def seg_dist(p, b):
    h, t = b.head_local, b.tail_local
    d = t - h
    s = max(0.0, min(1.0, (p - h).dot(d) / max(d.length_squared, 1e-9)))
    return (p - (h + d * s)).length


from mathutils import kdtree as _kdtree
_seen = set()
pieces_ = []
for f0 in bm.faces:
    if f0 in _seen:
        continue
    stack, comp = [f0], []
    _seen.add(f0)
    while stack:
        g = stack.pop()
        comp.append(g)
        for e in g.edges:
            for h in e.link_faces:
                if h not in _seen:
                    _seen.add(h)
                    stack.append(h)
    pieces_.append(comp)
big_v = []
for comp in pieces_:
    vs = {v for g in comp for v in g.verts}
    if len(vs) >= SMALL_PIECE:
        for g in comp:
            for v in g.verts:
                big_v.append((v.co.copy(), flab[g]))
kdb = _kdtree.KDTree(len(big_v))
for i, (c_, _l) in enumerate(big_v):
    kdb.insert(c_, i)
kdb.balance()
n_small = 0
for comp in pieces_:
    vs = list({v for g in comp for v in g.verts})
    if len(vs) >= SMALL_PIECE:
        continue
    # a small piece follows the armour it TOUCHES (the label of the nearest big piece), so it can never drift away from it
    votes = {}
    for v in vs[:: max(1, len(vs) // 24)]:
        _c, j, _d = kdb.find(v.co)
        votes[big_v[j][1]] = votes.get(big_v[j][1], 0) + 1
    lab = max(votes, key=votes.get)
    # a cap sitting AT a hinge (elbow point, knee cap, heel spur) belongs to the lower bone, like the real elbow point is part of the
    # forearm: hanging off the end of the upper arm, an elbow cap floated in the air when the elbow bent
    cen = sum((v.co for v in vs), Vector()) / len(vs)
    for sd in "LR":
        for jb, R_ in (("forearm." + sd, 0.13), ("shin." + sd, 0.15), ("foot." + sd, 0.13)):
            if (cen - B[jb].head_local).length < R_ and lab in (seg_of(B[jb].parent.name), seg_of(jb)):
                if lab != seg_of(jb):
                    print("   CAP at", jb, "->", seg_of(jb), "(was", lab + ")", len(vs), "verts")
                lab = seg_of(jb)
    for g in comp:
        flab[g] = lab
    n_small += 1
print("SMALL pieces attached to the armour they touch:", n_small)

# ---------------------------------------------------------------- 2b. no specks: a small patch of faces whose label differs from everything
# around it (left by the noisy original weights) would be cut loose and fly off with another bone as a floating fleck; it takes the label of
# the faces it borders most instead. A whole separate small piece (a bolt) keeps its own label: it moves as one.
MIN_FACES = 80


def label_patches():
    seen = set()
    out = []
    for f in bm.faces:
        if f in seen:
            continue
        lab = flab[f]
        stack, comp = [f], []
        seen.add(f)
        while stack:
            g = stack.pop()
            comp.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen and flab[h] == lab:
                        seen.add(h)
                        stack.append(h)
        out.append(comp)
    return out


for _it in range(4):
    fixed = 0
    for comp in label_patches():
        if len(comp) >= MIN_FACES:
            continue
        cs = set(comp)
        cnt = {}
        for g in comp:
            for e in g.edges:
                for h in e.link_faces:
                    if h not in cs:
                        cnt[flab[h]] = cnt.get(flab[h], 0) + 1
        if not cnt:
            continue
        best = max(cnt, key=cnt.get)
        for g in comp:
            flab[g] = best
        fixed += 1
    print("SPECKS pass", _it, "patches relabelled", fixed)
    if fixed == 0:
        break

# ---------------------------------------------------------------- 3. split the mesh where segments meet; rigid weights for limb segments
SPLIT_PAIRS_OK = lambda a, b: a != b and not (a == "torso" and b == "torso") and \
    not ({a, b} == {"forearm.L", "hand.L"} or {a, b} == {"forearm.R", "hand.R"}) and \
    not ({a, b} == {"torso", "upperarm.L"} or {a, b} == {"torso", "upperarm.R"})       # the shoulders stay smooth
SHOULDERS = [B["upperarm.L"].head_local, B["upperarm.R"].head_local]
near_shoulder = lambda co, r: min((co - s).length for s in SHOULDERS) < r
cut_edges = [e for e in bm.edges if len(e.link_faces) == 2 and SPLIT_PAIRS_OK(flab[e.link_faces[0]], flab[e.link_faces[1]])]
pre_islands = len(islands_of_bm(bm)) if 'islands_of_bm' in globals() else None
bmesh.ops.split_edges(bm, edges=cut_edges)
print("SPLIT edges", len(cut_edges))
# build check: the cut must not leave tiny loose pieces (each would be a floating fleck)
_seen = set(); _tiny = 0
for v0 in bm.verts:
    if v0 in _seen:
        continue
    stack, n_ = [v0], 0
    _seen.add(v0)
    while stack:
        x = stack.pop(); n_ += 1
        for e in x.link_edges:
            y = e.other_vert(x)
            if y not in _seen:
                _seen.add(y); stack.append(y)
    if n_ < 12:
        _tiny += 1
print("CHECK tiny loose pieces after the cut (also counts tiny pieces the original mesh already had):", _tiny)
bm.verts.ensure_lookup_table()
nr = nt_ = 0
TORSO_IDX = {gidx[n] for n in TORSO if n in gidx}
P0_, up_ = B['pelvis'].head_local, (B['neck'].head_local - B['pelvis'].head_local)
Lax_ = up_.length; up_ = up_ / Lax_
for v in bm.verts:
    labs = {flab[f] for f in v.link_faces}
    if len(labs) != 1:
        continue
    lab = labs.pop()
    d = v[dl]
    old = dict(d.items())
    rs_ = min((v.co - s).length for s in SHOULDERS)
    new = None
    if lab.split(".")[0] in RIGID or lab.startswith("forearm."):
        new = {gidx[lab]: 1.0}
        if lab.startswith("upperarm."):
            # the top of the upper arm blends smoothly into the shoulder: old (smooth) weights at the joint, rigid from 22 cm down
            x = min(1.0, max(0.0, (rs_ - 0.10) / 0.12)); x = x * x * (3 - 2 * x)
            new = {k: w * (1 - x) for k, w in old.items()}
            new[gidx[lab]] = new.get(gidx[lab], 0.0) + x
        nr += 1
    elif lab == "torso":
        # torso armour follows torso bones only (no pull toward a limb, so it cannot stretch when a leg or arm swings) - except around
        # the shoulders, where the body side of the joint keeps its smooth weights, fading out between 18 and 30 cm from the joint
        keep_ = {k: w for k, w in old.items() if k in TORSO_IDX}
        if len(keep_) != len(old):
            if not keep_:
                t_ = (v.co - P0_).dot(up_) / Lax_
                keep_ = {gidx["pelvis" if t_ < 0.3 else "chest"]: 1.0}
            s_ = sum(keep_.values())
            freed = {k: w / s_ for k, w in keep_.items()}
            x = min(1.0, max(0.0, (rs_ - 0.18) / 0.12)); x = x * x * (3 - 2 * x)
            new = {k: old.get(k, 0.0) * (1 - x) + freed.get(k, 0.0) * x for k in set(old) | set(freed)}
            nt_ += 1
    if new is not None:
        tot = sum(new.values())
        for k in list(d.keys()):
            del d[k]
        for k, w in new.items():
            if w / tot > 1e-4:
                d[k] = w / tot
print("RIGID vertices", nr, " torso vertices freed from limb weights", nt_)
bm.to_mesh(body.data)
bm.free()
body.data.update()

# ---------------------------------------------------------------- 4. the original shading (custom normals) back onto the cut mesh
dt = body.modifiers.new("NormalsFromOriginal", 'DATA_TRANSFER')
dt.object = ref
dt.use_loop_data = True
dt.data_types_loops = {'CUSTOM_NORMAL'}
dt.loop_mapping = 'POLYINTERP_NEAREST'
dt.mix_mode = 'REPLACE'
body.modifiers.move(len(body.modifiers) - 1, 0)          # before the armature: transfer in the rest shape
with bpy.context.temp_override(object=body, active_object=body, selected_objects=[body]):
    bpy.ops.object.modifier_apply(modifier=dt.name)
bpy.data.objects.remove(ref, do_unlink=True)
print("NORMALS transferred; mesh now", len(body.data.vertices), "verts", len(body.data.polygons), "faces")

# ---------------------------------------------------------------- 5. the inner body
tor = json.load(open(OUT + r"\anim_cache5\relic_torso.json"))
P0, P1 = B['pelvis'].head_local, B['neck'].head_local
up = (P1 - P0); Lax = up.length; up.normalize()
lf = B['upperarm.L'].head_local - B['upperarm.R'].head_local
lf = (lf - up * lf.dot(up)).normalized()
fw = lf.cross(up).normalized()
INNER = 0.74                     # the undersuit sits at this fraction of the armour's outer reach
mats = bpy.data.materials.get("RelicUndersuit") or bpy.data.materials.new("RelicUndersuit")
mats.use_nodes = True
bs = mats.node_tree.nodes["Principled BSDF"]
bs.inputs["Base Color"].default_value = (0.018, 0.017, 0.016, 1.0)       # dark and matte: reads as a recess between the plates
bs.inputs["Metallic"].default_value = 0.25
bs.inputs["Roughness"].default_value = 0.8
us_me = bpy.data.meshes.new("Relic_Undersuit")
ub = bmesh.new()
SEG = 20
vgroups = {}            # vertex -> {bone: w}


def ring(center, a1, a2, ext, seg=SEG):
    """ellipse ring: a1/a2 unit axes; ext = (+a1, -a1, +a2, -a2) reaches"""
    vs = []
    for i in range(seg):
        th = 2 * math.pi * i / seg
        c, s = math.cos(th), math.sin(th)
        r1 = ext[0] if c >= 0 else ext[1]
        r2 = ext[2] if s >= 0 else ext[3]
        vs.append(ub.verts.new(center + a1 * (c * r1) + a2 * (s * r2)))
    return vs


def bridge(r0, r1):
    n = len(r0)
    for i in range(n):
        ub.faces.new((r0[i], r0[(i + 1) % n], r1[(i + 1) % n], r1[i]))


def cap(r, center, flip=False):
    c = ub.verts.new(center)
    n = len(r)
    for i in range(n):
        ub.faces.new((r[(i + 1) % n], r[i], c) if not flip else (r[i], r[(i + 1) % n], c))
    return c


def weigh(verts, wfun):
    for v in verts:
        vgroups[v] = wfun(v.co)


# torso tube along the spine: rings at the measured heights, inside the armour
rows = [r for r in tor['rows'] if -0.2 <= r['t'] <= 1.05]
tb = {n: ((B[n].head_local - P0).dot(up) / Lax) for n in ("spine", "chest", "neck")}
breaks = [("pelvis", "spine", tb["spine"]), ("spine", "chest", tb["chest"]), ("chest", "neck", tb["neck"])]


def torso_w(co):
    t = (co - P0).dot(up) / Lax
    w = {"pelvis": 1.0}
    for a, b, tb_ in breaks:
        x = min(1.0, max(0.0, (t - (tb_ - 0.07)) / 0.14))
        x = x * x * (3 - 2 * x)
        if x > 0:
            w = {k: v * (1 - x) for k, v in w.items()}
            w[b] = w.get(b, 0.0) + x
    return {k: v for k, v in w.items() if v > 1e-3}


prev = None
first = None
rows = [r for r in rows if r['t'] >= 0.0]                 # not below the pelvis: the hips are filled by the thigh tubes and hip balls
for k, r in enumerate(rows):
    ce = P0 + up * (r['t'] * Lax)
    half = min(r['left'], r['right'])                     # the body is symmetric (one hip plate made the right side read 32 cm)
    f_ = INNER * (0.85 if r['t'] < 0.2 else 1.0)          # a little slimmer at the pelvis
    ext = tuple(max(0.05, x) for x in (r['front'] * f_, r['back'] * f_, half * f_, half * f_))
    if k == 0:
        # rounded bottom: a smaller ring a little below, so the end is a dome, not a flat lid
        rb_ = ring(ce - up * 0.05, fw, lf, tuple(x * 0.6 for x in ext))
        weigh(rb_, torso_w)
        first = (rb_, ce - up * 0.07)
        prev = rb_
    rr = ring(ce, fw, lf, ext)
    weigh(rr, torso_w)
    bridge(prev, rr)
    prev = rr
weigh([cap(first[0], first[1], flip=True)], torso_w)
last_c = P0 + up * (rows[-1]['t'] * Lax)
weigh([cap(prev, last_c + up * 0.02)], torso_w)
print("UNDERSUIT torso rings", len(rows))

# neck tube to the head
hc = P0 + up * (tor['head']['center_t'] * Lax) + fw * tor['head']['center_f']


def tube(a, b, ra, rb, bone, overlap=0.6, seg=16):
    d = (b - a); L = d.length; d.normalize()
    a1 = d.orthogonal().normalized(); a2 = d.cross(a1).normalized()
    a_ = a - d * (ra * overlap); b_ = b + d * (rb * overlap)
    r0 = ring(a_, a1, a2, (ra, ra, ra, ra), seg)
    r1 = ring(b_, a1, a2, (rb, rb, rb, rb), seg)
    bridge(r0, r1)
    c0 = cap(r0, a_ - d * ra * 0.4, flip=True)
    c1 = cap(r1, b_ + d * rb * 0.4)
    weigh(r0 + r1 + [c0, c1], lambda co, bone=bone: {bone: 1.0})


# limb radii: how far in the bone's own armour starts (12th percentile of radial distance), sampled at both ends
me = body.data
co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
gi2 = {g.index: g.name for g in body.vertex_groups}
own = {}
for v in me.vertices:
    if v.groups:
        g = max(v.groups, key=lambda g: g.weight)
        own.setdefault(gi2[g.group], []).append(v.index)


def inner_r(bone, at):
    b = B[bone]
    h, t = np.array(b.head_local), np.array(b.tail_local)
    ax = (t - h) / np.linalg.norm(t - h)
    p = h + (t - h) * at
    idx = own.get(bone, []) + (own.get("hand." + bone.split(".")[1], []) if bone.startswith("forearm") else [])
    if not idx:
        return 0.04
    d = co[idx] - p
    s = d @ ax
    near = np.abs(s) < 0.04
    if near.sum() < 15:
        return 0.04
    rad = np.linalg.norm(d[near] - np.outer(s[near], ax), axis=1)
    return float(np.clip(np.percentile(rad, 12) * 0.88, 0.025, 0.14))


made = []
for sd in "LR":
    for bone, child in (("upperarm." + sd, None), ("forearm." + sd, None), ("thigh." + sd, None), ("shin." + sd, None), ("foot." + sd, None)):
        b = B[bone]
        ra, rb = inner_r(bone, 0.15), inner_r(bone, 0.85)
        tube(b.head_local, b.tail_local, ra, rb, bone)
        made.append((bone, round(ra, 3), round(rb, 3)))
    # shoulder bridge: from the upper chest to the shoulder joint, so a raised arm never opens a hole under the shoulder plate
    sh = B["upperarm." + sd].head_local
    tube(P0 + up * (0.78 * Lax), sh, 0.11, inner_r("upperarm." + sd, 0.1), "clavicle." + sd, overlap=0.3)
tube(P0 + up * (0.95 * Lax), hc, 0.10, 0.09, "neck", overlap=0.2)
print("UNDERSUIT limbs", made)
ub.to_mesh(us_me)
us = bpy.data.objects.new("Relic_Undersuit", us_me)
body.users_collection[0].objects.link(us)
us.parent = rig
us.matrix_parent_inverse = Matrix.Identity(4)
us.matrix_world = body.matrix_world.copy()               # same space as the body (the armature's rest space)
us.data.materials.append(mats)
for p in us.data.polygons:
    p.use_smooth = True
# weights: vgroups was filled in vertex-creation order, which is the mesh's vertex order
vg = {}
ub.free()
order = list(vgroups.values())
assert len(order) == len(us_me.vertices), (len(order), len(us_me.vertices))
for i, w in enumerate(order):
    for bn, x in w.items():
        if bn not in vg:
            vg[bn] = us.vertex_groups.new(name=bn)
        vg[bn].add([i], x, 'REPLACE')
am = us.modifiers.new("Armature", 'ARMATURE'); am.object = rig
# recompute normals outward on the undersuit
bm2 = bmesh.new(); bm2.from_mesh(us_me); bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces); bm2.to_mesh(us_me); bm2.free()
print("UNDERSUIT", len(us_me.vertices), "verts")
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\fight_v07.blend")
print("MODEL v7 SAVED")
