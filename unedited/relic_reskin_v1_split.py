# relic_reskin.py : stop Relic's armor from stretching at the limb joints.
#   The AI-made mesh has big plates that cross the shoulder / elbow / wrist / hip / knee / ankle with smoothly blended weights, so when the joint bends
#   the faces over it stretch into long slivers (up to 0.37 m).  Here every face that belongs to a LIMB bone is cut away from the faces that belong to the
#   next limb bone and each piece is attached rigidly (weight 1.0) to its own bone, like real armor plates: pieces slide over each other at the joint,
#   nothing stretches.  Torso, neck and head keep their smooth weights.
#   blender.exe -b fight_v01.blend --python relic_reskin.py   -> saves fight_v03.blend
import bpy, bmesh, numpy as np, math

OUT = r"C:\ai\unedited"
LIMB = ["clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe"]


def is_limb(n):
    return n.split(".")[0] in LIMB


def reskin(ob):
    me = ob.data
    gi = {g.index: g.name for g in ob.vertex_groups}
    # dominant bone per vertex
    dom = []
    for v in me.vertices:
        dom.append(gi[max(v.groups, key=lambda g: g.weight).group] if v.groups else "none")
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table(); bm.edges.ensure_lookup_table()
    # face cluster = most common dominant bone of its vertices (ties: first)
    fclu = []
    for f in bm.faces:
        names = [dom[v.index] for v in f.verts]
        fclu.append(max(set(names), key=names.count))
    # two smoothing passes: a face whose neighbours all agree on another bone joins them (removes confetti)
    for _ in range(2):
        new = list(fclu)
        for f in bm.faces:
            nb = [fclu[o.index] for e in f.edges for o in e.link_faces if o is not f]
            if nb:
                top = max(set(nb), key=nb.count)
                if nb.count(top) >= max(2, len(nb) - 0) and top != fclu[f.index]:
                    new[f.index] = top
        fclu = new
    # cut along edges between faces of different limb clusters (only where at least one side is a limb bone)
    cut = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            a, b = fclu[e.link_faces[0].index], fclu[e.link_faces[1].index]
            if a != b and (is_limb(a) or is_limb(b)) and not ({a, b} <= {"chest", "spine", "pelvis", "neck", "head"}):
                cut.append(e)
    n_before = len(bm.verts)
    bmesh.ops.split_edges(bm, edges=cut)
    bm.verts.index_update(); bm.faces.index_update(); bm.edges.index_update()
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    # a face's cluster is unchanged by the split; recompute per-vertex cluster from its faces (all faces of a vertex now share one cluster across a cut)
    vclu = {}
    for f in bm.faces:
        c = fclu[f.index] if f.index < len(fclu) else None
    # (faces keep their order, so fclu still indexes them)
    for f in bm.faces:
        for v in f.verts:
            vclu.setdefault(v.index, []).append(fclu[f.index])
    bm.to_mesh(me)
    bm.free()
    me.update()
    # weights: limb clusters rigid; the rest keep their previous (smooth) weights where the vertex was not duplicated... simpler: rebuild all weights
    return vclu, n_before, len(me.vertices)


def apply():
    rig = bpy.data.objects["Relic_AI_Rig"]
    ob = bpy.data.objects["Relic_AI"]
    # remember the old smooth weights per original vertex (by position) so non-limb vertices keep them
    me0 = ob.data
    gi = {g.index: g.name for g in ob.vertex_groups}
    old = {}
    for v in me0.vertices:
        old[tuple(np.round(v.co, 5))] = {gi[g.group]: g.weight for g in v.groups}
    vclu, n0, n1 = reskin(ob)
    me = ob.data
    # clear and rebuild the groups
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    groups = {}
    def grp(n):
        if n not in groups:
            groups[n] = ob.vertex_groups.new(name=n)
        return groups[n]
    rigid = 0
    for v in me.vertices:
        cl = vclu.get(v.index, [])
        top = max(set(cl), key=cl.count) if cl else None
        if top is not None and (is_limb(top) or top == 'pelvis'):
            grp(top).add([v.index], 1.0, 'REPLACE')
            rigid += 1
        else:
            w = old.get(tuple(np.round(v.co, 5)))
            if w:
                for n, x in w.items():
                    grp(n).add([v.index], x, 'REPLACE')
            elif top is not None:
                grp(top).add([v.index], 1.0, 'REPLACE')
    print("RESKIN verts %d -> %d, rigid limb verts %d" % (n0, n1, rigid))
    # the mud crust copy follows the same mesh
    cr = bpy.data.objects.get("Relic_AI_Crust")
    if cr is not None:
        cr.data = ob.data.copy()
        for g in list(cr.vertex_groups):
            cr.vertex_groups.remove(g)
        for g in ob.vertex_groups:
            cr.vertex_groups.new(name=g.name)
        for v in ob.data.vertices:
            for g in v.groups:
                cr.vertex_groups[ob.vertex_groups[g.group].name].add([v.index], g.weight, 'REPLACE')


apply()
bpy.ops.wm.save_as_mainfile(filepath=OUT + r"\fight_v03.blend")
print("RESKIN SAVED")
