import bpy, sys, json
sc = bpy.data.scenes["Street"]
who = sys.argv[sys.argv.index("--") + 1]
moves = sys.argv[sys.argv.index("--") + 2:]
spans = json.load(open(r"C:\ai\unedited\anim_cache5\fm_spans_%s.json" % who))
for nm in moves:
    s, dur, hit = spans[nm]
    offs = sorted({0, int(hit * 0.4), int(hit * 0.75), hit, hit + 3, dur})
    for o in offs:
        sc.frame_set(int(s + o))
        sc.render.filepath = r"C:\ai\unedited\renders\fm_%s_%s_%02d.png" % (who, nm, o)
        bpy.ops.render.render(write_still=True)
    print("OFFS", nm, offs)
