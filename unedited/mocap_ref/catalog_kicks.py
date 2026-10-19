import numpy as np, json
d = np.load(r"C:\ai\mocap\data\seq30.npz", allow_pickle=True)
loc = d["local"].astype(np.float32); root = d["root"]; starts = list(d["starts"]) + [len(loc)]; clips = list(d["clips"]); desc = d["desc"]
J = {n: k for k, n in enumerate(d["joints"])}
cands = ["135_04","144_05","144_06","144_09","144_10","74_03","74_04","74_05","74_06","143_24","111_19","113_13","141_14","86_01","86_03","86_06","86_08","75_16","02_05","14_01","13_17","10_01","10_02","90_05","87_01"]
out = []
for nm in cands:
    i = clips.index(nm); L = loc[starts[i]:starts[i+1]]; n = len(L)
    for side in "LR":
        a = L[:, J["ankle_" + side]]
        fw = -a[:, 1]; up = a[:, 2]; lat = a[:, 0] * (1 if side == "L" else -1)   # + = outward
        mask = (up > -0.45) & (fw > 0.30)
        k = 0
        while k < n:
            if mask[k]:
                s = k
                while k < n and mask[k]: k += 1
                e = k
                pk = s + int(np.argmax(up[s:e] + fw[s:e]))
                out.append(dict(clip=nm, desc=str(desc[i])[:40], side=side, peak=int(pk), start=int(max(0, s - 22)), end=int(min(n - 1, e + 18)), height=float(up[pk]), fwd=float(fw[pk]), lat=float(lat[pk]), dur=int(e - s)))
            else:
                k += 1
print(len(out))
json.dump(out, open("kicks.json", "w"))
for o in sorted(out, key=lambda o: (o["clip"], o["peak"])): print(o["clip"], o["side"], o["peak"], "h%.2f f%.2f lat%.2f dur%d" % (o["height"], o["fwd"], o["lat"], o["dur"]), "|", o["desc"])
