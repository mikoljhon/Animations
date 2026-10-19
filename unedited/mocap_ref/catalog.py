import numpy as np, json, sys
d = np.load(r"C:\ai\mocap\data\seq30.npz", allow_pickle=True)
loc = d["local"].astype(np.float32); root = d["root"]; starts = list(d["starts"]) + [len(loc)]; clips = list(d["clips"]); desc = d["desc"]
J = {n: k for k, n in enumerate(d["joints"])}
LEG = float(d["leg"])
def clip(name):
    i = clips.index(name); return loc[starts[i]:starts[i+1]], root[starts[i]:starts[i+1]], str(desc[i])
cands = ["13_17","13_18","14_01","14_02","14_03","15_13","17_10","79_08","80_10","143_23","144_13","144_14","144_20","144_21","02_05","111_19","113_13","141_14","86_01","86_06","86_08","56_03","76_01","76_02"]
out = []
for nm in cands:
    L, R, ds = clip(nm)
    n = len(L)
    for side in ("L", "R"):
        w = L[:, J["wrist_" + side]]
        sh = L[:, J["shoulder_" + side]]
        fwd = -(w[:, 1] - sh[:, 1])            # forward reach of the fist beyond the shoulder (units of leg)
        # peaks of fwd with prominence
        k = 0
        while k < n:
            if fwd[k] > 0.50:
                a = k
                while k < n and fwd[k] > 0.50: k += 1
                b = k
                pk = a + int(np.argmax(fwd[a:b]))
                lat = np.abs(w[max(0,pk-12):pk+1, 0] - w[max(0,pk-12), 0]).max()
                out.append(dict(clip=nm, desc=ds[:40], side=side, peak=int(pk), start=int(max(0, pk-20)), end=int(min(n-1, pk+16)), reach=float(fwd[pk]), lat=float(lat), hipsz=float(R[pk,2])))
            else:
                k += 1
print(len(out), "punch candidates")
json.dump(out, open("punches.json", "w"))
import collections
c = collections.Counter((o["clip"], o["side"]) for o in out)
print(sorted(c.items())[:60])
