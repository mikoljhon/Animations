import numpy as np, json
exec(open("anatomy.py").read())
d = np.load(r"C:\ai\mocap\data\seq30.npz", allow_pickle=True)
loc = d["local"].astype(np.float32); root = d["root"].astype(np.float32); head = d["heading"].astype(np.float32)
starts = list(d["starts"]) + [len(loc)]; clips = list(d["clips"]); desc = d["desc"]; LEG = float(d["leg"])
names = [str(n) for n in d["joints"]]
J = {n: k for k, n in enumerate(names)}
SW = {}
for n in names:
    if n.endswith("_L"): SW[J[n]] = J[n[:-2] + "_R"]
    elif n.endswith("_R"): SW[J[n]] = J[n[:-2] + "_L"]
perm = np.arange(len(names))
for a, b in SW.items(): perm[a] = b

def mirror(L, R, H):
    L = L[:, perm].copy(); L[:, :, 0] *= -1.0
    # mirror the root path and heading across the start heading axis
    h0 = H[0]; left0 = np.array([-h0[1], h0[0]])
    d = R[:, :2] - R[0, :2]
    f, l = d @ h0, d @ left0
    R2 = R.copy(); R2[:, :2] = R[0, :2] + np.outer(f, h0) - np.outer(l, left0)
    hf, hl = H @ h0, H @ left0
    H2 = np.outer(hf, h0) - np.outer(hl, left0)
    return L, R2, H2

KEYJ = [J[n] for n in ("wrist_L", "wrist_R", "ankle_L", "ankle_R", "knee_L", "knee_R", "head", "elbow_L", "elbow_R")]
def speed(L):
    v = np.linalg.norm(np.diff(L[:, KEYJ], axis=0), axis=2).sum(1)
    v = np.convolve(v, np.ones(5) / 5.0, mode="same")
    return np.concatenate([[v[0]], v])

def cut(name, clip, peak, mirror_it=False, before=40, after=40, fixed=None):
    i = clips.index(clip); s = starts[i]; e = starts[i + 1]
    L = loc[s:e]; R = root[s:e]; H = head[s:e]
    if fixed:
        a, b = fixed
    else:
        v = speed(L)
        lo = max(0, peak - before); hi = min(len(L) - 1, peak + after)
        a = lo + int(np.argmin(v[lo:max(lo + 1, peak - 8)])) if peak - 8 > lo else lo
        b = peak + 6 + int(np.argmin(v[peak + 6:hi])) if hi > peak + 7 else hi
    Ls, Rs, Hs = L[a:b + 1], R[a:b + 1], H[a:b + 1]
    if mirror_it: Ls, Rs, Hs = mirror(Ls, Rs, Hs)
    # some takes of the BVH conversion have a whole arm flipped (the elbow bends backward): repair it (anatomy.py)
    Ls, fixed = repair_flipped_arms(Ls, J)
    np.savez("segs/%s.npz" % name, local=Ls, root=Rs, heading=Hs, leg=LEG, fps=30.0, desc=str(desc[i]), clip=clip, a=a, b=b,
             arm_repair=str(fixed))
    print("%-14s %s [%d..%d] %d frames (%.1fs) mirror=%s  arms repaired: %s" % (name, clip, a, b, len(Ls), len(Ls) / 30.0, mirror_it, fixed or "none"))

jobs = json.load(open("jobs.json"))
for j in jobs:
    cut(**j)
