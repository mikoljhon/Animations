# raw_strip.py : draw a cut segment straight from its .npz (no Blender, no retarget) as stick figures, side and front view,
#   to judge the raw capture itself.   python raw_strip.py <seg name> [every_n_frames]   -> raw_<seg>.png
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
NAMES = ['hips', 'spine', 'chest', 'neck', 'head', 'eye_L', 'eye_R', 'clav_L', 'shoulder_L', 'elbow_L', 'wrist_L', 'knuckle_L', 'thumb_L', 'clav_R',
         'shoulder_R', 'elbow_R', 'wrist_R', 'knuckle_R', 'thumb_R', 'hip_L', 'knee_L', 'ankle_L', 'toe_L', 'hip_R', 'knee_R', 'ankle_R', 'toe_R']
J = {n: i for i, n in enumerate(NAMES)}
B = [("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"), ("chest", "clav_L"), ("clav_L", "shoulder_L"), ("shoulder_L", "elbow_L"),
     ("elbow_L", "wrist_L"), ("wrist_L", "knuckle_L"), ("chest", "clav_R"), ("clav_R", "shoulder_R"), ("shoulder_R", "elbow_R"), ("elbow_R", "wrist_R"),
     ("wrist_R", "knuckle_R"), ("hips", "hip_L"), ("hip_L", "knee_L"), ("knee_L", "ankle_L"), ("ankle_L", "toe_L"), ("hips", "hip_R"), ("hip_R", "knee_R"),
     ("knee_R", "ankle_R"), ("ankle_R", "toe_R")]
seg = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 6
first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
last = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 6
d = np.load(r"C:\ai\unedited\mocap_ref\segs\%s.npz" % seg, allow_pickle=True)
L = d["local"]                     # (F, J, 3): x = the person's left, y = back, z = up
F = len(L)
frs = list(range(first, min(F, last + 1), every))[:8]
fig, ax = plt.subplots(2, len(frs), figsize=(3.2 * len(frs), 9), squeeze=False)
for c, i in enumerate(frs):
    P = L[i]
    for r, (u, nm) in enumerate(((1, "side (facing left)"), (0, "front"))):
        a = ax[r, c]
        for p, q in B:
            col = "tab:blue" if p.endswith("_L") or q.endswith("_L") else ("tab:red" if p.endswith("_R") or q.endswith("_R") else "k")
            hu = P[J[p], u] if u == 1 else -P[J[p], 0]
            hv = P[J[q], u] if u == 1 else -P[J[q], 0]
            a.plot([hu, hv], [P[J[p], 2], P[J[q], 2]], color=col, lw=2)
        a.set_aspect("equal"); a.set_xlim(-0.8, 0.8); a.set_ylim(-0.3, 1.1); a.grid(True, alpha=0.3)
        a.axvline(P[J["neck"], u] if u == 1 else -P[J["neck"], 0], color="gray", lw=0.6, ls="--")
        a.set_title("f%d  %s" % (i, nm), fontsize=10)
plt.suptitle("%s  (%s)  raw capture: blue = left, red = right;  side view: the person faces LEFT" % (seg, str(d["desc"])), fontsize=10)
plt.tight_layout()
plt.savefig(r"C:\ai\unedited\renders\raw_%s_%d.png" % (seg, first), dpi=60)
print("ok", F, "frames")
