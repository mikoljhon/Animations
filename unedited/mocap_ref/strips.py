import numpy as np, json, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.load(r"C:\ai\mocap\data\seq30.npz", allow_pickle=True)
loc = d["local"].astype(np.float32); root = d["root"]; starts = list(d["starts"]) + [len(loc)]; clips = list(d["clips"]); desc = d["desc"]
J = {n: k for k, n in enumerate(d["joints"])}
CH = [["hips","spine","chest","neck","head"],["neck","clav_L","shoulder_L","elbow_L","wrist_L","knuckle_L"],["neck","clav_R","shoulder_R","elbow_R","wrist_R","knuckle_R"],
      ["hips","hip_L","knee_L","ankle_L","toe_L"],["hips","hip_R","knee_R","ankle_R","toe_R"]]
COL = ["#222","#36c","#d33","#36c","#d33"]
def draw(ax, p, view):
    ix = 1 if view == "side" else 0
    for ch, c in zip(CH, COL):
        pts = np.array([p[J[n]] for n in ch])
        xs = -pts[:, 1] if view == "side" else pts[:, 0]
        ax.plot(xs, pts[:, 2], color=c, lw=2.5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.2, 1.1); ax.set_aspect("equal"); ax.axis("off")
def strip(items, path, n=7):
    fig, axs = plt.subplots(len(items) * 2, n, figsize=(n * 1.6, len(items) * 3.3), dpi=60)
    for r, it in enumerate(items):
        i = clips.index(it["clip"]); s = starts[i]
        fr = np.linspace(it["start"], it["end"], n).astype(int)
        for c, f in enumerate(fr):
            p = loc[s + f]
            draw(axs[r * 2, c], p, "side"); draw(axs[r * 2 + 1, c], p, "front")
        axs[r * 2, 0].set_title("%s %s %s reach %.2f lat %.2f" % (it["clip"], it["side"], it["desc"][:18], it["reach"], it["lat"]), fontsize=7, loc="left")
    fig.savefig(path); plt.close(fig)
if __name__ == "__main__":
    P = json.load(open("punches.json"))
    straight = sorted([p for p in P if p["lat"] < 0.30 and p["reach"] > 0.62], key=lambda p: -p["reach"])
    hooks = sorted([p for p in P if p["lat"] >= 0.45 and p["reach"] > 0.5], key=lambda p: -p["lat"])
    print(len(straight), len(hooks))
    pick = []
    seen = set()
    for p in straight:
        k = (p["clip"], p["side"])
        if k in seen: continue
        seen.add(k); pick.append(p)
        if len(pick) >= 6: break
    strip(pick, "strips_straight.png")
    pick = []; seen = set()
    for p in hooks:
        k = (p["clip"], p["side"])
        if k in seen: continue
        seen.add(k); pick.append(p)
        if len(pick) >= 4: break
    strip(pick, "strips_hook.png")
