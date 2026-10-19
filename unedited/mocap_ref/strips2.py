import numpy as np, json, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("strips.py").read().split("def strip")[0])
def strip2(items, path, n=9, views=("side","front")):
    fig, axs = plt.subplots(len(items), n * len(views), figsize=(n * len(views) * 0.95, len(items) * 1.6), dpi=70)
    for r, (nm, a, b, label) in enumerate(items):
        i = clips.index(nm); s = starts[i]
        fr = np.linspace(a, b, n).astype(int)
        for c, f in enumerate(fr):
            for v, view in enumerate(views):
                draw(axs[r, c * len(views) + v], loc[s + f], view)
        axs[r, 0].set_title(label, fontsize=7, loc="left")
    fig.subplots_adjust(wspace=0, hspace=0.25, left=0.01, right=0.99, top=0.96, bottom=0.01)
    fig.savefig(path); plt.close(fig)
if __name__ == "__main__":
    items = [("144_05", 461-20, 461+16, "144_05 R front kick @461"), ("144_06", 291-20, 291+16, "144_06 R @291"), ("135_04", 94-20, 94+16, "135_04 R front kick @94"),
             ("141_14", 64-18, 64+14, "141_14 R @64 lat .65"), ("141_14", 106-18, 106+14, "141_14 R @106"), ("113_13", 247-20, 247+16, "113_13 R @247"),
             ("86_06", 877-20, 877+16, "86_06 R @877"), ("144_09", 624-20, 624+16, "144_09 L @624")]
    strip2(items, "strips_kicks.png", n=7)
