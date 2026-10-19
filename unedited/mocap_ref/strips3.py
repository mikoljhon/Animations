import json
exec(open("strips2.py").read().split('if __name__')[0])
P = json.load(open("punches.json"))
def first(clip, side, k=3, lat=None):
    xs = [p for p in P if p["clip"] == clip and p["side"] == side]
    return xs[:k]
items = []
for clip, side in (("144_20","R"),("144_13","L"),("143_23","R"),("14_03","R"),("13_17","R"),("80_10","L")):
    for p in first(clip, side, 2):
        items.append((clip, p["start"], p["end"], "%s %s peak %d reach %.2f lat %.2f" % (clip, side, p["peak"], p["reach"], p["lat"])))
strip2(items, "strips_punch2.png", n=7)
# whole-clip overviews of defensive things
ov = []
for clip, a, b in (("76_03",0,270),("76_04",0,104),("77_09",0,245),("134_01",0,89),("139_09",0,246),("144_07",0,300),("144_26",0,300),("102_22",0,61),("102_26",0,111),("102_28",0,175),("77_03",0,245)):
    ov.append((clip, a, b, clip + " " + str(desc[clips.index(clip)])))
strip2(ov, "strips_defense.png", n=8)
print(len(items))
