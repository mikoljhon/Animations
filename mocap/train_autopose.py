"""Train AutoPose: give it a few joint positions (any subset, usually hips, neck, wrists and ankles) and it predicts a
natural position for every joint. One network handles any subset: each joint's input is its position times a
"given" flag, plus the flag itself, so training hides random joints and asks the network to fill them in.

Poses come from prep_cmu.py (hips at the origin, facing -Y, Z up, in leg lengths). Training uses the CPU.
Exports autopose.npz (plain numpy weights, run inside Blender without torch) and a training report.

Usage: python train_autopose.py <data folder> <out folder> [minutes] [hips|feet]
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

CORE = ["neck", "wrist_L", "wrist_R", "ankle_L", "ankle_R"]          # with the hips (the origin): Cascadeur's six points
# hips anchor: poses relative to the hips (the hips are always given). feet anchor: relative to the midpoint of the
# ankles, so the network also places the hips (drag a hand down and the body crouches)
ANCHORS = {"hips": {"always": ["hips"], "core": {c: 0.92 for c in CORE}},
           "feet": {"always": ["ankle_L", "ankle_R"], "core": {"wrist_L": 0.92, "wrist_R": 0.92, "neck": 0.7, "hips": 0.25}}}
GROUPS = {"pelvis": ["hip_L", "hip_R", "spine"], "look": ["head", "eye_L", "eye_R"]}
WIDTH, BLOCKS = 512, 3
GIVEN = {"hips": CORE, "feet": ["wrist_L", "wrist_R", "neck"]}


class AutoPose(nn.Module):
    def __init__(self, J, width=WIDTH, blocks=BLOCKS):
        super().__init__()
        self.inp = nn.Linear(J * 4, width)
        self.blocks = nn.ModuleList([nn.ModuleDict({"ln": nn.LayerNorm(width), "a": nn.Linear(width, width),
                                                    "b": nn.Linear(width, width)}) for _ in range(blocks)])
        self.ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, J * 3)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.inp(x))
        for blk in self.blocks:
            h = h + blk["b"](self.act(blk["a"](blk["ln"](h))))
        return self.out(self.act(self.ln(h)))


def mirror_index(names):
    swap = []
    for n in names:
        if n.endswith("_L"):
            swap.append(names.index(n[:-2] + "_R"))
        elif n.endswith("_R"):
            swap.append(names.index(n[:-2] + "_L"))
        else:
            swap.append(names.index(n))
    return np.array(swap)


def rigid_pairs(poses, names):
    """joint pairs whose distance never changes (real bones), with that length"""
    cand = [("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"), ("head", "eye_L"), ("head", "eye_R"),
            ("chest", "clav_L"), ("clav_L", "shoulder_L"), ("shoulder_L", "elbow_L"), ("elbow_L", "wrist_L"), ("wrist_L", "knuckle_L"),
            ("chest", "clav_R"), ("clav_R", "shoulder_R"), ("shoulder_R", "elbow_R"), ("elbow_R", "wrist_R"), ("wrist_R", "knuckle_R"),
            ("hips", "hip_L"), ("hip_L", "knee_L"), ("knee_L", "ankle_L"), ("ankle_L", "toe_L"),
            ("hips", "hip_R"), ("hip_R", "knee_R"), ("knee_R", "ankle_R"), ("ankle_R", "toe_R")]
    sample = poses[np.random.default_rng(0).choice(len(poses), 20000, replace=False)].astype(np.float32)
    out = []
    for a, b in cand:
        d = np.linalg.norm(sample[:, names.index(a)] - sample[:, names.index(b)], axis=1)
        med = float(np.median(d))
        if np.std(d) / med < 0.02:
            out.append((names.index(a), names.index(b), med))
    return out


def make_masks(n, names, rng, anchor="hips"):
    J = len(names)
    m = np.zeros((n, J), dtype=np.float32)
    for a in ANCHORS[anchor]["always"]:
        m[:, names.index(a)] = 1.0
    for c, p in ANCHORS[anchor]["core"].items():
        m[:, names.index(c)] = np.maximum(m[:, names.index(c)], rng.random(n) < p)
    p_extra = np.where(rng.random(n) < 0.5, 0.0, np.where(rng.random(n) < 0.6, rng.random(n) * 0.25, 0.25 + rng.random(n) * 0.35))
    m = np.maximum(m, (rng.random((n, J)) < p_extra[:, None]).astype(np.float32))
    for g, members in GROUPS.items():
        on = rng.random(n) < (0.3 if g == "pelvis" else 0.2)
        for name in members:
            m[on, names.index(name)] = 1.0
    return m


def batch_inputs(P, M, noise, rng):
    X = P + noise * rng.standard_normal(P.shape).astype(np.float32)
    X = X * M[:, :, None]
    return np.concatenate([X.reshape(len(P), -1), M], axis=1)


def evaluate(model, P, names, given, rng, leg_cm, anchor="hips"):
    """mean joint error (cm) on the joints that were not given"""
    M = np.zeros((len(P), len(names)), dtype=np.float32)
    for n in ANCHORS[anchor]["always"] + given:
        M[:, names.index(n)] = 1.0
    with torch.no_grad():
        Y = model(torch.from_numpy(batch_inputs(P, M, 0.0, rng))).numpy().reshape(P.shape)
    err = np.linalg.norm(Y - P, axis=2)
    hidden = M == 0
    return float((err * hidden).sum() / hidden.sum() * leg_cm), {n: round(float(err[:, i].mean() * leg_cm), 1)
                                                                    for i, n in enumerate(names) if M[0, i] == 0}


def export(model, path, names, swap, pairs, leg, report, anchor="hips"):
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    np.savez(path, **{k.replace(".", "__"): v for k, v in sd.items()}, joints=np.array(names), mirror=swap,
             pairs=np.array([[a, b] for a, b, _l in pairs]), pair_len=np.array([l for _a, _b, l in pairs], dtype=np.float32),
             leg=np.float32(leg), width=WIDTH, blocks=BLOCKS, core=np.array(CORE), anchor=anchor, report=json.dumps(report))


def main(data, out, minutes=30.0, anchor="hips"):
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    d = np.load(os.path.join(data, "poses.npz"))
    poses, clip, clips, names = d["poses"], d["clip"], [str(c) for c in d["clips"]], [str(n) for n in d["joints"]]
    if anchor == "feet":
        poses = poses.astype(np.float32)
        poses = (poses - 0.5 * (poses[:, names.index("ankle_L")] + poses[:, names.index("ankle_R")])[:, None]).astype(np.float16)
    subject = np.array([int(c.split("_")[0]) for c in clips])[clip]
    rng = np.random.default_rng(1)
    held = set(rng.choice(np.unique(subject), size=8, replace=False).tolist()) | {141, 142, 143, 144}
    val = np.isin(subject, list(held))
    train_P = poses[~val].astype(np.float32)
    val_P = poses[val].astype(np.float32)[rng.choice(int(val.sum()), 20000, replace=False)]
    swap = mirror_index(names)
    pairs = rigid_pairs(poses, names)
    leg = 0.822
    print("train %d poses, validation %d (subjects %s), %d rigid bones" % (len(train_P), len(val_P), sorted(held), len(pairs)), flush=True)

    model = AutoPose(len(names))
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 2048
    pa = torch.tensor([a for a, _b, _l in pairs])
    pb = torch.tensor([b for _a, b, _l in pairs])
    pl = torch.tensor([l for _a, _b, l in pairs])
    t0 = time.time()
    budget = minutes * 60.0
    step, log = 0, []
    est_steps = None
    while True:
        el = time.time() - t0
        if el > budget:
            break
        if step == 200:
            est_steps = int(budget / (el / 200.0))
            print("about %d steps fit in %.0f minutes" % (est_steps, minutes), flush=True)
        if est_steps:
            frac = min(step / est_steps, 1.0)
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (2e-3 - 1e-5) * (1 + np.cos(np.pi * frac))
        idx = rng.integers(0, len(train_P), bs)
        P = train_P[idx]
        flip = rng.random(bs) < 0.5
        P[flip] = P[flip][:, swap] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        M = make_masks(bs, names, rng, anchor)
        X = torch.from_numpy(batch_inputs(P, M, 0.004, rng))
        T = torch.from_numpy(P)
        Y = model(X).view(bs, -1, 3)
        loss_pos = ((Y - T) ** 2).sum(-1).mean()
        bl = torch.linalg.norm(Y[:, pa] - Y[:, pb], dim=-1)
        loss_bone = ((bl - pl) ** 2).mean()
        loss = loss_pos + 0.5 * loss_bone
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 500 == 0:
            model.eval()
            e6, _ = evaluate(model, val_P, names, GIVEN[anchor], rng, leg * 100, anchor)
            model.train()
            log.append({"step": step, "minutes": round((time.time() - t0) / 60, 1), "loss": round(loss.item(), 5), "val_cm_6pts": round(e6, 2)})
            print(log[-1], flush=True)
    model.eval()
    e6, per6 = evaluate(model, val_P, names, GIVEN[anchor], rng, leg * 100, anchor)
    e10, per10 = evaluate(model, val_P, names, GIVEN[anchor] + ["elbow_L", "elbow_R", "knee_L", "knee_R"], rng, leg * 100, anchor)
    report = {"steps": step, "minutes": round((time.time() - t0) / 60, 1), "train_poses": int(len(train_P)),
              "val_subjects": sorted(int(s) for s in held), "val_cm_6_points": round(e6, 2), "val_cm_10_points": round(e10, 2),
              "per_joint_cm_6_points": per6, "log": log}
    stem = "autopose" if anchor == "hips" else "autopose_" + anchor
    torch.save(model.state_dict(), os.path.join(out, stem + ".pt"))
    export(model, os.path.join(out, stem + ".npz"), names, swap, pairs, leg, report, anchor)
    with open(os.path.join(out, stem + "_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("done:", {k: v for k, v in report.items() if k != "log"}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 30.0, sys.argv[4] if len(sys.argv) > 4 else "hips")
