"""Turn the CMU motion capture library (cgspeed's Daz-friendly BVH release) into training data for AutoPose.

Every motion is parsed straight from its zip, run through forward kinematics and reduced to 27 key joints.
Saved, at 30 fps:
  seq30.npz   every frame of every clip: joints relative to the hips in the hips' heading frame (facing -Y, Z up,
              in leg lengths), plus the root position and heading in metres, clip starts, names and descriptions
  poses.npz   the AutoPose training set: the same hips-relative poses with near-duplicates (idle stretches) removed
The library is all retargeted onto one skeleton (Daz Aiko3), so a single leg length normalises everything.

Usage: python prep_cmu.py <zip folder> <output folder>
"""

import glob
import io
import os
import re
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np

CANON = [("hips", "hip"), ("spine", "abdomen"), ("chest", "chest"), ("neck", "neck"), ("head", "head"),
         ("eye_L", "leftEye"), ("eye_R", "rightEye"),
         ("clav_L", "lCollar"), ("shoulder_L", "lShldr"), ("elbow_L", "lForeArm"), ("wrist_L", "lHand"),
         ("knuckle_L", "lMid1"), ("thumb_L", "lThumb2"),
         ("clav_R", "rCollar"), ("shoulder_R", "rShldr"), ("elbow_R", "rForeArm"), ("wrist_R", "rHand"),
         ("knuckle_R", "rMid1"), ("thumb_R", "rThumb2"),
         ("hip_L", "lThigh"), ("knee_L", "lShin"), ("ankle_L", "lFoot"), ("toe_L", "lFoot_end"),
         ("hip_R", "rThigh"), ("knee_R", "rShin"), ("ankle_R", "rFoot"), ("toe_R", "rFoot_end")]
NAMES = [c[0] for c in CANON]
IX = {n: i for i, n in enumerate(NAMES)}
STEP = 4            # 120 fps -> 30 fps
DEDUP = 0.012       # drop a pose when no joint moved more than this (in leg lengths) since the last kept one


def parse_bvh(text):
    head, motion = text.split("MOTION", 1)
    tok = head.split()
    joints, stack, cur, i = [], [], None, 0
    while i < len(tok):
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            cur = {"name": tok[i + 1], "parent": stack[-1] if stack else -1, "offset": (0.0, 0.0, 0.0), "channels": []}
            joints.append(cur)
            i += 2
        elif t == "End":
            cur = {"name": joints[stack[-1]]["name"] + "_end", "parent": stack[-1], "offset": (0.0, 0.0, 0.0), "channels": []}
            joints.append(cur)
            i += 2
        elif t == "{":
            stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            cur["offset"] = tuple(float(x) for x in tok[i + 1:i + 4])
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            cur["channels"] = tok[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    m = re.search(r"Frames:\s*(\d+)\s*Frame Time:\s*([0-9.eE+-]+)", motion)
    nframes, ftime = int(m.group(1)), float(m.group(2))
    body = motion[m.end():]
    vals = np.fromstring(body, dtype=np.float32, sep=" ")
    ncol = sum(len(j["channels"]) for j in joints)
    nframes = min(nframes, len(vals) // ncol)
    return joints, vals[:nframes * ncol].reshape(nframes, ncol), ftime


def _rot(axis, deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    M = np.zeros((len(r), 3, 3), dtype=np.float32)
    a, b = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    k = 3 - a - b
    M[:, k, k] = 1.0
    M[:, a, a] = c
    M[:, a, b] = -s
    M[:, b, a] = s
    M[:, b, b] = c
    return M


def forward_kinematics(joints, D):
    """global joint positions (F, J, 3) in the file's own axes and units"""
    F = len(D)
    GR, Gp = [None] * len(joints), [None] * len(joints)
    col = 0
    for j, J in enumerate(joints):
        R = np.broadcast_to(np.eye(3, dtype=np.float32), (F, 3, 3)).copy()
        t = np.broadcast_to(np.array(J["offset"], dtype=np.float32), (F, 3)).copy()
        for ch in J["channels"]:
            v = D[:, col]
            col += 1
            if ch.endswith("position"):
                t[:, "XYZ".index(ch[0])] += v
            else:
                R = R @ _rot(ch[0], v)
        p = J["parent"]
        if p < 0:
            GR[j], Gp[j] = R, t
        else:
            Gp[j] = Gp[p] + np.einsum("fij,fj->fi", GR[p], t)
            GR[j] = GR[p] @ R
    return np.stack(Gp, axis=1)


def heading_frame(P):
    """per frame: hips position and the rotation whose columns are left, back, up (character faces -back)"""
    left = P[:, IX["hip_L"]] - P[:, IX["hip_R"]]
    left[:, 2] = 0.0
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
    up = np.zeros_like(left)
    up[:, 2] = 1.0
    back = np.cross(up, left)
    return np.stack([left, back, up], axis=2)


def leg_length(joints):
    by = {j["name"]: j for j in joints}
    return float(np.linalg.norm(by["lShin"]["offset"]) + np.linalg.norm(by["lFoot"]["offset"])) / 100.0


def process(args):
    zpath, member = args
    with zipfile.ZipFile(zpath) as z:
        text = z.read(member).decode("latin-1")
    joints, D, ftime = parse_bvh(text)
    if len(D) < STEP * 4:
        return None
    step = max(1, int(round((1.0 / 30.0) / ftime)))
    D = D[1::step]                                     # frame 0 is the T-pose the converter adds
    P = forward_kinematics(joints, D)
    names = [j["name"] for j in joints]
    P = P[:, [names.index(src) for _n, src in CANON]]
    P = np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1) / 100.0      # Y-up cm -> Z-up metres
    L = leg_length(joints)
    R = heading_frame(P)
    root = P[:, IX["hips"]].copy()
    local = np.einsum("fjc,fcd->fjd", P - root[:, None, :], R) / L
    head = np.stack([R[:, 0, 0], R[:, 1, 0]], axis=1)                       # the 'left' axis, as cos/sin
    keep = [0]
    for k in range(1, len(local)):
        if np.abs(local[k] - local[keep[-1]]).max() > DEDUP:
            keep.append(k)
    clip = os.path.splitext(os.path.basename(member))[0]
    return clip, local.astype(np.float16), root.astype(np.float32), head.astype(np.float32), np.array(keep, dtype=np.int32), L


def read_descriptions(zpaths):
    desc = {}
    for zp in zpaths:
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.lower().endswith("index-text.txt"):
                    for line in z.read(n).decode("latin-1").splitlines():
                        m = re.match(r"\s*(\d+_\d+)\s+(.*\S)", line)
                        if m:
                            desc[m.group(1)] = m.group(2)
                    break
        if desc:
            break
    return desc


def main(zdir, out):
    os.makedirs(out, exist_ok=True)
    zpaths = sorted(glob.glob(os.path.join(zdir, "*.zip")))
    jobs = []
    for zp in zpaths:
        with zipfile.ZipFile(zp) as z:
            jobs += [(zp, n) for n in z.namelist() if n.lower().endswith(".bvh")]
    print("%d zips, %d motions" % (len(zpaths), len(jobs)), flush=True)
    t0 = time.time()
    results = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(process, jobs, chunksize=4)):
            if r is not None:
                results.append(r)
            if (i + 1) % 200 == 0:
                print("  %d / %d  (%.0f s)" % (i + 1, len(jobs), time.time() - t0), flush=True)
    results.sort(key=lambda r: r[0])
    desc = read_descriptions(zpaths)
    clips = [r[0] for r in results]
    starts = np.cumsum([0] + [len(r[1]) for r in results])
    np.savez_compressed(os.path.join(out, "seq30.npz"), local=np.concatenate([r[1] for r in results]),
                        root=np.concatenate([r[2] for r in results]), heading=np.concatenate([r[3] for r in results]),
                        starts=starts, clips=np.array(clips), desc=np.array([desc.get(c, "") for c in clips]),
                        joints=np.array(NAMES), leg=np.float32(results[0][5]), fps=np.float32(30.0))
    poses = np.concatenate([r[1][r[4]] for r in results])
    pclip = np.concatenate([np.full(len(r[4]), i, dtype=np.int32) for i, r in enumerate(results)])
    np.savez_compressed(os.path.join(out, "poses.npz"), poses=poses, clip=pclip, clips=np.array(clips), joints=np.array(NAMES))
    print("done in %.0f s: %d clips, %d frames at 30 fps, %d training poses" % (
        time.time() - t0, len(results), int(starts[-1]), len(poses)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
