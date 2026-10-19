#!/usr/bin/env bash
# qa_clips.sh : run every motion clip through the fight pipeline's QA gate, one clip at a time, on Relic (or the KillDroid),
# and write the approved library: mocap_ref/approved_<who>.json  (only clips that PASS may be used in a fight).
#   bash qa_clips.sh relic                 # all clips in mocap_ref/segs
#   bash qa_clips.sh relic punch_14_03_a   # just the named clips
# Blender runs at below-normal priority, one at a time, and writes no .blend (PIPE_NOSAVE) - QA only.
#   GATED=1 bash qa_clips.sh kd ...         # through run_gated.ps1 (waits until the PC is free, pauses while the user works)
#   BASE=fight_v07.blend bash qa_clips.sh   # another base (default fight_v08.blend: Relic v7 + KillDroid v02)
cd "$(dirname "$0")"
WHO=${1:-relic}; shift
BASE=${BASE:-fight_v08.blend}
RUNNER=run_py.ps1; [ -n "$GATED" ] && RUNNER=run_gated.ps1
if [ $# -gt 0 ]; then CLIPS="$@"; else CLIPS=$(ls mocap_ref/segs/*.npz | xargs -n1 basename | sed 's/\.npz$//'); fi
mkdir -p anim_cache5
for c in $CLIPS; do
  PIPE_NOSAVE=1 powershell -NoProfile -ExecutionPolicy Bypass -File "C:/ai/unedited/$RUNNER" -blend $BASE -script pipe_test.py \
      -log "anim_cache5/qa_one_${WHO}_$c.txt" -a "$WHO one_${WHO}_$c $c"
  r=$(grep -E '^QA ' "anim_cache5/qa_one_${WHO}_$c.txt" | cut -c4-7)
  echo "$c ${r:-ERROR}"
  grep -E '\[XX\]' "anim_cache5/qa_one_${WHO}_$c.txt" | sed 's/^/     /'
done | tee "anim_cache5/qa_clips_${WHO}.txt"
python - "$WHO" <<'PYEOF'
import json, sys, re
who = sys.argv[1]
ok, bad, cur = [], {}, None
for line in open("anim_cache5/qa_clips_%s.txt" % who):
    m = re.match(r"^(\S+) (PASS|FAIL|ERROR)", line)
    if m:
        cur = m.group(1)
        if m.group(2) == "PASS":
            ok.append(cur)
        else:
            bad.setdefault(cur, [])
    elif line.strip().startswith("[XX]") and cur in bad:
        bad[cur].append(line.strip())
json.dump({"approved": ok, "rejected": bad}, open("mocap_ref/approved_%s.json" % who, "w"), indent=1)
print("approved:", ok)
print("rejected:", list(bad))
PYEOF
