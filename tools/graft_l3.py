"""Search helper for Level 3 (demonstration generator, not RL): re-use a short action segment that solved a timed obstacle in
one route (e.g. 'shoot the disk, then run past the gate before it closes') from states of another route at the same spot.
For each archive state near (ROOM, X, Y) at the given stage it waits 0..WMAX no-op steps (the disk / gate phase differs), plays
the segment and keeps the result if the hero is alive and past the goal x. Successful states are added to the archive.
usage: python tools/graft_l3.py ARCHIVE SEG_JSON ROOM X Y STAGE GOAL_X [WMAX=80]"""
import json, os, pickle, sys, numpy as np, stable_retro as retro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pv.env_l3 import GAME, COMBOS
arch_f, seg = sys.argv[1], json.load(open(sys.argv[2])); room, X, Y, stg, gx = [int(v) for v in sys.argv[3:8]]
wmax = int(sys.argv[8]) if len(sys.argv) > 8 else 80
arch = pickle.load(open(arch_f, "rb"))
e = retro.make(GAME, state="Level3", render_mode=None); e.reset(); em = e.unwrapped.em; names = e.unwrapped.buttons
A = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: A[i, names.index(b)] = 1
R = lambda: e.unwrapped.get_ram().astype(int)
FLAGS = [513, 514, 515, 516]
cands = [v for k, v in arch.items() if k[0] == room and k[3] == stg and abs(v["x"] - X) <= 12 and abs(v["y"] - Y) <= 8]
print(len(cands), "start states"); found = 0
for v in sorted(cands, key=lambda v: -v["q"])[:20]:
    for w in range(wmax):
        em.set_state(v["state"]); lives = R()[247]; acts = [0] * w + seg; ok = True
        for a in acts:
            for _ in range(4): e.step(A[a])
            r = R()
            if r[247] < lives or r[252] == 0: ok = False; break
        r = R(); xx = r[86] + 256 * r[87]
        if not ok or r[75] != room or (xx > gx if gx < X else xx < gx): continue
        st = em.get_state(); dead = False                                  # survival check
        for _ in range(40):
            e.step(A[0])
            if R()[247] < lives or R()[252] == 0: dead = True; break
        if dead: continue
        em.set_state(st); t = v["t"] + len(acts); q = 4 * r[252] + 5 * (r[647] * 60 + r[648] * 10 + r[649]) + 1200 * r[247]
        k = (r[75], xx // 16, r[88] // 16, int(sum(r[j] > 0 for j in FLAGS)), r[250], 0)
        arch[k] = dict(state=st, t=t, traj=v["traj"] + acts, q=q, visits=0, rooms=v.get("rooms", frozenset()), x=xx, y=r[88], room=r[75],
                       ev=v.get("ev", -999), born=max(c["born"] for c in arch.values()), score=r[244] * 65536 + r[245] * 256 + r[246])
        found += 1; print("success: wait", w, "from t", v["t"], "-> x", xx, "q", q, flush=True); break
    if found >= 3: break
pickle.dump(arch, open(arch_f, "wb")); print("added", found)
