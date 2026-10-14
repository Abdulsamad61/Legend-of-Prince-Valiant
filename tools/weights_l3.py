"""Level 3 helper (demonstration generator, not RL): the two pulley weights outside the cave (room 31). Walk to the
weights and headbutt them (jump while standing under one when it is low), switching between the two, until the game sets
the 'weights done' flag RAM[513] (the vine to the right starts working). Random controller settings, keeps the fastest.
usage: python tools/weights_l3.py PROGRESS_PKL PLAN_JSON  (inserts a [31, 3] waypoint into the plan and the progress)"""
import json, os, pickle, sys, numpy as np, stable_retro as retro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pv.env_l3 import GAME, COMBOS
pf, planf = sys.argv[1], sys.argv[2]; P = pickle.load(open(pf, "rb")); plan = json.load(open(planf))
e = retro.make(GAME, state="Level3", render_mode=None); e.reset(); em = e.unwrapped.em; names = e.unwrapped.buttons
A = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: A[i, names.index(b)] = 1
R = lambda: e.unwrapped.get_ram().astype(int)
rng = np.random.default_rng(0); best = None
for trial in range(150):
    em.set_state(P["state"]); seq = []; thr = int(rng.integers(50, 90)); sw = int(rng.integers(1, 6)); tgt = 465; jumps = 0
    for t in range(700):
        r = R()
        if r[513] or r[252] == 0: break
        x = r[86] + 256 * r[87]; y = (r[674] if tgt == 465 else r[672]) - 9          # RAM 672 / 674: left / right weight height
        if abs(x - tgt) > 3: act = 1 if x < tgt else 2
        elif y >= thr and r[94] == 0: act = 6; jumps += 1
        else: act = 0
        if jumps >= sw: tgt = 417 if tgt == 465 else 465; jumps = 0; sw = int(rng.integers(1, 6))
        for _ in range(4): e.step(A[act])
        seq.append(act)
    r = R()
    if r[513] and r[252] > 0 and (best is None or len(seq) < len(best[0])): best = (seq, em.get_state()); print(trial, len(seq), "energy", r[252])
assert best, "weights not solved"
i = len(P["done"]); plan.insert(i, [31, 3, 0, 0, 9999, 0, 999]); json.dump(plan, open(planf, "w"))
P.setdefault("states", []).append(P["state"]); P["state"] = best[1]; P["traj"] = P["traj"] + best[0]; em.set_state(best[1]); r = R()
P["done"].append(dict(goal=[31, 3], steps=len(best[0]), t=len(P["traj"]), time_left=int(r[647] * 60 + r[648] * 10 + r[649]), energy=int(r[252]), lives=int(r[247]), note="weights controller"))
pickle.dump(P, open(pf, "wb")); print("inserted at plan index", i, P["done"][-1])
