"""Go-Explore route search for Level 2 (The Forest) - produces a demonstration, NOT RL.
Starts from the Level 2 save state with exactly the timing of pv.env_l2 (reset + 1 no-op frame; 4-frame steps,
fire pressed on the first frame only). Cell = (distance//4, kills), distance = RAM 24 + 256*RAM 25; kills count as progress where the screen is locked. Keeps, per cell, the state with the most energy
and time left; never accepts a lost life (a death resets the distance). Exploration policy: AIMP (default 50 %) "aim assist"
(steer the crosshair to the nearest robber in the sprite table, fire when on target), the rest random actions.
When the level ends (level byte RAM 71 > 1) the button list is written to runs/search_l2/level2_clear_*.json.
usage: python tools/go_explore_l2.py SEED SECONDS [AIM_PROB=0.5]"""
import json, os, pickle, sys, time, numpy as np, stable_retro as retro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pv.env_l2 import GAME, COMBOS, cross_xy

seed, budget = int(sys.argv[1]), float(sys.argv[2]); AIMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5; rng = np.random.default_rng(seed)
os.makedirs("runs/search_l2", exist_ok=True); out = f"runs/search_l2/arch_{seed}.pkl"
e = retro.make(GAME, state="Level2", render_mode=None); e.reset(); em = e.unwrapped.em; names = e.unwrapped.buttons
_r0 = e.unwrapped.get_ram()
FIRE = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: FIRE[i, names.index(b)] = 1
AIM = FIRE.copy(); AIM[:, names.index("A")] = 0; AIM[:, names.index("B")] = 0     # A / B pressed on the first frame only
DIR = {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4, (-1, -1): 5, (1, -1): 6, (-1, 1): 7, (1, 1): 8}

def ram(): return e.unwrapped.get_ram()
SCORE0 = int('%02x%02x%02x' % (_r0[244], _r0[245], _r0[246]))
def KILLS(r): return max(0, (int('%02x%02x%02x' % (r[244], r[245], r[246])) - SCORE0) // 75)
def tleft(r): return int(r[647]) * 60 + int(r[648]) * 10 + int(r[649])
def act(a):
    for i in range(4): e.step(FIRE[a] if i == 0 else AIM[a])
LAST_SHOT = [-99]
def aim_action(r, step):
    """Aim assist: steer to the centre of the nearest robber sprite cluster (+8 px down: measured hit window is
    x +-8, y -10..+40 around the sprite centre); fire when on target, at most every 3 steps."""
    oam = r[0x700:0x800].reshape(64, 4).astype(int)
    pts = [(x + 4, y + 4) for y, tile, _, x in oam if y < 158 and tile not in (0, 1, 2, 3, 4) and not (8 <= tile <= 15 or tile == 22)]
    groups = []
    for p in pts:
        for g in groups:
            if any(abs(p[0] - q[0]) <= 12 and abs(p[1] - q[1]) <= 12 for q in g): g.append(p); break
        else: groups.append([p])
    if not groups: return int(rng.integers(9))
    cx, cy = cross_xy(r)
    tx, ty = min(((np.mean([q[0] for q in g]), np.mean([q[1] for q in g]) + 8) for g in groups), key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    dx, dy = tx - cx, ty - cy
    d = DIR[(int(np.sign(dx)) if abs(dx) > 3 else 0, int(np.sign(dy)) if abs(dy) > 4 else 0)]
    if abs(dx) <= 6 and abs(dy) <= 12 and step - LAST_SHOT[0] >= 3: LAST_SHOT[0] = step; return d + 9
    return d

if os.path.exists(out):
    arch = pickle.load(open(out, "rb")); print("resumed", len(arch))
else:
    e.step(FIRE[0] * 0); r = ram()
    arch = {(0, 0): dict(state=em.get_state(), t=0, traj=[], q=int(r[252]) + 5 * tleft(r), dist=int(r[24]), visits=0)}
t0 = time.time(); tl = t0; it = 0; cleared = False
while time.time() - t0 < budget and not cleared:
    it += 1; keys = list(arch); k = np.array([kk[0] + 0.5 * kk[1] for kk in keys], float)   # progress = distance, then kills
    qq = np.array([arch[x]["q"] for x in keys], float)
    w = np.exp((k - k.max()) / 6) * (0.2 + qq / max(qq.max(), 1)) / (1 + np.array([arch[x]["visits"] for x in keys])); w /= w.sum()   # prefer healthy states
    c = arch[keys[rng.choice(len(keys), p=w)]]; c["visits"] += 1
    em.set_state(c["state"]); traj = list(c["traj"]); t = c["t"]; lives = int(ram()[247])
    a = int(rng.integers(len(COMBOS))); sticky = False
    for step in range(120):
        r = ram()
        if rng.random() < 0.3:                                             # sticky actions: let the crosshair travel
            sticky = rng.random() >= AIMP; a = int(rng.integers(len(COMBOS)))
        if not sticky: a = aim_action(r, t)
        act(a); traj.append(a); t += 1; r = ram()
        if int(r[247]) < lives: break                                     # never accept a lost life
        if int(r[71]) > 1:
            cleared = True; f = f"runs/search_l2/level2_clear_seed{seed}_t{t}.json"; json.dump(traj, open(f, "w"))
            print("LEVEL 2 CLEARED at step", t, "->", f, flush=True); break
        D = int(r[24]) + 256 * int(r[25]) if t > 20 else 0; kk = (D // 4, min(KILLS(r), 60));   # first frames: stale Level 1 scroll
        q = int(r[252]) + 5 * tleft(r)
        if kk not in arch or q > arch[kk]["q"]:
            arch[kk] = dict(state=em.get_state(), t=t, traj=list(traj), q=q, dist=D, visits=0)
    if time.time() - tl > 60:
        tl = time.time(); pickle.dump(arch, open(out, "wb")); b = arch[max(arch)]
        print(round(tl - t0), "best (distance, kills)", max(arch), "dist", b["dist"], "step", b["t"], "q", b["q"], "cells", len(arch), flush=True)
pickle.dump(arch, open(out, "wb")); print("done; best", max(arch), "dist", arch[max(arch)]["dist"], "cleared", cleared)
