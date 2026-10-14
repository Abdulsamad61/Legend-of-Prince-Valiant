"""Scripted aim-assist route for Level 2 (demonstration generator, NOT RL - same role as tools/go_explore.py in Level 1).
Reads the sprite table, steers the crosshair to the nearest robber and fires when it is on target. Uses exactly the
timing of pv.env_l2 (reset + 1 no-op frame; 4-frame steps; A pressed on the first frame only) and writes the action
indices (pv.env_l2.COMBOS) to runs/search_l2/level2_clear_aim.json when the level byte (RAM 71) changes.
usage: python tools/aim_route_l2.py [OY=8 LO=-4 HI=20 COOLDOWN=3]"""
import json, os, sys, numpy as np, stable_retro as retro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pv.env_l2 import GAME, COMBOS, DIRS, cross_xy

OY, LO, HI, CD = [int(x) for x in sys.argv[1:5]] if len(sys.argv) > 4 else (8, -4, 20, 3)
e = retro.make(GAME, state="Level2", render_mode=None); e.reset(); names = e.unwrapped.buttons
FIRE = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: FIRE[i, names.index(b)] = 1
AIM = FIRE.copy(); AIM[:, names.index("A")] = 0; AIM[:, names.index("B")] = 0
e.step(FIRE[0] * 0)                                                          # = Level2.reset(): one no-op frame


def policy(r, t, last):
    oam = r[0x700:0x800].reshape(64, 4).astype(int)
    pts = [(x + 4, y + 4) for y, tl, _, x in oam if y < 158 and tl not in (0, 1, 2, 3, 4) and not (8 <= tl <= 15 or tl == 22)]
    groups = []
    for p in pts:
        for g in groups:
            if any(abs(p[0] - q[0]) <= 12 and abs(p[1] - q[1]) <= 12 for q in g): g.append(p); break
        else: groups.append([p])
    if not groups: return 0, False
    cx, cy = cross_xy(r)
    tx, ty = min(((np.mean([q[0] for q in g]), np.mean([q[1] for q in g])) for g in groups),
                 key=lambda p: (p[0] - cx) ** 2 + (p[1] + OY - cy) ** 2)
    dx, dy = tx - cx, cy - ty
    keys = (["RIGHT"] if dx > 3 else ["LEFT"] if dx < -3 else []) + (["DOWN"] if dy < OY - 4 else ["UP"] if dy > OY + 4 else [])
    keys = sorted(keys, key=lambda k: k in ("LEFT", "RIGHT"))                  # DIRS order: vertical first
    d = DIRS.index(keys)
    fire = abs(dx) <= 6 and LO <= dy <= HI and t - last >= CD
    return d + 9 * fire, fire


traj, last = [], -99
for t in range(6000):
    r = e.unwrapped.get_ram()
    a, fire = policy(r, t, last); last = t if fire else last
    for i in range(4): e.step(FIRE[a] if i == 0 else AIM[a])
    traj.append(int(a)); r = e.unwrapped.get_ram()
    if t % 250 == 0: print(t, "dist", int(r[24]) + 256 * int(r[25]), "lives", r[247], "energy", r[252], "time", r[647], r[648], r[649], flush=True)
    if int(r[71]) > 1:
        os.makedirs("runs/search_l2", exist_ok=True); f = "runs/search_l2/level2_clear_aim.json"; json.dump(traj, open(f, "w"))
        print("LEVEL 2 CLEARED at step", t + 1, "lives", r[247], "->", f); break
else: print("not cleared")
