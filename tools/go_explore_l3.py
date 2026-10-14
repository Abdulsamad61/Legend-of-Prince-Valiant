"""Go-Explore route search for Level 3 (The Viking) - savestates; produces a demonstration, NOT RL.
Starts from the Level 3 save state with the exact timing of pv.env_l3 (reset + 1 no-op frame, 4-frame steps, 12 actions).
The cave is a maze of rooms joined by doors; the route has to come back through rooms it already visited (dagger room,
Bear, Viking). Cell = (room RAM[75], x // 16, y // 16, stage, tokens RAM[250], score event in the last 100 steps, damage dealt to
the Bear / speed of the pulley weights); stage = items /
quest flags collected (RAM[514] = dagger). A state with the dagger, a token, or just after shooting a gate disk (the
gate is open for a few seconds) is a new cell. Selection favours the highest stage, rarely reached 64x64 regions,
recently found cells, few visits and more energy / time left.
Exploration: sticky random actions with random hold lengths (hold UP to enter a door, DOWN+B to crouch-shoot,
repeated jumps, a springboard routine, and when a gate disk is on screen: face it and fire repeatedly, standing or crouched). A lost life is allowed (the hero respawns at a checkpoint with full energy, quest flags kept) but costs 1200 in
the state quality q = 4 * energy + 5 * seconds left + 1200 * lives; the last life is never spent. A state is archived only
if the hero survives 10 idle steps from it. When the level byte RAM[71] passes 2 the route is written to
runs/search_l3/level3_clear_*.json.
usage: python tools/go_explore_l3.py SEED SECONDS [TEMP=0.5] [FOCUS=room,stage,xmin,xmax[,ymax] | auto]  (only start from cells there;
       auto = the room the route reached last at the highest stage)"""
import json, os, pickle, sys, time, numpy as np, stable_retro as retro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pv.env_l3 import GAME, COMBOS

seed, budget = int(sys.argv[1]), float(sys.argv[2]); TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
fr = None
G = 32 if os.environ.get("COARSE") else 16                            # COARSE=1: 32 px cells, no token count (fewer cells: faster time optimisation)
AUTO = len(sys.argv) > 4 and sys.argv[4] == "auto"                     # auto: focus on the most recently discovered room of the highest stage
FOCUS = tuple(int(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 and not AUTO else None
rng = np.random.default_rng(seed); os.makedirs("runs/search_l3", exist_ok=True); out = f"runs/search_l3/arch_{seed}.pkl"
e = retro.make(GAME, state="Level3", render_mode=None); e.reset(); em = e.unwrapped.em; names = e.unwrapped.buttons
ACTS = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: ACTS[i, names.index(b)] = 1
R = lambda: e.unwrapped.get_ram().astype(int)
tleft = lambda r: r[647] * 60 + r[648] * 10 + r[649]
FLAGS = [513, 514, 515, 516]                                           # quest flags: 514 = dagger, 515 = Bear beaten, 513 = pulley weights done, 516 = Viking beaten (517 is a counter, not a flag)
stage = lambda r: int(sum(r[j] > 0 for j in FLAGS))
def boss(r):
    """Progress inside a room where position does not change: Bear damage (room 28, RAM[163] = health 255 -> 0) and the
    speed of the two pulley weights outside (room 31, RAM[1028] = signed speed; headbutts speed them up)."""
    if r[75] == 28: return (255 - r[163]) // 16
    if r[75] == 31: v = r[1028] - 256 if r[1028] > 127 else r[1028]; return 4 * min(abs(v), 4)
    return 0


def safe():
    st = em.get_state(); l0 = R()[247]; ok = R()[252] > 0                 # energy 0 = dying (the life is taken ~150 frames later)
    for _ in range(40):
        if not ok: break
        e.step(ACTS[0]); rr = R()
        if rr[247] < l0 or rr[252] == 0: ok = False
    em.set_state(st); return ok


def disk_x(r):
    """Screen x of a gate disk (sprite tiles 246 / 247) if one is on screen, else None."""
    oam = r[0x700:0x800].reshape(64, 4)
    xs = [x for y, tl, _, x in oam if y < 240 and tl in (246, 247)]
    return float(np.mean(xs)) if xs else None


def choose(r):
    u = rng.random(); dx = disk_x(r)
    if dx is not None and u < 0.15:                                       # shooting macro: face the disk, fire as it passes
        right = dx > (r[86] + 256 * r[87] - r[24] - 256 * r[25]) % 256
        if rng.random() < 0.5: plan = [5 if right else 11] + [4, 0, 0, 0] * int(rng.integers(3, 8))
        else: plan = [5 if right else 11] + [10, 3, 3, 3] * int(rng.integers(3, 8))
        return plan[::-1]
    if u < 0.05: return [9] * int(rng.integers(2, 10))                     # hold UP: enter a door / climb
    if u < 0.09: return [10] * int(rng.integers(2, 8))                     # crouch-shoot
    if u < 0.12: return [3] * int(rng.integers(3, 15))                     # duck (arrows, swinging blades)
    if u < 0.15: return [0] * int(rng.integers(3, 20))                     # wait for a moving obstacle
    if u < 0.17: return sum(([6] + [0] * int(rng.integers(5, 12)) for _ in range(int(rng.integers(2, 6)))), [])   # repeated jumps
    if u < 0.21:                                                           # springboard routine (see spring())
        SPR.update(y0=r[88], prev=r[88], w=3, n=0, drift=int(rng.integers(2, 7)), d=int(rng.choice([1, 2, 7, 8])), k=0)
        return [-1] * int(rng.integers(60, 140))
    return [int(rng.integers(len(COMBOS)))] * int(rng.geometric(0.25))     # sticky random action


SPR = {}


def spring(r):
    """Springboard: jump straight up and jump again just before each landing (gains height every bounce); after a few
    bounces drift sideways at the top to land on the high platform."""
    y = r[88]; desc = y > SPR["prev"]; SPR["prev"] = y
    if SPR["w"] > 0: SPR["w"] -= 1; SPR["y0"] = y; return 0             # stop first (no sideways momentum)
    if SPR["n"] >= SPR["drift"] and not desc and y < SPR["y0"] - 90:
        SPR["k"] += 1; return SPR["d"] if SPR["k"] < 8 else 0
    if desc and y >= SPR["y0"] - 26 or SPR["n"] == 0 and y >= SPR["y0"]: SPR["n"] += 1; return 6
    return 0


def save(a):
    pickle.dump(a, open(out + ".tmp", "wb")); os.replace(out + ".tmp", out)          # atomic: readers never see a half-written file


if os.path.exists(out):
    arch = pickle.load(open(out, "rb")); print("resumed", len(arch))
else:
    e.step(ACTS[0]); r = R()
    arch = {(r[75], 0, 0, 0, 1, 0, 0): dict(state=em.get_state(), t=0, traj=[], q=4 * r[252] + 5 * tleft(r) + 1200 * r[247], visits=0, rooms=frozenset([r[75]]), x=0, y=0, room=r[75], ev=-999, born=0, score=r[244] * 65536 + r[245] * 256 + r[246])}
t0 = time.time(); tl = t0; it = max(c["born"] for c in arch.values());   # continue the "found at" clock of a resumed archive
cleared = False; allrooms = set(); PICK = []
while time.time() - t0 < budget and not cleared:
    it += 1
    if PICK:
        c = arch[PICK.pop()]
    else:
      keys = list(arch)
      if AUTO and (it % 50 == 1 or fr is None):
          smax = max(k[3] for k in keys); first = {}
          for k in keys:
              if k[3] == smax: first[k[0]] = min(first.get(k[0], 1e18), arch[k]["first"] if "first" in arch[k] else arch[k]["t"])
          fr = max(first, key=first.get)                                   # room reached last along the route (largest first-arrival step)
      if AUTO: keys = [k for k in keys if k[0] == fr and k[3] == smax] or list(arch)
      if FOCUS: keys = [k for k in keys if k[0] == FOCUS[0] and k[3] == FOCUS[1] and FOCUS[2] <= arch[k]["x"] <= FOCUS[3] and arch[k]["y"] <= (FOCUS[4] if len(FOCUS) > 4 else 999)] or list(arch)
      n = np.array([k[3] + k[6] / 16 for k in keys], float); v = np.array([arch[k]["visits"] for k in keys], float)   # n = stage
      q = np.array([arch[k]["q"] for k in keys], float)
      born = np.array([arch[k]["born"] for k in keys], float)
      reg = [(k[0], k[1] * G // 64, k[2] * G // 64, k[3], k[6]) for k in keys]; cnt = {}
      for g in reg: cnt[g] = cnt.get(g, 0) + 1
      rc = np.array([cnt[g] for g in reg], float)                        # cells in the same 64x64 region: rare regions first
      w = np.exp((n - n.max()) / TEMP) * np.exp((born - born.max()) / 3000) / np.sqrt(v + 1) / rc * (0.3 + q / q.max()); w /= w.sum()
      PICK = [keys[i] for i in rng.choice(len(keys), size=20, p=w)]          # sample 20 start cells per weight computation
      c = arch[PICK.pop()]
    c["visits"] += 1
    em.set_state(c["state"]); traj = list(c["traj"]); t = c["t"]; rooms = set(c["rooms"]); lives = R()[247]; plan = []; ev = c["ev"]; sc = c["score"]
    for step in range(100):
        if not plan: plan = choose(R())
        a = plan.pop()
        if a == -1: a = spring(R())
        for _ in range(4): e.step(ACTS[a])
        traj.append(a); t += 1; r = R()
        if r[247] <= 1 and r[252] == 0: break                             # never spend the last life; energy 0 = dying (states are archived again after the respawn)
        if r[71] > 2:
            cleared = True; f = f"runs/search_l3/level3_clear_seed{seed}_t{t}.json"; json.dump(traj, open(f, "w"))
            print("LEVEL 3 CLEARED at step", t, "->", f, flush=True); break
        rooms.add(r[75]); s2 = r[244] * 65536 + r[245] * 256 + r[246]
        if s2 > sc: ev = t                                                # last score event (gate disk hit, token, enemy)
        sc = s2
        if r[75] not in allrooms: allrooms.add(r[75]); print("NEW ROOM", r[75], "at step", t, "rooms visited", len(rooms), flush=True)
        k = (r[75], (r[86] + 256 * r[87]) // G, r[88] // G, stage(r), r[250] if G == 16 else 0, int(t - ev < 100), boss(r)); qq = 4 * r[252] + 5 * tleft(r) + 1200 * r[247]   # RAM[250] = tokens
        if (k not in arch or qq > arch[k]["q"] + 20) and safe():
            arch[k] = dict(state=em.get_state(), t=t, traj=list(traj), q=qq, visits=0, rooms=frozenset(rooms), x=r[86] + 256 * r[87], y=r[88], room=r[75], ev=ev, born=it, score=sc)
    if time.time() - tl > 60:
        tl = time.time(); save(arch); b = max(arch, key=lambda k: (k[3], arch[k]["q"]))
        print(round(tl - t0), "cells", len(arch), "rooms seen", len(allrooms), "highest stage", b[3], "tokens", b[4], "room", b[0],
              "step", arch[b]["t"], "q", arch[b]["q"], flush=True)
save(arch); print("done cells", len(arch), "cleared", cleared)
