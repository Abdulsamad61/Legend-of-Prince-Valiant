"""Level 3 route builder, room by room (demonstration generator, NOT RL).
The whole-level search (tools/go_explore_l3.py) found WHICH rooms to pass and in which order, but its route is too slow
for the single 10:00 timer. This tool rebuilds the route one room at a time: from the state where the hero enters a room,
a small Go-Explore search (same exploration routines: sticky random actions, door/duck/wait holds, disk shooting,
springboard) looks for the fastest way to the next room of the plan (and, where the plan says so, to a quest flag),
guided along the slow route's path (cells further along it are explored first), keeping energy and lives in mind: segment score = 5 * seconds left + 2 * energy + 1200 * lives.
The best result becomes the start of the next segment. Progress is saved after every segment (resumable).
usage: python tools/segments_l3.py PLAN_JSON OUT_DIR [SECONDS_PER_SEGMENT=300] [REFERENCE_ROUTE_JSON]
PLAN_JSON: list of [room_to_reach, flags_required(, extra seconds to collect(, xmin, xmax, ymin, ymax: waypoint region in that room))] e.g. [[18,0],[14,0],[8,0],[14,1],...]; flags = number of quest flags set."""
import json, os, pickle, sys, time, numpy as np, stable_retro as retro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pv.env_l3 import GAME, COMBOS

plan = json.load(open(sys.argv[1])); outd = sys.argv[2]; SEG = float(sys.argv[3]) if len(sys.argv) > 3 else 300
os.makedirs(outd, exist_ok=True); rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
e = retro.make(GAME, state="Level3", render_mode=None); e.reset(); em = e.unwrapped.em; names = e.unwrapped.buttons
ACTS = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: ACTS[i, names.index(b)] = 1
R = lambda: e.unwrapped.get_ram().astype(int)
tleft = lambda r: r[647] * 60 + r[648] * 10 + r[649]
FLAGS = [513, 514, 515, 516]
stage = lambda r: int(sum(r[j] > 0 for j in FLAGS))
score = lambda r: 5 * tleft(r) + 2 * r[252] + 1200 * r[247]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "go_explore_l3.py")).read()
ns = {"np": np, "rng": rng, "R": R, "COMBOS": COMBOS}
exec(src[src.index("def boss"):src.index("def save")], ns)                # boss(), disk_x(), choose(), spring(): shared routines
boss, choose, spring, SPR = ns["boss"], ns["choose"], ns["spring"], ns["SPR"]


def safe():
    st = em.get_state(); l0 = R()[247]; ok = R()[252] > 0
    for _ in range(40):
        if not ok: break
        e.step(ACTS[0]); rr = R()
        if rr[247] < l0 or rr[252] == 0: ok = False
    em.set_state(st); return ok


GUIDE = np.zeros((0, 3))


def progress(r):
    """How far along the slow route's path (GUIDE = its positions in this room) the hero is: index of the furthest
    guide point within 12 px horizontally and 16 px vertically, or -1."""
    if not len(GUIDE): return 0
    x, y = r[86] + 256 * r[87], r[88]
    m = (GUIDE[:, 0] == r[75]) & (np.abs(GUIDE[:, 1] - x) <= 12) & (np.abs(GUIDE[:, 2] - y) <= 16)
    idx = np.nonzero(m)[0]
    return int(idx.max()) if len(idx) else -1


def roll(arch, tr, acts, room0, goal_room, goal_stage, best, bonus=0, t_start=0, region=None):
    """Play acts from the current emulator state; archive new / better cells; return (best, finished)."""
    ev = -999; sc0 = R()[244:247].tolist()
    for a in acts:
        if a == -1: a = spring(R())
        for _ in range(4): e.step(ACTS[a])
        tr.append(a); r = R()
        if r[247] <= 1 and r[252] == 0: return best, True
        if r[244:247].tolist() != sc0: ev = len(tr); sc0 = r[244:247].tolist()
        got = tleft(r) + len(tr) * 4 / 60 >= t_start + bonus                # extra-time goal: a T item collected
        inreg = region is None or (region[0] <= r[86] + 256 * r[87] <= region[1] and region[2] <= r[88] <= region[3])
        if r[71] > 2 or (r[75] == goal_room and stage(r) >= goal_stage and r[252] > 0 and got and inreg):
            sc = score(r) + (100000 if r[71] > 2 else 0)
            if best is None or sc > best[0] + 4: best = (sc, em.get_state(), list(tr), time.time())
            return best, True
        if r[75] != room0 and r[75] != goal_room: return best, True          # left the room the wrong way
        k = (r[75], (r[86] + 256 * r[87]) // 16, r[88] // 16, stage(r), int(len(tr) - ev < 100), boss(r)); qq = score(r)
        if r[252] > 0 and (k not in arch or qq > arch[k]["q"] + 10) and safe():
            arch[k] = dict(state=em.get_state(), traj=list(tr), q=qq, visits=0, prog=progress(r))
    return best, False


def segment(start, base, goal_room, goal_stage, bonus=0, guide=None, region=None):
    """Fastest way from start to goal_room (with goal_stage quest flags). base = the actions the slow whole-level route
    used here; replayed first (if it still works from this start it is the baseline and its states seed the search)."""
    global GUIDE
    GUIDE = np.array(guide if guide else [], float).reshape(-1, 3)
    em.set_state(start); r = R(); room0 = r[75]; ts = tleft(r)
    arch = {("s",): dict(state=start, traj=[], q=score(r), visits=0, prog=0)}; best = None; t0 = time.time()
    best, _ = roll(arch, [], list(base), room0, goal_room, goal_stage, best, bonus, ts, region)
    if best: print("   baseline works:", len(best[2]), "steps", flush=True)
    while time.time() - t0 < (SEG if best else 3 * SEG):
        if best is not None and time.time() - best[3] > SEG / 2: break           # no better finish for a while
        keys = list(arch); v = np.array([arch[k]["visits"] for k in keys], float); q = np.array([arch[k]["q"] for k in keys], float)
        reg = [k[:3] if len(k) > 2 else k for k in keys]; cnt = {}
        for g in reg: cnt[g] = cnt.get(g, 0) + 1
        rc = np.array([cnt[g] for g in reg], float)
        pg = np.array([arch[k]["prog"] for k in keys], float)                # guided: prefer cells further along the slow route's path
        qn = (q - q.min()) / (q.max() - q.min() + 1)
        w = np.exp((pg - pg.max()) / (len(GUIDE) / 15 + 1)) * (0.5 + qn) / np.sqrt(v + 1) / rc if best is None else (0.05 + qn) ** 3 / np.sqrt(v + 1) / rc   # once the goal is reached: hunt for faster states
        w /= w.sum()
        c = arch[keys[rng.choice(len(keys), p=w)]]; c["visits"] += 1
        em.set_state(c["state"]); acts = []
        while len(acts) < 100: acts += choose(R())[::-1]
        best, _ = roll(arch, list(c["traj"]), acts[:100], room0, goal_room, goal_stage, best, bonus, ts, region)
    return best, len(arch)


prog_f = os.path.join(outd, "progress.pkl")
if os.path.exists(prog_f): P = pickle.load(open(prog_f, "rb")); print("resume at segment", len(P["done"]))
else:
    e.reset(); e.step(ACTS[0]); P = dict(state=em.get_state(), traj=[], done=[])
# reference route (the slow whole-level one): its room entries give each segment's baseline actions
ref = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else []
e.reset(); e.step(ACTS[0]); entries = [(0, R()[75], 0)]; pos = []
for t, a in enumerate(ref):
    for _ in range(4): e.step(ACTS[a])
    r = R(); pos.append((r[75], r[86] + 256 * r[87], r[88]))
    if r[75] != entries[-1][1]: entries.append((t + 1, r[75], stage(r)))
ptr = 0; bases = []; guides = []
for room, stg, *extra in plan:
    if len(extra) > 1: bases.append([]); guides.append([]); continue      # waypoint inside a room: no baseline
    j = next((j for j in range(ptr + 1, len(entries)) if entries[j][1] == room and entries[j][2] >= stg), None)
    bases.append(ref[entries[ptr][0]:entries[j][0]] if j else []); guides.append(pos[entries[ptr][0]:entries[j][0]] if j else []); ptr = j if j else ptr
for i in range(len(P["done"]), len(plan)):
    room, stg = plan[i][:2]; bonus = plan[i][2] if len(plan[i]) > 2 else 0; region = plan[i][3:7] if len(plan[i]) > 3 else None
    best, n = segment(P["state"], [] if region else bases[i], room, stg, bonus, None if region else guides[i], region)
    if best is None: print("segment", i, "->", plan[i], "NOT reached (cells", n, ")", flush=True); break
    P.setdefault("states", []).append(P["state"]); P["state"], P["traj"] = best[1], P["traj"] + best[2]; em.set_state(P["state"]); r = R()
    P["done"].append(dict(goal=plan[i], steps=len(best[2]), t=len(P["traj"]), time_left=tleft(r), energy=r[252], lives=r[247]))
    pickle.dump(P, open(prog_f, "wb")); json.dump(P["traj"], open(os.path.join(outd, "route.json"), "w"))
    print("segment", i, plan[i], "steps", len(best[2]), "total", len(P["traj"]), "time left", tleft(r), "energy", r[252], "lives", r[247], flush=True)
    if r[71] > 2: print("LEVEL 3 CLEARED, route", len(P["traj"]), "steps"); json.dump(P["traj"], open("runs/search_l3/level3_clear_segments.json", "w")); break
