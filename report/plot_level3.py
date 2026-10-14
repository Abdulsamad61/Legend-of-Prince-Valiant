"""Figure 5: Level 3 - time left on the 10:00 level timer along the routes, and imitation completion per checkpoint."""
import json, pickle, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, stable_retro as retro
sys.path.insert(0, ".")
from pv.env_l3 import GAME, COMBOS
e = retro.make(GAME, state="Level3", render_mode=None); names = e.unwrapped.buttons
A = np.zeros((len(COMBOS), len(names)), np.int8)
for i, c in enumerate(COMBOS):
    for b in c: A[i, names.index(b)] = 1
def timeline(route):
    e.reset(); e.step(A[0]); out = []
    for a in route:
        for _ in range(4): e.step(A[a])
        r = e.unwrapped.get_ram(); out.append(int(r[647]) * 60 + int(r[648]) * 10 + int(r[649]))
    return np.array(out)
final = timeline(json.load(open("optional_demo/demo_level3_route.json")))
first = timeline(json.load(open("optional_demo/level3_first_search_route.json")))
log = pickle.load(open("runs/l3_v1/bc_log.pkl", "rb"))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(np.arange(len(first)) / 15, first, color="#999", label="first whole-level search route (time runs out)")
ax[0].plot(np.arange(len(final)) / 15, final, color="#1c5cab", label="final route (room-by-room rebuilt, T items)")
ax[0].set_xlabel("game time (s)"); ax[0].set_ylabel("seconds left on the timer"); ax[0].set_title("Level 3 timer along the route", fontsize=10)
ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
ep = [l["epoch"] for l in log]
ax[1].plot(ep, [100 * l["acc"] for l in log], "s-", color="#888", label="actions matching the route (%)")
ax[1].plot(ep, [100 * l["cleared"] for l in log], "o-", color="#c0392b", label="Level 3 cleared (%)")
ax[1].set_xlabel("imitation epoch"); ax[1].set_ylim(-5, 105); ax[1].set_title("Level 3 imitation", fontsize=10); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("report/fig_level3.png", dpi=150); print("ok")
