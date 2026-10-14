"""Figure 4: Level 2 - PPO curves (TensorBoard) and imitation completion per checkpoint."""
import glob, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(glob.glob("runs/l2_ppo/tb/*/events*")[0]); ea.Reload()
g = lambda t: (np.array([s.step for s in ea.Scalars(t)]) / 1e6, np.array([s.value for s in ea.Scalars(t)]))
log = pickle.load(open("runs/l2_v1/bc_log.pkl", "rb"))
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for a, (t, lab) in zip(ax[:3], [("rollout/ep_rew_mean", "episode reward (one life)"), ("rollout/ep_len_mean", "episode length (steps)"),
                                 ("game/kills_per_episode", "robbers hit per episode")]):
    x, y = g(t); a.plot(x, y, color="#1c5cab", lw=1.2); a.set_title("PPO: " + lab, fontsize=10); a.set_xlabel("env steps (M)"); a.grid(alpha=.3)
x, y = g("game/clear_rate"); ax[3].plot(x / x.max() * 0 + np.linspace(0, 1, len(x)), y * 100, color="#999", lw=1.2, label="PPO (fraction of 1.53M steps)")
ep = [l["epoch"] for l in log]; ax[3].plot(np.array(ep) / max(ep), [100 * l["cleared"] for l in log], "o-", color="#c0392b", label="imitation (fraction of 60 epochs)")
ax[3].set_title("Level 2 completion rate (%)", fontsize=10); ax[3].set_xlabel("training progress"); ax[3].set_ylim(-5, 105); ax[3].legend(fontsize=7.5); ax[3].grid(alpha=.3)
plt.tight_layout(); plt.savefig("report/fig_level2.png", dpi=150); print("ok")
