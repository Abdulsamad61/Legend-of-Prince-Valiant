"""Training curves for the report: episode reward, episode length (TensorBoard, PPO runs) and level-completion
rate across checkpoints (checkpoint_eval.json from eval_checkpoints.py; imitation runs also use bc_log.pkl).
Usage: python plot_curves.py --runs runs/v1 runs/v2 runs/v5 --out results/curves.png"""
import argparse, glob, json, os, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p = argparse.ArgumentParser(); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", default="results/curves.png")
a = p.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for run in a.runs:
    name = os.path.basename(run.rstrip("/"))
    ev = sorted(glob.glob(f"{run}/tb/*/events*"))
    if ev:
        ea = EventAccumulator(os.path.dirname(ev[-1])); ea.Reload(); tags = ea.Tags()["scalars"]
        for i, tag in enumerate(["rollout/ep_rew_mean", "rollout/ep_len_mean"]):
            if tag in tags:
                s = ea.Scalars(tag); ax[i].plot([e.step for e in s], [e.value for e in s], lw=2, label=name)
    if os.path.exists(f"{run}/bc_log.pkl"):                     # imitation run: x = samples seen
        log = pickle.load(open(f"{run}/bc_log.pkl", "rb"))
        ax[0].plot([], [])                                       # keep colours aligned
    if os.path.exists(f"{run}/checkpoint_eval.json"):
        ce = json.load(open(f"{run}/checkpoint_eval.json"))
        xs = [r["x"] for r in ce]; xlabel = "imitation epoch" if "bc_epoch" in ce[0]["checkpoint"] else "env steps"
        ax[2].plot(xs, [100 * r["completion"] for r in ce], lw=2, marker="o", label=f"{name} ({xlabel})")
ax[0].set_title("Episode reward (mean of last 100)", loc="left"); ax[0].set_xlabel("env steps")
ax[1].set_title("Episode length (steps)", loc="left"); ax[1].set_xlabel("env steps")
ax[2].set_title("Level 1 completion rate (normal start, deterministic)", loc="left"); ax[2].set_ylabel("%"); ax[2].set_ylim(-5, 105)
for x in ax: x.grid(alpha=.3); x.legend(frameon=False, fontsize=9); x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)
