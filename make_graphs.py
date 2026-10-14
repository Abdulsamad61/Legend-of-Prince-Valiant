"""The report's training curves: episode reward, episode length and level completion rate, for every saved checkpoint
(PPO baseline and imitation agent, Levels 1-3), plus the PPO entropy curve (exploration -> exploitation).
Each checkpoint plays one episode with deterministic actions (the game is deterministic, so one episode is enough).
Usage: python make_graphs.py            -> report/graph_level1.png, graph_level2.png, graph_level3.png, graph_ppo_entropy.png"""
import argparse, glob, os, re, numpy as np, torch, matplotlib; matplotlib.use("Agg"); torch.set_num_threads(2)
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pv.utils import make_flat, TimeFeature

p = argparse.ArgumentParser()
p.add_argument("--ppo1", default="runs/ppo_baseline"); p.add_argument("--ppo2", default="runs/l2_ppo")
p.add_argument("--imit1", default="runs/v5"); p.add_argument("--imit2", default="runs/l2_v1"); p.add_argument("--imit3", default="runs/l3_v1")
p.add_argument("--points", type=int, default=8, help="max checkpoints per curve")
a = p.parse_args(); os.makedirs("report", exist_ok=True)
DEMO_LEN = {1: 3611, 2: 2704, 3: 12533}           # route length = training samples per imitation epoch


def make_env(level, channels):
    if level == 1: return make_flat("Level1", noop_max=0, time_feature=channels == 5)
    if level == 2:
        from pv.env_l2 import make_l2; e = make_l2(episodic_life=False); return TimeFeature(e) if channels == 9 else e
    from pv.env_l3 import make_l3; e = make_l3(episodic_life=False); return TimeFeature(e, bits=14) if channels == 5 else e


def play(path, level):
    """One deterministic episode from the level start -> (episode reward, episode length, cleared 0/1)."""
    m = PPO.load(path[:-4], device="cpu"); env = make_env(level, m.observation_space.shape[0])
    o, _ = env.reset(seed=0); R, t, done = 0.0, 0, False
    while not done and t < 15000:
        act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
        o, r, te, tr, info = env.step(int(act[0])); R += r; t += 1; done = te or tr
    env.close(); return R, t, int(info["cleared"])


def curve(files, x_of, level, name):
    files = sorted(files, key=x_of); xs, res = [], []
    if len(files) > a.points: files = [files[round(i)] for i in np.linspace(0, len(files) - 1, a.points)]   # keep it light
    for f in files:
        xs.append(x_of(f)); res.append(play(f, level))
        print(f"Level {level} {name} {os.path.basename(f)}: reward {res[-1][0]:.1f}, length {res[-1][1]}, cleared {res[-1][2]}", flush=True)
    return np.array(xs), np.array(res)


def ppo_curve(run, level):
    return curve(glob.glob(f"{run}/ckpt/ppo_*_steps.zip"), lambda f: int(re.findall(r"_(\d+)_steps", f)[0]), level, "PPO")


def imit_curve(run, level):
    return curve(glob.glob(f"{run}/bc_epoch*.zip"), lambda f: int(re.findall(r"bc_epoch(\d+)", f)[0]) * DEMO_LEN[level], level, "imitation")


for level, ppo_run, imit_run in [(1, a.ppo1, a.imit1), (2, a.ppo2, a.imit2), (3, None, a.imit3)]:
    rows = [("PPO baseline (RL)", "training steps", ppo_curve(ppo_run, level))] if ppo_run and os.path.isdir(ppo_run) else []
    rows += [("Imitation agent", "training samples", imit_curve(imit_run, level))] if os.path.isdir(imit_run) else []
    if not rows: continue
    fig, ax = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)), squeeze=False)
    for i, (title, xlabel, (x, r)) in enumerate(rows):
        for j, (ylabel, y) in enumerate([("episode reward", r[:, 0]), ("episode length (steps)", r[:, 1]), ("level completion rate", r[:, 2])]):
            ax[i, j].plot(x, y, "o-"); ax[i, j].set_title(f"{title}: {ylabel}"); ax[i, j].set_xlabel(xlabel); ax[i, j].grid(alpha=0.3)
            if j == 2: ax[i, j].set_ylim(-0.05, 1.05)
    fig.suptitle(f"Level {level}"); fig.tight_layout(); fig.savefig(f"report/graph_level{level}.png", dpi=120); plt.close(fig)
    print("saved", f"report/graph_level{level}.png", flush=True)

# PPO entropy from TensorBoard: high = exploring (random keys), falling = exploiting (sure of its keys)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fig, ax = plt.subplots(figsize=(6, 3.6)); found = False
for level, run in [(1, a.ppo1), (2, a.ppo2)]:
    for ev in glob.glob(f"{run}/tb/*/events.out.tfevents.*"):
        acc = EventAccumulator(ev); acc.Reload()
        if "train/entropy_loss" in acc.Tags()["scalars"]:
            s = acc.Scalars("train/entropy_loss"); ax.plot([e.step for e in s], [-e.value for e in s], label=f"Level {level} PPO"); found = True
if found:
    ax.set_xlabel("training steps"); ax.set_ylabel("policy entropy"); ax.set_title("PPO: exploration -> exploitation"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("report/graph_ppo_entropy.png", dpi=120); print("saved report/graph_ppo_entropy.png")
