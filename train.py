import argparse, json, os, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from pv.utils import make_flat
from pv.env import load_starts

p = argparse.ArgumentParser()
p.add_argument("--state", default="Level1"); p.add_argument("--steps", type=int, default=5_000_000)
p.add_argument("--envs", type=int, default=8); p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="runs/level1")
p.add_argument("--curriculum", type=float, default=0.0, help="share of episodes that start from a curriculum savestate (needs --starts)")
p.add_argument("--starts", default=None, help="pickle of savestates, e.g. runs/bc/demo_starts.pkl written by bc.py")
p.add_argument("--init", default=None, help="policy weights to start from (e.g. runs/bc/bc_policy.pt)")
p.add_argument("--ent", type=float, default=0.02)
a = p.parse_args()

starts = load_starts(a.starts) if a.curriculum > 0 else None
def mk(i): return lambda: make_flat(a.state, starts=starts, p_start=a.curriculum)
venv = VecTransposeImage(VecMonitor(SubprocVecEnv([mk(i) for i in range(a.envs)], start_method="fork")))
venv.seed(a.seed)                                   # gives every sub-env its own seed (seed + i)
model = PPO("CnnPolicy", venv, n_steps=256, batch_size=256, n_epochs=4, gamma=0.99, gae_lambda=0.95,
            learning_rate=lambda f: 2.5e-4 * f,          # linear LR decay
            ent_coef=a.ent, clip_range=0.1, seed=a.seed, tensorboard_log=f"{a.out}/tb",
            device="cuda" if torch.cuda.is_available() else "cpu", verbose=1)
if a.init: model.policy.load_state_dict(torch.load(a.init, map_location=model.device))   # warm start from demos
os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)  # reproducibility
# checkpoints every ~5% of budget -> early / mid / final for the video
model.learn(a.steps, callback=CheckpointCallback(max(a.steps // 20 // a.envs, 1), f"{a.out}/ckpt", "ppo"), tb_log_name="ppo")
model.save(f"{a.out}/final")
