"""Level 2 PPO training - pure RL (no demonstrations, no imitation).
Episodes start from the Level 2 save state (allowed by the task: 'training may start from a save state at the
beginning of each level'). Logs episode kills / distance / clear rate to TensorBoard next to reward and length.
Usage: python train_l2.py --steps 2000000 --envs 8 --seed 0 --out runs/l2_ppo"""
import argparse, json, os, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from pv.env_l2 import make_l2

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=2_000_000); p.add_argument("--envs", type=int, default=8)
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="runs/l2_ppo")
p.add_argument("--ent", type=float, default=0.01); p.add_argument("--init", default=None)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)


class EpisodeStats(BaseCallback):
    """Logs game-specific episode statistics: kills, distance reached, level-clear rate."""
    def __init__(self): super().__init__(); self.buf = []
    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done: self.buf.append((info.get("kills", 0), info.get("distance", 0), float(info.get("cleared", False))))
        if len(self.buf) >= 20:
            k, d, c = np.mean(self.buf, axis=0)
            self.logger.record("game/kills_per_episode", k); self.logger.record("game/distance", d); self.logger.record("game/clear_rate", c)
            self.buf = []
        return True


venv = VecTransposeImage(VecMonitor(SubprocVecEnv([lambda: make_l2(episodic_life=True) for _ in range(a.envs)], start_method="fork")))
venv.seed(a.seed)
model = PPO("CnnPolicy", venv, n_steps=128, batch_size=256, n_epochs=4, gamma=0.99, gae_lambda=0.95,
            learning_rate=lambda f: 2.5e-4 * f, ent_coef=a.ent, clip_range=0.1, seed=a.seed, tensorboard_log=f"{a.out}/tb",
            device="cuda" if torch.cuda.is_available() else "cpu", verbose=1)
if a.init: model.set_parameters(a.init)
model.learn(a.steps, callback=[CheckpointCallback(max(a.steps // 20 // a.envs, 1), f"{a.out}/ckpt", "ppo"), EpisodeStats()], tb_log_name="ppo")
model.save(f"{a.out}/final")
