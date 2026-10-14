"""Record agent gameplay: native 240x224 NES screen upscaled 4x (nearest) -> 960x896, with label/reward overlay.
Uses the exact training env (make_flat) so the agent sees the same observations it was trained on.
Usage: python record_video.py --model runs/level1/ckpt/ppo_X_steps.zip --label "step 250k" --out clips/early.mp4"""
import argparse, cv2, numpy as np, torch; torch.set_num_threads(1)
from stable_baselines3 import PPO
from pv.utils import make_flat
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--state", default="Level1")
p.add_argument("--label", default=""); p.add_argument("--out", required=True); p.add_argument("--max", type=int, default=3000)
p.add_argument("--stochastic", action="store_true"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--noop", type=int, default=30)
a = p.parse_args()
env = make_flat(a.state, noop_max=a.noop); raw = env.unwrapped
m = PPO.load(a.model.removesuffix(".zip"), device="cpu"); o, _ = env.reset(seed=a.seed)
m.set_random_seed(a.seed)
S = 4; h, w = raw.get_screen().shape[:2]; W, H = w * S, h * S
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H)); R, done, n = 0.0, False, 0
while not done and n < a.max:
    act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=not a.stochastic)
    o, r, te, tr, info = env.step(int(act[0])); R += r; done = te or tr; n += 1
    fr = cv2.resize(raw.get_screen(), (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.putText(fr, f"{a.label}  reward {R:.1f}  sec {info['section']}  x {info['progress'] % 2048}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
vw.release(); print(f"{a.out}: {n} steps, reward {R:.1f}, progress {info['progress']}")
