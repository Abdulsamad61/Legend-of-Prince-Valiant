"""Level 2 PPO baseline video: early / middle / final PPO checkpoint, each one uncut episode from the Level 2 start
(all lives, deterministic actions). Overlay: checkpoint training steps and episode reward.
Usage: python make_video_ppo_l2.py --run runs/l2_ppo --out clips/ppo_level2.mp4"""
import argparse, glob, os, re, subprocess, cv2, numpy as np, torch; torch.set_num_threads(2)
from stable_baselines3 import PPO
from pv.env_l2 import make_l2
from pv.utils import TimeFeature
p = argparse.ArgumentParser(); p.add_argument("--run", default="runs/l2_ppo"); p.add_argument("--out", default="clips/ppo_level2.mp4")
p.add_argument("--fps", type=int, default=25); p.add_argument("--max_steps", type=int, default=3000)
a = p.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
S = 4
steps = lambda f: int(re.findall(r"_(\d+)_steps", f)[0])
ck = sorted(glob.glob(f"{a.run}/ckpt/ppo_*_steps.zip"), key=steps)
SEGMENTS = [("EARLY", ck[len(ck) // 3 - 1]), ("MIDDLE", ck[2 * len(ck) // 3 - 1]), ("FINAL", ck[-1])] if len(ck) >= 3 else [("FINAL", f"{a.run}/final.zip")]


def put(img, txt, y, scale=0.8, col=(255, 255, 0)):
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4); cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)


tmp = a.out.replace(".mp4", "_raw.mp4"); vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (240 * S, 224 * S))
for title, path in SEGMENTS:
    m = PPO.load(path[:-4], device="cpu"); env = make_l2(episodic_life=False)
    if m.observation_space.shape[0] == 9: env = TimeFeature(env)
    o, info = env.reset(seed=0); raw = env.unwrapped; n = steps(path) if "_steps" in path else "final"
    R, done, t = 0.0, False, 0
    while not done and t < a.max_steps:
        act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
        o, r, te, tr, info = env.step(int(act[0])); R += r; done = te or tr; t += 1
        fr = cv2.cvtColor(cv2.resize(raw.get_screen(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        put(fr, f"PPO {title} | {n:,} training steps" if n != "final" else f"PPO {title}", 34)
        put(fr, f"episode reward {R:7.1f}   kills {info['kills']}   lives {info['lives']}", 70)
        vw.write(fr)
    print(title, path, "reward", round(R, 1), "kills", info["kills"], "distance", info["distance"], "cleared", info["cleared"], flush=True)
    env.close()
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", a.out], check=True)
os.remove(tmp); print("wrote", a.out)
