"""Progress video per the task: early / middle / final checkpoint, then one full uncut Level 1 run by the final agent.
MP4, 4x nearest-neighbour upscale of the native 240x224 NES frame (960x896, original aspect), overlay with checkpoint
training count and running episode reward. Played at 20 fps = 1.33x real time (one agent step = 4 NES frames) so the
whole video fits in 3-5 minutes. Usage: python make_video.py --out clips/progress_video.mp4"""
import argparse, subprocess, cv2, numpy as np, torch; torch.set_num_threads(2)
from stable_baselines3 import PPO
from pv.utils import make_flat
import glob, os, re
p = argparse.ArgumentParser(); p.add_argument("--run", default="runs/v5", help="run folder with bc_epoch*.zip or ckpt/ppo_*_steps.zip and final.zip")
p.add_argument("--out", default="clips/progress_video.mp4"); p.add_argument("--fps", type=int, default=20); p.add_argument("--demo_len", type=int, default=3611)
a = p.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
S, DEMO_LEN = 4, a.demo_len
num = lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[0])
cks = sorted(glob.glob(f"{a.run}/bc_epoch*.zip") or glob.glob(f"{a.run}/ckpt/ppo_*_steps.zip"), key=num)
assert cks, "no checkpoints found in " + a.run
SEGMENTS = [("EARLY checkpoint", cks[len(cks) // 3][:-4]), ("MIDDLE checkpoint", cks[(2 * len(cks)) // 3][:-4]),
            ("FINAL agent - full Level 1 run", f"{a.run}/final" if os.path.exists(f"{a.run}/final.zip") else cks[-1][:-4])]
def label(path):
    if "bc_epoch" in path: e = num(path); return f"imitation epoch {e} ({e * DEMO_LEN // 1000}k samples)"
    if "ppo_" in path: return f"{num(path):,} env steps"
    return "final model"
tmp = a.out.replace(".mp4", "_raw.mp4"); vw = None

def put(img, txt, y, scale=0.8, col=(255, 255, 0)):
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4); cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)

def card(lines, secs=2.5):
    for _ in range(int(secs * a.fps)):
        img = np.zeros((224 * S, 240 * S, 3), np.uint8)
        for i, l in enumerate(lines): put(img, l, 330 + 60 * i, 1.0, (255, 255, 255))
        vw.write(img)

for title, path in SEGMENTS:
    m = PPO.load(path, device="cpu"); lab = label(path) if not path.endswith("final") else label(cks[-1][:-4]) + " (final)"
    env = make_flat("Level1", noop_max=0, time_feature=m.observation_space.shape[0] == 5); o, info = env.reset(seed=0); raw = env.unwrapped
    if vw is None:
        h, w = raw.get_screen().shape[:2]; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w * S, h * S))
    card([title, lab, "normal game start, deterministic actions"])
    R, done, t = 0.0, False, 0
    while not done and t < 6000:
        act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
        o, r, te, tr, info = env.step(int(act[0])); R += r; done = te or tr; t += 1
        fr = cv2.cvtColor(cv2.resize(raw.get_screen(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        put(fr, f"{title.split(' -')[0].split(' checkpoint')[0]} | {lab}", 34)
        put(fr, f"episode reward {R:7.1f}   section {info['section']}   lives {info['lives']}   step {t}", 70)
        vw.write(fr)
    end = "LEVEL 1 CLEARED" if info["cleared"] else f"episode over: section {info['section']}, x {info['progress'] % 2048}"
    if info["cleared"]:                      # let the game's own LEVEL 1 COMPLETE screen play, still uncut
        for i in range(260):
            for _ in range(4): raw.step(np.zeros(9, np.int8))
            fr = cv2.cvtColor(cv2.resize(raw.get_screen(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            put(fr, f"FINAL | episode reward {R:.1f} | LEVEL 1 CLEARED", 34); vw.write(fr)
    card([end, f"episode reward {R:.1f}, {t} steps"], 2.0); env.close()
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", a.out], check=True)
os.remove(tmp); print("wrote", a.out)
