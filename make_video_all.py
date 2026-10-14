"""One progress video for the task (MP4, 960x896 = 4x the NES frame, original aspect ratio).
Level 1: early and middle imitation checkpoints (each played until it loses its first life), then the final agent's
full run; then the final agents' full runs of Level 2 and Level 3. Every run is shown from start to end without cuts,
as a time-lapse (every 3rd agent step = 6x real time) so all three levels fit in about 4-5 minutes.
The overlay shows the level, checkpoint, training samples, step and episode reward.
Usage: python make_video_all.py --out clips/progress_video.mp4"""
import argparse, glob, os, re, subprocess, cv2, numpy as np, torch
from stable_baselines3 import PPO
from pv.utils import make_flat, TimeFeature
from pv.env_l2 import make_l2
from pv.env_l3 import make_l3

torch.set_num_threads(2)
p = argparse.ArgumentParser(); p.add_argument("--out", default="clips/progress_video.mp4")
p.add_argument("--stride", type=int, default=3); p.add_argument("--fps", type=int, default=30)
a = p.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
S = 4
epoch = lambda f: int(re.findall(r"bc_epoch(\d+)", f)[0])
l1 = sorted(glob.glob("runs/v5/bc_epoch*.zip"), key=epoch)
SEGMENTS = [  # (level, title, model, demo length, play until first life lost?)
    (1, "EARLY checkpoint", l1[len(l1) // 3][:-4], 3611, True),
    (1, "MIDDLE checkpoint", l1[2 * len(l1) // 3][:-4], 3611, True),
    (1, "FINAL agent", "runs/v5/final", 3611, False),
    (2, "FINAL agent", "runs/l2_v1/final", 2704, False),
    (3, "FINAL agent", "runs/l3_v1/final", 12533, False)]


def make_env(level):
    if level == 1: return make_flat("Level1", noop_max=0, time_feature=True)
    if level == 2: return TimeFeature(make_l2(episodic_life=False))
    return TimeFeature(make_l3(episodic_life=False), bits=14)


def put(img, txt, y, scale=0.8, col=(255, 255, 0)):
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)


tmp = a.out.replace(".mp4", "_raw.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (240 * S, 224 * S))


def card(lines, secs=2.5):
    for _ in range(int(secs * a.fps)):
        img = np.zeros((224 * S, 240 * S, 3), np.uint8)
        for i, l in enumerate(lines): put(img, l, 330 + 60 * i, 1.0, (255, 255, 255))
        vw.write(img)


for level, title, path, demo_len, first_life in SEGMENTS:
    m = PPO.load(path, device="cpu"); env = make_env(level); o, info = env.reset(seed=0); raw = env.unwrapped
    ep = epoch(path) if "bc_epoch" in path else None
    train = f"imitation epoch {ep} ({ep * demo_len // 1000}k samples)" if ep else "final model"
    card([f"LEVEL {level} - {title}", train, "deterministic actions, " + ("until the first life is lost" if first_life else "full run"),
          f"time-lapse: {2 * a.stride}x real time"])
    R, done, t, lives0 = 0.0, False, 0, info.get("lives")
    while not done and t < 15000:
        act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
        o, r, te, tr, info = env.step(int(act[0])); R += r; done = te or tr; t += 1
        if first_life and info.get("lives") is not None and info["lives"] < lives0: done = True
        if t % a.stride == 0 or done:
            fr = cv2.cvtColor(cv2.resize(raw.get_screen(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            put(fr, f"LEVEL {level} | {title} | {train}", 34)
            put(fr, f"step {t}   episode reward {R:7.1f}   lives {info.get('lives')}", 70)
            vw.write(fr)
    card([f"LEVEL {level} CLEARED" if info.get("cleared") else f"LEVEL {level}: episode over", f"{t} steps, episode reward {R:.1f}"], 2.0)
    env.close()
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", a.out], check=True)
os.remove(tmp); print("wrote", a.out)
