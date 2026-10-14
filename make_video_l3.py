"""Level 3 progress video: early / middle / final checkpoint, each a complete uncut episode from the Level 3 start
(all lives, deterministic actions), the final one being the full Level 3 run (12,533 steps). MP4, 4x nearest-neighbour upscale of the
native 240x224 frame (960x896, original aspect), overlay with checkpoint, training samples, episode reward.
Played at 50 fps = 3.3x real time (one agent step = 4 NES frames) so the 10-minute level fits in a 3-5 minute video.
Usage: python make_video_l3.py --run runs/l3_v1 --out clips/progress_video_level3.mp4"""
import argparse, json, os, subprocess, cv2, numpy as np, torch; torch.set_num_threads(2)
from stable_baselines3 import PPO
from pv.env_l3 import make_l3
from pv.utils import TimeFeature
p = argparse.ArgumentParser(); p.add_argument("--run", default="runs/l3_v1"); p.add_argument("--out", default="clips/progress_video_level3.mp4")
p.add_argument("--fps", type=int, default=50); p.add_argument("--demo", default="optional_demo/demo_level3_route.json")
a = p.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
S, DEMO_LEN = 4, len(json.load(open(a.demo)))
ck = sorted(int(f[8:-4]) for f in os.listdir(a.run) if f.startswith("bc_epoch") and f.endswith(".zip"))
last = ck[-1]; early, mid = ck[max(0, len(ck) // 3 - 1)], ck[max(0, 2 * len(ck) // 3 - 1)]
SEGMENTS = [("EARLY checkpoint", early, f"{a.run}/bc_epoch{early}"), ("MIDDLE checkpoint", mid, f"{a.run}/bc_epoch{mid}"),
            ("FINAL agent - full Level 3 run", last, f"{a.run}/final")]
tmp = a.out.replace(".mp4", "_raw.mp4"); vw = None


def put(img, txt, y, scale=0.8, col=(255, 255, 0)):
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4); cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)


def card(lines, secs=2.5):
    for _ in range(int(secs * a.fps)):
        img = np.zeros((224 * S, 240 * S, 3), np.uint8)
        for i, l in enumerate(lines): put(img, l, 330 + 60 * i, 1.0, (255, 255, 255))
        vw.write(img)


for title, epoch, path in SEGMENTS:
    m = PPO.load(path, device="cpu"); env = TimeFeature(make_l3(episodic_life=not title.startswith("FINAL")), bits=14); o, info = env.reset(seed=0); raw = env.unwrapped   # early / middle: until the first life is lost (keeps the video short)
    if vw is None:
        h, w = raw.get_screen().shape[:2]; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w * S, h * S))
    card([title, f"imitation epoch {epoch}  ({epoch * DEMO_LEN:,} training samples)", "Level 3 start, deterministic actions" + ("" if title.startswith("FINAL") else ", until the first life is lost")])
    R, done, t = 0.0, False, 0
    while not done and t < 15000:
        act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
        o, r, te, tr, info = env.step(int(act[0])); R += r; done = te or tr; t += 1
        fr = cv2.cvtColor(cv2.resize(raw.get_screen(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        put(fr, f"{title.split(' -')[0]} | epoch {epoch} ({epoch * DEMO_LEN // 1000}k samples)", 34)
        put(fr, f"episode reward {R:7.1f}   rooms {info['rooms']}   lives {info['lives']}   step {t}", 70)
        vw.write(fr)
    end = "LEVEL 3 CLEARED" if info["cleared"] else f"episode over: {info['rooms']} rooms visited"
    if info["cleared"]:                      # let the game's level-complete screen play, still uncut
        for i in range(60):
            for _ in range(4): raw.step(np.zeros(9, np.int8))
            fr = cv2.cvtColor(cv2.resize(raw.get_screen(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            put(fr, f"FINAL | episode reward {R:.1f} | LEVEL 3 CLEARED", 34); vw.write(fr)
    card([end, f"episode reward {R:.1f}, {t} steps"], 2.0); env.close()
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", a.out], check=True)
os.remove(tmp); print("wrote", a.out)
