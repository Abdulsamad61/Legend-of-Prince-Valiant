"""Level-completion rate across training checkpoints (for the 'completion rate' curve the task asks for).
Evaluates every checkpoint of a run from the NORMAL start with deterministic actions.
  PPO runs:       runs/<name>/ckpt/ppo_<steps>_steps.zip
  imitation runs: runs/<name>/bc_epoch<N>.zip
Usage: python eval_checkpoints.py --run runs/v5 --episodes 3"""
import argparse, glob, json, re, numpy as np, torch; torch.set_num_threads(1)
from stable_baselines3 import PPO
from pv.utils import make_flat
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--episodes", type=int, default=3)
p.add_argument("--noop", type=int, default=0); p.add_argument("--max_steps", type=int, default=6000)
a = p.parse_args()
cks = glob.glob(f"{a.run}/ckpt/ppo_*_steps.zip") + glob.glob(f"{a.run}/bc_epoch*.zip")
key = lambda f: int(re.findall(r"(\d+)", f.split("/")[-1])[0])
out = []
for ck in sorted(cks, key=key):
    m = PPO.load(ck[:-4], device="cpu"); tf = m.observation_space.shape[0] == 5
    env = make_flat("Level1", noop_max=a.noop, time_feature=tf); res = []
    for ep in range(a.episodes):
        o, _ = env.reset(seed=ep); done, t, R = False, 0, 0.0
        while not done and t < a.max_steps:
            act, _ = m.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
            o, r, te, tr, info = env.step(int(act[0])); R += r; done = te or tr; t += 1
        res.append((bool(info["cleared"]), int(info["section"]), int(info["progress"]), R, t))
    env.close()
    row = dict(checkpoint=ck, x=key(ck), completion=float(np.mean([r[0] for r in res])), section=float(np.mean([r[1] for r in res])),
               progress=float(np.mean([r[2] for r in res])), reward=float(np.mean([r[3] for r in res])), length=float(np.mean([r[4] for r in res])))
    out.append(row); print(row, flush=True)
json.dump(out, open(f"{a.run}/checkpoint_eval.json", "w"), indent=1)
