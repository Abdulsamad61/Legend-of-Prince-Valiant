"""Imitation learning until the policy reproduces a full-level demonstration exactly.

The emulator is deterministic, so a policy whose greedy action equals the demo action at every visited
state replays the whole route and clears Level 1 from the normal start. We train the PPO CnnPolicy
(same network PPO uses) with cross-entropy on (observation, action) pairs from the demo, using the
time-aware observation, and every few epochs run the real evaluation (normal start, deterministic).
Usage: python bc_exact.py --demos states/demo_level1_full.json --out runs/v5"""
import argparse, os, pickle, time, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from pv.utils import make_flat

p = argparse.ArgumentParser()
p.add_argument("--demos", required=True); p.add_argument("--demo", type=int, default=0)
p.add_argument("--out", default="runs/v5"); p.add_argument("--max_epochs", type=int, default=400)
p.add_argument("--check_every", type=int, default=10); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)

if a.demos.endswith(".json"):
    import json; d = json.load(open(a.demos)); traj = d if isinstance(d[0], int) else d[a.demo]
else:
    traj = pickle.load(open(a.demos, "rb"))[a.demo]
env = make_flat("Level1", noop_max=0, time_feature=True)
O, A = [], []
o, info = env.reset(seed=0)
for act in traj:
    O.append(np.transpose(o, (2, 0, 1))); A.append(act)
    o, r, te, tr, info = env.step(act)
    if te or tr: break
assert info["cleared"], "demo does not clear the level in the training env"
O, A = torch.tensor(np.stack(O)), torch.tensor(A)
print(f"dataset: {len(A)} steps, demo clears level: {info['cleared']}")

model = PPO("CnnPolicy", VecTransposeImage(DummyVecEnv([lambda: env])), device="cpu", seed=a.seed)
pol = model.policy; opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

def evaluate(n_steps=6000):
    """The task's evaluation: normal start, deterministic actions."""
    o, info = env.reset(seed=0); t = 0; done = False
    while not done and t < n_steps:
        act, _ = model.predict(np.transpose(o, (2, 0, 1))[None], deterministic=True)
        o, r, te, tr, info = env.step(int(act[0])); done = te or tr; t += 1
    return info, t

log = []; t0 = time.time()
for ep in range(1, a.max_epochs + 1):
    idx = torch.randperm(len(A)); tot = 0.0
    for i in range(0, len(A), 128):
        b = idx[i:i + 128]
        _, logp, _ = pol.evaluate_actions(O[b].float(), A[b])
        loss = -logp.mean(); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    sched.step()
    if ep % a.check_every == 0:
        with torch.no_grad():
            pred = torch.cat([pol.get_distribution(O[i:i + 512].float()).distribution.probs.argmax(1) for i in range(0, len(A), 512)])
        wrong = (pred != A).nonzero().flatten().tolist()
        info, steps = evaluate()
        log.append(dict(epoch=ep, loss=tot / len(A), acc=1 - len(wrong) / len(A), first_wrong=wrong[:1], cleared=bool(info["cleared"]),
                        section=int(info["section"]), progress=int(info["progress"]), steps=steps, minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
        model.save(f"{a.out}/bc_epoch{ep}")
        if info["cleared"]:
            model.save(f"{a.out}/final"); print("CLEARS LEVEL 1 from the normal start with deterministic actions"); break
pickle.dump(log, open(f"{a.out}/bc_log.pkl", "wb"))
