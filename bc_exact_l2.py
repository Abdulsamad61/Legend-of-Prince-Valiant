"""Level 2: imitation learning until the policy reproduces the demonstration route exactly (same method as bc_exact.py).
The route comes from tools/aim_route_l2.py (scripted aim assist, a demonstration generator). The policy is the PPO
CnnPolicy on the Level 2 observation (4 x playfield + 4 x crosshair zoom) plus the time channel. Every few epochs the
real evaluation runs: Level 2 start state, all lives, deterministic actions.
Usage: python bc_exact_l2.py --demo runs/search_l2/level2_clear_aim.json --out runs/l2_v1"""
import argparse, json, os, pickle, time, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from pv.env_l2 import make_l2
from pv.utils import TimeFeature

p = argparse.ArgumentParser()
p.add_argument("--demo", required=True); p.add_argument("--out", default="runs/l2_v1")
p.add_argument("--max_epochs", type=int, default=400); p.add_argument("--check_every", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count()))

traj = json.load(open(a.demo))
env = TimeFeature(make_l2(episodic_life=False))
O, A = [], []
o, info = env.reset(seed=0)
for act in traj:
    O.append(np.transpose(o, (2, 0, 1))); A.append(act)
    o, r, te, tr, info = env.step(act)
    if te or tr: break
assert info["cleared"], "demo does not clear Level 2 in the training env"
O, A = torch.tensor(np.stack(O)), torch.tensor(A)
print(f"dataset: {len(A)} steps, demo clears Level 2: {info['cleared']}", flush=True)

model = PPO("CnnPolicy", VecTransposeImage(DummyVecEnv([lambda: env])), device="cpu", seed=a.seed)
pol = model.policy; opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)


def evaluate(n_steps=6000):
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
        info, steps = evaluate() if len(wrong) <= 20 else ({"cleared": False, "distance": -1, "kills": -1, "lives": -1}, 0)
        log.append(dict(epoch=ep, loss=tot / len(A), acc=1 - len(wrong) / len(A), first_wrong=wrong[:1], cleared=bool(info["cleared"]),
                        distance=int(info["distance"]), kills=int(info["kills"]), lives=int(info["lives"]), steps=steps,
                        minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
        model.save(f"{a.out}/bc_epoch{ep}")
        if info["cleared"]:
            model.save(f"{a.out}/final"); print("CLEARS LEVEL 2 from its start state with deterministic actions"); break
pickle.dump(log, open(f"{a.out}/bc_log.pkl", "wb"))
