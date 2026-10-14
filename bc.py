"""Behaviour cloning from search demonstrations, then hand the weights to PPO (train.py --init).

A demo is a list of action indices (pv.env.COMBOS) that starts from the exact Shaped reset (noop_max=0).
We replay each demo in the training env, record (observation, action) pairs, check the replay really
reaches the same place, and fit the PPO CnnPolicy by maximising log pi(a|s).
Also writes evenly spaced emulator savestates along the demos -> dense curriculum for PPO."""
import argparse, pickle, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from pv.utils import make_flat

p = argparse.ArgumentParser()
p.add_argument("--demos", required=True, help="pickle: list of action lists")
p.add_argument("--out", default="runs/bc"); p.add_argument("--epochs", type=int, default=30)
p.add_argument("--every", type=int, default=20, help="savestate spacing (agent steps) for the curriculum")
p.add_argument("--balance", action="store_true", help="class-balanced loss (off: demos contain random filler actions)")
a = p.parse_args()

if a.demos.endswith(".json"):
    import json; _d = json.load(open(a.demos)); demos = [_d] if isinstance(_d[0], int) else _d
else:
    demos = pickle.load(open(a.demos, "rb"))
env = make_flat("Level1", noop_max=0)
O, A, starts = [], [], []
for d, traj in enumerate(demos):
    o, info = env.reset(seed=0); lives0 = info["lives"]
    shared = max([next((i for i, (u, v) in enumerate(zip(traj, prev)) if u != v), min(len(traj), len(prev))) for prev in demos[:d]] or [0])
    for t, act in enumerate(traj):
        if t < shared: o, r, te, tr, info = env.step(act); continue      # identical prefix of an earlier demo: already recorded
        if t % a.every == 0: starts.append(env.unwrapped.em.get_state())
        O.append(np.transpose(o, (2, 0, 1))); A.append(act)
        o, r, te, tr, info = env.step(act)
        if te or tr: break
    print(f"demo {d}: {len(traj)} steps (new samples from step {shared}) -> section {info['section']}, cleared {info.get('cleared')}, lives {info['lives']}/{lives0}")
O, A = np.stack(O), np.array(A)
print("dataset", O.shape, "action counts", np.bincount(A, minlength=env.action_space.n))

env.close()                                      # one emulator per process
venv = VecTransposeImage(DummyVecEnv([lambda: make_flat("Level1")]))
model = PPO("CnnPolicy", venv, device="cpu", seed=0)
pol = model.policy; opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
w = torch.tensor(1.0 / np.maximum(np.bincount(A, minlength=env.action_space.n), 1), dtype=torch.float32)
w = w / w.sum() * len(w) if a.balance else torch.ones_like(w)
obs_t, act_t = torch.tensor(O), torch.tensor(A)
idx = np.arange(len(A)); rng = np.random.default_rng(0)
for ep in range(a.epochs):
    rng.shuffle(idx); tot, acc = 0.0, 0.0
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        _, logp, ent = pol.evaluate_actions(obs_t[b].float(), act_t[b])
        loss = -(w[act_t[b]] * logp).mean() - 0.001 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    with torch.no_grad():
        pred = torch.cat([pol.get_distribution(obs_t[i:i + 256].float()).distribution.probs.argmax(1) for i in range(0, len(A), 256)])
        acc = (pred == act_t).float().mean().item()
    if ep % 5 == 0 or ep == a.epochs - 1: print(f"epoch {ep:2d} loss {tot / len(A):.3f} train acc {acc:.2f}")
import os; os.makedirs(a.out, exist_ok=True)
torch.save(pol.state_dict(), f"{a.out}/bc_policy.pt")
pickle.dump([dict(name=f"demo_{i}", state=s) for i, s in enumerate(starts)], open(f"{a.out}/demo_starts.pkl", "wb"))
print("saved", f"{a.out}/bc_policy.pt", "and", len(starts), "curriculum states")
