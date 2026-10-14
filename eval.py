"""Evaluation as defined by the task: N episodes from the game's NORMAL start (no save states, no changes to
lives / timer / invincibility). Completion = info["cleared"] (Winged Demon killed / level byte advances).
  --temperature 0   deterministic (greedy) actions            [task: "deterministic or low-temperature"]
  --temperature T   sample from softmax(logits / T)
  --noop K          optional robustness test: 0..K random no-op steps after reset (default 0 = exact normal start)
  --level 2 / 3     Level 2 / 3 agent: starts from that level's save state (start of the level), plays with all lives
Usage: python eval.py --model runs/v5/final.zip --episodes 10
       python eval.py --model runs/l2_v1/final.zip --level 2 --episodes 10"""
import argparse, json, numpy as np, torch; torch.set_num_threads(1)
from stable_baselines3 import PPO
from pv.utils import make_flat
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--state", default="Level1")
p.add_argument("--episodes", type=int, default=10); p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--noop", type=int, default=0); p.add_argument("--level", type=int, default=1); p.add_argument("--max_steps", type=int, default=15000)
a = p.parse_args()
m = PPO.load(a.model.removesuffix(".zip"), device="cpu")
if a.level == 3:
    from pv.env_l3 import make_l3; from pv.utils import TimeFeature
    env = make_l3(episodic_life=False, noop_max=a.noop); env = TimeFeature(env, bits=14) if m.observation_space.shape[0] == 5 else env
elif a.level == 2:
    from pv.env_l2 import make_l2; from pv.utils import TimeFeature
    env = make_l2(episodic_life=False, noop_max=a.noop); env = TimeFeature(env) if m.observation_space.shape[0] == 9 else env
else:
    tf = m.observation_space.shape[0] == 5                     # policy trained with the time-aware channel
    env = make_flat(a.state, noop_max=a.noop, time_feature=tf)
res = []
for ep in range(a.episodes):
    o, _ = env.reset(seed=ep); rng = np.random.default_rng(ep); done, R, t = False, 0.0, 0
    while not done and t < a.max_steps:
        x = torch.as_tensor(np.transpose(o, (2, 0, 1))[None]).float()
        with torch.no_grad(): logits = m.policy.get_distribution(x).distribution.logits[0].numpy()
        if a.temperature <= 0: act = int(logits.argmax())
        else: pr = np.exp((logits - logits.max()) / a.temperature); act = int(rng.choice(len(pr), p=pr / pr.sum()))
        o, r, te, tr, info = env.step(act); R += r; done = te or tr; t += 1
    res.append(dict(ep=ep, cleared=bool(info["cleared"]), reward=round(R, 1), steps=t, lives=int(info["lives"]),
                    **({"distance": int(info["distance"]), "kills": int(info["kills"])} if a.level == 2 else
                       {"rooms": int(info["rooms"]), "room": int(info["room"])} if a.level == 3 else
                       {"section": int(info["section"]), "progress": int(info["progress"])})))
    print(res[-1], flush=True)
rate = np.mean([r["cleared"] for r in res])
print(f"completion rate: {rate:.0%} ({sum(r['cleared'] for r in res)}/{len(res)})  temperature={a.temperature} noop={a.noop}")
json.dump(dict(args=vars(a), completion_rate=rate, episodes=res),
          open(a.model.removesuffix(".zip") + f".eval_L{a.level}_T{a.temperature}_noop{a.noop}.json", "w"), indent=1)
