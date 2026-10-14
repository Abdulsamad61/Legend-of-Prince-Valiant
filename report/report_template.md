# Report – RL agent for Legend of Prince Valiant (NES)
Candidate: Abdul Samad · Alphabots GmbH
## 1. Algorithm choice  (PPO; rejected DQN/Rainbow/IMPALA – why)
## 2. Reward design  (table: term | weight | purpose; anti-hacking: pay only new max progress, time penalty, step cap)
## 3. Exploration → exploitation  (ent_coef 0.02, LR linear decay, clip 0.1; point to entropy/ep_rew curves; steps/hours)
## 4. Hurdles  (no position/level in stock integration → RAM probing; x saturates at 200, scroll at 54; headless render; torch/proxy; CPU-only)
## 5. Results  (curves: ep reward, ep length, completion rate; eval 10 eps/level; time + hardware)
## 6. Next steps
## Appendix  (RAM map, hyperparameters, bonus experiments)
