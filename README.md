# Legend of Prince Valiant (NES) – RL agent

PPO experiments (v1–v4) and the final **search-then-imitate** agent (v5) that clears **Level 1** from the game's
normal start in **10/10** evaluation episodes with deterministic actions, and the same technique for **Level 2** (The Forest) and
**Level 3** (The Viking), each cleared in **10/10** episodes from the start of the level.

## Results

Fill in from your own runs (`eval.py` prints the completion rate; `eval_checkpoints.py` + `plot_curves.py` give the curves).

| Model | Method | Level 1 completion (normal start, 10 episodes) |
|---|---|---|
| PPO baseline | PPO from pixels, shaped reward | _your result_ |
| BC + PPO (optional) | demonstrations as warm start + curriculum | _your result_ |
| Final agent (v5) | exact imitation of the search-found route + time channel | _your result_ |

| Level 2 model | Method | Level 2 completion (Level 2 start, 10 episodes) |
|---|---|---|
| PPO baseline (`runs/l2_ppo`) | PPO from pixels, shaped reward + aim shaping | _your result_ |
| Final Level 2 agent (`runs/l2_v1`) | exact imitation of the scripted aim-assist route + time channel | _your result_ |

| Level 3 model | Method | Level 3 completion (Level 3 start, 10 episodes) |
|---|---|---|
| Final Level 3 agent (`runs/l3_v1`) | exact imitation of the search-found route + 14-bit time channel | _your result_ |

Reference values from the development run (2-core cloud CPU): PPO-only agents did not clear Level 1 (best: section 2 in
3/10); the final agent cleared 10/10 deterministic, 10/10 at T=0.1, 7/10 at T=0.5, and 0/10 with a 0–30 no-op shifted
start (it reproduces one route and does not generalise). Level 2: PPO 0/10 (≈5 robbers, ≈150 distance per life after 1.5M
steps); final Level 2 agent 10/10 deterministic, 10/10 T=0.1, 10/10 T=0.5, 0/10 with a shifted start. Level 3: 10/10
deterministic, 10/10 T=0.1, 10/10 T=0.5, 0/10 with a shifted start (12,533-step route, all lives kept).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # on Google Colab: pip install -r requirements-colab.txt
mkdir roms && cp /path/to/your/Legend_of_Prince_Valiant.nes roms/   # your own legally owned ROM (not included)
python -m stable_retro.import roms/
```
Or open `colab_run.ipynb` in Google Colab and run the cells top to bottom.

## Commands

```bash
# Evaluate the final agent exactly as the task defines it (normal start, deterministic, 10 episodes)
python eval.py --model runs/v5/final.zip --episodes 10
python eval.py --model runs/v5/final.zip --episodes 10 --temperature 0.1   # low temperature
python eval.py --model runs/v5/final.zip --episodes 10 --noop 30           # robustness test

# Re-train the final agent from the demonstration (≈2–5 min on CPU)
python bc_exact.py --demos optional_demo/demo_level1_full.json --out runs/v5 --seed 0

# Progress video (early / middle / final checkpoint + full uncut Level 1 run)
python make_video.py --run runs/v5 --out clips/progress_video.mp4
python eval_checkpoints.py --run runs/v5 --episodes 1 && python plot_curves.py --runs runs/ppo_baseline runs/v5

# PPO training (v2/v3/v4 style); TensorBoard logs in runs/<name>/tb
python train.py --steps 1000000 --envs 8 --seed 0 --out runs/ppo_baseline
python bc.py --demos optional_demo/demo_level1_full.json --out runs/bc --epochs 60
python train.py --steps 1500000 --envs 8 --seed 0 --curriculum 0.7 --starts runs/bc/demo_starts.pkl \
                --init runs/bc/bc_policy.pt --ent 0.01 --out runs/bc_ppo
tensorboard --logdir runs

# Route search that produced the demonstration (savestate Go-Explore; hours; resumes its archive - re-run to continue)
python tools/go_explore.py 41 3600 40                      # until the section-2 vine chain, and again for section 3 + boss
python tools/go_explore.py 41 3600 40 2,850,1200 20        # vine chain: focus region, weight kills
# on the boss kill the route is written to runs/search/level1_clear_*.json
```

### Level 2

```bash
python train_l2.py --steps 1500000 --envs 8 --seed 0 --out runs/l2_ppo         # PPO baseline (pure RL)
python tools/aim_route_l2.py                                                   # demonstration route (~1 min) -> runs/search_l2/level2_clear_aim.json
python bc_exact_l2.py --demo runs/search_l2/level2_clear_aim.json --out runs/l2_v1 --seed 0
python eval.py --model runs/l2_v1/final.zip --level 2 --episodes 10            # deterministic; add --temperature 0.1 / --noop 30
python report/plot_level2.py                                                   # Level 2 curves -> report/fig_level2.png
python make_video_l2.py --run runs/l2_v1 --out clips/progress_video_level2.mp4
```

### Level 3

```bash
python bc_exact_l3.py --demo optional_demo/demo_level3_route.json --out runs/l3_v1 --seed 0     # ~15 min
python eval.py --model runs/l3_v1/final.zip --level 3 --episodes 10
python report/plot_level3.py; python make_video_l3.py --run runs/l3_v1 --out clips/progress_video_level3.mp4
# route search (hours): whole-level search, then the room-by-room rebuild against the 10:00 timer
python tools/go_explore_l3.py 1 14400
python tools/segments_l3.py optional_demo/level3_room_plan.json runs/search_l3/seg 480 optional_demo/level3_first_search_route.json
```

## Repository layout

| Path | Content |
|---|---|
| `pv/env.py` | stable-retro env: frame skip 4, grayscale 84×84, 10 actions, shaped reward, `cleared` detection, curriculum starts, RAM map |
| `pv/utils.py` | frame-stack flattening, time-aware observation channel |
| `train.py` | PPO training (SB3), checkpoints every 5 % |
| `bc.py`, `bc_exact.py` | behaviour cloning (warm start for PPO) / exact imitation (v5) |
| `eval.py` | task evaluation: normal start, deterministic or temperature sampling, optional no-op robustness |
| `make_video.py`, `record_video.py` | progress video / single clips |
| `tools/go_explore.py` | savestate route search (demonstration generator, not RL) |
| `eval_checkpoints.py`, `plot_curves.py` | completion rate per checkpoint; reward / length / completion curves |
| `pv/env_l2.py` | Level 2 env: 19 actions (aim × fire + cover), two-channel observation (playfield + crosshair zoom), shaped reward, RAM map |
| `train_l2.py`, `bc_exact_l2.py`, `make_video_l2.py` | Level 2 PPO / exact imitation / progress video |
| `tools/aim_route_l2.py` | scripted aim-assist route for Level 2 (demonstration generator, not RL); `tools/go_explore_l2.py` = earlier savestate search |
| `states/Level2.state` | save state at the start of Level 2 (training and evaluation start for Level 2) |
| `optional_demo/demo_level2_aim.json` | the Level 2 route (2,704 actions) |
| `pv/env_l3.py` | Level 3 env: 12 actions, 84×84 frames, exploration reward, RAM map |
| `bc_exact_l3.py`, `make_video_l3.py` | Level 3 exact imitation / progress video |
| `tools/go_explore_l3.py`, `tools/segments_l3.py` | Level 3 route search (whole level) and room-by-room rebuild; `graft_l3.py`, `weights_l3.py` = helpers used during the search |
| `states/Level3.state`, `optional_demo/demo_level3_route.json` | Level 3 start state; the Level 3 route (12,533 actions) |
| `colab_run.ipynb` | the whole pipeline on Google Colab, Levels 1–3, top to bottom |
| `make_video_all.py` | one progress video for Levels 1–3 (early / middle / final checkpoint + full runs) |
| `optional_demo/demo_level1_full.json` | route found by the search (3,611 actions) – use it instead of re-running the search |
| `runs/` (created by training) | models, checkpoints, TensorBoard logs, eval results |

## Environment details

* Actions (10): noop, RIGHT, LEFT, DOWN (duck), B (shoot), RIGHT+B, A (jump), RIGHT+A, LEFT+A, UP.
* Observation: 4 stacked 84×84 grayscale frames (v5 adds one channel encoding the step counter in binary).
* Reward (`Shaped.W`): +0.1 per pixel of **new** max progress, +20 per new section, +0.01 per score point,
  +1 per boss hit, +50 level clear, −5 per life lost, −0.001 per step.
* RAM map (found by probing): x = 86/87, scroll = 24/25, y = 88, jump timer = 94, section = 75, lives = 247,
  level = 71, boss health = 183, timer = 647–649.
* Level 2: actions = 9 aim directions × fire/no fire + B (take cover); A/B pressed on the first of the 4 frames (fire is
  edge-triggered). Reward: +3 per robber, +0.25 per unit of new distance, ±0.01 per energy point, −3 per life, +50 clear,
  −0.001 per step, potential-based aim shaping. RAM: crosshair x = (RAM[86] − RAM[24]) mod 256, y = 88; energy = 252;
  distance = 24/25; sprite table 0x700.
* Seeds: `--seed 0` for every run; the emulator itself is deterministic.

## Compute

Report the machine you actually trained on (`colab_run.ipynb` section 1 writes `hardware.txt`).
Reference throughput on a 2-core CPU: PPO ≈ 220–260 env steps/s with 8 envs; exact imitation ≈ 3 min; search: hours.
