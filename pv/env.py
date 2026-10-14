"""Prince Valiant env: stable-retro + reward shaping + standard Atari-style wrappers.

RAM map (verified, see report Appendix):
  86/87  player world-x lo/hi (little-endian; lo byte wraps at 256)
  24/25  camera scroll lo/hi
  88     player y (103 = standing on ground, smaller = higher)
  94     jump timer (one jump = 33 frames, ~33 px; holding A longer does NOT jump further)
  75     Marsh section (1-based). x resets to ~24 when a new section starts
  247    lives
  71     level index (0 = Level 1, 1 = Level 2 ...)
  183    Winged Demon health in the boss arena (section 3, x >= 1180); 24 hits; killing blow = +1000 score
  647-649 level timer m:s:s (4:00 for the WHOLE level, timeout = lose a life and restart the level; 'T' item = +60 s)
Buttons: A = jump (also grabs / releases the vine), B = shoot (not while crouching), DOWN = duck.
"""
import pickle, numpy as np, gymnasium as gym, cv2, stable_retro as retro
from gymnasium.wrappers import FrameStackObservation

GAME = "LegendOfPrinceValiant-Nes-v0"
RAM_X, RAM_SCROLL, RAM_Y, RAM_SECTION, RAM_LIVES = 86, 24, 88, 75, 247
RAM_LEVEL, RAM_BOSS_HP = 71, 183
SECTION_LEN = 2048          # > widest section (~1785 px), so progress is monotone across sections
# Action set from the walkthrough: walk, shoot ("shoot the moment you land"), duck ("duck and wait for the arrow"),
# jump in place (vine grab/release, moving stone timing), jump forward (trap doors), jump back ("watch your back").
COMBOS = [[], ["RIGHT"], ["LEFT"], ["DOWN"], ["B"], ["RIGHT", "B"], ["A"], ["RIGHT", "A"], ["LEFT", "A"], ["UP"]]


class Shaped(gym.Wrapper):
    """Frame skip, reduced actions, grayscale 84x84, shaped reward. All weights in W.
    starts: optional list of emulator savestates (curriculum); p_start = chance an episode begins from one.
    noop_max: random 0..noop_max no-op steps after reset (makes deterministic eval episodes differ)."""
    W = dict(progress=0.1, section=20.0, score=0.01, death=-5.0, time=-0.001, clear=50.0, boss_hit=1.0)

    def __init__(self, env, skip=4, max_steps=5000, w=None, starts=None, p_start=0.0, noop_max=0):
        super().__init__(env)
        self.W = {**self.W, **(w or {})}
        self.skip, self.max_steps = skip, max_steps
        self.starts, self.p_start, self.noop_max = starts or [], p_start, noop_max
        names = env.unwrapped.buttons
        self.acts = np.zeros((len(COMBOS), len(names)), np.int8)
        for i, c in enumerate(COMBOS):
            for b in c: self.acts[i, names.index(b)] = 1
        self.action_space = gym.spaces.Discrete(len(COMBOS))
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)

    def _obs(self, o):
        return cv2.resize(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)[..., None]

    def _ram(self):
        return self.env.unwrapped.get_ram()

    def _section(self):
        return max(int(self._ram()[RAM_SECTION]), 1)

    def _prog(self):
        r = self._ram()
        return (self._section() - 1) * SECTION_LEN + int(r[RAM_X]) + 256 * int(r[RAM_X + 1])

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        rng = self.np_random
        if self.starts and rng.random() < self.p_start:          # curriculum: start right before a hard spot
            self.env.unwrapped.em.set_state(self.starts[rng.integers(len(self.starts))])
        o, _, _, _, info = self.env.step(self.acts[0])           # reset() omits RAM vars; read them via a no-op step
        for _ in range(rng.integers(self.noop_max + 1) if self.noop_max else 0):
            for _ in range(self.skip): o, _, _, _, info = self.env.step(self.acts[0])
        self.t, self.best = 0, self._prog(); self.last, self.sec = self.best, self._section()
        self.lives, self.score = info["lives"], info["score"]; self.boss_hp = int(self._ram()[RAM_BOSS_HP])
        info["progress"], info["section"], info["cleared"] = self.best, self.sec, False
        return self._obs(o), info

    def step(self, a):
        rew, term, trunc = 0.0, False, False
        for _ in range(self.skip):
            o, _, term, trunc, info = self.env.step(self.acts[a])
            if term or trunc: break
        self.t += 1
        sec = self._section()
        if sec > self.sec:                       # reached the next Marsh section
            rew += self.W["section"]; self.sec = sec; self.last = self.best = self._prog()  # bonus only, no x-jump reward
        p, last = self._prog(), self.last
        self.last = p
        if p - last > 64: p = last               # ignore glitch values (e.g. 0xFFFF during death animation)
        if p > self.best:                        # pay only for NEW max progress -> no pacing/standing hacks
            rew += self.W["progress"] * (p - self.best); self.best = p
        rew += self.W["score"] * (info["score"] - self.score); self.score_prev = self.score; self.score = info["score"]
        rew += self.W["time"]
        if info["lives"] < self.lives:
            rew += self.W["death"]; self.lives = info["lives"]
        # boss: each hit lowers RAM 183 by one; the killing blow empties it and scores +1000
        ram = self._ram(); hp = int(ram[RAM_BOSS_HP]); in_arena = self.sec == 3 and (p % SECTION_LEN) >= 1180
        cleared = int(ram[RAM_LEVEL]) > 0
        if in_arena and hp == self.boss_hp - 1: rew += self.W["boss_hit"]
        if in_arena and info["score"] - self.score_prev >= 1000: cleared = True   # killing blow on the Winged Demon: +1000 (only he is worth that much)
        self.boss_hp = hp; self.score_prev = info["score"]
        if cleared: rew += self.W["clear"]; term = True
        if info["lives"] <= 0: term = True
        if self.t >= self.max_steps: trunc = True
        info["progress"], info["section"], info["cleared"] = self.best, self.sec, cleared
        return self._obs(o), rew, term, trunc, info


def load_starts(path="states/curriculum.pkl"):
    return [s["state"] for s in pickle.load(open(path, "rb"))]


def make_env(state="Level1", stack=4, **kw):
    e = retro.make(GAME, state=state, render_mode=None)
    return FrameStackObservation(Shaped(e, **kw), stack)
