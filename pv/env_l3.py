"""Level 3 ("The Viking") environment - side-view cave made of rooms (wooden towers, swinging ropes, spike pits, tokens).

RAM map (found by probing, Level 3):
  75        room id (the cave is a set of connected rooms; 30 = intro walk before the fall into the cave)
  86 / 87   hero x in the room (low / high byte);  88 = hero y
  24 / 25   horizontal camera scroll
  250       tokens collected (shown as TOKENS on the HUD)
  252       energy (255 full); 0 = dying - the life is taken ~150 frames later
  247       lives;  71 = level index (2 = Level 3);  647-649 = timer (10:00 for the level)
12 actions (Level 1's 10 + DOWN+B crouch-shoot, LEFT+B); frame skip 4; 84x84 grayscale; stack of 4.
Starts from states/Level3.state = first frame of Level 3 (timer running), reached by playing Levels 1-2.
"""
import os, shutil, numpy as np, gymnasium as gym, cv2, stable_retro as retro
from gymnasium.wrappers import FrameStackObservation

GAME = "LegendOfPrinceValiant-Nes-v0"
# Level 1's 10 actions + crouch-and-shoot (needed to hit the gate disks) + shoot left
COMBOS = [[], ["RIGHT"], ["LEFT"], ["DOWN"], ["B"], ["RIGHT", "B"], ["A"], ["RIGHT", "A"], ["LEFT", "A"], ["UP"], ["DOWN", "B"], ["LEFT", "B"]]


def install_level3_state():
    """Copy states/Level3.state into the stable-retro game folder so retro.make(GAME, state="Level3") finds it."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "states", "Level3.state")
    try: dst = os.path.join(os.path.dirname(retro.data.get_file_path(GAME, "rom.sha")), "Level3.state")
    except Exception: return
    if os.path.exists(src) and not os.path.exists(dst): shutil.copy(src, dst)


install_level3_state()


class Level3(gym.Wrapper):
    """Frame skip 4, grayscale 84x84, 10 actions. Reward: exploration of the cave (count-free: +1 for each new
    room, +0.05 for each new 16x16 cell of a room), +2 per token, +50 level clear, -5 per life, -0.001 per step."""
    W = dict(room=1.0, cell=0.05, token=2.0, death=-5.0, clear=50.0, time=-0.001)

    def __init__(self, env, skip=4, max_steps=15000, episodic_life=True, noop_max=0, w=None):
        super().__init__(env)
        self.W = {**self.W, **(w or {})}; self.skip, self.max_steps, self.episodic_life, self.noop_max = skip, max_steps, episodic_life, noop_max
        names = env.unwrapped.buttons
        self.acts = np.zeros((len(COMBOS), len(names)), np.int8)
        for i, c in enumerate(COMBOS):
            for b in c: self.acts[i, names.index(b)] = 1
        self.action_space = gym.spaces.Discrete(len(COMBOS))
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)

    def _obs(self, o):
        return cv2.resize(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)[..., None]

    def _read(self):
        r = self.env.unwrapped.get_ram().astype(int)
        return dict(room=r[75], x=r[86] + 256 * r[87], y=r[88], tokens=r[250], energy=r[252], lives=r[247], level=r[71])

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        o, *_ = self.env.step(self.acts[0])
        for _ in range(self.np_random.integers(self.noop_max + 1) if self.noop_max else 0):
            for _ in range(self.skip): o, *_ = self.env.step(self.acts[0])
        self.s = self._read(); self.t = 0; self.rooms = {self.s["room"]}; self.cells = set()
        return self._obs(o), self._info(False)

    def _info(self, cleared):
        return dict(**self.s, rooms=len(self.rooms), cleared=cleared)

    def step(self, a):
        rew, term, trunc = 0.0, False, False
        for _ in range(self.skip):
            o, _, term, trunc, _ = self.env.step(self.acts[a])
            if term or trunc: break
        self.t += 1; n = self._read(); p = self.s
        if n["lives"] < p["lives"]:
            rew += self.W["death"]
            if self.episodic_life: term = True
        elif n["energy"] > 0:
            if n["room"] not in self.rooms: self.rooms.add(n["room"]); rew += self.W["room"]
            c = (n["room"], n["x"] // 16, n["y"] // 16)
            if c not in self.cells: self.cells.add(c); rew += self.W["cell"]
            rew += self.W["token"] * max(0, n["tokens"] - p["tokens"])
        cleared = n["level"] > 2
        if cleared: rew += self.W["clear"]; term = True
        if n["lives"] <= 0: term = True
        rew += self.W["time"]
        if self.t >= self.max_steps: trunc = True
        self.s = n
        return self._obs(o), rew, term, trunc, self._info(cleared)


class Flat(gym.ObservationWrapper):
    def __init__(s, e):
        super().__init__(e); s.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
    def observation(s, o): return np.moveaxis(o[..., 0], 0, -1)


def make_l3(episodic_life=True, noop_max=0, **kw):
    e = retro.make(GAME, state="Level3", render_mode=None)
    return Flat(FrameStackObservation(Level3(e, episodic_life=episodic_life, noop_max=noop_max, **kw), 4))
