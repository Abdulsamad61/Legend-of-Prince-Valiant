"""Level 2 ("The Forest") environment - first-person shooting gallery, trained with pure RL.

RAM map (found by probing, Level 2):
  86 / 88   crosshair x (+ scroll RAM 24, mod 256) / y; D-pad moves it ~2 px per frame
  252       energy: 255 full, -24 per hit; when it runs out a life is lost and it refills
  24 / 25   distance walked in this life, low / high byte (grows while the screen scrolls; resets on life loss).
            The forest is ~512 units long; the screen stops scrolling while robbers are left alive.
  244-246   score (BCD); a robber is worth 75
  247       lives;  71 = level index (1 = Level 2)
  647-649   timer (1:30 per life)
  0x700     sprite table (y, tile, attr, x) x 64: crosshair = tiles 1-4, crossbow = y >= 167, robbers/items = the rest
Buttons: A = fire the crossbow (edge-triggered: one shot per press); B = take cover (tapping it cuts damage ~4x); D-pad = aim.
"""
import numpy as np, gymnasium as gym, cv2, stable_retro as retro
from gymnasium.wrappers import FrameStackObservation

GAME = "LegendOfPrinceValiant-Nes-v0"


def install_level2_state():
    """Copy states/Level2.state (save state at the start of Level 2) into the stable-retro game folder so that
    retro.make(GAME, state="Level2") finds it. Needs the ROM to be imported first (python -m stable_retro.import roms/)."""
    import os, shutil
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "states", "Level2.state")
    try: dst = os.path.join(os.path.dirname(retro.data.get_file_path(GAME, "rom.sha")), "Level2.state")
    except Exception: return
    if os.path.exists(src) and not os.path.exists(dst): shutil.copy(src, dst)


install_level2_state()
DIRS = [[], ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"], ["UP", "LEFT"], ["UP", "RIGHT"], ["DOWN", "LEFT"], ["DOWN", "RIGHT"]]
COMBOS = [d for d in DIRS] + [d + ["A"] for d in DIRS] + [["B"]]   # 19 actions: aim (9) x fire (2) + take cover (B)


def cross_xy(r):
    """Crosshair position in sprite coordinates. RAM 86 is relative to the scroll: screen x = (RAM 86 - RAM 24) mod 256."""
    return (int(r[86]) - int(r[24])) % 256, int(r[88])


def bcd(r):
    return int("%02x%02x%02x" % (r[244], r[245], r[246]))       # decimal score from BCD bytes


class Level2(gym.Wrapper):
    """Frame skip 4, grayscale 84x84, 18 actions, shaped reward. Weights in W.
    episodic_life=True (training): an episode ends when a life is lost (progress resets anyway).
    Evaluation uses episodic_life=False and plays with all lives."""
    W = dict(kill=3.0, distance=0.25, energy=0.01, death=-3.0, clear=50.0, time=-0.001, aim=1.0)

    def __init__(self, env, skip=4, max_steps=6000, episodic_life=True, noop_max=0, w=None):
        super().__init__(env)
        self.W = {**self.W, **(w or {})}; self.skip, self.max_steps = skip, max_steps
        self.episodic_life, self.noop_max = episodic_life, noop_max
        names = env.unwrapped.buttons
        self.acts = np.zeros((len(COMBOS), len(names)), np.int8)
        for i, c in enumerate(COMBOS):
            for b in c: self.acts[i, names.index(b)] = 1
        self.action_space = gym.spaces.Discrete(len(COMBOS))
        self.aim = self.acts.copy(); self.aim[:, names.index("A")] = 0; self.aim[:, names.index("B")] = 0   # A / B are pressed on the first frame only
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 2), np.uint8)

    def _obs(self, o):
        """Two channels: (0) playfield without HUD, contrast-equalised, crosshair drawn as a bright marker;
        (1) 2x zoom of the 64x64 region around the crosshair (is a robber under the aim?).
        The crosshair is visible on the real screen but disappears when the frame is shrunk to 84x84."""
        g = cv2.equalizeHist(cv2.cvtColor(o[:184], cv2.COLOR_RGB2GRAY))            # rows 184+ = HUD
        r = self.env.unwrapped.get_ram(); cx, cy = cross_xy(r); cx -= 8; cy -= 3          # crosshair centre in screen pixels (sprite x is in 256-px NES coords; screen is cropped by 8)
        full = cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA)
        px, py = int(np.clip(cx * 84 / 240, 1, 82)), int(np.clip(cy * 84 / 184, 1, 82))
        full[py, max(px - 2, 0):px + 3] = 255; full[max(py - 2, 0):py + 3, px] = 255
        pad = cv2.copyMakeBorder(g, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=0)
        cx, cy = min(max(cx, 0), 290), min(max(cy, 0), 200)                       # never an empty zoom window
        crop = cv2.resize(pad[cy:cy + 64, cx:cx + 64], (84, 84), interpolation=cv2.INTER_NEAREST)
        return np.stack([full, crop], -1)

    def _read(self):
        r = self.env.unwrapped.get_ram()
        return dict(score=bcd(r), energy=int(r[252]), dist=int(r[24]) + 256 * int(r[25]), lives=int(r[247]), level=int(r[71]))

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        o, *_ = self.env.step(self.acts[0])
        for _ in range(self.np_random.integers(self.noop_max + 1) if self.noop_max else 0):
            for _ in range(self.skip): o, *_ = self.env.step(self.acts[0])
        self.s = self._read(); self.s["dist"] = 0; self.best_dist = 0;   # RAM 24/25 still hold Level 1's scroll for a few frames
        self.t = 0; self.kills = 0; self.ep_dist = 0; self.phi = self._phi()
        return self._obs(o), self._info(False)

    def _phi(self):
        """Potential = -(distance from crosshair to the nearest robber / item) / 100 px; None if nothing to aim at.
        Targets = sprites above the crossbow (sprite y < 158) that are not the crosshair (tiles 1-4).
        Potential-based shaping (Ng et al., 1999) leaves the optimal policy unchanged."""
        r = self.env.unwrapped.get_ram(); oam = r[0x700:0x800].reshape(64, 4).astype(int)
        t = [(x + 4, y + 5) for y, tile, _, x in oam if y < 158 and tile not in (1, 2, 3, 4)]
        if not t: return None
        cx, cy = cross_xy(r); cy -= 3
        return -min(np.hypot(x - cx, y - cy) for x, y in t) / 100.0

    def _info(self, cleared):
        return dict(lives=self.s["lives"], energy=self.s["energy"], distance=getattr(self, 'ep_dist', 0), kills=self.kills,
                    score=self.s["score"], cleared=cleared, level=self.s["level"])

    def step(self, a):
        rew, term, trunc = 0.0, False, False
        for i in range(self.skip):
            # fire is edge-triggered: holding A never shoots, so A is pressed only on the first frame of each step
            o, _, term, trunc, _ = self.env.step(self.acts[a] if i == 0 else self.aim[a])
            if term or trunc: break
        self.t += 1; n = self._read(); p = self.s
        if n["score"] > p["score"]:
            k = max(1, round((n["score"] - p["score"]) / 75)); self.kills += k; rew += self.W["kill"] * k
        if n["lives"] < p["lives"]:
            rew += self.W["death"]; self.best_dist = 0
            if self.episodic_life: term = True
        else:
            rew += self.W["energy"] * (n["energy"] - p["energy"])                  # hit < 0, "E" icon > 0
            if n["dist"] > self.best_dist:
                rew += self.W["distance"] * (n["dist"] - self.best_dist); self.best_dist = n["dist"]; self.ep_dist = max(self.ep_dist, self.best_dist)
        cleared = n["level"] > 1
        if cleared: rew += self.W["clear"]; term = True
        phi = self._phi()                                                      # aiming hint (potential-based shaping)
        if phi is not None and self.phi is not None: rew += self.W["aim"] * (phi - self.phi)
        self.phi = phi
        if n["lives"] <= 0: term = True
        rew += self.W["time"]
        if self.t >= self.max_steps: trunc = True
        self.s = n
        return self._obs(o), rew, term, trunc, self._info(cleared)


class Flat(gym.ObservationWrapper):
    """(4 frames, 84, 84, 2) -> (84, 84, 8): 4 stacked playfield frames + 4 stacked crosshair zooms."""
    def __init__(s, e):
        super().__init__(e); s.observation_space = gym.spaces.Box(0, 255, (84, 84, 8), np.uint8)
    def observation(s, o): return np.concatenate([np.moveaxis(o[..., 0], 0, -1), np.moveaxis(o[..., 1], 0, -1)], -1)


def make_l2(episodic_life=True, noop_max=0, **kw):
    e = retro.make(GAME, state="Level2", render_mode=None)
    return Flat(FrameStackObservation(Level2(e, episodic_life=episodic_life, noop_max=noop_max, **kw), 4))
