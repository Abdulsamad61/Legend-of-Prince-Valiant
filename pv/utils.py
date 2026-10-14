import numpy as np, gymnasium as gym
from pv.env import make_env


class Flat(gym.ObservationWrapper):
    """(4, 84, 84, 1) frame stack -> (84, 84, 4) image for the CNN."""
    def __init__(s, e):
        super().__init__(e)
        s.observation_space = gym.spaces.Box(0, 255, (84, 84, e.observation_space.shape[0]), np.uint8)
    def observation(s, o): return np.moveaxis(o[..., 0], 0, -1)


class TimeFeature(gym.Wrapper):
    """Time-aware observation (Pardo et al., 2018): appends one channel that encodes the episode step
    counter in binary (13 horizontal bands of 6 px rows). Needed because timed waits in the demonstration
    produce identical frame stacks that call for different actions; the counter disambiguates them."""
    BITS, BAND = 13, 6
    def __init__(s, e, bits=13):
        super().__init__(e); s.BITS = bits                                   # 13 bits = 8191 steps; Level 3 needs 14 (bands of 6 rows fill 84)
        h, w, c = e.observation_space.shape
        s.observation_space = gym.spaces.Box(0, 255, (h, w, c + 1), np.uint8); s.t = 0
    def _obs(s, o):
        ch = np.zeros(o.shape[:2] + (1,), np.uint8)
        for b in range(s.BITS):
            if (s.t >> b) & 1: ch[b * s.BAND:(b + 1) * s.BAND] = 255
        return np.concatenate([o, ch], -1)
    def reset(s, **kw):
        o, info = s.env.reset(**kw); s.t = 0; return s._obs(o), info
    def step(s, a):
        o, r, te, tr, info = s.env.step(a); s.t += 1; return s._obs(o), r, te, tr, info


def make_flat(state="Level1", time_feature=False, **kw):
    e = Flat(make_env(state, **kw))
    return TimeFeature(e) if time_feature else e
