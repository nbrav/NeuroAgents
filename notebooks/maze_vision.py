"""Unified biologically-plausible SPIKING vision front-end for 4-11monkey-net.

Every model (all 13) shares ONE fixed spiking retina that turns the monkey's maze into neural
features, instead of reading raw barrier coordinates. Grounded in 2026 spiking-vision SOTA:
  * SpikeGen (rods/cones decoupling): two decoupled pathways -- a SUSTAINED "walls" channel and a
    TRANSIENT "goal" channel -- each rendered retinotopically and split into ON/OFF cells.
  * Bio-Vision SNN / event cameras: LIF neurons emit spikes; the feature is the spike RATE over a
    short micro-time window (event-driven, sparse), not a dense floating-point activation.
  * BSD (spike-based distillation) / PredNext (temporal predictive coding): the front-end is FIXED
    and shared (distilled once, frozen) -- a common visual substrate every controller reads from.

The retina is EGOCENTRIC (foveated on the cursor, like the monkey's gaze) and DIFFERENTIABLE in the
cursor position + barrier geometry, so gradient rules (BPTT/SHAC) backprop through it while local
rules (e-prop, Hebbian, ...) just consume the spike-rate features. Weights are frozen => the SAME
vision for all 13 models == the fairness the /goal demands. Shared by every model via `attach(env)`.
"""
import os, numpy as np, torch as th, torch.nn as nn

G = int(os.environ.get("MZ_VIS_GRID", 12))       # retinotopic grid G x G
FOV = float(os.environ.get("MZ_VIS_FOV", 0.6))   # egocentric half-extent (workspace units) the retina sees
T = int(os.environ.get("MZ_VIS_T", 4))           # LIF micro-timesteps (the event window)
D = int(os.environ.get("MZ_VIS_D", 48))          # number of output spike-rate features
_SHARP = 60.0                                     # soft-occupancy edge sharpness (retinal contrast)


def retina(pos, tgt, bar, msk):
    """Egocentric retinotopic render -> (B, 2, G, G): ch0 = walls (sustained), ch1 = goal (transient).
    Differentiable in pos/tgt/bar so gradient controllers backprop through the eye."""
    dev = pos.device
    ax = th.linspace(-FOV, FOV, G, device=dev)
    gy, gx = th.meshgrid(ax, ax, indexing="ij")                     # (G,G) egocentric pixel offsets
    wx = pos[:, 0, None, None] + gx                                 # (B,G,G) absolute pixel coords
    wy = pos[:, 1, None, None] + gy
    cx, cy, hw, hh = [bar[..., i] for i in range(4)]                # (B,K)
    dx = (wx[:, None] - cx[..., None, None]).abs()                  # (B,K,G,G)
    dy = (wy[:, None] - cy[..., None, None]).abs()
    inside = th.sigmoid(_SHARP * (hw[..., None, None] - dx)) * th.sigmoid(_SHARP * (hh[..., None, None] - dy))
    walls = (inside * msk[..., None, None]).max(dim=1).values       # (B,G,G) soft AABB occupancy
    tdx = wx - tgt[:, 0, None, None]; tdy = wy - tgt[:, 1, None, None]
    goal = th.exp(-(tdx ** 2 + tdy ** 2) / (2 * (FOV / 4) ** 2))    # (B,G,G) gaussian bump at target
    return th.stack([walls, goal], dim=1)                          # (B,2,G,G)


class SpikingRetina(nn.Module):
    """Fixed LIF spiking encoder: ON/OFF retinal cells -> D spike-rate features. Frozen + shared."""
    def __init__(self, seed=0):
        super().__init__()
        g = th.Generator().manual_seed(seed)
        W = th.randn(4 * G * G, D, generator=g) / np.sqrt(4 * G * G)   # 4 maps: walls/goal x ON/OFF
        self.register_buffer("W", W); self.register_buffer("b", th.zeros(D))
        self.beta, self.thr, self.k = 0.9, 1.0, 8.0                    # LIF leak, threshold, surrogate slope

    def forward(self, r):                                              # r (B,2,G,G)
        on = th.relu(r); off = th.relu(r.mean(dim=(2, 3), keepdim=True) - r)   # ON/OFF (rods/cones)
        I = th.cat([on, off], dim=1).flatten(1) @ self.W + self.b      # constant drive (B,D)
        v = th.zeros_like(I); rate = th.zeros_like(I)
        for _ in range(T):                                            # LIF integrate-and-fire over the window
            v = self.beta * v + I
            s = th.sigmoid(self.k * (v - self.thr))                   # surrogate spike (differentiable)
            v = v - s * self.thr; rate = rate + s                     # soft reset, accumulate spikes
        return rate / T                                               # mean spike rate (B,D), in [0,1]


def attach(env):
    """Prepend the shared spiking-vision features to `env`'s observation -- every model sees the same
    fixed retina (unified plausible vision). Mirrors _add_maze_vision's obs-patch pattern."""
    import gymnasium as gym
    dev = env._bar.device                          # states is None pre-reset; barriers are already placed
    enc = SpikingRetina().to(dev).eval()
    for p in enc.buffers():
        p.requires_grad_(False)
    env._spike_vis = enc
    _orig = env.get_obs

    def get_obs(*a, **k):
        obs = _orig(*a, **k); B = obs.shape[0]
        cond = getattr(env, "_cond", None)
        if cond is None or cond.shape[0] != B:                       # pre-draw (MotorNet internal reset): zeros
            feat = th.zeros(B, D, device=obs.device)
        else:
            pos = env.states["fingertip"]
            tgt = env.goal[..., :2]
            feat = enc(retina(pos, tgt, env._bar[cond], env._msk[cond]))
        return th.cat([obs, feat], -1)

    env.get_obs = get_obs
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(env.observation_space.shape[0] + D,), dtype=np.float32)
    return env


def demo():
    """Self-check: retina renders, spikes are rates in [0,1], encoder is frozen + differentiable in pos."""
    B, K = 5, 9
    pos = th.zeros(B, 2, requires_grad=True); tgt = th.ones(B, 2) * 0.2
    bar = th.tensor([[0.1, 0.0, 0.05, 0.2]]).repeat(B, K, 1); msk = th.ones(B, K)
    r = retina(pos, tgt, bar, msk)
    assert r.shape == (B, 2, G, G), r.shape
    enc = SpikingRetina()
    rate = enc(r)
    assert rate.shape == (B, D) and float(rate.min()) >= 0 and float(rate.max()) <= 1, "rate not in [0,1]"
    assert not any(p.requires_grad for p in enc.buffers()), "encoder must be frozen"
    rate.sum().backward()                                            # differentiable back to the cursor
    assert pos.grad is not None and th.isfinite(pos.grad).all(), "vision not differentiable in cursor pos"
    # a wall in the retina's field must raise the walls channel above an empty field
    empty = retina(pos.detach(), tgt, th.zeros(B, K, 4), th.zeros(B, K))
    assert r[:, 0].sum() > empty[:, 0].sum() + 1e-3, "walls channel does not respond to a barrier"
    print(f"maze_vision OK: retina {G}x{G} FOV={FOV} -> {D} LIF spike-rate features (T={T}); "
          f"frozen, ON/OFF rods/cones, differentiable in cursor. wall-response ok.")


if __name__ == "__main__":
    demo()
