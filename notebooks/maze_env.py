"""The monkey's OWN maze puzzles, rebuilt inside MotorNet.

WHY
---
Until now the AI models reached to free-space targets while the monkey solved barrier mazes.
Comparing their neural activity across two different tasks confounds "different learning rule"
with "different problem". This module puts the models in the monkey's task: the SAME 108 maze
configurations Jenkins solved in MC_Maze (Churchland+ 2012 / NLB MC_Maze, DANDI 000128), with
the same barrier geometry and the same active target.

MotorNet is a read-only submodule, so nothing here edits it: `MazeReach` SUBCLASSES the
project's ReachEnv and adds the maze on top.

THE 108 PUZZLES
---------------
MC_Maze is 36 maze_ids x 3 trial_versions = 108 distinct conditions. Each carries
  target_pos   (n_targets, 2)  candidate targets, mm
  active_target                index of the one that is actually cued
  barrier_pos  (n_barriers, 4) rectangles (cx, cy, half_w, half_h), mm; 0/6/7/8/9 per maze
Positions are converted mm -> m and re-centred on the MotorNet workspace, because the monkey's
hand coordinates and the plant's coordinates have different origins and extents.

COLLISION
---------
The point mass cannot be given true rigid contact without editing the plant, so barriers act
through a DIFFERENTIABLE penetration penalty: for each rectangle, penetration depth is
    pen = relu(half_w - |dx|) * relu(half_h - |dy|)   (>0 only strictly inside)
summed over barriers. It is differentiable, so a gradient rule feels it through the plant, and
it is a plain scalar, so a local rule feels it through the same shared objective. Every model
therefore gets the identical maze cost -- see `collision_penalty`.
"""
import os
import numpy as np
import torch as th

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save", "mc_maze_configs.npz")
NWB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "000128",
                   "sub-Jenkins", "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb")
MM_TO_M = 1e-3
# The maze centre-hold in raw maze coords = the monkey's mean CURSOR position at movement onset
# across the 108 mazes (std <=4 mm). MC_Maze is navigated by the cursor, and the cursor holds at
# the maze origin (~0,0) -- NOT the hand (which sits ~38 mm below). Barriers/targets are defined
# relative to this cursor origin, so a model compared to the monkey must start here too.
MAZE_HOLD = (-0.001, -0.003)


def hold_reset_options(env, batch):
    """reset() options that place the point mass at the monkey's centre-hold (in the env's plant
    frame), so a model solves the maze from the SAME start the monkey did."""
    import torch as _th
    hx = MAZE_HOLD[0] * env.maze_scale + float(env.maze_centre[0])
    hy = MAZE_HOLD[1] * env.maze_scale + float(env.maze_centre[1])
    js = _th.tensor([[hx, hy, 0.0, 0.0]], dtype=_th.float32).repeat(batch, 1)   # x,y,vx,vy (CPU)
    return {"batch_size": batch, "joint_state": js}


def _add_nlb_to_path():
    import sys
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools")
    if p not in sys.path:
        sys.path.insert(0, p)


def load_maze_nwb(nwb_path=NWB):
    """The MC_Maze NWBDataset, with nlb_tools put on the path first. Robust to the caller's
    working directory and to nlb_tools not already being importable (the qualitative demo needs
    the monkey's real hand trajectories, which live in this NWB)."""
    _add_nlb_to_path()
    from nlb_tools.nwb_interface import NWBDataset
    return NWBDataset(nwb_path)


def extract_configs(nwb_path=NWB, cache=CACHE, force=False):
    """Pull the 108 (maze_id, version) puzzles out of the NWB once and cache them."""
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools"))
    from nlb_tools.nwb_interface import NWBDataset
    ti = NWBDataset(nwb_path).trial_info

    keys, targets, barriers, nbar = [], [], [], []
    for (mz, ver), g in ti.groupby(["maze_id", "trial_version"]):
        r = g.iloc[0]
        tp = np.asarray(r["target_pos"], dtype=np.float64).reshape(-1, 2)
        ai = int(r["active_target"])
        ai = ai if 0 <= ai < len(tp) else 0
        bp = np.asarray(r["barrier_pos"], dtype=np.float64).reshape(-1, 4) \
            if int(r["num_barriers"]) > 0 else np.zeros((0, 4))
        keys.append((int(mz), int(ver))); targets.append(tp[ai])
        barriers.append(bp); nbar.append(len(bp))

    B = max(nbar) if nbar else 0
    bar = np.zeros((len(keys), B, 4)); msk = np.zeros((len(keys), B))
    for i, b in enumerate(barriers):
        if len(b): bar[i, :len(b)] = b; msk[i, :len(b)] = 1.0
    out = dict(keys=np.array(keys), targets=np.array(targets) * MM_TO_M,
               barriers=bar * MM_TO_M, mask=msk, n_barriers=np.array(nbar))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


def maze_split(seed=0, frac_train=0.6, frac_val=0.2, n=None):
    """Deterministic 60/20/20 split of the 108 MC-Maze conditions -> (train, val, test) index arrays.

    FIXED by `seed`, so the same puzzles are always used for training, for hyperparameter tuning
    (val, in 4-tuning-net), and for the held-out TEST report. The TEST puzzles are NEVER seen during
    training or tuning -- the model is fit on `train`, tuned against `val`, and reported on `test`.
    Returns integer index arrays into the 108 conditions (indices are what make_maze_env's
    `conditions=` expects)."""
    if n is None:
        n = len(extract_configs()["targets"])                    # 108
    g = np.random.default_rng(int(seed))
    perm = g.permutation(n)
    n_tr = int(round(frac_train * n)); n_va = int(round(frac_val * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def collision_penalty(pos, barriers, mask):
    """Summed penetration depth of `pos` into each barrier rectangle. 0 outside every barrier.

    pos      (B, 2)          fingertip
    barriers (B, K, 4)       (cx, cy, half_w, half_h)
    mask     (B, K)          1 for a real barrier, 0 for padding
    Differentiable in `pos`, so an analytic-gradient rule can descend it; a scalar, so a local
    rule can consume it through the same shared objective.
    """
    d = (pos[:, None, :] - barriers[..., :2]).abs()
    pen = th.relu(barriers[..., 2] - d[..., 0]) * th.relu(barriers[..., 3] - d[..., 1])
    return (pen * mask).sum(-1)


def collision_force(pos, barriers, mask):
    """Outward push = -gradient of the penetration penalty w.r.t. `pos`. Zero outside every barrier;
    a point INSIDE a barrier feels a force toward the nearest exit -- the flexor-withdrawal /
    obstacle-avoidance reflex the monkey's spinal cord also has. Same shapes as collision_penalty;
    returns (B, 2). Analytic (no autograd) so a local rule can use it under no_grad, exactly like
    the reach reflex. d(px*py)/dx = -sign(dx)*[px>0]*py, so the OUTWARD force is +sign(dx)*[px>0]*py."""
    o = pos[:, None, :] - barriers[..., :2]                       # (B,K,2) signed offset from centre
    px = th.relu(barriers[..., 2] - o[..., 0].abs())              # x-penetration (>0 only inside)
    py = th.relu(barriers[..., 3] - o[..., 1].abs())              # y-penetration
    fx = (th.sign(o[..., 0]) * (px > 0).float() * py * mask).sum(-1)
    fy = (th.sign(o[..., 1]) * (py > 0).float() * px * mask).sum(-1)
    return th.stack([fx, fy], -1)


class MazeReach:
    """Mixin that turns any ReachEnv subclass into the monkey's maze task.

    Use `make_maze_env(...)`; this is kept separate from the plant so MotorNet is untouched.
    """
    def _init_maze(self, cfg, scale=1.0, collide_w=6.0, conditions=None):
        self.cfg = cfg
        self.collide_w = float(collide_w)
        idx = np.arange(len(cfg["targets"])) if conditions is None else np.asarray(conditions)
        self.cond_idx = idx
        dev = self.device
        # monkey hand coords -> plant workspace: centre both, then scale to fit
        lo = self.effector.pos_lower_bound.detach().cpu().numpy()
        hi = self.effector.pos_upper_bound.detach().cpu().numpy()
        ctr = (hi + lo) / 2.0
        span_plant = float(np.min((hi - lo) / 2.0))
        span_monkey = float(np.abs(cfg["targets"]).max())
        k = scale * span_plant / max(span_monkey, 1e-6)
        self.maze_scale, self.maze_centre = k, ctr
        self._tg = th.tensor(cfg["targets"][idx] * k + ctr, dtype=th.float32, device=dev)
        b = cfg["barriers"][idx].copy()
        b[..., :2] = b[..., :2] * k + ctr        # centres move with the workspace
        b[..., 2:] = b[..., 2:] * k              # half-extents only scale
        self._bar = th.tensor(b, dtype=th.float32, device=dev)
        self._msk = th.tensor(cfg["mask"][idx], dtype=th.float32, device=dev)
        self._cond = None

    def force_conditions(self, idx):
        """Pin the maze conditions used by the NEXT reset(s) to `idx` (indices into cond_idx).
        Pass None to release back to the env's default sampling."""
        self._forced_cond = None if idx is None else np.asarray(idx)

    def sample_conditions(self, batch, generator=None, fixed=None):
        n = self._tg.shape[0]
        if fixed is not None:
            c = th.as_tensor(fixed, device=self._tg.device).long().reshape(-1)
            c = c.repeat((batch + len(c) - 1) // len(c))[:batch]
        else:
            c = th.randint(0, n, (batch,), device=self._tg.device, generator=generator)
        self._cond = c
        return c

    def maze_goal(self):
        return self._tg[self._cond]

    def maze_collision(self, pos):
        if self._cond is None:
            return th.zeros(pos.shape[0], device=pos.device)
        return collision_penalty(pos, self._bar[self._cond], self._msk[self._cond])

    def maze_collision_force(self, pos):
        """Outward obstacle-avoidance force at `pos` (B,2), for the spinal avoidance reflex."""
        if self._cond is None:
            return th.zeros_like(pos)
        return collision_force(pos, self._bar[self._cond], self._msk[self._cond])


# The monkey's maze objective, applied IDENTICALLY to every model (fairness by construction):
# reach the cued target FAST + with LEAST movement + LEAST endpoint error + WITHOUT hitting barriers.
#   step cost  =  |fingertip - goal|_1            (least cm; summed over the reach => rewards SPEED,
#                                                  since reaching sooner accrues less error)
#              +  collide_w * barrier_penetration  (avoid the barriers -- the monkey's collision)
#              +  effort_w  * mean(action^2)        (least movement / least muscle drive)
# It is one differentiable scalar, so a gradient rule descends it; a plain scalar, so a local rule
# consumes it through the same reflex; and env reward = -step_cost, so model-free RL optimises it too.
EFFORT_W = 0.05        # "least movement" weight (small: position error must still dominate so RL reaches)

def make_maze_env(dev, mass_set=None, conditions=None, collide_w=6.0, scale=0.85,
                  random_cond=True, effort_w=EFFORT_W, vision=None, **kw):
    # vision=True adds the maze's barrier layout to the observation (the monkey's VISION -> a 1:1
    # input mapping; used by 4-11monkey-net). Default is off (the normal 4-monkey-net setup); the
    # MZ_VISION env var flips it for whole-pipeline runs (maze_train / maze_tune / maze_eprop).
    if vision is None:
        vision = bool(os.environ.get("MZ_VISION"))
    """A MotorNet env running the monkey's 108 maze puzzles, on the monkey's own objective:
    reach the target fast, with the least movement and least endpoint error, WITHOUT hitting the
    barriers -- the same cursor-via-muscle task (obs = goal + proprioception, action = muscles).

    Subclasses the project's MassReach (MotorNet is never edited). On reset it draws a maze
    condition and OVERRIDES the goal with that maze's active target; the reward is the negative of
    the composite maze cost above, so the maze objective is the ONE shared objective every model
    optimises -- fair, identical input (obs) and output (muscles) for all 13.
    """
    import motor_zoo as mz
    cfg = extract_configs()

    class _MazeEnv(mz.MassReach, MazeReach):
        def reset(self, *a, **k):
            obs, info = super().reset(*a, **k)
            B = self.states["fingertip"].shape[0]
            # a caller can pin the exact conditions for this reset (e.g. the qualitative demo
            # rolling a model out on chosen mazes) via `env.force_conditions(idx)`; otherwise
            # draw randomly (training) or tile deterministically (eval).
            forced = getattr(self, "_forced_cond", None)
            fixed = forced if forced is not None else (None if random_cond
                                                       else np.arange(B) % len(self.cond_idx))
            self.sample_conditions(B, fixed=fixed)
            g = self.maze_goal()
            self.goal = g if self.goal.shape[-1] == g.shape[-1] else \
                th.cat([g, th.zeros_like(g)], -1)[..., :self.goal.shape[-1]]
            info["goal"] = self.goal
            return self.get_obs(), info

        def reward(self, action):
            ft = self.states["fingertip"]
            pos = -th.sum(th.abs(self.goal[..., :ft.shape[-1]] - ft), dim=-1, keepdim=True)
            effort = action.pow(2).mean(-1, keepdim=True)
            return pos - self.collide_w * self.maze_collision(ft)[:, None] - effort_w * effort

    env = _MazeEnv(effector=mz.mn.effector.ReluPointMass24(), max_ep_duration=1.0,
                   mass_set=mass_set, **kw)
    env = mz.env_to(env, dev)
    env._init_maze(cfg, scale=scale, collide_w=collide_w, conditions=conditions)
    if vision:
        _add_maze_vision(env)
    return env


def _add_maze_vision(env):
    """MONKEY'S VISION: append the current maze's barrier layout to the observation, so the policy
    SEES the maze the monkey saw -- a 1:1 input mapping -- instead of the barriers only entering
    through the reflex/objective reading env internals. Each of the K padded barriers is added as
    [cx, cy, half_w, half_h] in workspace coordinates (absent barriers = 0); the cursor (fingertip)
    and target are already in the base obs, so the policy has the full monkey-visible scene. This is
    an INPUT, exactly like the monkey's vision -- no privileged simulator metadata."""
    import gymnasium as gym
    K = env._bar.shape[1]                          # max barriers per maze (padded, e.g. 9)
    env._maze_vis_dim = K * 4
    _orig = env.get_obs

    def get_obs(*a, **k):
        obs = _orig(*a, **k); B = obs.shape[0]
        cond = getattr(env, "_cond", None)
        if cond is None or cond.shape[0] != B:     # not yet drawn (MotorNet's internal reset): zeros
            vis = th.zeros(B, env._maze_vis_dim, device=obs.device)
        else:
            vis = env._bar[cond].reshape(B, -1)    # (B, K*4) absolute barrier boxes = the visible walls
        return th.cat([obs, vis], -1)

    env.get_obs = get_obs
    d = env.observation_space.shape[0] + env._maze_vis_dim
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32)


def demo():
    """Self-check: geometry loads, collision fires inside a barrier and not outside."""
    cfg = extract_configs()
    n = len(cfg["targets"])
    assert n == 108, f"expected 108 MC_Maze conditions, got {n}"
    assert cfg["barriers"].shape[-1] == 4
    # a point at a barrier's centre must collide; one far outside must not
    i = int(np.argmax(cfg["n_barriers"]))
    bar = th.tensor(cfg["barriers"][i:i + 1], dtype=th.float32)
    msk = th.tensor(cfg["mask"][i:i + 1], dtype=th.float32)
    inside = bar[0, 0, :2][None]
    outside = th.tensor([[1e3, 1e3]])
    assert collision_penalty(inside, bar, msk).item() > 0, "no penalty at a barrier centre"
    assert collision_penalty(outside, bar, msk).item() == 0, "penalty far outside a barrier"
    # differentiable
    p = inside.clone().requires_grad_(True)
    collision_penalty(p, bar, msk).sum().backward()
    assert p.grad is not None and th.isfinite(p.grad).all()
    print(f"maze_env OK: {n} puzzles, barriers/maze {cfg['n_barriers'].min()}-"
          f"{cfg['n_barriers'].max()}, penalty differentiable")


if __name__ == "__main__":
    demo()
