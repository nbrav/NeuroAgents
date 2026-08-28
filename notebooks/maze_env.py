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
The real MC_Maze barriers were VIRTUAL: the monkey moved a planar manipulandum with no physical
walls, and a trial whose cursor crossed a barrier was ABORTED (fail-on-contact). So a *successful*
monkey trajectory never entered a barrier. We enforce that exactly, two ways that agree:

  * SOLID WALLS (SWEPT hard collision): each step the cursor is clamped at the FIRST barrier face its
    motion (prev->pos) crosses -- a ray-AABB sweep, so it cannot TUNNEL through a thin wall at speed
    (an endpoint-inside test missed that, and a fast model would drive through). Velocity into the
    blocked face is zeroed (tangential kept, so it slides). Done in the env's `step` (MotorNet is a
    read-only submodule) -- see `hard_collision_resolve`. Resolved on the grad-carrying position so a
    plant-backprop learner gets the CORRECT physics gradient (~0 into a blocked wall; a straight-through
    '1 everywhere' told long-horizon BPTT the walls did not exist and it stalled).
  * MONKEY-FAIR SPEED CAP: the cursor may move no faster than the monkey did (`MAX_SPEED` ~= the
    MC_Maze p99 cursor speed), enforced as a per-step displacement cap -- fairness across monkey / ANN
    / human, and it also keeps steps below a wall's width so nothing tunnels.
  * LOSS (reward = -loss, summed over the reach): DISTANCE-to-target-centroid (L1, cm) + TIME
    (per-step) + collide_w * WALL-CONTACT (a bit higher weight; contact = how far the intended move was
    blocked, |attempted - clamped|, differentiable) + effort. See `_MazeEnv.reward`.
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


def maze_split_family(seed=0, n_train=24, n_val=4):
    """LEAK-FREE split by maze FAMILY (maze_id): the two walled versions of a maze share identical
    walls, so a condition-level split leaks held-out wall layouts into training. Splits the 36
    maze_ids 24/4/8 (72/12/24 conditions); every version of a held-out maze is held out."""
    cfg = extract_configs(); keys = np.asarray(cfg["keys"])
    ids = np.array(sorted(set(int(m) for m in keys[:, 0])))
    g = np.random.default_rng(int(seed)); perm = g.permutation(len(ids))
    tr, va, te = set(ids[perm[:n_train]]), set(ids[perm[n_train:n_train + n_val]]), set(ids[perm[n_train + n_val:]])
    pick = lambda s: np.array([i for i, m in enumerate(keys[:, 0]) if int(m) in s])
    return pick(tr), pick(va), pick(te)


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


def hard_collision_resolve(pos, vel, prev, barriers, mask, eps=5e-3):
    """SWEPT solid-wall collision: clamp the cursor at the FIRST barrier face its motion (prev->pos)
    crosses, so it cannot tunnel through a thin wall at speed (the endpoint-inside test missed that),
    and zero the velocity INTO that face (tangential kept, so it can slide along the wall).

    pos, vel, prev (B, 2)   current position, current velocity, position ONE STEP EARLIER (outside).
    barriers (B, K, 4)      (cx, cy, half_w, half_h);   mask (B, K)  1 real / 0 padding.
    Returns (pos', vel'). Slab (ray-AABB) method -> robust to ANY step size. Differentiable in pos
    (the clamp point p0 + t*(pos-p0) is smooth in pos), so a gradient rule still learns through it.
    """
    p0, p1 = prev, pos                                           # (B,2) swept segment
    d = p1 - p0                                                  # (B,2) motion this step
    lo = barriers[..., :2] - barriers[..., 2:]                  # (B,K,2) AABB min corner
    hi = barriers[..., :2] + barriers[..., 2:]                  # (B,K,2) AABB max corner
    d3 = d[:, None, :]
    safe = th.where(d3.abs() < 1e-9, th.full_like(d3, 1e-9), d3)
    t_lo = (lo - p0[:, None, :]) / safe                         # (B,K,2) t to each min face
    t_hi = (hi - p0[:, None, :]) / safe                         # (B,K,2) t to each max face
    t_near = th.minimum(t_lo, t_hi)                             # (B,K,2)
    t_far = th.maximum(t_lo, t_hi)                              # (B,K,2)
    t_entry = t_near.amax(-1)                                   # (B,K) segment enters the AABB
    t_exit = t_far.amin(-1)                                     # (B,K) segment exits the AABB
    entry_axis = t_near.argmax(-1)                             # (B,K) which axis is the entered face
    crosses = (t_entry <= t_exit) & (t_exit >= 0.0) & (t_entry < 1.0) & (t_entry >= -0.02) & (mask > 0)
    t_hit = th.where(crosses, t_entry.clamp(0.0, 1.0), th.full_like(t_entry, 2.0))   # 2 => no hit
    B = p0.shape[0]; bi = th.arange(B, device=p0.device)
    tmin, kmin = t_hit.min(1)                                   # (B,) earliest crossing over barriers
    blocked = tmin < 1.5                                        # (B,) hit a wall this step
    ax = entry_axis[bi, kmin]                                   # (B,) entry-face axis of the first hit
    dnorm = d / (th.linalg.vector_norm(d, dim=-1, keepdim=True) + 1e-9)
    hitpos = p0 + tmin[:, None] * d - eps * dnorm               # stop just BEFORE the face
    newpos = th.where(blocked[:, None], hitpos, p1)
    axmask = th.nn.functional.one_hot(ax, 2).to(vel.dtype)     # (B,2) 1 on the blocked axis
    newvel = th.where(blocked[:, None], vel * (1.0 - axmask), vel)   # zero into-wall vel, keep tangential
    return newpos, newvel


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
        """Outward obstacle-avoidance force at `pos` (B,2), for the spinal avoidance reflex.
        (A proximity/anticipatory avoidance term was tried to lift the ceiling under solid walls but
        REVERTED: unit-scale repulsion over-avoids -> 0% reach, and it cannot help TEST anyway because a
        no-vision policy never sees the held-out maze's barriers. Kept inside-only, which is honest.)"""
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
TIME_W = 0.02          # explicit per-step TIME cost -> finish the puzzle sooner (summed over the reach)
MAX_SPEED = 6.0        # monkey-FAIR cursor speed cap (plant units/s ~= MC_Maze p99 cursor speed 0.84 m/s);
                       # the ANN/human cursor may move no faster than the monkey did -- also kills tunnelling.

def make_maze_env(dev, mass_set=None, conditions=None, collide_w=6.0, scale=0.85,
                  random_cond=True, effort_w=EFFORT_W, vision=None, hard_collision=None, **kw):
    # vision=True adds the maze's barrier layout to the observation (the monkey's VISION -> a 1:1
    # input mapping; used by 4-11monkey-net). Default is off (the normal 4-monkey-net setup); the
    # MZ_VISION env var flips it for whole-pipeline runs (maze_train / maze_tune / maze_eprop).
    if vision is None:
        vision = bool(os.environ.get("MZ_VISION"))
    # SOLID WALLS by default (the monkey's maze -- a successful reach never crosses a barrier). Off
    # only if explicitly disabled (hard_collision=False or MZ_SOFT=1), e.g. to reproduce old runs.
    if hard_collision is None:
        hard_collision = not bool(os.environ.get("MZ_SOFT"))
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
            # START AT THE MONKEY'S CENTRE HOLD: MC-Maze targets + barriers are defined relative to
            # the cursor's centre-hold at movement onset, so every reach must begin there (not at a
            # random position). Inject the hold as the initial joint state unless the caller pinned one.
            if hasattr(self, "maze_scale"):
                opts = dict(k.get("options") or {})
                if "joint_state" not in opts:
                    B = int(opts.get("batch_size", 1))
                    hx = MAZE_HOLD[0] * self.maze_scale + float(self.maze_centre[0])
                    hy = MAZE_HOLD[1] * self.maze_scale + float(self.maze_centre[1])
                    opts["joint_state"] = th.tensor([[hx, hy, 0.0, 0.0]], dtype=th.float32).repeat(B, 1)
                    k = dict(k); k["options"] = opts
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
            # The loss every model minimises (reward = -loss), summed over the reach:
            #   DISTANCE-to-target-centroid (L1, cm)  +  TIME (per-step)  +  collide_w * WALL-CONTACT
            #   (a bit higher weight)  +  effort. Wall-contact = how far the intended move was BLOCKED by
            #   a wall this step (differentiable: |attempted - clamped|); 0 in free space.
            ft = self.states["fingertip"]
            dist = th.sum(th.abs(self.goal[..., :ft.shape[-1]] - ft), dim=-1, keepdim=True)
            effort = action.pow(2).mean(-1, keepdim=True)
            contact = getattr(self, "_contact", None)
            if contact is None or contact.shape[0] != ft.shape[0]:        # None / stale (super().step's
                contact = self.maze_collision(ft)[:, None]                # pre-clamp call) -> soft fallback
            return -(dist + TIME_W + self.collide_w * contact + effort_w * effort)

        def step(self, action, *a, **k):
            # advance the plant, THEN swept-block the cursor at any wall its motion crossed (solid walls,
            # no tunneling), and score the reward on the BLOCKED position incl. the wall-contact penalty.
            if getattr(self, "hard_collision", False) and getattr(self, "_cond", None) is not None:
                prev = self.states["fingertip"].detach().clone()          # last (outside-the-walls) pos
                obs, r, term, trunc, info = super().step(action, *a, **k)
                self._apply_hard_collision(prev)                          # SWEPT: clamp at the face, set _contact
                obs = self.get_obs()                                      # obs now reflects the blocked pos
                r = self.reward(action)                                   # loss on blocked pos + wall-contact + time
                return obs, r, term, trunc, info
            return super().step(action, *a, **k)

        def _apply_hard_collision(self, prev):
            st = self.effector.states                                     # env.states IS effector.states
            pos, vel = st["joint"][:, :2], st["joint"][:, 2:]             # attempted (pre-clamp) pos/vel
            # MONKEY-FAIR SPEED CAP: no ANN cursor may move faster than the monkey did -- limit the
            # per-step displacement (and velocity) to MAX_SPEED*dt (fairness + it prevents tunnelling).
            disp = pos - prev; dist = th.linalg.vector_norm(disp, dim=-1, keepdim=True)
            pos = prev + disp * (MAX_SPEED * self.dt / dist.clamp(min=1e-9)).clamp(max=1.0)
            vel = vel * (MAX_SPEED / th.linalg.vector_norm(vel, dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            # SWEPT clamp on the grad-carrying (capped) position (physics gradient stays correct; a
            # straight-through '1 everywhere' told long-horizon BPTT the walls did not exist and it stalled).
            npos, nvel = hard_collision_resolve(pos, vel, prev,
                                                self._bar[self._cond], self._msk[self._cond])
            # WALL CONTACT (differentiable) = how far the intended move was blocked by a wall = |attempted-clamped|.
            self._contact = th.linalg.vector_norm(pos - npos, dim=-1, keepdim=True)
            self._wall_contact = (self._contact.squeeze(-1) > 1e-4).float().detach()   # % of steps blocked (metric)
            st["joint"] = th.cat([npos, nvel], -1)
            st["cartesian"] = st["joint"]
            st["fingertip"] = npos

    env = _MazeEnv(effector=mz.mn.effector.ReluPointMass24(), max_ep_duration=1.0,
                   mass_set=mass_set, **kw)
    env = mz.env_to(env, dev)
    env._init_maze(cfg, scale=scale, collide_w=collide_w, conditions=conditions)
    env.hard_collision = bool(hard_collision)
    if os.environ.get("MZ_CONTACT_OBS"):           # EMBODIED afferent: expose the binary wall-contact flag
        _add_contact_obs(env)
    if os.environ.get("MZ_SPIKE_VISION"):          # 4-11monkey: unified plausible SPIKING retina (all models)
        import maze_vision; maze_vision.attach(env)
    elif vision:
        _add_maze_vision(env)                       # raw egocentric barrier coords (4-monkey option)
    return env


def _add_contact_obs(env):
    """Append the binary wall-contact flag (1 = the intended move was blocked this step) to the
    observation: the agent's only afferent evidence of a wall in the BLIND condition. Zero pre-reset
    and on the first step; sized to the batch so it is safe in every reset path."""
    import gymnasium as gym
    _orig = env.get_obs

    def get_obs(*a, **k):
        obs = _orig(*a, **k); B = obs.shape[0]
        c = getattr(env, "_wall_contact", None)
        flag = c[:, None] if (c is not None and c.shape[0] == B) else th.zeros(B, 1, device=obs.device)
        return th.cat([obs, flag.to(obs.device)], -1)

    env.get_obs = get_obs
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(env.observation_space.shape[0] + 1,), dtype=np.float32)


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
            bar = env._bar[cond].clone()           # (B, K, 4) [cx, cy, half_w, half_h] absolute
            ft = env.states["fingertip"][:, None, :].detach()      # sensory input (cursor position)
            bar[..., :2] = bar[..., :2] - ft       # EGOCENTRIC: barrier offset from the cursor (dynamic,
            vis = bar.reshape(B, -1)               # foveated on the cursor -- how the monkey saw the walls)
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
    # hard collision: a cursor sent straight into a wall's long face must be BLOCKED, not pass through
    # (inject velocity so the test is free of the muscles' diagonal pull; compare soft vs solid).
    import motor_zoo as mz
    ci = int(np.argmax(cfg["n_barriers"]))
    def _drive_into_wall(hard):
        e = make_maze_env(mz.DEVICE, random_cond=False, hard_collision=hard)
        e.force_conditions([ci]); e.reset(options={"batch_size": 1})
        b = e._bar[e._cond][0][e._msk[e._cond][0] > 0][0]        # a real barrier (cx,cy,hw,hh)
        cx, cy, hw, hh = [float(v) for v in b]
        long_x = hw >= hh                                        # approach perpendicular to the long face
        dev = e.effector.states["joint"].device
        if long_x:  st, vel = [cx, cy - hh - 0.03, 0., 25.0], "y"  # below a long-in-x bar, drive +y FAST
        else:       st, vel = [cx - hw - 0.03, cy, 25.0, 0.], "x"  # left of a long-in-y bar, drive +x FAST
        _ = vel                                                    # (high speed => would TUNNEL without swept CD)
        e.effector.states["joint"][0] = th.tensor(st, device=dev)
        e.effector.states["fingertip"][0] = th.tensor(st[:2], device=dev)
        a = th.zeros(1, e.action_space.shape[0], device=dev); mx = 0.0
        for _ in range(25):
            e.step(a); mx = max(mx, float(e.maze_collision(e.states["fingertip"])[0]))
        ft = e.states["fingertip"][0]
        past = (float(ft[1]) > cy + hh) if long_x else (float(ft[0]) > cx + hw)   # crossed to the far side?
        return past, mx
    soft_past, _ = _drive_into_wall(False)
    hard_past, hard_pen = _drive_into_wall(True)
    assert soft_past, f"test invalid: soft wall did not let the fast cursor through (past={soft_past})"
    assert not hard_past and hard_pen < 0.01, f"SOLID wall failed to block a FAST cursor (tunnelled: past={hard_past}, pen={hard_pen:.4f})"
    print(f"maze_env OK: {n} puzzles, barriers/maze {cfg['n_barriers'].min()}-"
          f"{cfg['n_barriers'].max()}; SWEPT solid walls block even a FAST cursor "
          f"(soft: tunnels through; hard: blocked at the face, pen={hard_pen:.4f})")


if __name__ == "__main__":
    demo()
