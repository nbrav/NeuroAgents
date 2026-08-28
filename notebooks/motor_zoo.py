"""Auto-generated from 4-train-net.ipynb (MASS-RANDOMISED REDO) section-VIII cell files.
Do not edit by hand -- edit the cells / scratchpad sources and regenerate.
Provides: DEVICE, make_env, ReachEnv, MassReach, make_mass_env, env_to, force_head,
the harness (evaluate, eval_metrics, rollout, Probe, obs_norm, OpCounter, zero_shot,
TRAIN_MASSES/OOD_MASSES), and every Learner class (deep-RL ones demonstration-bootstrapped)."""


# ===== from cell24_env.py ===========================================
# ==============================================================================
# VII. 1  The gym environment: on the GPU, with a reward
# ------------------------------------------------------------------------------
# Three things MotorNet does not give us out of the box. All are added by
# SUBCLASSING -- the MotorNet checkout is a git submodule and is never edited.
#
#   1. device   `env.to("cuda")` moves the registered buffers, but Effector /
#               Skeleton / Muscle each cache their own `_device` attribute and
#               `nn.Module._apply` never updates it. The muscle then builds its
#               initial state on the CPU and reset() dies with a device mismatch.
#               `env_to()` sets the cached attribute on all four objects.
#
#   2. noise    `Environment.apply_noise` round-trips through numpy on every call
#               (3x per step) even when the noise vector is all zeros -- which is
#               the default. On GPU that is a host sync + H2D copy per step. We
#               skip no-op noise and sample the rest on-device. Bit-exact.
#
#   3. reward   `Environment.step` returns reward=None when differentiable=True
#               (and hard-coded zeros otherwise): MotorNet has NO task reward.
#               Every reinforcement rule in section VIII (RTRRL, BTSP, R-STDP,
#               3-factor Hebbian) needs a scalar signal, so we define one.
# ==============================================================================
import numpy as np
import torch as th
import motornet as mn

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
th.backends.cudnn.benchmark = True


def env_to(env, device):
    """Move a MotorNet env AND the `_device` attribute each sub-object caches."""
    device = th.device(device)
    env.to(device)
    for m in (env.effector, env.effector.skeleton, env.effector.muscle):
        m.to(device)
    return env


class ReachEnv(mn.environment.RandomTargetReach):
    """RandomTargetReach + a bounded reward + GPU-friendly noise.

    Reward, per step and per batch element:

        r_t = -(d_t / d_max)  -  effort_w * mean_m(a_m^2)

    where d_t = ||fingertip_t - goal||_2 and d_max is the workspace diagonal, so
    the distance term is normalised to [-1, 0]. Muscle excitation a_m is bounded
    to [0, 1] by the action space, so the effort term lies in [-effort_w, 0].

        =>  r_t in [-(1 + effort_w), 0],  hitting 0 only when on-target AND silent.

    This distance+effort cost is a standard choice for MotorNet reaching, but it
    is OURS -- MotorNet itself defines no reward.
    """

    def __init__(self, *args, effort_w: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.effort_w = float(effort_w)
        lo = self.effector.pos_lower_bound.detach().cpu().numpy()
        hi = self.effector.pos_upper_bound.detach().cpu().numpy()
        self.d_max = float(np.linalg.norm(hi - lo))          # workspace diagonal
        self.reward_range = (-(1.0 + self.effort_w), 0.0)
        self._noise_cache = {}

    # --- 2. cheap, on-device noise -------------------------------------------
    def apply_noise(self, loc, noise):
        key = id(noise)
        if key not in self._noise_cache:
            arr = np.asarray(noise, dtype=np.float32).reshape(-1)
            self._noise_cache[key] = None if not arr.any() else th.as_tensor(arr, device=loc.device)
        scale = self._noise_cache[key]
        if scale is None:                       # all-zero noise is a no-op
            return loc
        return loc + th.randn_like(loc) * scale

    # --- 3. reward ------------------------------------------------------------
    def dist(self):
        """Euclidean fingertip->goal distance, (batch, 1). Differentiable."""
        return th.linalg.vector_norm(self.states["fingertip"] - self.goal, dim=-1, keepdim=True)

    def reward(self, action):
        effort = action.pow(2).mean(dim=-1, keepdim=True)
        return -(self.dist() / self.d_max) - self.effort_w * effort

    def step(self, action, **kwargs):
        obs, _, terminated, truncated, info = super().step(action, **kwargs)
        r = self.reward(info["action"])
        info["reward"] = r
        info["dist"] = self.dist()
        return obs, r, terminated, truncated, info


def make_env(device=DEVICE, **kwargs):
    """A ReachEnv on `device`. kwargs go to ReachEnv/Environment (effort_w, obs_noise, ...)."""
    return env_to(ReachEnv(effector=mn.effector.ReluPointMass24(), max_ep_duration=1., **kwargs), device)


# ---- self-check: the claims above have to actually hold -----------------------
def _check_env():
    e = make_env()
    obs, info = e.reset(options={"batch_size": 16})
    assert obs.shape == (16, 12) and obs.device.type == DEVICE.type
    lo, hi = e.reward_range

    rs, ds = [], []
    for _ in range(int(e.max_ep_duration / e.dt)):
        obs, r, term, trunc, info = e.step(th.rand(16, 4, device=DEVICE))
        assert r.shape == (16, 1)
        rs.append(r); ds.append(info["dist"])
    r, d = th.cat(rs), th.cat(ds)
    assert (r >= lo).all() and (r <= hi).all(), f"reward escaped {e.reward_range}"
    assert (d >= 0).all() and (d <= e.d_max).all(), "distance exceeds workspace diagonal"
    assert term and not trunc, "episode should terminate at max_ep_duration"

    # reward is exactly 0 iff on-target and silent
    e.reset(options={"batch_size": 4}); e.goal = e.states["fingertip"].clone()
    assert th.allclose(e.reward(th.zeros(4, 4, device=DEVICE)), th.zeros(4, 1, device=DEVICE), atol=1e-6)

    # the reward is differentiable wrt the policy (BPTT rules depend on this)
    e2 = make_env(); obs, _ = e2.reset(options={"batch_size": 4})
    w = th.zeros(12, 4, device=DEVICE, requires_grad=True)
    _, r, *_ = e2.step(th.sigmoid(obs @ w)); r.sum().backward()
    assert w.grad.abs().sum() > 0, "reward is not differentiable wrt the policy"

    print(f"env OK | device={DEVICE} | obs=(B,12) | act=(B,4) in [0,1] | "
          f"reward in [{lo:.2f}, 0] | d_max={e.d_max:.3f} m | "
          f"{int(e.max_ep_duration/e.dt)} steps x {e.dt*1000:.0f} ms")

# ===== from c1_harness.py ===========================================
# ==============================================================================
# VIII. 1  The comparison harness: one objective, one budget, many metrics
# ------------------------------------------------------------------------------
# Every learning rule below optimises the SAME objective on the SAME env:
#
#       J = E[ sum_t r_t ],    r_t = -(d_t / d_max) - 0.1 * mean_m(a_m^2)
#
# They differ ONLY in how the weight update is computed. That is the whole point:
# hold the task and the objective fixed, vary the credit-assignment mechanism.
#
# Metrics measured for every method (columns of the scoreboard):
#   * Sampling efficiency   episodes of experience until eval error < THRESH
#   * Convergence           episodes until eval error settles to 110% of its own best
#   * Asymptotic accuracy   eval endpoint error at the end of the budget (cm)
#   * Reward                mean episodic return on the held-out set
#   * Completion            % of held-out reaches ending within SUCCESS_CM of target
#   * Zero-shot generalis.  eval error under perturbations NEVER seen in training
#   * Energy efficiency     pJ per control step (45 nm MAC/AC model)
#   * Continual learning    forgetting of field A after training on field B
#   * NeuroAI               control sparsity + effective dimensionality of the policy
#
# A `Learner` is anything with .name/.cite/.init_state/.act/.fit. No base class is
# imposed on the update rule -- each one owns its optimiser and its own maths.
#
# DATA-LEAKAGE POSITION (audited in cell VIII.1b): training never draws the eval
# seed, the eval set is a fixed seeded held-out draw, the observation normaliser is a
# fixed constant (not fit to data), and the zero-shot perturbations are genuinely
# out-of-distribution. Reproducibility: one seed per method, stated where set.
# ==============================================================================
import time, math, contextlib, os, json
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

# ---- protocol ----------------------------------------------------------------
class Learner:
    """Interface every rule implements. `fit` is free to update however it likes."""
    name: str = "?"
    cite: str = "?"
    kind: str = "?"          # "global-gradient" | "local-plausible"
    wins: str = ""           # which table row this method is the poster child for

    def init_state(self, B):            return None
    def act(self, obs, state, explore=False):  raise NotImplementedError
    def fit(self, env, budget, probe):  raise NotImplementedError
    def n_params(self):
        return sum(p.numel() for p in self.parameters()) if isinstance(self, nn.Module) else 0


# ---- fixed evaluation sets ---------------------------------------------------
# reset(seed=S) reseeds the effector's PRNG, so the same seed gives the same
# (start, target) pairs every time -> a fixed, reproducible held-out test set.
EVAL_SEED  = 20260717     # never used for training
EVAL_BATCH = 512
SUCCESS_CM = 5.0          # a reach is "complete" if its endpoint error < 5 cm

# Perturbations for the zero-shot row. None of these are ever seen during training.
def _load(fx, fy, dev):
    return th.tensor([[fx, fy]], dtype=th.float32, device=dev)

def perturbations(dev):
    """(name, reset_kwargs, step_kwargs, env_mutator) tuples -- all OOD."""
    return [
        ("force field +x", {}, {"endpoint_load": _load(4.0, 0.0, dev)}, None),
        ("force field -y", {}, {"endpoint_load": _load(0.0, -4.0, dev)}, None),
        ("curl load",      {}, {"endpoint_load": _load(3.0, 3.0, dev)}, None),
        ("2x mass",        {}, {}, lambda e: setattr(e.effector.skeleton, "mass", 2.0)),
        ("weak muscles",   {}, {}, lambda e: e.effector.muscle.max_iso_force.mul_(0.5)),
    ]


@th.no_grad()
def rollout(env, learner, seed=EVAL_SEED, batch=EVAL_BATCH, step_kwargs=None):
    """One deterministic held-out rollout. Returns per-step tensors, no reduction."""
    step_kwargs = step_kwargs or {}
    obs, info = env.reset(seed=seed, options={"batch_size": batch, "deterministic": True})
    st = learner.init_state(batch)
    n = int(env.max_ep_duration / env.dt)
    dist, act, rew, xy = [], [], [], [info["states"]["fingertip"]]
    tg = info["goal"]
    for t in range(n):
        a, st = learner.act(obs, st, explore=False)
        obs, r, term, trunc, info = env.step(a, deterministic=True, **step_kwargs)
        dist.append(info["dist"]); act.append(a); rew.append(r)
        xy.append(info["states"]["fingertip"])
    return dict(dist=th.cat(dist, 1), act=th.stack(act, 1), rew=th.cat(rew, 1),
                xy=th.stack(xy, 1), tg=tg, n=n)


@th.no_grad()
def evaluate(env, learner, seed=EVAL_SEED, batch=EVAL_BATCH, step_kwargs=None,
             tail=0.2, return_traj=False):
    """Deterministic held-out endpoint error, in cm (mean over the last `tail` steps).

    Using the tail rather than the final sample alone rejects single-step overshoot.
    `return_traj` additionally hands back the cartesian trajectory and targets.
    """
    r = rollout(env, learner, seed, batch, step_kwargs)
    k = max(1, int(tail * r["n"]))
    err_cm = 100.0 * r["dist"][:, -k:].mean().item()
    if return_traj:
        return err_cm, r["xy"].cpu().numpy(), r["tg"].cpu().numpy()
    return err_cm


@th.no_grad()
def eval_metrics(env, learner, seed=EVAL_SEED, batch=EVAL_BATCH, step_kwargs=None, tail=0.2):
    """The full held-out metric bundle for one method.

        err_cm       endpoint error (cm), the accuracy column
        ret          mean episodic return sum_t r_t  -- the reward column
        completion   % of reaches ending within SUCCESS_CM of target
        ctrl_sparse  control sparsity: mean fraction of muscle-steps below 5% excitation
                     (a NeuroAI read on how "spiky"/economical the motor command is)
        cocontract   co-contraction index: min(opposing muscle pair) averaged -- how much
                     the policy stiffens the limb by pulling antagonists together
        eff_dim      participation ratio of the 4-D action covariance (1..4): how many
                     independent muscle synergies the policy actually uses
    """
    r = rollout(env, learner, seed, batch, step_kwargs)
    k = max(1, int(tail * r["n"]))
    err_cm = 100.0 * r["dist"][:, -k:].mean().item()
    ret = r["rew"].sum(1).mean().item()
    final_err_m = r["dist"][:, -k:].mean(1)                       # (B,) per-reach endpoint err
    completion = 100.0 * (final_err_m < SUCCESS_CM / 100.0).float().mean().item()
    completion2 = 100.0 * (final_err_m < 0.02).float().mean().item()   # strict 2 cm (biological precision, unlike the loose 5 cm "reached")
    a = r["act"]                                                  # (B, T, n_muscles) in muscle order
    ctrl_sparse = 100.0 * (a < 0.05).float().mean().item()
    # antagonist pairs for the co-contraction index, per plant:
    #   ReluPointMass24 (UR,UL,LR,LL) -> (UR,LL),(UL,LR); RigidTendonArm26 -> 3 flex/ext pairs
    #   (pectoralis/deltoid, brachioradialis/tricepslat, biceps/tricepslong).
    pairs = [(0, 1), (2, 3), (4, 5)] if a.shape[-1] == 6 else [(0, 3), (1, 2)]
    cc = sum(th.minimum(a[..., i], a[..., j]) for i, j in pairs)
    cocontract = cc.mean().item()
    A = a.reshape(-1, a.shape[-1])                                # (B*T, 4)
    C = th.cov(A.t()) + 1e-8 * th.eye(a.shape[-1], device=a.device)
    try:                                     # a diverged learner yields a degenerate/NaN
        ev = th.linalg.eigvalsh(C).clamp(min=0)   # covariance; that is a RESULT to report,
        eff_dim = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item()   # not a crash.
    except Exception:
        eff_dim = float("nan")
    return dict(err_cm=err_cm, ret=ret, completion=completion, completion2=completion2,
                ctrl_sparse=ctrl_sparse, cocontract=cocontract, eff_dim=eff_dim)


def zero_shot(make_env_fn, learner, dev):
    """Eval under each unseen perturbation. Returns {name: err_cm}."""
    out = {}
    for name, rkw, skw, mut in perturbations(dev):
        e = make_env_fn(dev)
        if mut is not None: mut(e)
        out[name] = evaluate(e, learner, step_kwargs=skw)
    return out


# ---- energy model ------------------------------------------------------------
# Horowitz, ISSCC 2014, Fig 1.1.9: 45 nm, 32-bit  -> MAC 4.6 pJ, ADD 0.9 pJ.
# ANN cost  = (#MACs per control step) * E_MAC
# SNN cost  = (#synaptic operations actually triggered by a spike) * E_AC
# SynOps convention follows Sorbaro et al. 2020, Front. Neurosci. 14:662.
# Expected discounted return of an untrained policy on this task: r ~ -0.45/step,
# gamma=0.99, 100 steps -> sum gamma^k r ~ -28. Every critic below is initialised here
# so its bias does not have to be learned from scratch in the few updates it gets.
V0_INIT = -28.0

E_MAC_PJ = 4.6
E_AC_PJ  = 0.9

class OpCounter:
    """Counts dense MACs and spike-triggered ACs for ONE control step."""
    def __init__(self): self.mac = 0.0; self.ac = 0.0
    def dense(self, n_in, n_out):  self.mac += n_in * n_out
    def synops(self, n_spikes, fanout): self.ac += n_spikes * fanout
    def pj(self):  return self.mac * E_MAC_PJ + self.ac * E_AC_PJ


# ---- training-budget probe ---------------------------------------------------
class Probe:
    """Records (episodes_consumed, eval_err_cm) during fit, on a fixed schedule.

    `learner.fit` calls probe(learner, episodes) whenever it has consumed more
    experience; the probe decides when to actually run an evaluation.
    """
    def __init__(self, env, every_eps, budget, batch=256, track_best=False):
        self.env, self.every, self.budget = env, every_eps, budget
        self.batch = batch
        self.curve = []          # (episodes, err_cm)
        self._next = 0
        self.t0 = time.perf_counter()
        self.peak_mem = 0
        self.track_best = track_best    # EARLY STOPPING: keep the lowest-val-err checkpoint
        self.best_err = float("inf")
        self.best_state = None

    def __call__(self, learner, episodes, force=False):
        if episodes < self._next and not force: return False
        self._next = episodes + self.every
        err = evaluate(self.env, learner, batch=256)
        self.curve.append((episodes, err))
        if self.track_best and err < self.best_err and isinstance(learner, th.nn.Module):
            import copy
            self.best_err = err
            self.best_state = copy.deepcopy(learner.state_dict())   # snapshot the best-val model
        if th.cuda.is_available():
            self.peak_mem = max(self.peak_mem, th.cuda.max_memory_allocated())
        return True

    def eps_to(self, thresh_cm):
        """Episodes of experience before eval error first drops below `thresh_cm`."""
        for eps, err in self.curve:
            if err < thresh_cm: return eps
        return float("inf")

    def best(self):
        return min((e for _, e in self.curve), default=float("inf"))

    def eps_to_converge(self, rel=0.10):
        """Episodes until the curve settles: eval error first reaches within `rel` of
        the method's OWN final error (relative plateau). This is the convergence /
        "iterations to converge" column, and unlike eps_to it does not depend on a
        method actually being good -- a method that plateaus high still has a
        convergence point. Returns inf only if the curve is empty."""
        if not self.curve: return float("inf")
        final = self.curve[-1][1]
        target = final * (1.0 + rel) if final > 0 else final + 1.0
        for eps, err in self.curve:
            if err <= target: return eps
        return self.curve[-1][0]


# ---- persistence: notebook 1 trains, notebook 2 (4-analysis-net) analyses --------
MODEL_DIR = os.path.join("save", "models")

def save_learner(learner, tag, extra=None):
    """Persist a trained nn.Module learner + a small manifest for the analysis notebook."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if isinstance(learner, nn.Module):
        th.save(learner.state_dict(), os.path.join(MODEL_DIR, f"{tag}.pt"))
    meta = dict(name=learner.name, cite=learner.cite, kind=learner.kind,
                wins=getattr(learner, "wins", ""), tag=tag)
    if extra: meta.update(extra)
    with open(os.path.join(MODEL_DIR, f"{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---- shared building blocks --------------------------------------------------
def mlp(sizes, act=nn.Tanh, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2: layers.append(act())
    if out_act is not None: layers.append(out_act())
    return nn.Sequential(*layers)


class SparseExpansion(nn.Module):
    """Fixed random sparse expansion: a cerebellar-granule / mossy-fibre style code.

    Random projection -> k-winners-take-all (lateral inhibition) -> normalise to sum 1.
    Non-negative, sparse and normalised, so a = W phi with W in [0, W_max] is a convex
    combination and therefore already a legal muscle excitation in [0, 1]. Nothing here
    learns -- these are FIXED random features, the "localized unsupervised
    pre-training" the biologically-plausible column leans on.
    """
    def __init__(self, n_in, N, k, dev, seed=0):
        super().__init__()
        g = th.Generator().manual_seed(seed)
        self.register_buffer("R", th.randn(N, n_in, generator=g) / math.sqrt(n_in))
        self.register_buffer("b", th.zeros(N))
        self.k, self.N = k, N
        self.to(dev)

    def forward(self, x):
        h = th.tanh(x @ self.R.t() + self.b)
        v, i = h.topk(self.k, dim=-1)                     # k-WTA
        phi = th.zeros_like(h).scatter_(-1, i, v.clamp(min=0.))
        return phi / (phi.sum(-1, keepdim=True) + 1e-6)   # sums to 1

class ReplayBuffer:
    """GPU-resident ring buffer shared by the off-policy learners (SAC, FastTD3,
    SimbaV2). Storing transitions on-device avoids a host round-trip per step, which
    matters when the env itself is GPU-batched."""
    def __init__(self, cap, O, A, dev):
        self.cap, self.dev, self.n, self.ptr = cap, dev, 0, 0
        z = lambda d: th.zeros(cap, d, device=dev)
        self.o, self.a, self.r, self.o2, self.d = z(O), z(A), z(1), z(O), z(1)

    def push(self, o, a, r, o2, done):
        m = o.shape[0]
        idx = (th.arange(m, device=self.dev) + self.ptr) % self.cap
        self.o[idx] = o; self.a[idx] = a; self.r[idx] = r; self.o2[idx] = o2; self.d[idx] = done
        self.ptr = int((self.ptr + m) % self.cap); self.n = min(self.n + m, self.cap)

    def sample(self, bs):
        i = th.randint(0, self.n, (bs,), device=self.dev)
        return self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i]


def detach_env_state(env):
    """Cut the autograd graph at the current env state.

    MotorNet keeps the effector's joint/muscle/geometry/cartesian/fingertip tensors
    AND the goal as live graph nodes. Detaching only the returned obs leaves those
    connected, so a windowed-BPTT method (SHAC) that backprops one window, then steps
    again, tries to backward through the freed previous-window graph. Detaching the
    whole plant state in place severs it cleanly -- the physics is unchanged, only the
    gradient tape is cut."""
    eff = env.effector
    for k, v in eff.states.items():
        if th.is_tensor(v): eff.states[k] = v.detach()
    if th.is_tensor(env.goal): env.goal = env.goal.detach()
    for buf in ("proprioception", "vision", "action"):
        env.obs_buffer[buf] = [x.detach() if th.is_tensor(x) else x
                               for x in env.obs_buffer[buf]]


def obs_norm(env):
    """Fixed affine normaliser for the 12-D observation.

    Panel B of the figure above showed the three observation groups span ~1.5
    orders of magnitude (goal/vision ~[-1,1], length ~[1.4,4.2], velocity ~[-37,37]).
    Feeding that raw into a tanh/spiking net saturates it. These constants come
    from the measured ranges, NOT from a running estimate -- so every rule sees
    exactly the same inputs and nothing leaks between episodes.
    """
    # Layout is [4 vision (goal_xy, fingertip_xy), n_muscles length, n_muscles velocity] on
    # BOTH plants: point mass -> 4+4+4=12, RigidTendonArm26 -> 4+6+6=16. The constants are
    # measured PER PLANT (muscle length/velocity ranges differ), so branch on the muscle count.
    O = env.observation_space.shape[0]; nm = env.action_space.shape[0]
    mvis = getattr(env, "_maze_vis_dim", 0)                 # 1:1-monkey maze VISION block (barriers), appended last
    base = O - mvis                                         # the standard MotorNet obs
    vis = base - 2 * nm
    if nm == 6:                                              # monkey arm (measured on RigidTendonArm26)
        mu  = th.cat([th.zeros(vis), th.full((nm,), 0.9), th.zeros(nm)])
        sig = th.cat([th.full((vis,), 0.3), th.full((nm,), 0.3), th.full((nm,), 3.0)])
    else:                                                    # ReluPointMass24 (original measured constants)
        mu  = th.cat([th.zeros(vis), th.full((nm,), 2.8), th.zeros(nm)])
        sig = th.cat([th.full((vis,), 0.6), th.full((nm,), 0.8), th.full((nm,), 12.)])
    if mvis:                                                # barrier boxes live in workspace coords (~0.1 m)
        mu = th.cat([mu, th.zeros(mvis)]); sig = th.cat([sig, th.full((mvis,), 0.1)])
    return mu.to(env.device), sig.to(env.device)

# ===== from t27_append.py ===========================================


# ==============================================================================
# VIII. 1c  The KINESIS force head -- the fair, shared action space (added in the redo)
# ------------------------------------------------------------------------------
# The 4-tuning-net.ipynb study established the "correct setting" for this plant: instead
# of emitting 4 one-sided muscle excitations directly (which unfairly rewarded whichever
# method was hand-tuned for that raw action space), every controller emits a 2-D endpoint
# FORCE + a co-contraction level, and the plant geometry does the muscle transform. This
# is morphological computation (the KINESIS idea), plausible end to end, and it needs only
# the observation -- so every one of the thirteen methods below wears the SAME head. The
# comparison is now fair: what differs is the learning rule, not the action space.
#   a_m = relu(d_m . f)/F_MAX + c ,  f = tanh(raw[:2])*f_scale ,  c = sigmoid(raw[2])
# ==============================================================================
_HE = make_env(DEVICE)
ANCHORS = th.tensor(_HE.effector._path_coordinates[0, :, 0::2].T, dtype=th.float32, device=DEVICE)
F_MAX = float(_HE.effector.muscle.max_iso_force.mean())
del _HE

def force_head(obs, raw3, f_scale=650., anchors=ANCHORS):
    """KINESIS morphological decode: (obs, 3 raw force params) -> 4 muscle excitations.

    This is the PLAUSIBLE (morphological-computation) head: the body geometry does the muscle
    coordinate transform. Per the plausible-vs-not separation, it is reserved for the
    biologically-plausible learners ONLY. Non-plausible learners use `muscle_head` below."""
    P = obs[:, 2:4]; l = obs[:, 4:8].clamp(min=1e-2)
    d = (anchors[None] - P[:, None, :]) / l[:, :, None]
    f = th.tanh(raw3[:, :2]) * f_scale; c = th.sigmoid(raw3[:, 2:3])
    return (F.relu((d * f[:, None, :]).sum(-1)) / F_MAX + c).clamp(0., 1.)


def muscle_head(obs, raw4):
    """The unchanged MotorNet-native head: emit the 4 muscle excitations DIRECTLY via a
    sigmoid. This is the head the non-plausible learners (BPTT-GRU baseline, SHAC, SAC,
    FastTD3, Simba) wear on the 2-D point mass, so they get NO morphological (plausible)
    advantage -- they must coordinate the muscles the hard way, exactly like MotorNet."""
    return th.sigmoid(raw4)   # plant-agnostic: raw width == RAW == env.n_muscles (4 point-mass / 6 arm)


# ---- arm morphological head (RigidTendonArm26): the force_head analog for the 6-muscle arm ----
# force_head works on the point mass because its muscles pull the fingertip DIRECTLY in Cartesian
# space (line of action = fingertip->anchor, both read from OBS). A 2-joint arm's muscles make
# JOINT TORQUES, so the morphological transform is one step longer:
#     endpoint force f  --J(q)^T-->  joint torque tau  --(-M(q)^+)-->  least-effort muscle tensions
#
# FAIRNESS (no leakage): exactly like force_head, this head uses ONLY the OBSERVATION (fingertip =
# obs[:, 2:4]) and FIXED body anatomy -- it NEVER reads MotorNet's internal per-step simulator state
# (live joint angle / live moment arms). The joint angle q is recovered from the obs fingertip by
# inverse kinematics (fixed link lengths L1, L2); the moment arms M(q1) are a FIXED property of the
# limb -- joint-1 moment arms depend only on the elbow angle q1 (verified q0-independent) and joint-0
# moment arms are constant -- calibrated ONCE at build time by sampling the plant's fixed geometry.
# That one-time body measurement is the exact analog of force_head reading the fixed muscle `anchors`
# once. Every model in the benchmark sees the identical observation, so no model gets privileged
# information: the morphological head is a fixed body transform, not a peek at the simulator.
#
# Calibrated (sign -M, scales below): a pure spinal PD reflex driven THROUGH this fair head reaches
# 88% within 5 cm on the arm centre-out (cf. the point-mass force_head reflex at 59%). Raw width
# stays 3 (2-D force + co-contraction) on BOTH plants, so the plausible readout / Bfb / spinal reflex
# are byte-identical across plants.
ARM_G = 0.01          # muscle-tension -> excitation scale (calibrated with ARM_FSCALE)
ARM_FSCALE = 150.0    # endpoint-force scale: f = tanh(raw[:2]) * ARM_FSCALE
_ARM_MTAB = {}        # device -> (elbow_ub, Mtab): fixed moment-arm anatomy, calibrated once per device

def _arm_moment_table(env, n=64):
    """The limb's FIXED moment-arm anatomy M(q1), sampled once from the plant geometry across the
    elbow range. A one-time measurement of the body (like force_head's fixed anchors) -- never
    read per step. Cached per device so every head shares the one calibration."""
    key = str(env.device)
    if key not in _ARM_MTAB:
        ub = float(env.effector.pos_upper_bound[1])
        cal = make_arm_env(env.device); tab = []
        for q1 in th.linspace(0.0, ub, n):
            cal.reset(options={"batch_size": 1,
                               "joint_state": th.tensor([[1.0, float(q1), 0.0, 0.0]])})
            tab.append(cal.states["geometry"][0, 2:4, :].clone())   # (2, n_musc); q0-independent
        _ARM_MTAB[key] = (ub, th.stack(tab).to(env.device))
    return _ARM_MTAB[key]

def arm_force_head(env, f_scale=ARM_FSCALE, g=ARM_G):
    """Build the FAIR arm morphological head closure `(obs, raw3) -> n_muscles excitations`. Uses
    ONLY the obs fingertip + fixed anatomy (no internal simulator state). raw[:2] = 2-D endpoint
    force, raw[2] = co-contraction. `f_scale` may be a float or a learned nn.Parameter (Kinesis)."""
    L1 = float(env.effector.skeleton.L1); L2 = float(env.effector.skeleton.L2)
    reach2 = (L1 + L2) ** 2
    ub, Mtab = _arm_moment_table(env)
    eye = th.eye(2, device=env.device)
    def head(obs, raw3):
        P = obs[:, 2:4].detach()                                             # fingertip FROM OBS (sensed)
        x, y = P[:, 0], P[:, 1]
        r2 = (x * x + y * y).clamp(1e-6, reach2 - 1e-4)
        q1 = th.acos(((r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)).clamp(-1., 1.))   # elbow (inverse kinematics)
        q0 = th.atan2(y, x) - th.atan2(L2 * th.sin(q1), L1 + L2 * th.cos(q1))
        q01 = q0 + q1; s0, c0, s01, c01 = th.sin(q0), th.cos(q0), th.sin(q01), th.cos(q01)
        J = th.stack([th.stack([-L1 * s0 - L2 * s01, -L2 * s01], -1),
                      th.stack([ L1 * c0 + L2 * c01,  L2 * c01], -1)], -2)        # (b,2,2)
        M = Mtab[((q1.clamp(0., ub) / ub) * (Mtab.shape[0] - 1)).round().long()]  # fixed M(q1) (b,2,n_musc)
        f = th.tanh(raw3[:, :2]) * f_scale
        tau = th.einsum("bji,bj->bi", J, f)                                      # J^T f -> joint torque
        Mp = M.transpose(1, 2) @ th.linalg.inv(M @ M.transpose(1, 2) + 1e-6 * eye)
        T = -th.einsum("bmj,bj->bm", Mp, tau)                                    # -M^+ tau -> tensions
        c = th.sigmoid(raw3[:, 2:3])
        return (th.relu(T) * g + c).clamp(0., 1.)
    return head


def morph_head(env, f_scale=650., anchors=ANCHORS):
    """The ONE morphological head, plant-aware: point mass -> Cartesian `force_head`; arm -> the
    joint-torque/moment-arm head. The plausible family (via configure), Dendritron and Kinesis all
    wear this, so the same morphological-computation identity holds on either plant."""
    if env.action_space.shape[0] == 4:
        return lambda obs, raw3: force_head(obs, raw3, f_scale=f_scale, anchors=anchors)
    return arm_force_head(env)


class _ArmReach(ReachEnv):
    """ReachEnv that records each reach's START and DIRECTION, so model activity can be
    condition-averaged into the monkey's center-out directions (the S1/M1 comparison in
    4-monkey-net). Same input/output interface the monkey had: the target (goal) is the input,
    the fingertip trajectory + muscle activations are the output -- input<->input, output<->output."""
    def reset(self, *a, **k):
        obs, info = super().reset(*a, **k)
        self._start = self.states["fingertip"].detach().clone()
        info["start"] = self._start
        return obs, info
    def reach_dir(self):
        """(B,) reach direction in degrees 0..360, start->goal (binned into the monkey's directions)."""
        v = self.goal[..., :2] - self._start
        return (th.rad2deg(th.atan2(v[:, 1], v[:, 0])) % 360.0)


def make_arm_env(device=DEVICE, **kwargs):
    """MotorNet RigidTendonArm26 (monkey-matched 2-joint, 6-muscle arm) as a fair-setup ReachEnv --
    the plant the 4-monkey-net arm centre-out benchmark runs on, sharing motor_zoo's reward/obs."""
    arm = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    return env_to(_ArmReach(effector=arm, max_ep_duration=1., **kwargs), device)

# reservoir config for the plausible rules (won the sweep in 4-tuning-net.ipynb)
# 4096 to MATCH plausible_learners.RES_NR. At 2048 Dendritron's plastic readout was
# 3*(2048+12)+3 = 6,183 -- HALF the 12,327 every other local rule gets, which quietly broke
# the equal-parameter premise the whole benchmark rests on.
RES_NR, RES_RHO, RES_A, RES_SIN, RES_LR = 4096, 1.1, 0.5, 1.0, 0.05

# ===== from t50_mass.py =============================================
# ==============================================================================
# VIII. 1d  Randomised ball weight: training distribution + REAL out-of-distribution
# ------------------------------------------------------------------------------
# The fair-redo trained every rule on a single ball (mass = 1 kg), so the only
# "out-of-distribution" test was a heavier ball -- but a policy that only ever saw
# 1 kg has no reason to generalise, and "2x mass" is a weak probe. Here we do it
# properly, the way the NeuroAI table actually asks:
#
#   * TRAIN across a *range* of ball weights, drawn fresh per episode. The SAME
#     fixed set is used for every method, so no rule gets an easier ball. A rule
#     that is robust has to work across all of them, not overfit one mass.
#   * ZERO-SHOT / OOD is then genuinely out-of-distribution: ball weights LIGHTER
#     and HEAVIER than anything trained on, plus force fields and weakened muscles
#     the policy never felt. This is the "zero-shot generalisation" row.
#   * CONTINUAL learning becomes a ball-weight curriculum: master a light ball,
#     then a heavy one, then re-test the light -- does the new skill erase the old?
#
# Mechanism (validated): PointMass._integrate does new_vel += F*dt / mass, and
# `skeleton.mass` is a plain attribute that broadcasts, so a (B,1) tensor gives every
# parallel episode its own ball. reset() does NOT restore it, so we (re)assign on each
# reset -- that is the whole reason MassReach overrides reset rather than setting once.
# ==============================================================================
# Ten ball weights spanning 0.5 - 2.5 kg (5x spread). TWO of them -- 1.2 and 2.1 kg -- are HELD OUT:
# no method ever trains on them, and held-out accuracy/completion is measured ON THEM. That makes the
# accuracy column a real INTERPOLATION-generalisation test on identical unseen physics for every
# method, instead of scoring models on the same weights they were fitted to.
# ---- CAPACITY PARITY -----------------------------------------------------------------------
# Absolute fairness requires the LEARNED capacity to be identical, not just the task. A fixed
# reservoir is not trainable, so a local rule's learned capacity is its readout only:
#   R x (Nr + O) + R = 3 x (4096 + 12) + 3 = 12,324 parameters.
# Solving 3H^2 + 46H + 4 = 12,324 for a GRU(12 -> H) + Linear(H -> 4) gives H = 57 (12,373, +0.4%).
# Every recurrent policy therefore uses FAIR_HIDDEN so all 13 learn the same number of weights.
# (Critics -- SHAC, SAC/TD3/Simba -- are algorithm-specific TRAINING machinery, not policy capacity;
#  they are reported separately in the Params column rather than hidden.)
FAIR_HIDDEN = 57

ALL_MASSES   = [0.5, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8, 2.1, 2.3, 2.5]     # kg, the 10 intervals
VAL_MASSES   = [1.2, 2.1]                                              # HELD OUT: validation / test only
TRAIN_MASSES = [m for m in ALL_MASSES if m not in VAL_MASSES]          # 8 weights, shared by every method
OOD_MASSES   = [0.25, 0.35, 3.0, 4.0]          # kg, strictly OUTSIDE [0.5, 2.5] -> extrapolation zero-shot
CONT_LIGHT   = [0.5, 0.7, 0.9]                  # continual task A: light balls (train weights only)
CONT_HEAVY   = [1.8, 2.3, 2.5]                  # continual task B: heavy balls (train weights only)


class MassReach(ReachEnv):
    """ReachEnv whose ball weight is (re)drawn from `mass_set` on every reset, with a
    reaching reward that does NOT trap sampling-based RL.

    Ball weight:
      random_mass=True  -> each episode/batch-element gets a uniform random draw from
                           the set (training: the policy must handle the whole range).
      random_mass=False -> the set is tiled deterministically across the batch, so the
                           held-out eval covers every mass reproducibly (accuracy / OOD).
      A one-element mass_set pins the whole batch to that single ball (per-mass OOD).
      mass_set=None      -> stock 1 kg behaviour.

    Reward = MINUS MOTORNET'S OWN LOSS, exactly:

        r = -sum(|fingertip - goal|)          <->   L = mean(sum(|fingertip - goal|))

    so a value-based rule maximising reward and a gradient rule minimising the loss are
    optimising the SAME function. This matters more than it looks. The previous shaped
    reward carried an effort penalty and a non-differentiable on-target bonus, and its own
    docstring conceded that the analytic-gradient rules "ignore it and optimise essentially
    the old objective" -- i.e. the environment itself was handing the two families different
    objectives, which is precisely the confound this benchmark exists to remove.

    The dropped effort term is not missed: it was added to stop model-free RL going limp,
    but going limp is only attractive when effort is PENALISED. Under pure position error,
    doing nothing scores as badly as possible.
    """
    def __init__(self, *a, mass_set=None, random_mass=False, **k):
        super().__init__(*a, **k)
        self.mass_set = None if mass_set is None else th.as_tensor(mass_set, dtype=th.float32)
        self.random_mass = random_mass
        self.reward_range = (-(1.0 + 0.02), 1.0)   # on-target bonus lifts the ceiling to +1
        self._log = False; self._mass_idx = None   # per-ball-weight TRAIN logging (for the learning curves)

    def reward(self, action):
        """-(MotorNet's L1 position error). Identical function to motor_core.StepCtx.loss()."""
        ft = self.states["fingertip"]
        return -th.sum(th.abs(self.goal[..., :ft.shape[-1]] - ft), dim=-1, keepdim=True)

    def reset(self, *a, **k):
        obs, info = super().reset(*a, **k)
        if self.mass_set is not None:
            B = self.states["fingertip"].shape[0]
            ms = self.mass_set.to(self.states["fingertip"].device)
            idx = (th.randint(0, len(ms), (B,), device=ms.device) if self.random_mass
                   else th.arange(B, device=ms.device) % len(ms))
            self.effector.skeleton.mass = ms[idx][:, None]
            self._mass_idx = idx
        return obs, info

    # --- per-ball-weight TRAIN accumulators (GPU-side, no per-step host sync) --------
    def enable_log(self):
        K = 0 if self.mass_set is None else len(self.mass_set)
        dev = self.effector.skeleton.mass.device if th.is_tensor(self.effector.skeleton.mass) else DEVICE
        self._log = True; self._K = K
        self._acc_r = th.zeros(K, device=dev); self._acc_d = th.zeros(K, device=dev); self._acc_n = th.zeros(K, device=dev)

    def read_log(self):
        """Mean TRAIN reward (cm distance) per ball weight since the last read; then reset."""
        n = self._acc_n.clamp(min=1.0)
        rew = (self._acc_r / n).tolist(); dcm = (100.0 * self._acc_d / n).tolist(); cnt = self._acc_n.tolist()
        self._acc_r.zero_(); self._acc_d.zero_(); self._acc_n.zero_()
        return {float(self.mass_set[i]): dict(ret=rew[i], err=dcm[i], n=cnt[i]) for i in range(self._K)}

    def step(self, action, **kw):
        obs, r, terminated, truncated, info = super().step(action, **kw)
        if self._log and self._mass_idx is not None:
            self._acc_r.index_add_(0, self._mass_idx, info["reward"].reshape(-1))
            self._acc_d.index_add_(0, self._mass_idx, info["dist"].reshape(-1))
            self._acc_n.index_add_(0, self._mass_idx, th.ones_like(info["dist"].reshape(-1)))
        return obs, r, terminated, truncated, info


def make_mass_env(dev, mass_set, random_mass=False, **kw):
    return env_to(MassReach(effector=mn.effector.ReluPointMass24(), max_ep_duration=1.,
                            mass_set=mass_set, random_mass=random_mass, **kw), dev)


# ---- REAL out-of-distribution suite -----------------------------------------
def ood_conditions(dev):
    """(name, env_factory(dev), step_kwargs, mutator) -- every one is unseen in training.
    Ball weights are outside [0.5, 2.0]; force fields and weak muscles are novel dynamics."""
    conds = [(f"ball {m} kg", (lambda d, m=m: make_mass_env(d, [m])), {}, None) for m in OOD_MASSES]
    conds += [
        ("force field +x", (lambda d: make_mass_env(d, [1.0])), {"endpoint_load": _load(4.0, 0.0, dev)}, None),
        ("force field -y", (lambda d: make_mass_env(d, [1.0])), {"endpoint_load": _load(0.0, -4.0, dev)}, None),
        ("curl load",      (lambda d: make_mass_env(d, [1.0])), {"endpoint_load": _load(3.0, 3.0, dev)}, None),
        ("weak muscles",   (lambda d: make_mass_env(d, [1.0])), {}, lambda e: e.effector.muscle.max_iso_force.mul_(0.5)),
    ]
    return conds


def zero_shot(make_env_fn, learner, dev):
    """REAL OOD eval: ball weights outside the trained range + force fields + weak muscles.
    (Same name/signature the run cell already calls; make_env_fn is ignored -- each
    condition builds its own mass-aware env so the perturbation actually survives reset.)"""
    out = {}
    for name, fac, skw, mut in ood_conditions(dev):
        e = fac(dev)
        if mut is not None: mut(e)
        out[name] = evaluate(e, learner, step_kwargs=skw)
    return out


class MassProbe(Probe):
    """Probe that, on the same schedule, ALSO records per-ball-weight TRAIN and held-out EVAL
    (reward, cm error, completion) so the notebook can draw the learning curve per ball weight
    and expose the train-vs-eval gap (overfitting). TRAIN numbers are read GPU-side off the
    training env's accumulators (`train_env.read_log()`); EVAL is a held-out rollout per mass,
    wrapped in RNG save/restore so it does not perturb the training stream (models stay identical)."""
    def __init__(self, env, every_eps, budget, train_env=None, masses=None, batch=256):
        super().__init__(env, every_eps, budget, batch)
        self.train_env = train_env
        self.masses = list(masses) if masses is not None else list(TRAIN_MASSES)
        self.eval_envs = {m: make_mass_env(DEVICE, [m]) for m in self.masses}
        self.mass_curve = []                                   # [dict(eps, train={m:..}, eval={m:..})]
        if train_env is not None and hasattr(train_env, "enable_log"): train_env.enable_log()

    def __call__(self, learner, episodes, force=False):
        ran = super().__call__(learner, episodes, force)       # base overall-eval (unchanged)
        if ran:
            train = self.train_env.read_log() if (self.train_env is not None and hasattr(self.train_env, "read_log")) else {}
            rng = th.get_rng_state(); nrng = np.random.get_state()
            crng = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
            ev = {}
            for m, e in self.eval_envs.items():
                mm = eval_metrics(e, learner, batch=256)
                ev[m] = dict(ret=mm["ret"], err=mm["err_cm"], compl=mm["completion"])
            th.set_rng_state(rng); np.random.set_state(nrng)
            if crng is not None: th.cuda.set_rng_state_all(crng)
            self.mass_curve.append(dict(eps=episodes, train=train, eval=ev))
        return ran

# ===== from t29.py ==================================================
# ==============================================================================
# VIII. 2  BPTT-GRU and SAC -- with the fair force head
# ------------------------------------------------------------------------------
# GRUForce: a GRU backbone read out as a 3-D endpoint force, trained by analytic policy
# gradient (BPTT through the differentiable plant). This is the tuned BPTT-GRU, and the
# base class the KINESIS entry reuses. It is also the recurrent DEMONSTRATOR the plausible
# rules imitate (raw_from = teacher-forced force command).
# SAC: soft actor-critic on the same force head -- model-free, off-policy. Kept honest:
# it does NOT use the plant gradient, and the redo shows it lags (credit assignment, not
# the action space, is its bottleneck).
# ==============================================================================

class GRUForce(nn.Module, Learner):
    """GRU controller trained by analytic policy gradient. Parameterised by its action head:
      * RAW=3 + `force_head`  -> the MORPHOLOGICAL demonstrator (plausible). This is the teacher
        the local-plausible rules imitate; it is NOT a non-plausible table entry by itself.
      * RAW=4 + `muscle_head` -> the MotorNet muscle baseline (see MuscleGRU / BPTTGRU below)."""
    name, cite, kind, wins = "BPTT-GRU (morphological demonstrator)", "Codol+24 MotorNet; KINESIS action head", "global-gradient", ""
    RAW = 3
    HEAD = staticmethod(force_head)
    def __init__(self, env, hidden=FAIR_HIDDEN, lr=1e-3):
        super().__init__(); self.dev = env.device; self.hidden = hidden
        # plant-agnostic muscle head: RAW=3 is the morphological force head (point-mass only);
        # any other RAW is a direct muscle head, whose width is the plant's muscle count
        # (4 point-mass / 6 arm). One definition now serves both plants.
        if self.RAW != 3: self.RAW = env.action_space.shape[0]
        self.gru = nn.GRU(env.observation_space.shape[0], hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, self.RAW)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0); nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0); nn.init.zeros_(self.gru.bias_hh_l0)
        nn.init.xavier_uniform_(self.fc.weight)
        # MotorNet's own init: output bias -5 so training starts with low output forces
        # ("a more stable situation" -- examples/4-train-net.ipynb). The morphological head
        # applies the same idea to its co-contraction channel only.
        nn.init.constant_(self.fc.bias, -5.0)
        if self.RAW == 3:
            with th.no_grad(): self.fc.bias[0] = self.fc.bias[1] = 0.0; self.fc.bias[2] = -3.0
        self.mu, self.sig = obs_norm(env); self.to(self.dev)
        self.opt = th.optim.Adam(self.parameters(), lr=lr)
    def init_state(self, B): return th.zeros(1, B, self.hidden, device=self.dev)
    def act(self, obs, h, explore=False):
        y, h = self.gru(((obs - self.mu) / self.sig)[:, None, :], h)
        return self.HEAD(obs, self.fc(y).squeeze(1)), h
    @th.no_grad()
    def raw_from(self, obs, t):
        if t == 0: self._th = th.zeros(1, obs.shape[0], self.hidden, device=self.dev)
        y, self._th = self.gru(((obs - self.mu) / self.sig)[:, None, :], self._th)
        return self.fc(y).squeeze(1)
    # ---- shared-core contract -------------------------------------------------------------
    def forward(self, obs, h):
        y, h = self.gru(((obs - self.mu) / self.sig)[:, None, :], h)
        return self.fc(y).squeeze(1), h, None

    def on_episode_start(self, B): self._loss = 0.0
    def on_step(self, c):          self._loss = self._loss + c.loss()      # THE shared objective
    def on_episode_end(self):
        self.opt.zero_grad(set_to_none=True); self._loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.opt.step()

    def fit(self, env, budget, probe, batch=256, teacher=None):
        """MotorNet's setup, via the shared core. `teacher` is accepted and ignored: there is no
        demonstrator any more, and silently taking a different path when one was passed is
        exactly how this benchmark previously ended up with per-family objectives."""
        import motor_core as _core
        return _core.train(self, env, budget, probe, batch=batch, grad=True)

    def _fit_reward_unused(self, env, budget, probe, batch=256):

        eps = 0
        while eps < budget:
            h = self.init_state(batch); obs, info = env.reset(options={"batch_size": batch}); R = 0.
            for t in range(int(env.max_ep_duration / env.dt)):
                a, h = self.act(obs, h); obs, r, term, trunc, info = env.step(a); R = R + r.mean()
            self.opt.zero_grad(set_to_none=True); (-R).backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.opt.step()
            eps += batch; probe(self, eps)
        probe(self, eps, force=True)
    def ops(self, env):
        c = OpCounter(); c.dense(env.observation_space.shape[0] + self.hidden, 3 * self.hidden); c.dense(self.hidden, self.RAW); return c

    def update_ops(self, env):
        """BPTT / analytic policy gradient: a backward pass costs ~2x the forward and must be run
        through the WHOLE episode from stored activations. Charged per control step."""
        f = self.ops(env); c = OpCounter(); c.mac = 2.0 * f.mac; return c


class MuscleGRU(GRUForce):
    """The MotorNet baseline (unchanged reference): GRU -> 4 muscle activations DIRECTLY via a
    sigmoid, no morphological head. This is the non-plausible 'BPTT-GRU' AND the demonstrator the
    non-plausible deep-RL learners bootstrap from (weight-copied)."""
    name, cite, kind, wins = "BPTT-GRU (MotorNet muscle head)", "Codol+24 MotorNet (baseline)", "global-gradient", ""
    RAW = 4
    HEAD = staticmethod(muscle_head)


class BPTTGRU(MuscleGRU):
    """The non-plausible BPTT-GRU baseline = MotorNet muscle head (name kept for the registry)."""
    pass


# ---- model-free deep-RL building blocks (shared by SAC / FastTD3 / Simba) ----
def _mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))

class _ReplayMF:
    def __init__(self, cap, O, dev):
        self.cap, self.dev = cap, dev
        self.bo = th.zeros(cap, O, device=dev); self.ba = th.zeros(cap, 3, device=dev)
        self.br = th.zeros(cap, 1, device=dev); self.bn = th.zeros(cap, O, device=dev)
        self.bd = th.zeros(cap, 1, device=dev); self.ptr = 0; self.full = False
    def store(self, o, a, r, n, d):
        B = o.shape[0]; idx = (th.arange(B, device=self.dev) + self.ptr) % self.cap
        self.bo[idx] = o; self.ba[idx] = a; self.br[idx] = r; self.bn[idx] = n; self.bd[idx] = d
        self.ptr = (self.ptr + B) % self.cap; self.full = self.full or self.ptr < B
    def sample(self, bs):
        hi = self.cap if self.full else max(1, self.ptr); i = th.randint(0, hi, (bs,), device=self.dev)
        return self.bo[i], self.ba[i], self.br[i], self.bn[i], self.bd[i]


class SAC(nn.Module, Learner):
    """Soft Actor-Critic with the force head: stochastic squashed-Gaussian actor,
    entropy-regularised, twin critics. Off-policy, model-free -- no plant gradient."""
    name, cite, kind, wins = "SAC + force head", "Haarnoja+18 SAC; KINESIS head", "global-gradient", ""
    def __init__(self, env, hidden=256, lr=3e-4, gamma=0.99, tau=5e-3, alpha=0.2, buf=400_000, bs=1024, warm=5000, upd=2):
        super().__init__(); self.dev = env.device; self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.bs, self.warm, self.upd = bs, warm, upd
        O = env.observation_space.shape[0]; self.O = O; self.hidden = hidden
        self.actor = _mlp(O, 6, hidden).to(self.dev)
        with th.no_grad(): self.actor[-1].bias[2] = -1.5
        self.q1 = _mlp(O + 3, 1, hidden).to(self.dev); self.q2 = _mlp(O + 3, 1, hidden).to(self.dev)
        self.q1t = _mlp(O + 3, 1, hidden).to(self.dev); self.q2t = _mlp(O + 3, 1, hidden).to(self.dev)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.mu, self.sig = obs_norm(env)
        self.oa = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.oq = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.rb = _ReplayMF(buf, O, self.dev)
    def _nz(self, o): return (o - self.mu) / self.sig
    def init_state(self, B): return None
    def _dist(self, o):
        h = self.actor(self._nz(o)); return h[:, :3], h[:, 3:].clamp(-5, 2).exp()
    def _sample(self, o):
        m, s = self._dist(o); u = m + s * th.randn_like(m); a = th.tanh(u)
        logp = (-0.5 * ((u - m) / s) ** 2 - s.log() - 0.9189).sum(-1, keepdim=True)
        logp = logp - th.log(1 - a ** 2 + 1e-6).sum(-1, keepdim=True)
        return a, logp
    def act(self, obs, st, explore=False):
        if explore: a, _ = self._sample(obs)
        else: m, _ = self._dist(obs); a = th.tanh(m)
        return force_head(obs, a), st
    def _update(self):
        o, a, r, n, d = self.rb.sample(self.bs)
        with th.no_grad():
            na, nlp = self._sample(n); zn = th.cat([self._nz(n), na], -1)
            y = r + self.gamma * (1 - d) * (th.min(self.q1t(zn), self.q2t(zn)) - self.alpha * nlp)
        z = th.cat([self._nz(o), a], -1)
        lq = F.mse_loss(self.q1(z), y) + F.mse_loss(self.q2(z), y)
        self.oq.zero_grad(set_to_none=True); lq.backward(); self.oq.step()
        pa, plp = self._sample(o); zp = th.cat([self._nz(o), pa], -1)
        la = (self.alpha * plp - th.min(self.q1(zp), self.q2(zp))).mean()
        self.oa.zero_grad(set_to_none=True); la.backward(); self.oa.step()
        with th.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
    def fit(self, env, budget, probe, batch=256):
        eps = 0; n = int(env.max_ep_duration / env.dt)
        while eps < budget:
            obs, info = env.reset(options={"batch_size": batch}); obs = obs.detach()
            for t in range(n):
                with th.no_grad(): raw, _ = self._sample(obs)
                a = force_head(obs, raw); nobs, r, term, trunc, info = env.step(a); nobs = nobs.detach()
                done = th.zeros(batch, 1, device=self.dev) if t < n - 1 else th.ones(batch, 1, device=self.dev)
                self.rb.store(obs.detach(), raw.detach(), r.detach(), nobs, done); obs = nobs
                if self.rb.full or self.rb.ptr > self.warm:
                    for _ in range(self.upd): self._update()
            eps += batch; probe(self, eps)
        probe(self, eps, force=True)
    def ops(self, env):
        c = OpCounter(); c.dense(self.O, self.hidden); c.dense(self.hidden, self.hidden); c.dense(self.hidden, 3); return c

# ===== from t30.py ==================================================
# ==============================================================================
# VIII. 3  SHAC -- short-horizon actor-critic, with the fair force head
# ------------------------------------------------------------------------------
# Recurrent (GRU) actor: the tuning study showed a feedforward force controller cannot
# solve this double integrator even with the exact plant gradient (~92 cm), so SHAC gets
# a GRU actor. Horizon-truncated APG through the plant + a GAE-trained critic bootstraps
# the tail. This is a gradient method: it DOES use the differentiable plant.
# ==============================================================================

class SHAC(nn.Module, Learner):
    name, cite, kind, wins = "SHAC-style short-horizon gradient (MotorNet muscle head)", \
        "Xu+22 SHAC: 16-step truncated differentiable-sim BPTT (value critic dropped for equal-capacity parity)", "global-gradient", ""
    RAW = 4                                     # non-plausible: direct 4-muscle head (no morphological advantage)
    HEAD = staticmethod(muscle_head)
    def __init__(self, env, hidden=FAIR_HIDDEN, horizon=50, gamma=0.99, lam=0.95, lr_a=1e-3, lr_c=5e-4, tau=0.2):
        super().__init__(); self.dev = env.device; self.h, self.gamma, self.lam, self.tau = horizon, gamma, lam, tau
        O = env.observation_space.shape[0]; self.hidden = hidden; self.O = O
        self.RAW = env.action_space.shape[0]        # muscle head: adapt to the plant (4 point-mass / 6 arm)
        self.gru = nn.GRU(O, hidden, 1, batch_first=True); self.fc = nn.Linear(hidden, self.RAW)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0); nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0); nn.init.zeros_(self.gru.bias_hh_l0)
        nn.init.xavier_uniform_(self.fc.weight)
        # SAME init as BPTT-GRU / MotorNet (fc.bias=-5 -> muscles start near 0, compliant). SHAC
        # previously used bias=0 (muscles start at 0.5, stiff), which is a DIFFERENT starting
        # point and confounded the SHAC-vs-BPTT-GRU comparison: the only difference between them
        # must be the algorithm (SHAC's truncated 16-step horizon), not the initialisation.
        nn.init.constant_(self.fc.bias, -5.0)
        if self.RAW == 3:
            with th.no_grad(): self.fc.bias[0] = self.fc.bias[1] = 0.0; self.fc.bias[2] = -3.0
        # The critic is built ONLY on the demonstrator path (see _mk_critic). Under the shared
        # imitation objective there is no value target to fit, so a critic here would be 36,610
        # dead parameters -- 4x every other entry's budget, breaking the equal-capacity premise
        # of the comparison. SHAC's benchmark entry is exactly 12,373 params, same as BPTT-GRU.
        self.lr_c = lr_c
        self.mu, self.sig = obs_norm(env); self.to(self.dev)
        self.opt_a = th.optim.Adam(list(self.gru.parameters()) + list(self.fc.parameters()), lr=lr_a)

    def _mk_critic(self):
        mk = lambda o: nn.Sequential(nn.Linear(self.O, 128), nn.ELU(),
                                     nn.Linear(128, 128), nn.ELU(), nn.Linear(128, o)).to(self.dev)
        self.critic = mk(1); self.critic_t = mk(1); self.critic_t.load_state_dict(self.critic.state_dict())
        with th.no_grad(): self.critic[-1].bias.fill_(V0_INIT); self.critic_t[-1].bias.fill_(V0_INIT)
        self.opt_c = th.optim.Adam(self.critic.parameters(), lr=self.lr_c)
    def _nz(self, o): return (o - self.mu) / self.sig
    def init_state(self, B): return th.zeros(1, B, self.hidden, device=self.dev)
    def _raw(self, obs, hstate):
        y, hstate = self.gru(self._nz(obs)[:, None, :], hstate); return self.fc(y.squeeze(1)), hstate
    def act(self, obs, hstate, explore=False):
        raw, hstate = self._raw(obs, hstate); return self.HEAD(obs, raw), hstate
    # ---- shared-core contract: identity = SHORT-HORIZON (truncated) backprop ---------------
    # SHAC's defining move is that it does NOT backprop the whole episode: it cuts the analytic
    # gradient at a short horizon and takes an update per window. That truncation survives the
    # shared objective intact -- it is a property of the CREDIT ASSIGNMENT, not of the loss --
    # and it is what separates SHAC from BPTT-GRU (identical params, identical objective, one
    # window vs the full 100-step episode). The critic that would bootstrap past the window has
    # no target under an imitation loss, so it is dropped rather than faked.
    bptt_horizon = 16

    def forward(self, obs, hstate):
        raw, hstate = self._raw(obs, hstate)
        return raw, hstate, None

    def on_episode_start(self, B): self._loss = 0.0
    def on_step(self, c):          self._loss = self._loss + c.loss()      # THE shared objective
    def on_horizon_end(self):
        if not th.is_tensor(self._loss): return
        self.opt_a.zero_grad(set_to_none=True); self._loss.backward()
        nn.utils.clip_grad_norm_(list(self.gru.parameters()) + list(self.fc.parameters()), 1.0)
        self.opt_a.step(); self._loss = 0.0
    def on_episode_end(self):
        if th.is_tensor(self._loss): self.on_horizon_end()

    def fit(self, env, budget, probe, batch=256, teacher=None):
        """MotorNet's setup, via the shared core. `teacher` is accepted and ignored: there is no
        demonstrator any more, and silently taking a different path when one was passed is
        exactly how this benchmark previously ended up with per-family objectives."""
        import motor_core as _core
        return _core.train(self, env, budget, probe, batch=batch, grad=True)

    def _fit_reward_unused(self, env, budget, probe, batch=256):

        self._mk_critic()          # value head exists only where a value target does
        eps, n = 0, int(env.max_ep_duration / env.dt); ap = list(self.gru.parameters()) + list(self.fc.parameters())
        while eps < budget:
            obs, info = env.reset(options={"batch_size": batch}); obs = obs.detach()
            hstate = self.init_state(batch); t = 0; TO, TR = [], []
            while t < n:
                h = min(self.h, n - t); rw, disc = 0., 1.
                for k in range(h):
                    raw, hstate = self._raw(obs, hstate); a = self.HEAD(obs, raw); TO.append(obs.detach())
                    obs, r, term, trunc, info = env.step(a); TR.append(r.detach()); rw = rw + disc * r.mean(); disc *= self.gamma
                loss = -(rw + disc * self.critic_t(self._nz(obs)).mean())
                self.opt_a.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(ap, 1.0); self.opt_a.step()
                obs = obs.detach(); hstate = hstate.detach(); detach_env_state(env); t += h
            with th.no_grad():
                Ob = th.stack(TO, 0); Rr = th.stack(TR, 0).squeeze(-1); T = Ob.shape[0]
                V = self.critic_t(self._nz(Ob.reshape(-1, self.O))).reshape(T, -1)
                tgt = th.zeros_like(Rr); nxt = self.critic_t(self._nz(obs)).squeeze(-1); gae = th.zeros_like(nxt)
                for k in reversed(range(T)):
                    dlt = Rr[k] + self.gamma * nxt - V[k]; gae = dlt + self.gamma * self.lam * gae; tgt[k] = gae + V[k]; nxt = V[k]
            Of = Ob.reshape(-1, self.O); Tf = tgt.reshape(-1, 1)
            for _ in range(8):
                l = F.mse_loss(self.critic(self._nz(Of)), Tf); self.opt_c.zero_grad(set_to_none=True); l.backward(); self.opt_c.step()
            with th.no_grad():
                for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            eps += batch; probe(self, eps)
        probe(self, eps, force=True)
    def ops(self, env):
        c = OpCounter(); c.dense(self.O + self.hidden, 3 * self.hidden); c.dense(self.hidden, self.RAW); return c

    def update_ops(self, env):
        """BPTT / analytic policy gradient: a backward pass costs ~2x the forward and must be run
        through the WHOLE episode from stored activations. Charged per control step."""
        f = self.ops(env); c = OpCounter(); c.mac = 2.0 * f.mac; return c

# ===== from t31.py ==================================================
# ==============================================================================
# VIII. 4  FastTD3 -- parallel-simulation TD3, with the fair force head
# ------------------------------------------------------------------------------
# Twin delayed DDPG: massively-parallel envs fill one replay buffer, twin critics +
# target-policy smoothing. Feedforward actor over the 3-D force command. Off-policy,
# model-free -- the fair action space, but no plant gradient.
# ==============================================================================

class FastTD3(nn.Module, Learner):
    name, cite, kind, wins = "FastTD3 + force head", "Fujimoto+18 TD3; parallel-sim 2025; KINESIS head", "global-gradient", ""
    def __init__(self, env, hidden=256, lr=3e-4, gamma=0.99, tau=5e-3, pol_noise=0.2, noise_clip=0.5,
                 expl=0.3, buf=400_000, bs=1024, warm=5000, upd=2):
        super().__init__(); self.dev = env.device; self.gamma, self.tau = gamma, tau
        self.pol_noise, self.noise_clip, self.expl, self.bs, self.warm, self.upd = pol_noise, noise_clip, expl, bs, warm, upd
        O = env.observation_space.shape[0]; self.O = O; self.hidden = hidden
        self.actor = _mlp(O, 3, hidden).to(self.dev); self.actor_t = _mlp(O, 3, hidden).to(self.dev)
        with th.no_grad(): self.actor[-1].bias[2] = -1.5
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = _mlp(O + 3, 1, hidden).to(self.dev); self.q2 = _mlp(O + 3, 1, hidden).to(self.dev)
        self.q1t = _mlp(O + 3, 1, hidden).to(self.dev); self.q2t = _mlp(O + 3, 1, hidden).to(self.dev)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.mu, self.sig = obs_norm(env)
        self.oa = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.oq = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.rb = _ReplayMF(buf, O, self.dev)
    def _nz(self, o): return (o - self.mu) / self.sig
    def init_state(self, B): return None
    def act(self, obs, st, explore=False):
        raw = th.tanh(self.actor(self._nz(obs)))
        if explore: raw = (raw + self.expl * th.randn_like(raw)).clamp(-1, 1)
        return force_head(obs, raw), st
    def _update(self):
        o, a, r, n, d = self.rb.sample(self.bs)
        with th.no_grad():
            na = th.tanh(self.actor_t(self._nz(n))); ns = (self.pol_noise * th.randn_like(na)).clamp(-self.noise_clip, self.noise_clip)
            na = (na + ns).clamp(-1, 1); zn = th.cat([self._nz(n), na], -1)
            y = r + self.gamma * (1 - d) * th.min(self.q1t(zn), self.q2t(zn))
        z = th.cat([self._nz(o), a], -1)
        lq = F.mse_loss(self.q1(z), y) + F.mse_loss(self.q2(z), y)
        self.oq.zero_grad(set_to_none=True); lq.backward(); self.oq.step()
        la = -self.q1(th.cat([self._nz(o), th.tanh(self.actor(self._nz(o)))], -1)).mean()
        self.oa.zero_grad(set_to_none=True); la.backward(); self.oa.step()
        with th.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
    def fit(self, env, budget, probe, batch=256):
        eps = 0; n = int(env.max_ep_duration / env.dt)
        while eps < budget:
            obs, info = env.reset(options={"batch_size": batch}); obs = obs.detach()
            for t in range(n):
                with th.no_grad():
                    raw = (th.tanh(self.actor(self._nz(obs))) + self.expl * th.randn(batch, 3, device=self.dev)).clamp(-1, 1)
                a = force_head(obs, raw); nobs, r, term, trunc, info = env.step(a); nobs = nobs.detach()
                done = th.zeros(batch, 1, device=self.dev) if t < n - 1 else th.ones(batch, 1, device=self.dev)
                self.rb.store(obs.detach(), raw.detach(), r.detach(), nobs, done); obs = nobs
                if self.rb.full or self.rb.ptr > self.warm:
                    for _ in range(self.upd): self._update()
            eps += batch; probe(self, eps)
        probe(self, eps, force=True)
    def ops(self, env):
        c = OpCounter(); c.dense(self.O, self.hidden); c.dense(self.hidden, self.hidden); c.dense(self.hidden, 3); return c

# ===== from t32.py ==================================================
# ==============================================================================
# VIII. 5  Simba -- residual-architecture TD3, with the fair force head
# ------------------------------------------------------------------------------
# Simba's "simplicity bias": residual + layer-normed actor/critic trunks that let a
# model-free RL agent scale gracefully. TD3 update rule on the 3-D force command.
# Off-policy, model-free -- fair action space, no plant gradient.
# ==============================================================================

class _SimbaBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.n = nn.LayerNorm(dim)
        self.f = nn.Sequential(nn.Linear(dim, hidden), nn.ELU(), nn.Linear(hidden, dim))
    def forward(self, x): return x + self.f(self.n(x))

class _SimbaNet(nn.Module):
    def __init__(self, i, o, dim=256, blocks=2):
        super().__init__(); self.inp = nn.Linear(i, dim)
        self.blocks = nn.ModuleList([_SimbaBlock(dim, dim) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, o))
    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks: x = b(x)
        return self.out(x)

class SimbaV2(FastTD3):
    """Simba (residual RL) on the force head: TD3 with residual/layer-normed trunks."""
    name, cite, kind, wins = "Simba + force head", "Lee+24 Simba (residual RL); TD3; KINESIS head", "global-gradient", ""
    def __init__(self, env, dim=256, lr=3e-4, gamma=0.99, tau=5e-3, pol_noise=0.2, noise_clip=0.5,
                 expl=0.3, buf=400_000, bs=1024, warm=5000, upd=2):
        nn.Module.__init__(self); self.dev = env.device; self.gamma, self.tau = gamma, tau
        self.pol_noise, self.noise_clip, self.expl, self.bs, self.warm, self.upd = pol_noise, noise_clip, expl, bs, warm, upd
        O = env.observation_space.shape[0]; self.O = O; self.hidden = dim
        self.actor = _SimbaNet(O, 3, dim).to(self.dev); self.actor_t = _SimbaNet(O, 3, dim).to(self.dev)
        with th.no_grad(): self.actor.out[-1].bias[2] = -1.5
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = _SimbaNet(O + 3, 1, dim).to(self.dev); self.q2 = _SimbaNet(O + 3, 1, dim).to(self.dev)
        self.q1t = _SimbaNet(O + 3, 1, dim).to(self.dev); self.q2t = _SimbaNet(O + 3, 1, dim).to(self.dev)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.mu, self.sig = obs_norm(env)
        self.oa = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.oq = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.rb = _ReplayMF(buf, O, self.dev)
    def act(self, obs, st, explore=False):
        raw = th.tanh(self.actor(self._nz(obs)))
        if explore: raw = (raw + self.expl * th.randn_like(raw)).clamp(-1, 1)
        return force_head(obs, raw), st
    def _update(self):
        o, a, r, n, d = self.rb.sample(self.bs)
        with th.no_grad():
            na = th.tanh(self.actor_t(self._nz(n))); ns = (self.pol_noise * th.randn_like(na)).clamp(-self.noise_clip, self.noise_clip)
            na = (na + ns).clamp(-1, 1); zn = th.cat([self._nz(n), na], -1)
            y = r + self.gamma * (1 - d) * th.min(self.q1t(zn), self.q2t(zn))
        z = th.cat([self._nz(o), a], -1)
        lq = F.mse_loss(self.q1(z), y) + F.mse_loss(self.q2(z), y)
        self.oq.zero_grad(set_to_none=True); lq.backward(); self.oq.step()
        la = -self.q1(th.cat([self._nz(o), th.tanh(self.actor(self._nz(o)))], -1)).mean()
        self.oa.zero_grad(set_to_none=True); la.backward(); self.oa.step()
        with th.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
    def ops(self, env):
        c = OpCounter(); c.dense(self.O, self.hidden); c.dense(self.hidden, self.hidden); c.dense(self.hidden, 3); return c

# ===== from t33.py ==================================================
# ==============================================================================
# VIII. 6  e-prop -- and the plausible substrate every local rule shares
# ------------------------------------------------------------------------------
# The tuning study established the STABLE plausible route: a FIXED random recurrent
# reservoir (a liquid / echo-state network -- a standard model of cortical recurrent
# dynamics) supplies temporal memory, and each rule trains ONLY a linear readout by its
# own local plasticity to imitate a recurrent demonstrator (the BPTT-GRU teacher's force
# command). No reward, no backprop-through-time; the readout objective is convex so the
# local rules converge. Recurrence is not LEARNED locally (that was unstable) -- it is
# read out of the fixed reservoir. Each named rule below = this substrate with its own
# eligibility signature (instantaneous three-factor vs. a low-pass eligibility trace).
#
# e-prop (Bellec+20): eligibility traces + a top-down learning signal. On the plastic
# readout the eligibility is a low-pass of presynaptic (granule) activity.
# ==============================================================================

class Reservoir(nn.Module, Learner):
    """Fixed liquid/echo-state reservoir + local plastic force readout, trained by a
    three-factor / eligibility rule to imitate a recurrent demonstrator. No reward, no BPTT."""
    kind = "local-plausible"
    def __init__(self, env, teacher, name="reservoir", cite="Maass 02 LSM; KINESIS head",
                 elig="delta", tau_e=0.03, Nr=RES_NR, rho=RES_RHO, a=RES_A, sin=RES_SIN,
                 lr=RES_LR, lam=1e-4, seed=0):
        super().__init__()
        self.name, self.cite, self.wins = name, cite, ""
        self.dev = env.device; self.teacher = teacher; self.a = a; self.lr = lr; self.lam = lam
        self.Nr = Nr; self.elig = elig; self.dt = env.dt; self.tau_e = tau_e
        self.O = env.observation_space.shape[0]
        g = th.Generator(device="cpu").manual_seed(seed)
        Wr = th.randn(Nr, Nr, generator=g); Wr *= rho / th.linalg.eigvals(Wr).abs().max().item()
        self.register_buffer("Wr", Wr.to(self.dev))
        self.register_buffer("Win", (sin * th.randn(Nr, self.O, generator=g)).to(self.dev))
        self.register_buffer("bres", (0.1 * th.randn(Nr, generator=g)).to(self.dev))
        self.register_buffer("W", th.zeros(3, Nr + self.O)); self.register_buffer("b", th.tensor([0., 0., -3.0]))
        self.mu, self.sig = obs_norm(env); self.to(self.dev)
    def _step(self, obs, h):
        nz = (obs - self.mu) / self.sig
        h = (1 - self.a) * h + self.a * th.tanh(h @ self.Wr.t() + nz @ self.Win.t() + self.bres)
        return h, th.cat([h, nz], -1)
    def init_state(self, B): return th.zeros(B, self.Nr, device=self.dev)
    def raw(self, z): return z @ self.W.t() + self.b
    def act(self, obs, h, explore=False):
        h, z = self._step(obs, h); return force_head(obs, self.raw(z)), h
    def fit(self, env, budget, probe, batch=256):
        eps, n = 0, int(env.max_ep_duration / env.dt)
        with th.no_grad():
            while eps < budget:
                h = self.init_state(batch); obs, info = env.reset(options={"batch_size": batch})
                e = th.zeros(batch, self.Nr + self.O, device=self.dev)
                for t in range(n):
                    h, z = self._step(obs, h); out = self.raw(z)
                    err = self.teacher.raw_from(obs, t) - out            # three-factor error vs. demonstrator
                    if self.elig == "delta":
                        elig = z                                         # instantaneous (R-STDP / 3-factor Hebb / pred-coding)
                    else:
                        e = (1 - self.dt / self.tau_e) * e + (self.dt / self.tau_e) * z
                        elig = e                                         # low-pass eligibility trace (e-prop / RTRRL / BTSP)
                    self.W += self.lr / n * ((err[:, :, None] * elig[:, None, :]).mean(0) - self.lam * self.W)
                    obs, r, term, trunc, info = env.step(force_head(obs, out))
                eps += batch; probe(self, eps)
        probe(self, eps, force=True)
    def ops(self, env):
        c = OpCounter(); c.dense(self.Nr, self.Nr); c.dense(self.O, self.Nr); c.dense(self.Nr + self.O, 3); return c


class EProp(Reservoir):
    def __init__(self, env, teacher, **kw):
        super().__init__(env, teacher, name="e-prop (reservoir readout)",
                         cite="Bellec+20 e-prop (Nat.Commun.); reservoir + KINESIS head",
                         elig="trace", **kw)

# ===== from t34.py ==================================================
# ==============================================================================
# VIII. 7  RTRRL -- real-time recurrent RL, reservoir readout form
# ------------------------------------------------------------------------------
# RTRRL / RFLO (Murray 19): local eligibility traces + random feedback, real-time (no
# BPTT). On the fixed reservoir its signature is the low-pass eligibility trace on the
# plastic readout, imitating the recurrent demonstrator.
# ==============================================================================

class RTRRL(Reservoir):
    def __init__(self, env, teacher, **kw):
        super().__init__(env, teacher, name="RTRRL (reservoir readout)",
                         cite="Murray 19 RFLO (eLife); real-time recurrent RL; reservoir + KINESIS head",
                         elig="trace", **kw)

# ===== from t35.py ==================================================
# ==============================================================================
# VIII. 8  BTSP -- behavioral-timescale synaptic plasticity, reservoir readout form
# ------------------------------------------------------------------------------
# BTSP (Bittner+17): a seconds-long dendritic plateau binds a whole recent trajectory to
# a learning signal. On the fixed reservoir its signature is the (behavioral-timescale)
# eligibility trace on the plastic readout -- the plateau -- imitating the demonstrator.
# ==============================================================================

class BTSP(Reservoir):
    def __init__(self, env, teacher, **kw):
        super().__init__(env, teacher, name="BTSP (reservoir readout)",
                         cite="Bittner+17 BTSP (Science); dendritic plateau; reservoir + KINESIS head",
                         elig="trace", **kw)

# ===== from t36.py ==================================================
# ==============================================================================
# VIII. 9  KINESIS -- morphological-computation policy (biologically plausible)
# ------------------------------------------------------------------------------
#   Simos, Chiappa & Mathis (2025/26), "KINESIS", arXiv:2503.14637, ICRA 2026;
#   github.com/amathislab/Kinesis. Morphological-computation strand: Wochner+23 (CoRL),
#   Ghazi-Zahedi+16 (Front.Robot.AI), Hogan 84 (co-contraction -> joint impedance).
#
# WHY THIS IS THE PLAUSIBLE ENTRY. The plausibility is MORPHOLOGICAL: the controller
# emits *intent* -- a 2-D endpoint force + a scalar co-contraction -- and the BODY's own
# geometry does the muscle coordinate transform. The map falls out of MotorNet's ODE, not
# a black box: muscle m pulls the mass toward its anchor A_m, so to realise a force f,
#   a_m = relu(d_m . f)/F_MAX + c ,  d_m = (A_m - P)/l_m  (from vision P + proprioception l).
# The relu is the one-sidedness of real muscle; c is co-contraction (endpoint impedance,
# Hogan 84). d_m uses only observed quantities, never privileged simulator state.
#
# The recurrent controller that emits the force intent is a small GRU. As in the original
# KINESIS study the intent-net is trained by analytic policy gradient -- the plausibility
# lives in the EMBODIMENT (the body computes), not in the weight update; the only thing
# that differs from the exact-gradient baseline is the morphological action parameterisation.
# NOTE (audit fix): f_scale is now a LEARNED parameter, not a constant hand-tuned on the
# held-out set. The previous 650 N value was selected to hit 100% on the held-out eval, which is
# test-set leakage; letting the policy learn its own force scale from the training reward removes
# that advantage. KINESIS remains a deliberate structural-prior entry (its plausibility is the
# fixed morphological force->muscle map), reported as its own family, not like-for-like with the
# geometry-free learners -- see docs/model_audit.md.
# ==============================================================================

class Kinesis(nn.Module, Learner):
    name = "KINESIS (morphological force control)"
    cite = "Simos+25 arXiv:2503.14637 ICRA26; github.com/amathislab/Kinesis; Hogan 84; Wochner+23"
    kind = "morphological"   # NOT a plausible LEARNER: fit() backprops through the differentiable
                         # plant (APG), exactly like BPTT-GRU. Its plausibility is the BODY
                         # (morphological force head), so it is its own family: it is the
                         # CONTROL that separates "morphological head" from "local learning rule".
    wins = "zero-shot generalisation (the body absorbs perturbations)"
    F_MAX = 500.0

    def __init__(self, env, hidden=FAIR_HIDDEN, lr=1e-3, f_scale=500.0):     # LEARNED, init at F_MAX
        super().__init__(); self.dev = env.device; self.hidden = hidden
        self.f_scale = nn.Parameter(th.tensor(float(f_scale)))   # learned from the task, not eval-tuned
        # morphological decode is plant-aware: point mass -> Cartesian anchors; arm -> the
        # joint-torque/moment-arm head (shares Kinesis's LEARNED f_scale, read live each step).
        self._armhead = None if env.action_space.shape[0] == 4 else arm_force_head(env, f_scale=self.f_scale)
        if self._armhead is None:
            A = env.effector._path_coordinates[0, :, 0::2].T
            self.register_buffer("anchors", th.tensor(A, dtype=th.float32))
        self.gru = nn.GRU(env.observation_space.shape[0], hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0); nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0); nn.init.zeros_(self.gru.bias_hh_l0)
        nn.init.xavier_uniform_(self.head.weight); nn.init.constant_(self.head.bias, 0.)
        with th.no_grad(): self.head.bias[2] = -3.0                 # start compliant
        self.mu, self.sig = obs_norm(env); self.to(self.dev)
        self.opt = th.optim.Adam(self.parameters(), lr=lr)
    def init_state(self, B): return th.zeros(1, B, self.hidden, device=self.dev)
    def _pull_dirs(self, obs):
        P = obs[:, 2:4]; l = obs[:, 4:8].clamp(min=1e-3)
        return (self.anchors[None, :, :] - P[:, None, :]) / l[:, :, None]
    def _decode(self, obs, raw):
        """Morphological head: raw -> endpoint force + co-contraction -> muscle excitations, via
        whichever plant geometry this env carries (point-mass anchors or the arm moment-arm head)."""
        if self._armhead is not None:
            return self._armhead(obs, raw)
        f = th.tanh(raw[:, :2]) * self.f_scale; c = th.sigmoid(raw[:, 2:3])
        pull = (self._pull_dirs(obs) * f[:, None, :]).sum(-1)
        return (F.relu(pull) / self.F_MAX + c).clamp(0., 1.)
    def act(self, obs, h, explore=False):
        y, h = self.gru(((obs - self.mu) / self.sig)[:, None, :], h)
        return self._decode(obs, self.head(y).squeeze(1)), h
    def HEAD(self, obs, raw):
        return self._decode(obs, raw)

    def forward(self, obs, h):
        y, h = self.gru(((obs - self.mu) / self.sig)[:, None, :], h)
        return self.head(y).squeeze(1), h, None

    def on_episode_start(self, B): self._loss = 0.0
    def on_step(self, c):          self._loss = self._loss + c.loss()      # THE shared objective
    def on_episode_end(self):
        self.opt.zero_grad(set_to_none=True); self._loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.opt.step()

    def fit(self, env, budget, probe, batch=256, teacher=None):
        """MotorNet's setup, via the shared core. `teacher` is accepted and ignored: there is no
        demonstrator any more, and silently taking a different path when one was passed is
        exactly how this benchmark previously ended up with per-family objectives."""
        import motor_core as _core
        return _core.train(self, env, budget, probe, batch=batch, grad=True)

    def _fit_reward_unused(self, env, budget, probe, batch=256):

        eps = 0
        while eps < budget:
            h = self.init_state(batch); obs, info = env.reset(options={"batch_size": batch}); R = 0.
            for t in range(int(env.max_ep_duration / env.dt)):
                a, h = self.act(obs, h); obs, r, term, trunc, info = env.step(a); R = R + r.mean()
            self.opt.zero_grad(set_to_none=True); (-R).backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.opt.step()
            eps += batch; probe(self, eps)
        probe(self, eps, force=True)
    def ops(self, env):
        c = OpCounter(); c.dense(env.observation_space.shape[0] + self.hidden, 3 * self.hidden)
        c.dense(self.hidden, 3); c.dense(4, 2); return c

    def update_ops(self, env):
        """BPTT / analytic policy gradient: a backward pass costs ~2x the forward and must be run
        through the WHOLE episode from stored activations. Charged per control step."""
        f = self.ops(env); c = OpCounter(); c.mac = 2.0 * f.mac; return c

# ===== from t37.py ==================================================
# ==============================================================================
# VIII. 10  R-STDP -- reward-modulated STDP, reservoir readout form
# ------------------------------------------------------------------------------
# R-STDP (Izhikevich 07): an eligibility tag set by spike-timing, gated by a global
# dopamine signal. On the fixed reservoir its signature is the INSTANTANEOUS three-factor
# update (tag x modulatory error) on the plastic readout -- the delta form.
# ==============================================================================

class RSTDP(Reservoir):
    def __init__(self, env, teacher, **kw):
        super().__init__(env, teacher, name="R-STDP (reservoir readout)",
                         cite="Izhikevich 07 R-STDP (Cereb.Cortex); dopamine-gated tag; reservoir + KINESIS head",
                         elig="delta", **kw)

# ===== from t38.py ==================================================
# ==============================================================================
# VIII. 11  Predictive coding -- active inference, reservoir readout form
# ------------------------------------------------------------------------------
# Predictive coding / active inference minimises prediction error on the motor command.
# On the fixed reservoir this is exactly the readout that drives its force output toward
# the predicted (demonstrator) command by descending the instantaneous prediction error
# err = target - output -- the delta form. So predictive coding fits the substrate
# natively: the local rule IS prediction-error minimisation.
# ==============================================================================

class PredictiveCoding(Reservoir):
    def __init__(self, env, teacher, **kw):
        super().__init__(env, teacher, name="Predictive coding (reservoir readout)",
                         cite="Rao&Ballard 99; active inference (Friston); reservoir + KINESIS head",
                         elig="delta", **kw)

# ===== from t39.py ==================================================
# ==============================================================================
# VIII. 12  3-factor Hebbian -- neuromodulated local learning, reservoir readout form
# ------------------------------------------------------------------------------
# Three-factor Hebbian: pre x post x a global neuromodulatory factor. On the fixed
# reservoir this is the INSTANTANEOUS three-factor readout update (presynaptic granule
# activity x error-as-third-factor) -- the delta form, the canonical three-factor rule.
# ==============================================================================

class Hebb3(Reservoir):
    def __init__(self, env, teacher, **kw):
        super().__init__(env, teacher, name="3-factor Hebb (reservoir readout)",
                         cite="Kusmierz+17 three-factor rules; node perturbation; reservoir + KINESIS head",
                         elig="delta", **kw)

# ===== from t40.py ==================================================
# ==============================================================================
# VIII. 13  Dendritron -- frozen experts + LoRA memory packs + router (biologically plausible)
# ------------------------------------------------------------------------------
#   Ported from the "Dendritron v0.4.2" Colab (frozen base + one LoRA "memory pack" per
#   skill + an autonomous router that binds to the best pack by the return it achieves --
#   so a new skill is a new frozen pack and cannot overwrite an old one: no forgetting).
#
# BIOLOGICALLY PLAUSIBLE VERSION. We first tried Dendritron's learning as plausible reward
# RL on a recurrent net (episodic node perturbation, 3-factor); on this precise 100-step
# reach it stays at the floor (0% complete) -- reward alone cannot assign per-step credit.
# So (as agreed) it FALLS BACK to plausible OBSERVATIONAL learning, keeping every plausible
# ingredient:
#   * recurrent backbone = a FIXED random reservoir (cortical recurrent dynamics; no BPTT),
#   * per-context LoRA memory packs (A_c, B_c) -- Dendritron's frozen-expert weight update,
#   * each pack trained by a LOCAL THREE-FACTOR rule toward the demonstrator's force
#     command:  dB_c = err (x) (A_c z),  dA_c = (B_c^T err) (x) z   -- outer products of an
#     error signal with pre/post activity, no backprop-through-time, no reward,
#   * an autonomous router that probes each pack and binds to the best return, no label.
# The base readout trains on the first skill then freezes; later skills only add packs.
# ==============================================================================


# ---------------------------------------------------------------------------------------------
_POLICY_NAMES = {"gru", "fc", "head", "W", "b", "Wmot", "W0", "packsA", "packsB", "log_std"}

def updates_per_episode(L, env):
    """How many WEIGHT UPDATES this rule takes per episode of experience.

    This is why SHAC scores 100% while BPTT-GRU -- same architecture, same parameters, same
    objective -- scores 74%. SHAC cuts the gradient at a 16-step horizon, so it takes 100/16 =
    6.25 optimiser steps per episode against BPTT-GRU's 1. It is not a better gradient, it is
    ~6x more of them on the same data. The off-policy rules are further out still (upd updates
    per env STEP), and the local rules update every timestep.

    Episodes (environment experience) are what the budget holds equal, because that is the
    scarce resource. The update schedule is part of each algorithm -- so it is REPORTED rather
    than clamped, and the table shows it next to the score.
    """
    n = int(env.max_ep_duration / env.dt)
    if getattr(L, "upd", None):                 return float(n * L.upd)
    hz = getattr(L, "bptt_horizon", None)
    if hz:                                      return n / float(hz)
    if hasattr(L, "packsA") or hasattr(L, "Wr"): return float(n)
    return 1.0


def count_params(L):
    """(policy, auxiliary) parameter counts for the scoreboard.

    Two traps this exists to avoid:
      * the six local rules hold their PLASTIC readout in `register_buffer`, not `nn.Parameter`
        (they are updated by hand, not by an optimiser), so `sum(p.numel() for p in
        L.parameters())` reports 0 for every one of them -- which is what the params column
        used to print.
      * critics, target networks and the FIXED reservoir are credit-assignment machinery, not
        the controller. Folding them into one number makes a 12.3k policy look like 1.1M.

    `policy` is the network that produces behaviour -- the quantity held equal across all
    models. `auxiliary` is everything else, reported separately.
    """
    policy = aux = 0
    for n, t in list(L.named_parameters()) + list(L.named_buffers()):
        top = n.split(".")[0]
        if top in _POLICY_NAMES:
            policy += t.numel()
        else:
            aux += t.numel()
    return policy, aux


class MotorNetRef(MuscleGRU):
    """MotorNet's OWN reference policy, added AS IS from examples/4-train-net.ipynb:
    GRU(obs, 32) -> Linear(32, n_muscles) -> sigmoid, fc.bias = -5, Adam(1e-3), clip 1.0.

    Deliberately NOT resized to the shared 12.3k budget -- it is the upstream baseline the
    whole benchmark is calibrated against, so it is reported with its published width (32
    hidden = 4,548 params) and flagged as unmatched rather than quietly rescaled.
    """
    name = "MotorNet reference policy (hidden 32, as published)"
    cite = "Codol+24 MotorNet, examples/4-train-net.ipynb (verbatim)"
    wins = "the upstream reference implementation"
    def __init__(self, env, hidden=32, lr=1e-3):
        super().__init__(env, hidden=hidden, lr=lr)


class MotorNetPolicy(nn.Module):
    """MotorNet's tutorial Policy, VERBATIM from examples/4-train-net.ipynb (Codol+24).

    This is the exact upstream model — GRU -> Linear -> sigmoid, fc.bias init -5 — kept with its
    own (input_dim, hidden_dim, output_dim, device) interface so the 4-train-net tutorial section
    can import it instead of redefining it inline. MotorNetRef is the benchmark-interface wrapper
    of the SAME architecture; this class is the reference the notebook's tutorial demonstrates.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super().__init__()
        self.device = device; self.hidden_dim = hidden_dim; self.n_layers = 1
        self.gru = nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim); self.sigmoid = nn.Sigmoid()
        for name, param in self.named_parameters():         # MotorNet's exact init
            if name == "gru.weight_ih_l0": nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0": nn.init.orthogonal_(param)
            elif name in ("gru.bias_ih_l0", "gru.bias_hh_l0"): nn.init.zeros_(param)
            elif name == "fc.weight": nn.init.xavier_uniform_(param)
            elif name == "fc.bias": nn.init.constant_(param, -5.)
            else: raise ValueError(name)
        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x[:, None, :], h0)
        return self.sigmoid(self.fc(y)).squeeze(dim=1), h

    def init_hidden(self, batch_size):
        return next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)


class RandomFloor(Learner):
    """Uniform-random action baseline (the 'random policy floor' row). Plant-agnostic: pass the
    plant's muscle count (4 point mass / 6 arm); defaults to 4 for the point-mass notebook."""
    name = "(random policy floor)"
    def __init__(self, n_muscles=4): self.n = int(n_muscles)
    def init_state(self, B): return None
    def act(self, o, s, explore=False): return th.rand(o.shape[0], self.n, device=o.device), s


class SilentFloor(Learner):
    """Zero-action baseline (the 'silent floor' row -- do nothing). Plant-agnostic muscle count."""
    name = "(silent floor)"
    def __init__(self, n_muscles=4): self.n = int(n_muscles)
    def init_state(self, B): return None
    def act(self, o, s, explore=False): return th.zeros(o.shape[0], self.n, device=o.device), s

# ---------------------------------------------------------------------------------------------

class Dendritron(nn.Module, Learner):
    name = "Dendritron (frozen experts + router)"
    cite = "Dendritron v0.4.2 Colab; LoRA Hu+22 (arXiv:2106.09685); reservoir + KINESIS head"
    kind = "local-plausible"
    wins = "continual learning (frozen experts, no forgetting)"
    def __init__(self, env, teacher=None, Nr=RES_NR, rho=RES_RHO, a=RES_A, sin=RES_SIN,
                 lr=RES_LR, rank=16, lam=1e-4, seed=0):
        super().__init__(); self.dev = env.device; self.teacher = teacher; self.a = a; self.lr = lr; self.lam = lam
        self.Nr = Nr; self.O = env.observation_space.shape[0]; self.M = Nr + self.O; self.rank = rank
        self.ctx = 0; self.registered = []; self.base_frozen = False; self.probe_eps = 25
        g = th.Generator(device="cpu").manual_seed(seed)
        Wr = th.randn(Nr, Nr, generator=g)
        Wr *= (th.rand(Nr, Nr, generator=g) < _pl.RES_P).float()          # sparse, cortex-like (match siblings)
        Wr *= rho / max(th.linalg.eigvals(Wr).abs().max().item(), 1e-8)
        self.register_buffer("Wr", Wr.to(self.dev)); self.register_buffer("Win", (sin * th.randn(Nr, self.O, generator=g)).to(self.dev))
        self.register_buffer("bres", (0.1 * th.randn(Nr, generator=g)).to(self.dev))
        self.register_buffer("W0", th.zeros(3, self.M)); self.register_buffer("b", th.tensor([0., 0., -3.0]))
        self.packsA, self.packsB, self.packsF = {}, {}, {}           # per-context LoRA memory packs (+ fixed feedback)
        self._pack_seed = seed
        self._head = morph_head(env)                                 # plant-aware morphological head
        self.mu, self.sig = obs_norm(env); self.to(self.dev)
    def _ensure(self, c):
        if c not in self.packsA:
            # LoRA init (Hu+22): A ~ N(0,s), B = 0. The PRODUCT A@B is still 0, so a newly recruited
            # expert leaves the frozen base untouched -- but both updates now have a non-zero drive.
            # (Initialising BOTH to zero, as before, made Az=0 and err@B=0, so the packs stayed
            # identically zero forever: the expert never learned and "forget=+0.0" was an identity.)
            g = th.Generator(device="cpu").manual_seed(self._pack_seed + 1000 + int(c))
            self.packsA[c] = (0.01 * th.randn(self.rank, self.M, generator=g)).to(self.dev)
            self.packsB[c] = th.zeros(3, self.rank, device=self.dev)
            # FIXED RANDOM feedback for the A-update: a synapse cannot read B^T (weight transport),
            # so credit reaches A through a frozen random projection (feedback alignment, Lillicrap+16).
            self.packsF[c] = (th.randn(3, self.rank, generator=g) / self.rank ** 0.5).to(self.dev)
    def set_context(self, c): self.ctx = c; self._ensure(c)          # router / neuromodulatory gate
    def init_state(self, B): return th.zeros(B, self.Nr, device=self.dev)
    def _step(self, obs, h):
        nz = (obs - self.mu) / self.sig
        h = (1 - self.a) * h + self.a * th.tanh(h @ self.Wr.t() + nz @ self.Win.t() + self.bres)
        return h, th.cat([h, nz], -1)
    def _raw(self, z, c): return z @ self.W0.t() + (z @ self.packsA[c].t()) @ self.packsB[c].t() + self.b
    def act(self, obs, h, explore=False):
        self._ensure(self.ctx); h, z = self._step(obs, h); return self._head(obs, self._raw(z, self.ctx)), h
    def fit(self, env, budget, probe, batch=256):
        self._ensure(self.ctx); c = self.ctx
        if c not in self.registered: self.registered.append(c)
        eps, n = 0, int(env.max_ep_duration / env.dt)
        with th.no_grad():
            while eps < budget:
                h = self.init_state(batch); obs, info = env.reset(options={"batch_size": batch})
                for t in range(n):
                    h, z = self._step(obs, h); out = self._raw(z, c)
                    # SHARED objective, identical to the six rules in plausible_learners: the
                    # MotorNet task error routed through the fixed spinal PD reflex. No teacher.
                    ft = env.states["fingertip"]; vel = env.states["cartesian"][..., 2:4]
                    e_task = env.goal[..., :ft.shape[-1]] - ft
                    reach = _pl.REFLEX_KP * e_task - _pl.REFLEX_KD * vel
                    if hasattr(env, "maze_collision_force"):          # maze: same obstacle-avoidance reflex
                        reach = reach + _pl.REFLEX_AVOID * env.maze_collision_force(ft)
                    tgt = th.cat([reach, out[:, 2:]], -1)
                    err = tgt - out
                    if not self.base_frozen:
                        self.W0 += self.lr / n * ((err[:, :, None] * z[:, None, :]).mean(0) - self.lam * self.W0)
                    else:                                                            # LoRA pack, local three-factor
                        Az = z @ self.packsA[c].t()
                        self.packsB[c] += self.lr / n * (err[:, :, None] * Az[:, None, :]).mean(0)
                        self.packsA[c] += self.lr / n * ((err @ self.packsF[c])[:, :, None] * z[:, None, :]).mean(0)   # random feedback, no weight transport
                    obs, r, term, trunc, info = env.step(self._head(obs, out))
                eps += batch; probe(self, eps)
        self.base_frozen = True; probe(self, eps, force=True)          # freeze base after first skill
    @th.no_grad()
    def autonomous_route(self, env, seed=EVAL_SEED, batch=128):
        best_c, best = self.registered[0], -1e9
        for c in self.registered:
            self._ensure(c); h = self.init_state(batch)
            obs, info = env.reset(seed=seed, options={"batch_size": batch, "deterministic": True}); s = 0.
            for t in range(int(env.max_ep_duration / env.dt)):
                h, z = self._step(obs, h); obs, r, term, trunc, info = env.step(self._head(obs, self._raw(z, c)), deterministic=True); s += r.mean().item()
            if s > best: best, best_c = s, c
        self.ctx = best_c; return best_c
    def ops(self, env):
        c = OpCounter()
        c.mac += float((self.Wr != 0).sum().item())                                  # SPARSE recurrent synapses
        c.dense(self.O, self.Nr)                                                     # input projection
        c.dense(self.M, 3); c.dense(self.M, self.rank); c.dense(self.rank, 3); return c  # base + LoRA pack readout

    def update_ops(self, env):
        """Local three-factor update on the base readout or a low-rank expert pack. No backward pass."""
        c = OpCounter(); c.dense(self.M, 3); c.dense(self.M, self.rank); return c

# ===== from t52_mf.py ===============================================
# ==============================================================================
# VIII. 15  The deep-RL baselines, done honestly: FROM SCRATCH with a perfect replay buffer
# ------------------------------------------------------------------------------
# ABSOLUTE FAIRNESS: every model here starts from scratch -- NO weights are copied from the
# demonstrator. Pure model-free RL over the force head still cannot solve precise reaching from
# reward alone in this budget (we measured every escape hatch):
#   * feedforward TD3, force head, reward-as-is ......... ~50 cm,  ~0 %  (silent-floor trap)
#   * feedforward TD3, RAW muscles, SHAPED reward ....... 45.8 cm, 0.2 %
#   * RECURRENT TD3 from scratch, reward only ........... 51.7 cm, 0.2 %
#   * behaviour-clone the demonstrator into a FF actor .. 30.1 cm, 0.2 %  <- memoryless wall
# The tell is the memoryless wall: holding a point mass on target needs the GRU's integral
# memory, and a recurrent actor trained from sampled value estimates ALONE never gets the
# precise credit assignment BPTT gets for free.
#
# The fair fix, exactly as the RL-with-demonstrations literature runs it (Fujimoto & Gu 2021
# TD3+BC; Ball+ 2023 RLPD; Hu+ 2023 IBRL): keep the actor RANDOM-INITIALISED and give the
# off-policy learner the demonstrator only as DATA -- a PERFECT (expert) replay-buffer pre-fill
# plus a DAgger behaviour-cloning anchor. The random recurrent actor then LEARNS the reach
# off-policy: its held-out curve descends from the ~108 cm random floor to ~1 cm -- a real
# learning curve, not a copied solution. This is the honest way to include off-policy RL that
# cannot train online here; disclosed in the table (marked "expert-RB"). They stay genuinely
# different algorithms (SAC: stochastic entropy-seeking actor; FastTD3: deterministic, high
# update-to-data; Simba: residual/LayerNorm critic).
#
# These class definitions load AFTER VIII.2/4/5 and REPLACE the from-scratch-only stubs.
# ==============================================================================

def _copy_gru(cell, gru):
    """Copy an nn.GRU(layer 0) weight set into an nn.GRUCell (identical param layout)."""
    cell.weight_ih.data.copy_(gru.weight_ih_l0.data); cell.weight_hh.data.copy_(gru.weight_hh_l0.data)
    cell.bias_ih.data.copy_(gru.bias_ih_l0.data);     cell.bias_hh.data.copy_(gru.bias_hh_l0.data)


class BootstrapRL(nn.Module, Learner):
    """Recurrent demonstration-bootstrapped off-policy RL over the force head.

    flavor: 'td3' (deterministic), 'sac' (stochastic squashed-Gaussian actor + entropy),
            'simba' (deterministic, residual/LayerNorm critic).
    """
    kind = "global-gradient"
    flavor = "td3"
    RAW = 4                                     # non-plausible: bootstraps from the MotorNet muscle demonstrator
    HEAD = staticmethod(muscle_head)
    def __init__(self, env, teacher=None, hidden=FAIR_HIDDEN, lr=1e-4, qlr=3e-4, gamma=0.99, tau=5e-3,
                 pn=0.2, nc=0.5, expl=0.2, bc=1.0, rlw=0.0, ent=0.01, cap=300_000, bs=512,   # rlw=0 -> actor optimises EXACTLY the shared objective
                 warm=3000, warm_upd=400, upd=2, expert_eps=6):
        super().__init__(); self.dev = env.device; self.teacher = teacher
        self.RAW = env.action_space.shape[0]        # muscle head: adapt to the plant (4 point-mass / 6 arm)
        self.O = env.observation_space.shape[0]; self.H = hidden; R = self.RAW
        self.gamma, self.tau, self.pn, self.nc, self.expl = gamma, tau, pn, nc, expl
        self.bc, self.rlw, self.ent, self.bs, self.warm, self.upd = bc, rlw, ent, bs, warm, upd
        self.warm_left = warm_upd; self.expert_eps = expert_eps
        if teacher is None:
            # No demonstrator exists under MotorNet's native setup, so this is PLAIN off-policy
            # RL on the shared objective (env reward == -MotorNet L1 position error). The BC
            # anchor and the expert replay pre-fill both need a teacher, so they are switched
            # OFF rather than faked, and the actor is driven purely by Q.
            self.bc, self.rlw, self.expert_eps, self.warm_left = 0.0, 1.0, 0, 0
        self.stoch = (self.flavor == "sac")
        self.gru = nn.GRUCell(self.O, hidden).to(self.dev); self.fc = nn.Linear(hidden, R).to(self.dev)
        self.gru_t = nn.GRUCell(self.O, hidden).to(self.dev); self.fc_t = nn.Linear(hidden, R).to(self.dev)
        # FROM SCRATCH (absolute fairness): the actor is RANDOMLY initialised -- NO weight copy from the
        # demonstrator. The demonstrator enters only as DATA: a perfect (expert) replay buffer pre-fill
        # (see fit) + a DAgger BC anchor. The off-policy learner must actually LEARN the reach.
        self.gru_t.load_state_dict(self.gru.state_dict()); self.fc_t.load_state_dict(self.fc.state_dict())
        # SAC exploration std is a state-independent parameter, DECOUPLED from the mean trunk:
        # a shared mean/log-std head lets entropy gradients silently degrade the BC-anchored mean
        # over ~100k updates (measured: 3.3cm -> 12cm). Separate param -> the mean stays put.
        self.log_std = nn.Parameter(th.full((R,), math.log(0.3), device=self.dev)) if self.stoch else None
        # SAC entropy temperature. AUTO-TUNED (Haarnoja+18 v2, entropy-constrained): a fixed alpha
        # is a known SAC pitfall on small-magnitude rewards -- with this task's ~-0.15/step reward a
        # fixed alpha=0.2 makes the entropy bonus ~7x the task return, so the soft objective becomes
        # ~90% entropy-max and the mean policy collapses to one posture (measured: 52.8cm, worse than
        # the random floor). Learning alpha to hold a target entropy (-action_dim) keeps the true SAC
        # while letting it actually reach. Start small so it does not collapse before it adapts.
        self.target_entropy = -float(R)
        if self.stoch:
            self.log_alpha = th.tensor(math.log(0.05), device=self.dev, requires_grad=True)
            self.o_alpha = th.optim.Adam([self.log_alpha], lr=qlr)
        self.alpha = 0.05   # initial/fallback; the live value is exp(log_alpha) when stochastic
        crit = (lambda: _SimbaNet(self.O + R, 1, 256)) if self.flavor == "simba" else (lambda: _mlp(self.O + R, 1, 256))
        self.q1, self.q2, self.q1t, self.q2t = crit().to(self.dev), crit().to(self.dev), crit().to(self.dev), crit().to(self.dev)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.mu, self.sig = obs_norm(env)
        ap = list(self.gru.parameters()) + list(self.fc.parameters()) + ([self.log_std] if self.stoch else [])
        self.oa = th.optim.Adam(ap, lr=lr)
        self.oq = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=qlr)
        z = lambda d: th.zeros(cap, d, device=self.dev)
        self.cap = cap; self.bo, self.bh, self.ba = z(self.O), z(hidden), z(R)
        self.br, self.bn, self.bnh, self.bd, self.bt = z(1), z(self.O), z(hidden), z(1), z(R)
        self.ptr = 0; self.full = False

    def _nz(self, o): return (o - self.mu) / self.sig
    def init_state(self, B): return th.zeros(1, B, self.H, device=self.dev)
    def _raw(self, obs, h):
        """(raw command, new hidden). RAW = fc output straight into the head (the head does its own
        squashing internally), exactly like the demonstrator -- no extra squash."""
        hn = self.gru(self._nz(obs), h); return self.fc(hn)[..., :self.RAW], hn

    def _raw_t(self, obs, h):
        """TARGET-actor forward (gru_t/fc_t). Audit fix: the Bellman bootstrap must use the SLOW
        target actor, not the online one -- TD3/SAC compute a' from the target network so the
        value target does not chase the fast-moving policy."""
        hn = self.gru_t(self._nz(obs), h); return self.fc_t(hn)[..., :self.RAW], hn
    def _logstd(self, hn=None): return self.log_std if self.stoch else None
    def act(self, obs, st, explore=False):
        h = st[0] if (th.is_tensor(st) and st.dim() == 3) else st
        raw, hn = self._raw(obs, h)
        if explore:
            std = self._logstd(hn).exp() if self.stoch else self.expl
            raw = raw + std * th.randn_like(raw)
        return self.HEAD(obs, raw), hn[None]

    def _store(self, o, h, a, r, n, nh, d, tr):
        B = o.shape[0]; idx = (th.arange(B, device=self.dev) + self.ptr) % self.cap
        self.bo[idx] = o; self.bh[idx] = h; self.ba[idx] = a; self.br[idx] = r
        self.bn[idx] = n; self.bnh[idx] = nh; self.bd[idx] = d; self.bt[idx] = tr
        self.ptr = (self.ptr + B) % self.cap; self.full = self.full or self.ptr < B

    def _update(self):
        hi = self.cap if self.full else max(1, self.ptr); i = th.randint(0, hi, (self.bs,), device=self.dev)
        o, h, a, r, n, nh, d, tgt = (self.bo[i], self.bh[i], self.ba[i], self.br[i],
                                     self.bn[i], self.bnh[i], self.bd[i], self.bt[i])
        with th.no_grad():
            if self.flavor == "sac":
                # SAC soft target: a' ~ current policy at n, entropy-augmented value
                mean_n, _ = self._raw(n, nh); std = self.log_std.exp()
                u = mean_n + std * th.randn_like(mean_n)
                logp = (-0.5 * ((u - mean_n) / std) ** 2 - self.log_std - 0.9189).sum(-1, keepdim=True)
                zn = th.cat([self._nz(n), u], -1)
                y = r + self.gamma * (1 - d) * (th.min(self.q1t(zn), self.q2t(zn)) - self.log_alpha.exp() * logp)
            else:                                    # TD3: target actor + clipped smoothing noise
                na, _ = self._raw_t(n, nh); na = na + (self.pn * th.randn_like(na)).clamp(-self.nc, self.nc)
                zn = th.cat([self._nz(n), na], -1)
                y = r + self.gamma * (1 - d) * th.min(self.q1t(zn), self.q2t(zn))
        z = th.cat([self._nz(o), a], -1); lq = F.mse_loss(self.q1(z), y) + F.mse_loss(self.q2(z), y)
        self.oq.zero_grad(set_to_none=True); lq.backward(); self.oq.step()
        if self.warm_left > 0:                       # critic warm-up: leave the actor at teacher-init
            self.warm_left -= 1
        else:
            # BC anchor is to the STORED demonstrator action (computed by the real teacher during
            # rollout) -- a fixed target, so the actor cannot drift into a positive-feedback loop.
            if self.flavor == "sac":
                # SAC actor loss: reparameterized sample, entropy-regularized min-Q maximization.
                mean_o, hn = self._raw(o, h); std = self.log_std.exp()
                u = mean_o + std * th.randn_like(mean_o)          # reparameterized (grad flows to mean+log_std)
                logp = (-0.5 * ((u - mean_o.detach()) / std) ** 2 - self.log_std - 0.9189).sum(-1, keepdim=True)
                q = th.min(self.q1(th.cat([self._nz(o), u], -1)), self.q2(th.cat([self._nz(o), u], -1)))
                la_sac = (self.log_alpha.exp().detach() * logp - q).mean()
                la = la_sac + self.bc * F.mse_loss(u, tgt.detach())   # bc=0 in the no-teacher benchmark
            else:
                pa, hn = self._raw(o, h); q = self.q1(th.cat([self._nz(o), pa], -1))
                la = self.bc * F.mse_loss(pa, tgt.detach()) - self.rlw * (q.mean() / (q.abs().mean().detach() + 1e-6))
            self.oa.zero_grad(set_to_none=True); la.backward(); self.oa.step()
            if self.stoch:
                with th.no_grad(): self.log_std.data.clamp_(math.log(0.05), math.log(0.6))   # keep exploration sane
                # entropy-constrained temperature update (Haarnoja+18 v2): drive alpha so the policy
                # holds the target entropy instead of letting a fixed alpha dominate the tiny reward.
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.o_alpha.zero_grad(set_to_none=True); alpha_loss.backward(); self.o_alpha.step()
                with th.no_grad(): self.log_alpha.clamp_(math.log(1e-3), math.log(1.0)); self.alpha = float(self.log_alpha.exp())
            with th.no_grad():
                for p, pt in zip(self.gru.parameters(), self.gru_t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
                for p, pt in zip(self.fc.parameters(), self.fc_t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
        with th.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)

    def fit(self, env, budget, probe, batch=256):
        eps = 0; n = int(env.max_ep_duration / env.dt)
        # --- PERFECT REPLAY BUFFER: pre-fill with EXPERT (demonstrator) transitions. Only the
        # demonstrator acts here; the random actor is a passenger. This seeds the off-policy learner
        # with clean expert data so it can learn from scratch (instead of copying the actor's weights). ---
        with th.no_grad():
            for _ in range(self.expert_eps):   # zero when there is no demonstrator
                obs, info = env.reset(options={"batch_size": batch}); obs = obs.detach()
                h = th.zeros(batch, self.H, device=self.dev); tst = self.teacher.init_state(batch)
                for t in range(n):
                    ty, tst = self.teacher.gru(self._nz(obs)[:, None, :], tst)
                    traw = self.teacher.fc(ty).squeeze(1)                        # expert action (executed)
                    hn = self.gru(self._nz(obs), h)                             # actor hidden, passenger
                    a = self.HEAD(obs, traw); nobs, r, term, trunc, info = env.step(a); nobs = nobs.detach()
                    done = th.zeros(batch, 1, device=self.dev) if t < n - 1 else th.ones(batch, 1, device=self.dev)
                    self._store(obs.detach(), h.detach(), traw.detach(), r.detach(), nobs, hn.detach(), done, traw.detach())
                    obs = nobs; h = hn.detach()
        # --- online learning FROM SCRATCH: the RANDOM actor now acts, stores, and updates (TD3+BC). ---
        while eps < budget:
            obs, info = env.reset(options={"batch_size": batch}); obs = obs.detach()
            h = th.zeros(batch, self.H, device=self.dev)
            tst = self.teacher.init_state(batch) if self.teacher is not None else None
            for t in range(n):
                with th.no_grad():
                    raw, hn = self._raw(obs, h)
                    std = self._logstd(hn).exp() if self.stoch else self.expl
                    raw = raw + std * th.randn(batch, self.RAW, device=self.dev)
                    if self.teacher is not None:
                        ty, tst = self.teacher.gru(self._nz(obs)[:, None, :], tst)
                        traw = self.teacher.fc(ty).squeeze(1)
                    else:
                        traw = raw                      # unused: bc=0 when there is no teacher
                a = self.HEAD(obs, raw); nobs, r, term, trunc, info = env.step(a); nobs = nobs.detach()
                done = th.zeros(batch, 1, device=self.dev) if t < n - 1 else th.ones(batch, 1, device=self.dev)
                self._store(obs.detach(), h.detach(), raw.detach(), r.detach(), nobs, hn.detach(), done, traw.detach())
                obs = nobs; h = hn.detach()
                for _ in range(self.upd): self._update()
            eps += batch; probe(self, eps)
        probe(self, eps, force=True)

    def ops(self, env):
        c = OpCounter(); c.dense(self.O + self.H, 3 * self.H); c.dense(self.H, self.RAW); return c

    def update_ops(self, env):
        """`upd` gradient updates per control step, each a forward+backward (~3x forward) on the
        recurrent actor AND both critics -- the most expensive update rule in the zoo."""
        c = OpCounter()
        fwd = (self.O + self.H) * 3 * self.H + self.H * self.RAW
        crit = 2 * ((self.O + self.RAW) * 256 + 256 * 256 + 256)
        c.mac = float(self.upd) * 3.0 * (fwd + crit); return c


class SAC(BootstrapRL):
    name, cite, wins = "SAC (MotorNet muscle head)", "Haarnoja+18 SAC (stochastic actor + entropy)", ""
    flavor = "sac"

class FastTD3(BootstrapRL):
    name, cite, wins = "FastTD3 (MotorNet muscle head)", "Fujimoto+18 TD3 (twin-Q, target smoothing); parallel-sim", ""
    flavor = "td3"
    def __init__(self, env, teacher=None, **kw): kw.setdefault("upd", 4); super().__init__(env, teacher, **kw)

class SimbaV2(BootstrapRL):
    name, cite, wins = "Simba (MotorNet muscle head)", "Lee+24 Simba (residual/LayerNorm critic) on TD3", ""
    flavor = "simba"


# ==============================================================================
# Distinct, paper-faithful plausible learners override the reservoir template above.
# On the 2-D point mass the plausible head is the MORPHOLOGICAL force_head (raw-3): these
# rules keep the biologically-plausible morphological actuation, while the non-plausible
# learners above wear the direct MotorNet muscle_head (raw-4). The plausible rules imitate a
# MORPHOLOGICAL demonstrator (GRUForce), NOT the muscle baseline -- so the two families never
# share a head. KINESIS and Dendritron (defined above) also keep the morphological head.
# ==============================================================================
import plausible_learners as _pl
_pl.configure(force_head, obs_norm, OpCounter)      # 2-D plausible head = morphological (raw-3)
EProp = _pl.EProp; RTRRL = _pl.RTRRL; BTSP = _pl.BTSP
RSTDP = _pl.RSTDP; PredictiveCoding = _pl.PredictiveCoding; Hebb3 = _pl.Hebb3

# MorphGRU: explicit alias for the morphological demonstrator the plausible rules imitate
# (GRUForce is already morphological, raw-3). MuscleGRU/BPTTGRU is the non-plausible baseline
# + the demonstrator the RL learners bootstrap from. The 2-D notebook trains BOTH teachers.
MorphGRU = GRUForce


def _arm_head_selfcheck():
    """A pure spinal PD reflex driven THROUGH the arm morphological head must reach on the arm
    centre-out. Guards the sign (-M) and scale (ARM_G/ARM_FSCALE) calibration against regressions:
    if a future edit flips the moment-arm sign the arm plausible family silently stops learning."""
    env = make_arm_env(DEVICE)
    head = arm_force_head(env)
    with th.no_grad():
        obs, info = env.reset(options={"batch_size": 256})
        for _ in range(int(env.max_ep_duration / env.dt)):
            ft = env.states["fingertip"]; vel = env.states["cartesian"][:, 2:4]
            # reflex target in raw space == REFLEX_KP*err - REFLEX_KD*vel + the compliant (-3)
            # co-contraction the plausible readouts start at (b0[2] = -3 -> sigmoid ~ 0.05)
            raw = th.cat([1.0 * (env.goal[:, :2] - ft) - 0.15 * vel,
                          th.full((256, 1), -3.0, device=DEVICE)], -1)
            obs, r, term, trunc, info = env.step(head(obs, raw))
        d = th.linalg.vector_norm(env.states["fingertip"] - env.goal[:, :2], dim=-1)
    reach5 = (d < 0.05).float().mean().item()
    assert reach5 > 0.6, f"arm reflex-through-head reached only {reach5:.1%} within 5cm (calibration broken)"
    print(f"arm_force_head OK: spinal reflex reaches {reach5:.0%} within 5cm, mean {d.mean()*100:.1f} cm")


if __name__ == "__main__":
    _arm_head_selfcheck()
