"""MC_Maze e-prop -- a small, self-contained wrapper to train / evaluate / visualise the notebook's
e-prop network (EPropPolicy) on the monkey's 108 MC-Maze puzzles.

The monkey (Churchland/Kaufman/Shenoy; NLB **MC_Maze**, DANDI **000128**) moved a cursor from the centre
to a cued target THROUGH barriers. Here the same e-prop RNN as the notebook (leaky units, NO BPTT --
credit is assigned by eligibility traces x learning signals, Bellec et al. 2020) controls a point-mass
cursor via 4 muscles (obs = goal + proprioception) on those same 108 mazes, on the monkey's objective:
reach the target fast, with least movement, without hitting the walls.

Four modular functions (all tested):
    maze_split(seed)                     -> fixed train/val/test index split of the 108 puzzles
    train(train, val, test, episodes)    -> e-prop training (tqdm + brief loss), returns (policy, history)
    quantitative(policy, ..., history)   -> success rate, endpoint error (cm), time/episode + per-episode
                                            train/val/test/reward line-plots (bias/variance/overfitting)
    qualitative(policy, maze_id)         -> a GIF of the e-prop cursor solving one maze (default: random test)
"""
import os, sys, time, io, base64, urllib.request
import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "nlb_tools"))
import maze_env

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
SUCCESS_CM = 5.0
# DANDI 000128 (MC_Maze) landing page + the behaviour NWB the 108 maze configs are extracted from.
DANDI_URL = "https://dandiarchive.org/dandiset/000128"


# ============================================================
# e-prop network (identical algorithm to motornet_eprop.ipynb: leaky RNN, detached recurrence)
# ============================================================
class EPropPolicy(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=DEVICE, dt=0.01, tau=0.05):
        super().__init__()
        self.device = th.device(device); self.hidden_dim = hidden_dim
        self.alpha = float(dt) / float(tau)
        self.w_in = th.nn.Parameter(th.empty(hidden_dim, input_dim))
        self.w_rec = th.nn.Parameter(th.empty(hidden_dim, hidden_dim))
        self.b_rec = th.nn.Parameter(th.zeros(hidden_dim))
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        th.nn.init.xavier_uniform_(self.w_in); th.nn.init.orthogonal_(self.w_rec)
        with th.no_grad(): self.w_rec.mul_(0.9)
        th.nn.init.xavier_uniform_(self.fc.weight); th.nn.init.constant_(self.fc.bias, -5.0)
        self.to(self.device)

    def init_hidden(self, b):
        z = th.zeros(b, self.hidden_dim, device=self.device); return z, z.clone()

    def forward(self, obs, hidden):
        x_prev, r_prev = hidden
        x_prev, r_prev, obs = x_prev.detach(), r_prev.detach(), obs.detach()   # no BPTT
        x = (1 - self.alpha) * x_prev + self.alpha * (obs @ self.w_in.T + r_prev @ self.w_rec.T + self.b_rec)
        r = th.relu(x); action = th.sigmoid(self.fc(r))
        return action, (x.detach(), r.detach()), {"obs": obs, "r_prev": r_prev, "x": x, "r": r}

    def act(self, obs, hidden):                                    # eval convenience (no traces)
        return self.forward(obs, hidden)[:2]


class _Traces:
    """e-prop eligibility traces: low-passed pre-activity gated by the post pseudo-derivative."""
    def __init__(self, policy, b):
        d = policy.device; h = policy.hidden_dim
        self.p = policy; self.eps_in = None; self.b = b
        self.eps_in = th.zeros(b, h, policy.w_in.shape[1], device=d)
        self.eps_rec = th.zeros(b, h, h, device=d); self.eps_bias = th.zeros(b, h, device=d)

    @th.no_grad()
    def update(self, cache):
        a = self.p.alpha; dec = 1 - a; deriv = (cache["x"] > 0).float()
        self.eps_in = dec * self.eps_in + a * cache["obs"][:, None, :]
        self.eps_rec = dec * self.eps_rec + a * cache["r_prev"][:, None, :]
        self.eps_bias = dec * self.eps_bias + a
        return (deriv[:, :, None] * self.eps_in).clone(), (deriv[:, :, None] * self.eps_rec).clone(), (deriv * self.eps_bias).clone()


def _l1(x, y):
    return th.mean(th.sum(th.abs(x - y), dim=-1))


# ============================================================
# data  (108 MC-Maze puzzles; auto-download / -build if the cached configs are missing)
# ============================================================
def ensure_maze_data(verbose=True):
    """Make sure the 108 MC-Maze conditions are available. Uses the cached configs if present; else
    builds them from the DANDI 000128 behaviour NWB; else downloads that NWB from the DANDI API."""
    if os.path.exists(maze_env.CACHE):
        if verbose: print(f"MC-Maze configs ready ({maze_env.CACHE}); source: {DANDI_URL}")
        return maze_env.extract_configs()
    nwb = os.path.abspath(maze_env.NWB)
    if not os.path.exists(nwb):
        os.makedirs(os.path.dirname(nwb), exist_ok=True)
        name = os.path.basename(nwb)
        api = ("https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/"
               f"?path=sub-Jenkins/{name}&metadata=false")
        if verbose: print(f"downloading MC-Maze data from DANDI 000128 ...")
        meta = __import__("json").loads(urllib.request.urlopen(api, timeout=60).read())
        asset = meta["results"][0]
        url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
        urllib.request.urlretrieve(url, nwb)
        if verbose: print(f"  saved {nwb} ({os.path.getsize(nwb)/1e6:.1f} MB)")
    return maze_env.extract_configs(force=True)


def maze_split(seed=0):
    """Fixed (seeded) 60/20/20 split of the 108 puzzles -> (train, val, test) index arrays."""
    ensure_maze_data(verbose=False)
    return maze_env.maze_split(seed=seed)


def _env(conditions, random_cond, collide_w=6.0, effort_w=maze_env.EFFORT_W):
    return maze_env.make_maze_env(DEVICE, conditions=np.asarray(conditions), random_cond=random_cond,
                                  collide_w=collide_w, effort_w=effort_w)


# ============================================================
# evaluation  (the monkey's scorecard on a set of mazes)
# ============================================================
@th.no_grad()
def _score(policy, env, batch=256, seed=7):
    obs, info = env.reset(seed=seed, options={"batch_size": batch, "deterministic": True})
    obs = obs.to(DEVICE); h = policy.init_hidden(batch); n = int(env.max_ep_duration / env.dt)
    hit = 0.0; ret = 0.0; t0 = time.perf_counter()
    for _ in range(n):
        a, h = policy.act(obs, h); obs, r, term, trunc, info = env.step(a); obs = obs.to(DEVICE)
        hit += float((env.maze_collision(env.states["fingertip"]) > 0).float().mean()); ret += float(r.mean())
    d = th.linalg.vector_norm(env.states["fingertip"] - env.goal[:, :2], dim=-1)
    return dict(reach=100 * (d < SUCCESS_CM / 100.).float().mean().item(), err_cm=100 * d.mean().item(),
                reward=ret, in_wall=100 * hit / n, ms_per_ep=1000 * (time.perf_counter() - t0) / 1)


# ============================================================
# 1) TRAIN  (e-prop; brief loss every few episodes + a tqdm progress/ETA bar)
# ============================================================
def train(train_idx, val_idx, test_idx=None, episodes=3000, hidden=32, lr=1e-3, batch=32,
          tau=0.05, collide_w=6.0, effort_w=maze_env.EFFORT_W, log_every=None, seed=0):
    """Train the e-prop network on the TRAIN mazes; log train/val(/test) accuracy every few episodes.
    Returns (policy, history). history[k] = {episode, train_loss, splits{train/val/test:{reach,err,reward,ms}}}."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k): return x
    th.manual_seed(seed); np.random.seed(seed)
    envs = {"train": _env(train_idx, True, collide_w, effort_w),
            "val": _env(val_idx, False, collide_w, effort_w)}
    if test_idx is not None: envs["test"] = _env(test_idx, False, collide_w, effort_w)
    env = envs["train"]; n = int(env.max_ep_duration / env.dt)
    policy = EPropPolicy(env.observation_space.shape[0], hidden, env.action_space.shape[0], DEVICE, env.dt, tau)
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    log_every = log_every or max(1, episodes // 30)
    history, losses = [], []
    bar = tqdm(range(episodes), desc="e-prop / MC-Maze", unit="ep")
    for ep in bar:
        h = policy.init_hidden(batch); tr = _Traces(policy, batch)
        obs, info = env.reset(options={"batch_size": batch}); obs = obs.to(DEVICE)
        xy = [info["states"]["fingertip"][:, None, :]]; tg = [info["goal"][:, None, :]]
        rates, ein, erec, ebias, acts, cols = [], [], [], [], [], []
        for _ in range(n):
            a, h, cache = policy(obs, h)
            e_in, e_rec, e_bias = tr.update(cache)
            rates.append(cache["r"]); ein.append(e_in); erec.append(e_rec); ebias.append(e_bias)
            obs, r, term, trunc, info = env.step(a); obs = obs.to(DEVICE)
            ft = info["states"]["fingertip"]; xy.append(ft[:, None, :]); tg.append(info["goal"][:, None, :])
            acts.append(a); cols.append(env.maze_collision(ft))
        xy = th.cat(xy, 1); tg = th.cat(tg, 1)
        # THE maze objective (differentiable): reach + barrier + least-movement.
        loss = (_l1(xy, tg) + collide_w * th.stack(cols, 1).sum(1).mean()
                + effort_w * th.stack([x.pow(2).mean(-1) for x in acts], 1).sum(1).mean())
        grads = th.autograd.grad(loss, rates + [policy.fc.weight, policy.fc.bias])
        sig = grads[:len(rates)]; gfw, gfb = grads[-2], grads[-1]
        gwin = th.zeros_like(policy.w_in); gwrec = th.zeros_like(policy.w_rec); gbrec = th.zeros_like(policy.b_rec)
        for s, ei, er, eb in zip(sig, ein, erec, ebias):
            gwin += th.einsum("bj,bji->ji", s, ei) / batch
            gwrec += th.einsum("bj,bji->ji", s, er) / batch
            gbrec += (s * eb).mean(0)
        opt.zero_grad(set_to_none=True)
        policy.w_in.grad, policy.w_rec.grad, policy.b_rec.grad = gwin, gwrec, gbrec
        policy.fc.weight.grad, policy.fc.bias.grad = gfw, gfb
        th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if ep % log_every == 0 or ep == episodes - 1:
            sp = {k: _score(policy, e) for k, e in envs.items()}
            history.append(dict(episode=ep, train_loss=float(np.mean(losses[-log_every:])), splits=sp))
            bar.set_postfix(loss=f"{history[-1]['train_loss']:.3f}",
                            val_reach=f"{sp['val']['reach']:.0f}%", val_cm=f"{sp['val']['err_cm']:.1f}")
    return policy, history


# ============================================================
# 2) QUANTITATIVE  (scorecard + per-episode train/val/test line-plots)
# ============================================================
def quantitative(policy, train_idx, val_idx, test_idx, history=None, show=True):
    """Print the final success rate / endpoint error / time-per-episode on each split, and (if a
    training `history` is given) plot those three + reward vs training episode for train/val/test."""
    import pandas as pd
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}
    final = {k: _score(policy, _env(v, False)) for k, v in idx.items() if v is not None}
    tab = pd.DataFrame({k: {"success_%": v["reach"], "error_cm": v["err_cm"], "in_wall_%": v["in_wall"],
                            "reward": v["reward"], "ms/episode": v["ms_per_ep"]} for k, v in final.items()}).T
    if show:
        try:
            from IPython.display import display; display(tab.round(2))
        except Exception:
            print(tab.round(2))
    if history:
        import matplotlib.pyplot as plt
        eps = [h["episode"] for h in history]; splits = list(history[0]["splits"].keys())
        colors = {"train": "#e76f51", "val": "#2a9d8f", "test": "#6c5ce7"}
        panels = [("success rate (%)", "reach"), ("endpoint error (cm)", "err_cm"),
                  ("time / episode (ms)", "ms_per_ep"), ("episodic reward", "reward")]
        fig, axs = plt.subplots(1, 4, figsize=(19, 3.8))
        for ax, (title, key) in zip(axs, panels):
            for sp in splits:
                ax.plot(eps, [h["splits"][sp][key] for h in history], "-o", ms=2.5, lw=1.6,
                        color=colors.get(sp, None), label=sp)
            ax.set_title(title, fontsize=10); ax.set_xlabel("training episode"); ax.grid(alpha=.25)
        axs[0].legend(title="split (gap ⇒ over/under-fit)", fontsize=8)
        fig.suptitle("e-prop on the monkey's mazes — train vs val vs test over training", fontweight="bold")
        plt.tight_layout()
        if show: plt.show()
    return tab


# ============================================================
# 3) QUALITATIVE  (a GIF of the e-prop cursor solving one maze; default a random TEST maze)
# ============================================================
@th.no_grad()
def qualitative(policy, maze_id=None, test_idx=None, out=None, fps=14, seed=7):
    """Render the e-prop cursor solving MC-Maze condition `maze_id` (0..107). Default: a random maze
    from `test_idx` if given (held-out), else any of the 108. Returns the GIF path."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    ensure_maze_data(verbose=False)
    pool = list(map(int, test_idx)) if test_idx is not None else list(range(108))
    if maze_id is None:
        maze_id = pool[int(np.random.default_rng(seed).integers(len(pool)))]
    env = _env([maze_id], False)
    obs, info = env.reset(options={"batch_size": 1}); obs = obs.to(DEVICE); h = policy.init_hidden(1)
    n = int(env.max_ep_duration / env.dt)
    path = [env.states["fingertip"][0].cpu().numpy()]
    inside = [float(env.maze_collision(env.states["fingertip"])[0]) > 0]   # is the cursor in a wall?
    for _ in range(n):
        a, h = policy.act(obs, h); obs, *_ = env.step(a); obs = obs.to(DEVICE)
        path.append(env.states["fingertip"][0].cpu().numpy())
        inside.append(float(env.maze_collision(env.states["fingertip"])[0]) > 0)
    path = np.array(path); inside = np.array(inside)
    bar = env._bar[env._cond][0].cpu().numpy(); msk = env._msk[env._cond][0].cpu().numpy() > 0
    tg = env._tg[env._cond][0].cpu().numpy(); real = bar[msk]
    fig, ax = plt.subplots(figsize=(5, 5))
    for cx, cy, hw, hh in real:
        ax.add_patch(plt.Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fc="#5A6472", ec="#2b2b2b", lw=0.6, alpha=.9))
    ax.plot(*tg, "*", ms=20, color="#D1495B", zorder=6, label="target")
    ax.plot(*path[0], "o", ms=9, color="#2A9D8F", zorder=6, label="start (centre)")
    (ln,) = ax.plot([], [], "-", color="#457b9d", lw=2.0, zorder=4)
    (dot,) = ax.plot([], [], "o", ms=8, zorder=7)
    # axis frames the WHOLE maze (barriers + path + target), so nothing is cut off
    xs = [path[:, 0].min(), path[:, 0].max(), tg[0]]; ys = [path[:, 1].min(), path[:, 1].max(), tg[1]]
    if len(real):
        xs += [(real[:, 0] - real[:, 2]).min(), (real[:, 0] + real[:, 2]).max()]
        ys += [(real[:, 1] - real[:, 3]).min(), (real[:, 1] + real[:, 3]).max()]
    m = 0.05
    ax.set_xlim(min(xs) - m, max(xs) + m); ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper left", fontsize=8, framealpha=.9)
    key = maze_env.extract_configs()["keys"][maze_id]
    ax.set_title(f"e-prop · MC-Maze {key[0]} v{key[1]} · {int(msk.sum())} barriers "
                 f"(cursor turns red inside a wall)", fontsize=9)

    def _f(i):
        ln.set_data(path[:i + 1, 0], path[:i + 1, 1]); dot.set_data([path[i, 0]], [path[i, 1]])
        dot.set_color("#e63946" if inside[i] else "#457b9d")     # RED = colliding with a wall
        return ln, dot
    anim = animation.FuncAnimation(fig, _f, frames=len(path), interval=1000 / fps, blit=True)
    out = out or os.path.join(_HERE, "save", f"eprop_maze_{maze_id}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim.save(out, writer=animation.PillowWriter(fps=fps)); plt.close(fig)
    return out


def show_gif(path):
    """Embed a saved GIF in a notebook cell."""
    from IPython.display import HTML
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return HTML(f'<img src="data:image/gif;base64,{b64}" width="360">')


def load_eprop(weight_path, hidden=32, tau=0.05, device=DEVICE):
    """Rebuild an EPropPolicy from a saved state_dict (for 4-monkey-net, which passes a weights path)."""
    ensure_maze_data(verbose=False)
    env = _env(list(range(108)), False)
    p = EPropPolicy(env.observation_space.shape[0], hidden, env.action_space.shape[0], device, env.dt, tau)
    p.load_state_dict(th.load(weight_path, map_location=device)); return p
