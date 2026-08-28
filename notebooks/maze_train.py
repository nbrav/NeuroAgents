"""Reproducible FAIR trainer for the MONKEY'S OWN MAZE task (MC-Maze, 108 puzzles, WITH collision).

The monkey moved a cursor through barrier mazes to a cued target. Here every one of the 13 learners
controls the SAME point-mass cursor (obs = goal + proprioception, action = 4 muscles -- the
joystick-via-muscle interface) on the SAME 108 MC-Maze conditions, optimising the SAME composite
objective (fairness by construction, decided by the env, not per model):

    reach the target FAST + with LEAST movement + LEAST endpoint error + WITHOUT hitting the barriers.

  * gradient rules (BPTT / SHAC / KINESIS) descend the composite maze cost by BPTT,
  * local plausible rules regress onto a spinal reflex = reach reflex + obstacle-avoidance reflex,
  * model-free deep-RL optimises env reward = -(composite cost),
all from task-space signals only -- no plant Jacobian, no privileged simulator state, one objective.

Writes save_monkey/maze_models/{tag}.pt + save_monkey/maze_results.json.
Run:  MZ_BUDGET=20000 .venv/bin/python notebooks/maze_train.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools"))
import numpy as np
import torch as th
import motor_zoo as mz
import plausible_learners as pl
import maze_env

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# paths are env-overridable so the 4-11monkey SPIKING-VISION run writes its own files (MZ_SPIKE_VISION=1)
# without clobbering the no-vision maze run in 4-monkey-net.
MODEL_DIR = os.environ.get("MZ_MODEL_DIR") or os.path.join(REPO, "save_monkey", "maze_models")
RESULTS_JSON = os.environ.get("MZ_RESULTS_JSON") or os.path.join(REPO, "save_monkey", "maze_results.json")
BUDGET = int(os.environ.get("MZ_BUDGET", 20_000))
DEVICE = mz.DEVICE
REACH_CM = 3.0                                   # "reached" threshold for the reach-TIME metric

# per-model tuned hyperparameters from the Ray fine-tuning (4-tuning-net). If present, each plausible
# model is (re)built with ITS best hyperparameters (learning rate + rule constant + reflex gains),
# tuned on VAL, to maximise accuracy; the gradient models keep their robust defaults.
import maze_hp
BEST_HP_JSON = os.environ.get("MZ_BEST_JSON") or os.path.join(REPO, "save_monkey", "maze_best_hp.json")
BEST_HP = json.load(open(BEST_HP_JSON)) if os.path.exists(BEST_HP_JSON) else {}

# FIXED 60/20/20 split of the 108 puzzles (seeded). Train on `train` ONLY; track the learning
# curve on `val`; report the final scorecard on `test`, which is NEVER seen during training.
if os.environ.get("MZ_ALL"):
    # TRAIN-ON-ALL: no-vision cannot generalise to unseen mazes (obs-aliasing: 35 targets / 72 walled),
    # so to LEARN the monkey's own 108 puzzles the models train + are scored on ALL conditions (like the
    # monkey learned its own mazes). Not a generalisation claim -- a "can the rule store the motor programs" one.
    TRAIN_IDX = VAL_IDX = TEST_IDX = np.arange(len(maze_env.extract_configs()["keys"]))
    MZ_TRAIN = maze_env.make_maze_env(DEVICE, conditions=None, random_cond=True)     # all 108, sampled
    MZ_VAL   = maze_env.make_maze_env(DEVICE, conditions=None, random_cond=False)    # all 108, tiled
    MZ_TEST  = MZ_VAL
else:
    TRAIN_IDX, VAL_IDX, TEST_IDX = (maze_env.maze_split_family(seed=0) if os.environ.get("MZ_SPLIT") == "family"
                                    else maze_env.maze_split(seed=0))
    MZ_TRAIN = maze_env.make_maze_env(DEVICE, conditions=TRAIN_IDX, random_cond=True)    # 60% train
    MZ_VAL   = maze_env.make_maze_env(DEVICE, conditions=VAL_IDX,  random_cond=False)    # 20% val  (curve + tuning)
    MZ_TEST  = maze_env.make_maze_env(DEVICE, conditions=TEST_IDX, random_cond=False)    # 20% test (held out)
pl.configure(mz.morph_head(MZ_TRAIN), mz.obs_norm, mz.OpCounter)

LEARNERS = [
    (mz.MotorNetRef, "motornet_ref"), (mz.BPTTGRU, "bptt_gru"), (mz.SHAC, "shac"),
    (mz.SAC, "sac"), (mz.FastTD3, "fasttd3"), (mz.SimbaV2, "simbav2"),
    (pl.EProp, "eprop"), (pl.RTRRL, "rtrrl"), (pl.BTSP, "btsp"),
    (mz.Kinesis, "kinesis"), (pl.RSTDP, "rstdp"), (pl.PredictiveCoding, "predcode"),
    (pl.Hebb3, "hebb3"), (mz.Dendritron, "dendritron"),
]


@th.no_grad()
def maze_metrics(L, env, batch=512, seed=mz.EVAL_SEED):
    """Roll a trained learner over `env`'s mazes; measure the monkey's own scorecard."""
    MZ_EVAL = env
    obs, info = MZ_EVAL.reset(seed=seed, options={"batch_size": batch, "deterministic": True})
    has_bar = (MZ_EVAL._msk[MZ_EVAL._cond].sum(-1) > 0)          # ONE setup, two conditions: with-maze vs no-maze
    st = L.init_state(batch); n = int(MZ_EVAL.max_ep_duration / MZ_EVAL.dt); dt_ms = MZ_EVAL.dt * 1000.0
    prev = MZ_EVAL.states["fingertip"].clone()
    path = th.zeros(batch, device=DEVICE)                          # total fingertip travel (movement)
    in_barrier = th.zeros(batch, device=DEVICE)                    # steps spent inside a barrier
    reach_step = th.full((batch,), float(n), device=DEVICE)        # first step within REACH_CM
    effort = 0.0
    for t in range(n):
        a, st = L.act(obs, st); obs, r, term, trunc, info = MZ_EVAL.step(a)
        ft = MZ_EVAL.states["fingertip"]
        path += th.linalg.vector_norm(ft - prev, dim=-1); prev = ft.clone()
        # WALL CONTACT: with solid walls the cursor is projected out so post-step penetration is 0;
        # the env records whether it TRIED to enter a wall (was blocked) -- that is the real metric.
        contact = getattr(MZ_EVAL, "_wall_contact", None)
        in_barrier += contact if contact is not None else (MZ_EVAL.maze_collision(ft) > 0).float()
        d = th.linalg.vector_norm(ft - MZ_EVAL.goal[:, :2], dim=-1)
        reached = (d < REACH_CM / 100.0) & (reach_step == float(n))
        reach_step = th.where(reached, th.full_like(reach_step, float(t)), reach_step)
        effort += float(a.pow(2).mean())
    d = th.linalg.vector_norm(MZ_EVAL.states["fingertip"] - MZ_EVAL.goal[:, :2], dim=-1)
    reached5 = (d < 0.05).float()
    nan = float("nan")
    r5_maze = 100.0 * reached5[has_bar].mean().item() if bool(has_bar.any()) else nan     # with-maze subset
    r5_free = 100.0 * reached5[~has_bar].mean().item() if bool((~has_bar).any()) else nan  # no-maze subset
    return dict(
        err_cm=100.0 * d.mean().item(),                           # least cm
        reach5=100.0 * reached5.mean().item(),                    # % reached (<5 cm), all conditions
        reach5_maze=r5_maze,                                      # % reached on the WITH-barrier puzzles
        reach5_nomaze=r5_free,                                    # % reached on the barrier-FREE (no-maze) reaches
        reach2=100.0 * (d < 0.02).float().mean().item(),          # % reached (<2 cm, precise)
        reach_time_ms=dt_ms * reach_step.mean().item(),           # FAST: mean ms to first get within 3 cm
        path_cm=100.0 * path.mean().item(),                       # LEAST MOVEMENT: fingertip travel
        in_barrier_pct=100.0 * (in_barrier / n).mean().item(),    # AVOID: % of steps blocked pressing a wall
        effort=effort / n,
    )


def run_one(cls, tag, budget=BUDGET, bs=None):
    _s = int(os.environ.get("MZ_SEED", 0)); th.manual_seed(_s); np.random.seed(_s)
    if tag in BEST_HP:
        L = maze_hp.build(tag, BEST_HP[tag], MZ_TRAIN)     # tuned hyperparameters (VAL-selected)
        if bs is None: bs = int(BEST_HP[tag].get("batch", 256))   # per-model TUNED batch (plausible + KINESIS)
    else:
        L = cls(MZ_TRAIN)                                  # untouched baseline (gradient / deep-RL): NOT fine-tuned
        if bs is None: bs = 32                             # baseline keeps its original batch (no HP tuning)
    pr = mz.Probe(MZ_VAL, every_eps=max(1, budget // 20), budget=budget, track_best=True)  # EARLY STOP on VAL
    t0 = time.perf_counter(); L.fit(MZ_TRAIN, budget, pr, batch=bs); train_s = time.perf_counter() - t0
    if getattr(pr, "best_state", None) is not None:
        L.load_state_dict(pr.best_state)     # restore BEST-VAL checkpoint: the real (swept) maze is hard, so the
                                             # local rules can DIVERGE late (e-prop 36%->5% at full budget) -- this
                                             # keeps each model's best point; the gradient baselines need the budget.
    m = maze_metrics(L, MZ_TEST)                                            # final scorecard on HELD-OUT test
    val_err = maze_metrics(L, MZ_VAL)["err_cm"]                             # val error (matches the tuning objective)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if isinstance(L, th.nn.Module):
        th.save(L.state_dict(), os.path.join(MODEL_DIR, f"{tag}.pt"))
    return dict(name=L.name, cite=L.cite, kind=L.kind, wins=getattr(L, "wins", ""), tag=tag,
                curve=pr.curve, val_err_cm=val_err, params=mz.count_params(L)[0], train_s=train_s, **m)


def main():
    only = [t for t in os.environ.get("MZ_ONLY", "").split(",") if t]
    learners = [(c, t) for c, t in LEARNERS if (not only or t in only)]
    print(f"device {DEVICE} | budget {BUDGET:,} eps | maze obs {MZ_TRAIN.observation_space.shape[0]} "
          f"act {MZ_TRAIN.action_space.shape[0]} | 108 MC-Maze WITH collision, split "
          f"{len(TRAIN_IDX)}/{len(VAL_IDX)}/{len(TEST_IDX)} train/val/TEST (seed 0) | {len(learners)} model(s)\n",
          flush=True)
    rand = mz.RandomFloor(MZ_TRAIN.action_space.shape[0])
    print(f"random-floor (test): err={maze_metrics(rand, MZ_TEST)['err_cm']:.1f}cm\n", flush=True)
    order = [t for _, t in LEARNERS]
    prior = {r["tag"]: r for r in json.load(open(RESULTS_JSON))} if (only and os.path.exists(RESULTS_JSON)) else {}
    for cls, tag in learners:
        try:
            r = run_one(cls, tag)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{tag:14s} FAILED: {e}", flush=True); continue
        prior[tag] = r
        print(f"{r['name']:42s} err={r['err_cm']:5.1f}cm reach5={r['reach5']:4.0f}% "
              f"time={r['reach_time_ms']:5.0f}ms move={r['path_cm']:5.1f}cm wall={r['in_barrier_pct']:4.1f}% "
              f"{r['train_s']:5.0f}s", flush=True)
        json.dump([prior[t] for t in order if t in prior], open(RESULTS_JSON, "w"), indent=1)
    print(f"\nsaved {len([t for t in order if t in prior])} models -> {MODEL_DIR}\nresults -> {RESULTS_JSON}", flush=True)


def main_ray(budget=BUDGET, gpu_frac=0.5):
    """Train all models as PARALLEL INSTANCES across both GPUs (instance-level parallelism -- each
    model is one Ray task on a GPU fraction; the models are small so several pack per card). Same
    fair setup and same per-model tuned hyperparameters as the sequential main(); ~3x faster."""
    import ray
    only = [t for t in os.environ.get("MZ_ONLY", "").split(",") if t]
    tags = [t for _, t in LEARNERS if (not only or t in only)]
    if ray.is_initialized(): ray.shutdown()
    _ng = len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x != ""])
    ray.init(num_gpus=_ng, ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)   # respect a pinned GPU
    print(f"parallel training {len(tags)} model(s) across {_ng} GPU(s) (~{int(1/gpu_frac)} per card) | "
          f"budget {budget:,} | split {len(TRAIN_IDX)}/{len(VAL_IDX)}/{len(TEST_IDX)}\n", flush=True)
    NB = os.path.dirname(os.path.abspath(__file__))

    @ray.remote(num_gpus=gpu_frac, max_retries=0)
    def _one(tag):
        import sys
        sys.path.insert(0, NB); sys.path.insert(0, os.path.join(NB, "..", "nlb_tools"))
        import maze_train as MT                                   # re-imports -> builds its envs on THIS GPU
        cls = next(c for c, t in MT.LEARNERS if t == tag)
        return MT.run_one(cls, tag, budget=budget)

    futs = {tag: _one.remote(tag) for tag in tags}
    order = [t for _, t in LEARNERS]
    prior = {r["tag"]: r for r in json.load(open(RESULTS_JSON))} if (only and os.path.exists(RESULTS_JSON)) else {}
    for tag, fut in futs.items():
        try:
            r = ray.get(fut); prior[tag] = r
            print(f"{r['name']:42s} err={r['err_cm']:5.1f}cm reach5={r['reach5']:4.0f}% "
                  f"time={r['reach_time_ms']:5.0f}ms move={r['path_cm']:5.1f}cm wall={r['in_barrier_pct']:4.1f}% "
                  f"(TEST) val_err={r['val_err_cm']:.1f}cm", flush=True)
        except Exception as e:
            print(f"{tag:14s} FAILED: {type(e).__name__}: {e}", flush=True)
        json.dump([prior[t] for t in order if t in prior], open(RESULTS_JSON, "w"), indent=1)
    ray.shutdown()
    print(f"\nsaved {len([t for t in order if t in prior])} models (parallel) -> {MODEL_DIR}\nresults -> {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    (main_ray(gpu_frac=float(os.environ.get("MZ_GPU_FRAC", 0.25)))   # smaller => more models train concurrently
     if os.environ.get("MZ_RAY") else main())
