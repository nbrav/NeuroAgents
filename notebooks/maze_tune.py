"""Distributed PER-MODEL hyperparameter fine-tuning for the biologically-plausible maze learners.

The plausible rules are hard to tune, so each gets its OWN Ray Tune search (learning rate + a
rule-specific constant + the spinal-reflex gains, from maze_hp.SEARCH/REFLEX), fit on the TRAIN
mazes and selected on VAL. Many trials run as parallel INSTANCES across BOTH GPUs (the models are
tiny, ~12k params, so several pack per card) -- this is instance-level parallelism, NOT splitting one
training across GPUs. The maze OBJECTIVE (collide_w, effort_w) is held fixed so the comparison stays
fair; only each model's learning machinery + reflex calibration move. Best HP -> save_monkey/maze_best_hp.json,
which maze_train.py then applies. TEST puzzles are never touched here.

Run:  .venv/bin/python notebooks/maze_tune.py            (writes save_monkey/maze_best_hp.json)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools"))

NB_DIR = os.path.dirname(os.path.abspath(__file__))
NLB = os.path.join(NB_DIR, "..", "nlb_tools")
BEST_JSON = os.environ.get("MZ_BEST_JSON") or os.path.join(NB_DIR, "..", "save_monkey", "maze_best_hp.json")


def maze_trainable(config):
    """One trial: build config['model'] with its hyperparameters, fit on TRAIN mazes, score on VAL."""
    import sys
    sys.path.insert(0, config["nb_dir"]); sys.path.insert(0, config["nlb"])
    import torch as th
    import motor_zoo as mz, plausible_learners as pl, maze_env, maze_hp
    from ray import tune
    tr, va, te = maze_env.maze_split(seed=0)                                    # test (te) untouched
    MZT = maze_env.make_maze_env(mz.DEVICE, conditions=tr, random_cond=True)    # FIXED objective (fair)
    MZV = maze_env.make_maze_env(mz.DEVICE, conditions=va, random_cond=False)
    pl.configure(mz.morph_head(MZT), mz.obs_norm, mz.OpCounter)
    th.manual_seed(0)
    L = maze_hp.build(config["model"], config, MZT)
    L.fit(MZT, config["budget"], lambda *a, **k: None, batch=int(config["batch"]))
    with th.no_grad():
        o, i = MZV.reset(options={"batch_size": 256}); st = L.init_state(256)
        n = int(MZV.max_ep_duration / MZV.dt); hit = 0.0
        for _ in range(n):
            a, st = L.act(o, st); o, *_ = MZV.step(a)
            hit += float((MZV.maze_collision(MZV.states["fingertip"]) > 0).float().mean())
        d = th.linalg.vector_norm(MZV.states["fingertip"] - MZV.goal[:, :2], dim=-1)
        r5 = 100 * (d < 0.05).float().mean().item(); barr = 100 * hit / n; err = 100 * d.mean().item()
    tune.report({"score": r5 - barr, "val_reach5": r5, "val_in_barrier": barr, "val_err_cm": err})


def _space(tag, budget):
    from ray import tune
    import maze_hp
    d = {}
    for k, rng in maze_hp.full_space(tag).items():
        if len(rng) == 3 and rng[2] == "log":
            d[k] = tune.loguniform(rng[0], rng[1])
        elif len(rng) == 3 and rng[2] == "int":
            d[k] = tune.randint(rng[0], rng[1] + 1)     # inclusive integer knob (n_infer / rank / hidden)
        else:
            d[k] = tune.uniform(rng[0], rng[1])
    d["batch"] = tune.choice([64, 128, 256, 512])       # OPTIMAL BATCH IS MODEL-SPECIFIC -> tune it per model
    d.update(model=tag, budget=budget, nb_dir=NB_DIR, nlb=NLB)
    return d


def tune_all(samples=18, budget=6000, gpu_frac=0.33, models=None, verbose=True):
    """Per-model Ray fine-tuning across both GPUs. Returns {tag: best_hp}; also writes BEST_JSON."""
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch          # SOTA: TPE Bayesian search over the wide space
    import maze_hp
    tags = models or maze_hp.TUNABLE
    if ray.is_initialized(): ray.shutdown()
    _ng = len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x != ""])
    ray.init(num_gpus=_ng, ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
    best = json.load(open(BEST_JSON)) if os.path.exists(BEST_JSON) else {}   # MERGE: a subset re-tune keeps the rest
    rows = []
    for tag in tags:
        tuner = tune.Tuner(tune.with_resources(maze_trainable, {"gpu": gpu_frac}),
                           param_space=_space(tag, budget),
                           tune_config=tune.TuneConfig(num_samples=samples, metric="score", mode="max",
                                                       search_alg=OptunaSearch(metric="score", mode="max")))
        res = tuner.fit()
        b = res.get_best_result(metric="score", mode="max")
        best[tag] = {k: float(v) for k, v in b.config.items() if k not in ("model", "budget", "nb_dir", "nlb")}
        rows.append(dict(model=tag, val_reach5=b.metrics["val_reach5"], val_in_barrier=b.metrics["val_in_barrier"],
                         val_err_cm=b.metrics["val_err_cm"], **best[tag]))
        if verbose:
            print(f"{tag:11s} val reach={b.metrics['val_reach5']:4.0f}% wall={b.metrics['val_in_barrier']:4.1f}% "
                  f"err={b.metrics['val_err_cm']:4.1f}cm  |  "
                  f"{', '.join(f'{k}={v:.3g}' for k, v in best[tag].items())}", flush=True)
    ray.shutdown()
    os.makedirs(os.path.dirname(BEST_JSON), exist_ok=True)
    json.dump(best, open(BEST_JSON, "w"), indent=1)
    return best, rows


if __name__ == "__main__":
    _tags = [t for t in os.environ.get("MZ_TAGS", "").split(",") if t] or None   # subset override
    tune_all(samples=int(os.environ.get("MZ_TUNE_SAMPLES", 18)),
             budget=int(os.environ.get("MZ_TUNE_BUDGET", 6000)),
             gpu_frac=float(os.environ.get("MZ_GPU_FRAC", 0.33)),   # smaller => more trials pack per GPU
             models=_tags)
    print(f"\nsaved tuned hyperparameters -> {BEST_JSON}")
