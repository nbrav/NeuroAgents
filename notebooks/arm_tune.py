"""Per-model Optuna fine-tuning for the biologically-plausible learners on the NON-MAZE (reaching)
task in 4-monkey-net. Objective: MINIMISE endpoint error (cm) and time-to-reach, MAXIMISE clear rate.
Baselines are NOT tuned. Runs many trials in parallel on ONE GPU (set CUDA_VISIBLE_DEVICES=0).
Best HP -> save_monkey/reaching_best_hp.json, which arm_train.py then applies.

Run:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python notebooks/arm_tune.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools"))

NB_DIR = os.path.dirname(os.path.abspath(__file__))
NLB = os.path.join(NB_DIR, "..", "nlb_tools")
BEST_JSON = os.path.join(NB_DIR, "..", "save_monkey", "reaching_best_hp.json")


def arm_trainable(config):
    """One trial: build config['model'] with its HP, train on the arm reach task, score on reach."""
    import sys
    sys.path.insert(0, config["nb_dir"]); sys.path.insert(0, config["nlb"])
    import torch as th
    import motor_zoo as mz, plausible_learners as pl, maze_hp
    from ray import tune
    env = mz.make_arm_env(mz.DEVICE)
    pl.configure(mz.morph_head(env), mz.obs_norm, mz.OpCounter)
    th.manual_seed(0)
    L = maze_hp.build(config["model"], config, env)
    L.fit(env, config["budget"], lambda *a, **k: None, batch=int(config["batch"]))
    m = mz.eval_metrics(env, L)
    err, comp = m["err_cm"], m["completion"]
    # score = maximise clear rate, minimise cm error (time-to-reach is captured by the summed-error reward)
    tune.report({"score": comp - 0.8 * err, "err_cm": err, "completion": comp})


def _space(tag, budget):
    from ray import tune
    import maze_hp
    d = {}
    for k, rng in maze_hp.full_space(tag).items():
        if len(rng) == 3 and rng[2] == "log":
            d[k] = tune.loguniform(rng[0], rng[1])
        elif len(rng) == 3 and rng[2] == "int":
            d[k] = tune.randint(rng[0], rng[1] + 1)
        else:
            d[k] = tune.uniform(rng[0], rng[1])
    d["batch"] = tune.choice([64, 128, 256, 512])
    d.update(model=tag, budget=budget, nb_dir=NB_DIR, nlb=NLB)
    return d


def tune_all(samples=40, budget=20000, gpu_frac=0.12, models=None, verbose=True):
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    import maze_hp
    tags = models or maze_hp.TUNABLE
    if ray.is_initialized(): ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)   # 1 visible GPU
    best = json.load(open(BEST_JSON)) if os.path.exists(BEST_JSON) else {}      # MERGE (subset-safe)
    for tag in tags:
        tuner = tune.Tuner(tune.with_resources(arm_trainable, {"gpu": gpu_frac}),
                           param_space=_space(tag, budget),
                           tune_config=tune.TuneConfig(num_samples=samples, metric="score", mode="max",
                                                       search_alg=OptunaSearch(metric="score", mode="max")))
        res = tuner.fit()
        b = res.get_best_result(metric="score", mode="max")
        best[tag] = {k: float(v) for k, v in b.config.items() if k not in ("model", "budget", "nb_dir", "nlb")}
        if verbose:
            print(f"{tag:11s} reach err={b.metrics['err_cm']:5.1f}cm clear={b.metrics['completion']:4.0f}%  |  "
                  f"{', '.join(f'{k}={v:.3g}' for k, v in best[tag].items())}", flush=True)
        os.makedirs(os.path.dirname(BEST_JSON), exist_ok=True)
        json.dump(best, open(BEST_JSON, "w"), indent=1)                          # write after each model
    ray.shutdown()
    return best


if __name__ == "__main__":
    _tags = [t for t in os.environ.get("MZ_TAGS", "").split(",") if t] or None
    tune_all(samples=int(os.environ.get("MN_TUNE_SAMPLES", 40)),
             budget=int(os.environ.get("MN_TUNE_BUDGET", 20000)),
             gpu_frac=float(os.environ.get("MN_GPU_FRAC", 0.12)), models=_tags)
    print(f"\nsaved reaching HP -> {BEST_JSON}")
