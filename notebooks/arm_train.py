"""Reproducible FAIR arm trainer for 4-monkey-net.

Trains all 13 learners (one plant-agnostic definition each, from motor_zoo + plausible_learners)
on the monkey-matched MotorNet RigidTendonArm26, under the SAME fair setup as 4-train-net:
  * ONE shared objective -- MotorNet's L1 fingertip->goal task loss (motor_core), NO demonstrator
  * batch 32 (MotorNet's own setting), Adam(1e-3), grad-clip 1.0, equal-capacity policies
  * morphological/plausible family actuates through the LEAK-FREE arm head (obs + fixed anatomy)

Writes save_monkey/models/{tag}.pt + save_monkey/results.json -- the artifacts 4-monkey-net loads.
The original script that produced these was not in the repo; this fills that reproducibility gap.

Run:  MN_BUDGET=60000 .venv/bin/python notebooks/arm_train.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch as th
import motor_zoo as mz
import plausible_learners as pl

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_DIR = os.path.join(REPO, "save_monkey", "models")
RESULTS_JSON = os.path.join(REPO, "save_monkey", "results.json")
BUDGET = int(os.environ.get("MN_BUDGET", 60_000))   # episodes/method; 4-train-net uses 100k
THRESH = 15.0                                        # cm: "solved enough" line (sample-efficiency col)
DEVICE = mz.DEVICE

env = mz.make_arm_env(DEVICE)
pl.configure(mz.morph_head(env), mz.obs_norm, mz.OpCounter)   # plausible family -> fair arm head

# (class, tag). Every entry is teacher-free: cls(env). The deep-RL trio default to teacher=None
# (plain model-free off-policy RL on the shared reward). Same 13 as 4-train-net.
LEARNERS = [
    (mz.MotorNetRef, "motornet_ref"), (mz.BPTTGRU, "bptt_gru"), (mz.SHAC, "shac"),
    (mz.SAC, "sac"), (mz.FastTD3, "fasttd3"), (mz.SimbaV2, "simbav2"),
    (pl.EProp, "eprop"), (pl.RTRRL, "rtrrl"), (pl.BTSP, "btsp"),
    (mz.Kinesis, "kinesis"), (pl.RSTDP, "rstdp"), (pl.PredictiveCoding, "predcode"),
    (pl.Hebb3, "hebb3"), (mz.Dendritron, "dendritron"),
]


def arm_zero_shot(learner):
    """REAL OOD eval on the arm. motor_zoo.zero_shot is the point-mass ball-weight version (it
    ignores make_env_fn and builds mass-aware POINT-MASS envs), so the arm needs its own: force
    fields (endpoint loads the arm never trained under) + weakened muscles, on fresh arm envs."""
    perts = [
        ("force field +x", {"endpoint_load": mz._load(4.0, 0.0, DEVICE)}, None),
        ("force field -y", {"endpoint_load": mz._load(0.0, -4.0, DEVICE)}, None),
        ("curl load",      {"endpoint_load": mz._load(3.0, 3.0, DEVICE)}, None),
        ("weak muscles",   {}, lambda e: e.effector.muscle.max_iso_force.mul_(0.5)),
    ]
    out = {}
    for name, skw, mut in perts:
        e = mz.make_arm_env(DEVICE)
        if mut is not None: mut(e)
        out[name] = mz.evaluate(e, learner, step_kwargs=skw)
    return out


import maze_hp
# REACHING-specific tuned HP (arm_tune.py); the maze HP does NOT transfer to reaching, so this file
# is the reaching min-max result. Overridable for ablations via MN_BEST_JSON.
_BEST_HP_JSON = os.environ.get("MN_BEST_JSON") or os.path.join(REPO, "save_monkey", "reaching_best_hp.json")
BEST_HP = json.load(open(_BEST_HP_JSON)) if os.path.exists(_BEST_HP_JSON) else {}

def run_one(cls, tag, budget=BUDGET, bs=32):
    th.manual_seed(0); np.random.seed(0)
    if tag in BEST_HP:
        L = maze_hp.build(tag, BEST_HP[tag], env)          # Optuna-tuned HP (plausible + KINESIS)
        bs = int(BEST_HP[tag].get("batch", bs))
    else:
        L = cls(env)                                       # baselines untouched
    pr = mz.Probe(env, every_eps=max(1, budget // 40), budget=budget, track_best=True)   # early stop on val
    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    L.fit(env, budget, pr, batch=bs)
    if getattr(pr, "best_state", None) is not None:
        L.load_state_dict(pr.best_state)                   # best-val checkpoint (avoids late divergence)
    train_s = time.perf_counter() - t0
    m = mz.eval_metrics(env, L)
    zs = arm_zero_shot(L)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if isinstance(L, th.nn.Module):
        th.save(L.state_dict(), os.path.join(MODEL_DIR, f"{tag}.pt"))
    peak_gb = (pr.peak_mem / 2 ** 30) if th.cuda.is_available() else float("nan")
    r = dict(
        name=L.name, cite=L.cite, kind=L.kind, wins=getattr(L, "wins", ""), tag=tag,
        curve=pr.curve, acc=m["err_cm"], ret=m["ret"],
        completion=m["completion"], completion2=m["completion2"], completion5=m["completion"],
        ctrl_sparse=m["ctrl_sparse"], cocontract=m["cocontract"], eff_dim=m["eff_dim"],
        eps_to=pr.eps_to(THRESH), eps_conv=pr.eps_to_converge(),
        zs=zs, zs_mean=float(np.mean(list(zs.values()))),
        boot=False, obs=False, params=mz.count_params(L)[0], train_s=train_s, peak_gb=peak_gb,
    )
    # small per-model manifest (mirrors save_learner), for the analysis notebook
    with open(os.path.join(MODEL_DIR, f"{tag}.json"), "w") as f:
        json.dump(dict(name=L.name, cite=L.cite, kind=L.kind, wins=getattr(L, "wins", ""), tag=tag,
                       obs_dim=env.observation_space.shape[0], n_muscles=env.action_space.shape[0],
                       params=r["params"], acc=r["acc"], completion5=r["completion5"]), f, indent=2)
    return r


def main():
    print(f"device {DEVICE} | budget {BUDGET:,} eps/method | arm obs {env.observation_space.shape[0]} "
          f"muscles {env.action_space.shape[0]}\n", flush=True)
    nm = env.action_space.shape[0]
    # MN_ONLY=sac,btsp trains ONLY those tags and MERGES into the existing results.json (the other
    # seeded/deterministic rows reproduce identically, so retraining just a changed model is exact).
    only = [t for t in os.environ.get("MN_ONLY", "").split(",") if t]
    learners = [(c, t) for c, t in LEARNERS if (not only or t in only)]
    print(f"training {len(learners)} model(s){' (subset '+','.join(only)+')' if only else ''}")
    floors = dict(random=mz.evaluate(env, mz.RandomFloor(nm)), silent=mz.evaluate(env, mz.SilentFloor(nm)))
    print(f"floors: random {floors['random']:.1f} cm | silent {floors['silent']:.1f} cm\n", flush=True)
    order = [t for _, t in LEARNERS]
    prior = {r["tag"]: r for r in json.load(open(RESULTS_JSON))} if (only and os.path.exists(RESULTS_JSON)) else {}
    for cls, tag in learners:
        try:
            r = run_one(cls, tag)
        except Exception as e:                       # a rule that cannot run is a RESULT, not a crash
            import traceback; traceback.print_exc()
            print(f"{tag:14s} FAILED: {type(e).__name__}: {e}", flush=True); continue
        prior[tag] = r
        print(f"{r['name']:42s} acc={r['acc']:6.1f}cm compl5={r['completion5']:5.1f}% "
              f"compl2={r['completion2']:5.1f}% ood={r['zs_mean']:6.1f} params={r['params']:6d} "
              f"{r['train_s']:5.0f}s", flush=True)
        json.dump([prior[t] for t in order if t in prior], open(RESULTS_JSON, "w"), indent=1)   # incremental, keep order
    print(f"\nsaved {len([t for t in order if t in prior])} models -> {MODEL_DIR}\nresults -> {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
