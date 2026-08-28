"""Per-model maze hyperparameters for the biologically-plausible family, shared by the Ray tuner
(4-tuning-net) and maze_train.py. The plausible rules are hard to tune, so each gets its OWN
hyperparameter search (learning rate + a rule-specific constant + the spinal-reflex gains), fit on
the TRAIN mazes and selected on VAL, to squeeze out maximum accuracy. The maze OBJECTIVE constants
(collide_w, effort_w) are NOT tuned -- they are the task definition, held fixed for every model so
the comparison stays fair; only each model's own learning machinery + its reflex calibration move."""
import motor_zoo as mz
import plausible_learners as pl

# tag -> {hyperparameter: (low, high) uniform | (low, high, "log") loguniform | (low, high, "int") integer}
# WIDE, per-model search over EACH rule's real knobs (learning rate + dynamics/plasticity constants +
# modest capacity), so the fine-tuner explores genuinely DIFFERENT parameterisations toward the best
# achievable accuracy -- not the same narrow band repeated. Capacity knobs (kinesis hidden, dendritron
# rank) let the local rules add representational power; param counts are reported so fairness stays visible.
# CAPACITY (Nr = reservoir units, base RES_NR=4096 -> ~12.3k plastic params) and REGULARISATION
# (lam = L2 weight decay on the readout) are tuned per rule, so each can trade bias vs variance and
# land a bit above/below 12.6k params -- faithful to each paper (same rule + fixed sparse reservoir),
# just sized to its best. rstdp has no L2 term; kinesis is a small MLP tuned by its hidden width.
_NR = (3072, 6144, "int"); _LAM = (1e-5, 1e-2, "log")
SEARCH = {
    "eprop":      {"lr": (0.003, 0.08, "log"),  "tau_e": (0.02, 0.30),  "beta": (0.1, 0.9),  "rho_a": (0.80, 0.99), "Nr": _NR, "lam": _LAM},
    "rtrrl":      {"lr": (0.002, 0.05, "log"),  "tau_e": (0.03, 0.40),  "Nr": _NR, "lam": _LAM},
    "btsp":       {"lr": (0.004, 0.10, "log"),  "tau_slow": (0.3, 2.5), "p_plateau": (0.3, 1.0), "Nr": _NR, "lam": _LAM},
    "rstdp":      {"lr": (0.0003, 0.02, "log"), "tau_c": (0.03, 0.40),  "vth": (0.10, 0.60), "Nr": _NR},
    "predcode":   {"lr": (0.01, 0.30, "log"),   "lr_g": (0.001, 0.06, "log"), "n_infer": (3, 12, "int"), "Nrep": _NR, "lam": _LAM},
    "hebb3":      {"lr": (0.004, 0.08, "log"),  "gain": (1.5, 9.0),     "tau_e": (0.2, 1.5), "Nr": _NR, "lam": _LAM},
    "dendritron": {"lr": (0.005, 0.12, "log"),  "rho": (0.9, 1.4), "a": (0.2, 0.8), "sin": (0.5, 1.5), "rank": (8, 64, "int"), "Nr": _NR, "lam": _LAM},
    # KINESIS is morphological (analytic policy gradient, no spinal reflex) -- include it per the goal.
    "kinesis":    {"lr": (3e-4, 8e-3, "log"),   "f_scale": (150.0, 1000.0), "hidden": (48, 72, "int")},   # ~9k-18.8k params (matches the reservoir band; 57 = 12.3k baseline)
}
# shared spinal reflex, tuned per model (each plausible rule gets its own best-calibrated reach +
# avoidance reflex). KINESIS does not use it (it backprops the plant), so it is excluded there.
REFLEX = {"reflex_avoid": (2.0, 22.0), "reflex_kp": (0.5, 2.0), "reflex_kd": (0.04, 0.35)}

CLS = {"eprop": pl.EProp, "rtrrl": pl.RTRRL, "btsp": pl.BTSP, "rstdp": pl.RSTDP,
       "predcode": pl.PredictiveCoding, "hebb3": pl.Hebb3, "dendritron": mz.Dendritron,
       "kinesis": mz.Kinesis}

TUNABLE = list(SEARCH.keys())      # the biologically-plausible rules + KINESIS we fine-tune (NOT the
                                   # non-plausible baselines: BPTT-GRU / SHAC / SAC / FastTD3 / Simba / ref)


def full_space(tag):
    """The complete range dict for one model (model-specific kwargs + spinal-reflex gains, except
    KINESIS which has no reflex)."""
    base = dict(SEARCH[tag])
    if tag != "kinesis":
        base.update(REFLEX)
    return base


def apply_reflex(config):
    """Set the shared spinal-reflex gains from a config (module constants). Safe per-process (Ray
    trial) and per-model-in-sequence (maze_train sets them right before building each model)."""
    if config.get("reflex_avoid") is not None: pl.REFLEX_AVOID = float(config["reflex_avoid"])
    if config.get("reflex_kp") is not None:    pl.REFLEX_KP = float(config["reflex_kp"])
    if config.get("reflex_kd") is not None:    pl.REFLEX_KD = float(config["reflex_kd"])


def build(tag, config, env):
    """Build a learner with its tuned hyperparameters (reflex gains set first, model kwargs applied).
    Integer-typed knobs (n_infer, rank, hidden) are rounded so a continuous sample maps to a valid arg."""
    apply_reflex(config)
    kw = {}
    for k, rng in SEARCH[tag].items():
        v = config.get(k)
        if v is None:
            continue
        kw[k] = int(round(v)) if (len(rng) == 3 and rng[2] == "int") else v
    return CLS[tag](env, **kw)
