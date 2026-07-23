"""Per-model maze hyperparameters for the biologically-plausible family, shared by the Ray tuner
(4-tuning-net) and maze_train.py. The plausible rules are hard to tune, so each gets its OWN
hyperparameter search (learning rate + a rule-specific constant + the spinal-reflex gains), fit on
the TRAIN mazes and selected on VAL, to squeeze out maximum accuracy. The maze OBJECTIVE constants
(collide_w, effort_w) are NOT tuned -- they are the task definition, held fixed for every model so
the comparison stays fair; only each model's own learning machinery + its reflex calibration move."""
import motor_zoo as mz
import plausible_learners as pl

# tag -> {hyperparameter: (low, high) uniform, or (low, high, "log") loguniform}
# model-specific kwargs (consumed by the class __init__) + the shared spinal-reflex gains below.
SEARCH = {
    "eprop":      {"lr": (0.008, 0.05, "log"),  "tau_e": (0.02, 0.15)},
    "rtrrl":      {"lr": (0.004, 0.03, "log"),  "tau_e": (0.05, 0.25)},
    "btsp":       {"lr": (0.008, 0.06, "log"),  "tau_slow": (0.5, 1.8)},
    "rstdp":      {"lr": (0.0008, 0.01, "log"), "tau_c": (0.05, 0.25)},
    "predcode":   {"lr": (0.03, 0.18, "log"),   "lr_g": (0.002, 0.04, "log")},
    "hebb3":      {"lr": (0.008, 0.05, "log"),  "gain": (2.0, 7.0)},
    "dendritron": {"lr": (0.01, 0.08, "log")},
    # KINESIS is morphological (analytic policy gradient, no spinal reflex) -- include it per the goal.
    "kinesis":    {"lr": (5e-4, 5e-3, "log"),   "f_scale": (200.0, 900.0)},
}
# shared spinal reflex, tuned per model (each plausible rule gets its own best-calibrated reach +
# avoidance reflex). KINESIS does not use it (it backprops the plant), so it is excluded there.
REFLEX = {"reflex_avoid": (3.0, 18.0), "reflex_kp": (0.7, 1.7), "reflex_kd": (0.06, 0.28)}

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
    """Build a learner with its tuned hyperparameters (reflex gains set first, model kwargs applied)."""
    apply_reflex(config)
    kw = {k: config[k] for k in SEARCH[tag] if config.get(k) is not None}
    return CLS[tag](env, **kw)
