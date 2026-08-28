# Biologically-plausible NeuroAI — learning like the brain

![Plausible ANNs behave like the monkey and keep its advantages](save/conclusion_fig.png)

**One task, several nervous systems.** A monkey (S1 reaching + its own MC_Maze), **two human datasets** — an intracortical iBCI cursor and a non-invasive MEG reach — and 13 ANNs perform the *same* movements (free reaches and the monkey's own **MC_Maze** puzzles, real solid walls) with the same input, output, and objective (reach fast, precisely, without hitting walls).

**Brain-plausible ANNs behave like the biology.** Learning **only locally** — no global backprop, no weight transport — rules such as e-prop, RFLO, predictive coding and dendritic learning reach the targets and sit *right beside the monkey's and humans' behavioural precision* (measured for the monkey S1 reach and the human iBCI cursor; an able-bodied reference for the MEG reach), where accurate-but-non-biological backprop networks do not.

**And they keep the brain's advantages.** Because the learning is local, they **generalize** to new ball masses and mazes, **don't forget** old skills when retrained on new ones, and **degrade gracefully** under sensory noise and synaptic damage — exactly where global backprop is brittle.

## What actually solves the maze — a conclusive factorial

![Factorial: perception (vision) × plan (demonstrator) × task (maze vs free)](save/conclusion_factorial.png)

We separated **wall-model** (solid physical block vs weighted penalty), **perception** (spiking vision vs blind), and **plan** (monkey demonstrator vs autonomous), on **both settings** (walled maze + barrier-free reach), across all 108 puzzles:

* **Free reaches are solved by ANNs** (backprop 100 %, best plausible ~100 %).
* **SOLID-block walls make the maze hard.** The cursor is physically clamped, so a model must *route* — which needs to tell aliased mazes apart, but 35 targets drive 72 different walled routes (same input, different paths). On held-out puzzles autonomy caps at **~29 % (blind), ~15–21 % (vision)**; the monkey's demonstrated **plan** lifts it to **67 % (plausible) / 90 % (backprop)** (a plan helps more than sight).
* **Penalty walls — the monkey's REAL virtual MC-Maze — change everything.** When the cursor moves freely and crossing a wall is a weighted loss (exactly the task the monkey solved), a model only has to *reach* the target, so success **generalizes** and the **biologically-plausible local rules hit ~100 % on held-out puzzles, no vision** (Predictive coding & KINESIS 100 %, e-prop & RFLO 93 %, BTSP & 3-factor-Hebb 86 %) — matching backprop and the monkey/human.

> **Bottom line:** a local, brain-like learning rule matches backprop on the movements ANNs can do and inherits the brain's generalization, memory, and robustness — and on the monkey's *actual* (weighted-penalty) maze it reaches **~100 %, no vision, generalizing to held-out puzzles.** The one place the brain stays ahead is the artificial *solid-wall* variant, where physical blocking exposes an observability limit no ANN scales — a limit closed by giving the model the monkey's plan.
