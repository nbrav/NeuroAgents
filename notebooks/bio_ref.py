"""Biological behavioural precision (endpoint error, cm) — single source of truth for the reference
lines drawn on every model-vs-biology plot (4-monkey-net non-maze + 4-11monkey-net maze). Numbers are
MEASURED from our own datasets except where marked (lit.). Reproduction snippets are in the project notes.

NON-MAZE reaching (4-monkey-net):
  monkey  Area2_Bump S1   1.5 cm   MEASURED, n=364   (raw hand endpoint 1.47±0.77 cm to target centre)
  iBCI    human IC cursor 1.6 cm   MEASURED, n=185   (normalised 0.184 × 8.8 cm reach; tetraplegic BMI,
                                                      native units are screen a.u. -> mapped to the reach scale)
  MEG     human MEG reach ~1.0 cm  ABLE-BODIED (lit.) (our MEG .mat ships NO endpoint coords -- only 306-ch
                                                      MEG + 3 accelerometer axes; a healthy human's centre-out
                                                      reach endpoint error is ~0.5-1 cm, so this is a cited
                                                      reference, NOT a number derived from our data)
MAZE (4-11monkey-net):
  maze    MC_Maze M1      0.8 cm   MEASURED, n=2295  (cursor endpoint 0.83±0.47 cm; 0.53 cm acceptance window)

Model err_cm is on this SAME cm axis (motor_zoo err_cm = 100*dist_m; SUCCESS_CM=5, strict 2 cm), so the
monkey/maze measured cm values are directly comparable; the iBCI value is a reach-scaled normalisation and
the MEG value is an able-bodied literature reference (both labelled as such).
"""
from matplotlib.lines import Line2D

# key: (value_cm, legend_label, colour, linestyle, measured_from_our_data)
REACH = {
    "monkey": (1.5, "monkey S1 · Area2_Bump (measured)",        "#e9c46a", "-",  True),
    "ibci":   (1.6, "human iBCI cursor · IC (measured)",        "#c9960a", "-",  True),
    "meg":    (1.0, "human MEG reach · able-bodied (lit. ref)", "#9c6f00", ":",  False),
}
MAZE = {
    "maze":   (0.8, "monkey cursor · MC_Maze (measured)",       "#e9c46a", "-",  True),
}


def draw(ax, refs=REACH, band=True, label=True, fs=8):
    """Draw the biological reference lines (+ a spanning 'biological precision' band) on a cm-axis plot.
    Returns Line2D legend handles so the caller can fold them into its own legend."""
    vals = [v[0] for v in refs.values()]
    if band:
        ax.axhspan(0, max(vals), color="#457b9d", alpha=0.06, zorder=0)   # region where a model matches biology
    handles = []
    for v, lab, col, ls, meas in refs.values():
        ax.axhline(v, color=col, ls=ls, lw=2.2 if meas else 1.5, zorder=2)
        handles.append(Line2D([0], [0], color=col, ls=ls, lw=2.2 if meas else 1.5, label=lab))
    return handles


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.set_ylim(0, 10)
    h = draw(ax); assert len(h) == 3
    hm = draw(ax, MAZE, band=False); assert len(hm) == 1
    print("bio_ref OK:", {k: v[0] for k, v in REACH.items()}, "| maze", MAZE["maze"][0], "cm")
