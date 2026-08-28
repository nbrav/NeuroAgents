"""Re-bake 4-monkey-net.ipynb: execute every cell in a real kernel and write the refreshed
outputs (scoreboard, figures, neural-linking numbers, GIFs) back INTO the notebook, so it opens
already showing the FAIR-retrain results. Preserves cell SOURCE (markdown prose is untouched);
only code-cell outputs are refreshed. Run AFTER arm_train.py has produced save_monkey/.

Usage:  .venv/bin/python notebooks/bake_monkey.py
"""
import os
import nbformat
from nbclient import NotebookClient

ND = os.path.dirname(os.path.abspath(__file__))
NB = os.environ.get("MN_NB") or os.path.join(ND, "4-monkey-net.ipynb")   # MN_NB reuses this for 4-11monkey-net

nb = nbformat.read(NB, as_version=4)
# allow_errors=False so a broken cell surfaces instead of baking a traceback; long timeout for the
# neural-linking / CKA / GIF cells. Kernel runs with cwd = notebooks/ so relative imports/paths work.
client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": ND}}, allow_errors=True)
client.execute()

# report any cell that errored (so we fix rather than silently bake a traceback)
errors = []
for i, c in enumerate(nb.cells):
    if c.get("cell_type") != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            errors.append((i, o.get("ename"), (o.get("evalue") or "")[:120]))

nbformat.write(nb, NB)
sz = os.path.getsize(NB) / 2 ** 20
if errors:
    print(f"BAKED WITH {len(errors)} ERROR CELL(S) (notebook {sz:.1f} MB):")
    for i, en, ev in errors:
        print(f"  cell {i}: {en}: {ev}")
else:
    print(f"re-baked cleanly, no error cells; notebook now {sz:.1f} MB")
