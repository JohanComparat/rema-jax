"""Make every figure of docs/cosmology_sensitivity.rst (each script in its own process).

    python docs/figures/cosmology_sensitivity/make_all.py [--only fig_response ...]

Inputs: $REMA_WORK/cosmo_sens (see common.py); a figure whose inputs are missing is skipped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig_distances", "fig_response", "fig_rerun", "fig_counts", "fig_constraints", "results_table", "calib_table"]

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--only", nargs="+", default=SCRIPTS)
a = p.parse_args()
for s in a.only:
    print(f"== {s}")
    r = subprocess.run([sys.executable, str(HERE / f"{s}.py")], cwd=HERE)
    if r.returncode:
        raise SystemExit(f"{s} failed")
