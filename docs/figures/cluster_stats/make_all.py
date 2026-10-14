"""Make every figure of docs/cluster_statistics_dr11.rst (each script in its own process).

    python docs/figures/cluster_stats/make_all.py [--only fig_wp ...]

Inputs: $REMA_WORK/cluster_sumstat (see common.py) and, for fig_compare_c25, the measurements of
Comparat et al. (2025) in $CXG_DATA; a figure whose inputs are missing is skipped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig_wp", "fig_esd", "fig_xgal", "fig_compare_c25"]

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--only", nargs="+", default=SCRIPTS)
a = p.parse_args()
for s in a.only:
    print(f"== {s}")
    r = subprocess.run([sys.executable, str(HERE / f"{s}.py")], cwd=HERE)
    if r.returncode:
        raise SystemExit(f"{s} failed")
