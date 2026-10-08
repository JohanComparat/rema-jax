"""Make every figure of docs/photoz_finders.rst (each script in its own process).

    python docs/figures/photoz_finders/make_all.py [--only fig_null ...]

Inputs: $REMA_PHOTOZ and $REMA_EXTERNAL (see common.py); a figure whose inputs are missing is
skipped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig_errscale", "fig_null", "fig_counts", "fig_external", "fig_overlap", "fig_redshift",
           "fig_colour"]

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--only", nargs="+", default=SCRIPTS)
a = p.parse_args()
for s in a.only:
    print(f"== {s}")
    r = subprocess.run([sys.executable, str(HERE / f"{s}.py")], cwd=HERE)
    if r.returncode:
        raise SystemExit(f"{s} failed")
