"""SLURM driver (scripts/slurm/rema_dr11_blind.sh) with stub sbatch, squeue and rema stages.

The planning, status and bookkeeping commands (rema todo, regions, status) are the real ones;
the heavy stages are never run (sbatch only records its arguments).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rema.io.tables import write_catalog, write_table
from rema.pipeline import file_sha1, read_plan
from rema.sky.regions import sky_header, sweep_box

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "slurm" / "rema_dr11_blind.sh"
SWEEPS = [f"sweep-{ra:03d}m{d + 5:03d}-{ra + 5:03d}m{d:03d}.fits" for ra in (0, 5, 10, 350, 355)
          for d in (0, 5)]
SWEEPS = [s.replace("m000.fits", "p000.fits") for s in SWEEPS]


def _stubs(tmp: Path) -> Path:
    bin_ = tmp / "bin"
    bin_.mkdir()
    (bin_ / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        'n=$(( $(cat "$SB_COUNT" 2>/dev/null || echo 100) + 1 )); echo $n > "$SB_COUNT"\n'
        'printf "%s\\n" "$*" >> "$SB_LOG"\n'
        "echo $n\n")
    (bin_ / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
    real = shutil.which("rema") or f"{sys.executable} -m rema.cli"
    (bin_ / "rema").write_text(
        "#!/usr/bin/env bash\n"
        'case $1 in todo|status|regions) exec ' + real + ' "$@";; esac\n'
        'printf "rema %s\\n" "$*" >> "$REMA_LOG"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    return bin_


def _fake_inputs(tmp: Path):
    dr11 = tmp / "dr11"
    (dr11 / "sweep" / "11.0").mkdir(parents=True)
    for s in SWEEPS:
        (dr11 / "sweep" / "11.0" / s).touch()
    return dr11


def _galaxy_tables(out: Path):
    """Small per-sweep tables (100 galaxies each) with the headers rema ingest writes."""
    rng = np.random.default_rng(1)
    for s in SWEEPS:
        b = sweep_box(s)
        n = 100
        t = {"ID": np.arange(n, dtype=np.int64), "RA": rng.uniform(b.ra_min, b.ra_max, n),
             "DEC": rng.uniform(b.dec_min, b.dec_max, n), "ZSPEC": np.full(n, -1.0, np.float32)}
        write_table(out / "galaxies" / s, t, header={"NZSPEC": 0, "EBVMEAN": 0.03, **sky_header(b)},
                    extname="GALAXIES")


def _randoms_index(out: Path, nrand: int):
    d = out / "randoms_index"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(nrand):
        np.save(d / f"randoms-south-1-{k}.offsets.npy", np.zeros(12 * 64**2 + 1, np.int64))
        (d / f"randoms-south-1-{k}.json").write_text(json.dumps({"nside_index": 64, "nside_fine": 1024}))


def _run(tmp, mode, **env):
    site = ("LEGACYSURVEY_DIR", "CLUSTERS_DIR", "MEMBERS_DIR", "EXTRA_SBATCH", "PART_CPU", "PART_GPU",
            "GPU_GRES", "NRAND", "ACCOUNT", "CONFIG", "CALIB", "CALIB_BOX", "PLAN")
    e = {**{k: v for k, v in os.environ.items() if k not in site}, "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "SB_LOG": str(tmp / "sbatch.log"),
         "SB_COUNT": str(tmp / "count"), "REMA_LOG": str(tmp / "rema.log"), "DR11": str(tmp / "dr11"),
         "OUTDIR": str(tmp / "run"), "JAX_PLATFORMS": "cpu", **env}
    (tmp / "sbatch.log").write_text("")
    r = subprocess.run(["bash", str(DRIVER), mode], env=e, capture_output=True, text=True, timeout=600)
    calls = [l for l in (tmp / "sbatch.log").read_text().splitlines() if l]
    return r, calls


def _opt(call: str, name: str) -> str | None:
    for tok in call.split():
        if tok.startswith(f"--{name}="):
            return tok.split("=", 1)[1]
    return None


@pytest.fixture
def tmp(tmp_path):
    _stubs(tmp_path)
    _fake_inputs(tmp_path)
    return tmp_path


def test_prepare_submits_missing_ingest_and_randoms(tmp):
    r, calls = _run(tmp, "prepare", NRAND="3", CHUNK="4")
    assert r.returncode == 0, r.stderr
    assert len(calls) == 2
    assert _opt(calls[0], "array") == "0-2%20" and calls[0].endswith("ingest")      # 10 sweeps / 4
    assert _opt(calls[1], "array") == "0-2%20" and calls[1].endswith("randoms")
    assert "--calib-suggest" in r.stdout
    # Everything present: nothing to submit; with CALIB_BOX the calibration job runs.
    _galaxy_tables(tmp / "run")
    _randoms_index(tmp / "run", 3)
    r, calls = _run(tmp, "prepare", NRAND="3", CHUNK="4", CALIB_BOX="0 10 -10 0;350 360 -10 0")
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and calls[0].endswith("calib") and _opt(calls[0], "dependency") is None


def test_run_primes_then_array_then_merge(tmp):
    out = tmp / "run"
    _galaxy_tables(out)
    _randoms_index(out, 2)
    calib = tmp / "calib.fits"
    calib.write_bytes(b"calibration")
    r, calls = _run(tmp, "run", NRAND="2", TARGET_AREA="50", CALIB=str(calib), PART_GPU="gpuq",
                    GPU_CONSTRAINT="a100")
    assert r.returncode == 0, r.stderr
    plan, meta = read_plan(out / "regions.fits")
    n = len(plan["REGION_ID"])
    assert n >= 3
    # A region crossing RA 0 (tiles 350-360 and 0-15 are contiguous round the circle).
    assert any(plan["OWN_RA1"] > 360) or any(plan["OWN_RA0"] < 0) or \
        any((plan["OWN_RA0"] == 350) & (plan["OWN_RA1"] > 355))
    prime, regions, merge = calls
    assert prime.endswith("region") and _opt(prime, "array") is not None
    assert "--gres=gpu:1" in prime and _opt(prime, "partition") == "gpuq" and _opt(prime, "constraint") == "a100"
    assert _opt(regions, "dependency") == "afterany:101"
    arr = _opt(regions, "array").split("%")[0]
    ids = set()
    for part in arr.split(","):
        a, _, b = part.partition("-")
        ids |= set(range(int(a), int(b or a) + 1))
    assert ids | {int(_opt(prime, "array"))} == set(range(n))
    assert merge.endswith("merge") and _opt(merge, "dependency") == "afterany:101:102"

    # Two regions done (same plan and calibration), the JAX cache primed: only the others run.
    for rid in (0, 1):
        write_catalog(out / "regions" / f"{rid:04d}" / "clusters.fits", {}, {}, None,
                      {"PLANHASH": meta["PLANHASH"], "CALSHA1": file_sha1(calib)})
    (out / "jax_cache" / "cpu").mkdir(parents=True, exist_ok=True)
    (out / "jax_cache" / "cpu" / ".primed").touch()
    r, calls = _run(tmp, "run", NRAND="2", TARGET_AREA="50", CALIB=str(calib), DEVICE="cpu")
    assert r.returncode == 0, r.stderr
    regions, merge = calls
    assert "--gres" not in regions and _opt(regions, "cpus-per-task") == "16"
    arr = _opt(regions, "array").split("%")[0]
    assert arr == ("2" if n == 3 else f"2-{n - 1}")
    # Another calibration makes every region stale.
    calib.write_bytes(b"another calibration")
    r, calls = _run(tmp, "run", NRAND="2", TARGET_AREA="50", CALIB=str(calib), DEVICE="cpu")
    assert _opt(calls[0], "array").split("%")[0] == f"0-{n - 1}"


def test_run_requires_prepare_and_calib(tmp):
    r, _ = _run(tmp, "run", NRAND="2")
    assert r.returncode != 0 and "CALIB" in r.stderr
    calib = tmp / "calib.fits"
    calib.write_bytes(b"x")
    r, calls = _run(tmp, "run", NRAND="2", CALIB=str(calib))
    assert r.returncode != 0 and "prepare" in r.stderr and not calls


def _task(tmp, *args, **env):
    site = ("CLUSTERS_DIR", "MEMBERS_DIR", "JAX_COMPILATION_CACHE_DIR", "PLAN")
    e = {**{k: v for k, v in os.environ.items() if k not in site},
         "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "REMA_LOG": str(tmp / "rema.log"),
         "DR11": str(tmp / "dr11"), "OUTDIR": str(tmp / "run"), "CALIB": "calib.fits",
         "TMPDIR": str(tmp), **env}
    return subprocess.run(["bash", str(ROOT / "scripts" / "slurm" / "rema_task.sh"), *args], env=e,
                          capture_output=True, text=True, timeout=60)


def test_merge_writes_clusters_and_members_dirs(tmp):
    r = _task(tmp, "merge", CLUSTERS_DIR=str(tmp / "dr11" / "sweep" / "11.0-rm"),
              MEMBERS_DIR=str(tmp / "dr11" / "sweep" / "11.0-rm-mem"))
    assert r.returncode == 0, r.stderr
    call = (tmp / "rema.log").read_text()
    assert f"--out {tmp}/dr11/sweep/11.0-rm/clusters_dr11.fits" in call
    assert f"--members-out {tmp}/dr11/sweep/11.0-rm-mem/clusters_dr11_members.fits" in call


def test_region_deletes_checkpoint_and_compiles_privately(tmp):
    (tmp / "bin" / "rema").write_text(
        "#!/usr/bin/env bash\n"
        'echo "rema $1 cache=$JAX_COMPILATION_CACHE_DIR" >> "$REMA_LOG"\n')
    d = tmp / "run" / "regions" / "0003"
    (d / "checkpoint").mkdir(parents=True)
    (d / "checkpoint" / "firstpass.npz").write_bytes(b"x")
    shared = tmp / "run" / "jax_cache" / "gpu"
    # The priming task (cache not primed yet) compiles into the shared cache and primes it.
    r = _task(tmp, "region", "3")
    assert r.returncode == 0, r.stderr
    assert not (d / "checkpoint").exists() and (shared / ".primed").exists()
    assert f"rema blind cache={shared}\n" in (tmp / "rema.log").read_text()
    # Later tasks compile into a private copy in TMPDIR, deleted at the end.
    (tmp / "rema.log").write_text("")
    r = _task(tmp, "region", "3")
    assert r.returncode == 0, r.stderr
    line = [l for l in (tmp / "rema.log").read_text().splitlines() if l.startswith("rema blind")][0]
    private = line.split("cache=")[1]
    assert private.startswith(f"{tmp}/rema_jax.") and not Path(private).exists()


def test_clean_needs_a_merge(tmp):
    keep = tmp / "run" / "galaxies"
    keep.mkdir(parents=True)
    r = _task(tmp, "clean")
    assert r.returncode != 0 and "merge" in r.stderr and keep.exists()
