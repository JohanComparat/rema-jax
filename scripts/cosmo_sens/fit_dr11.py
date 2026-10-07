"""Fit the DR11 cluster counts N(lambda, z) for Omega_m and sigma_8 with the DES Y1 weak-lensing
calibration of the richness (rema.abundance), with and without the finder's response.

    python scripts/cosmo_sens/fit_dr11.py [--variant all|deep|shallow] [--lambda-min 20]
        [--response RESPONSE.fits] [--free Omega_m,ln10A_s,...] [--nuts N] [--tag NAME]

Steps: the data vector (common.data_vector); a first best fit with a Poisson covariance at the
defaults; the covariance (Poisson + super-sample) at that fit; the best fit again; the Fisher
matrix and the Laplace covariance at the best fit; optionally a NUTS chain. With ``--response``
the fit is done twice, without and with the response table (the finder's richness in the true
cosmology). Results: ``$REMA_COSMO/fit_<tag>.json`` (and ``chain_<tag>_<model>.npz``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosmo_common as C  # noqa: E402

FREE = ("Omega_m", "ln10A_s", "h", "n_s", "Omega_b", "mor_a", "mor_b", "mor_c", "sigma_int",
        "ln_s0", "s1", "dz_bias")


def fit_one(dv, zmap, keep, free, response=None, nuts=0, tag="", label="", ssv=True):
    from rema.abundance import fisher as FI
    from rema.abundance import sampling as SA
    from rema.abundance.counts import CountsModel, CountsSetup
    from rema.abundance.covariance import counts_covariance, sigma2_b_slabs
    from rema.abundance.likelihood import Likelihood

    t0 = time.time()
    model = CountsModel(CountsSetup.from_data(dv), response=response)
    lik = Likelihood(dv, model, free)
    x, info = SA.map_fit(lik)
    th = lik.theta(x)
    s2 = sigma2_b_slabs(zmap, dv.z_edges, th, pk=model.pk, pix_keep=keep) if ssv else None
    for _ in range(2):          # covariance at the fit, then the fit again
        pred = lik.predict(th)
        cov = counts_covariance(np.asarray(pred["N"]), np.asarray(pred["bias"]), s2)
        lik = Likelihood(dv, model, free, cov=cov)
        x, info = SA.map_fit(lik, x)
        th = lik.theta(x)
    jac = FI.jacobians(lik, x)
    F = FI.fisher(lik, x, jac)
    con = FI.constraints(F, free)
    try:
        lap = SA.laplace(lik, x)
        lap_err = dict(zip(free, np.sqrt(np.clip(np.diag(lap), 0, None)).tolist()))
    except np.linalg.LinAlgError:
        lap, lap_err = None, {}
    pred = lik.predict(th)
    out = {"model": label, "free": list(free), "best": dict(zip(free, map(float, x))),
           "fisher_err": {k: v for k, v in con.items() if k != "_cov"}, "laplace_err": lap_err,
           "sigma8": jac["sigma8_value"], "sigma8_err": FI.sigma8_error(jac, con["_cov"]),
           "chi2": lik.chi2(x), "map": info, "sigma2_b": None if s2 is None else s2.tolist(),
           "counts_model": np.asarray(pred["N"]).tolist(), "lnM_model": np.asarray(pred["lnM"]).tolist(),
           "bias_model": np.asarray(pred["bias"]).tolist(), "cov": con["_cov"].tolist(),
           "time": time.time() - t0}
    if nuts:
        ch = SA.nuts(lik, x, warmup=max(200, nuts // 2), samples=nuts, seed=1)
        np.savez(C.OUT / f"chain_{tag}_{label}.npz", **{k: np.asarray(v) for k, v in ch.items() if k != "names"},
                 names=np.array(free))
        out["nuts"] = {"accept": ch["accept"], "mean": dict(zip(free, ch["samples"].mean(0).tolist())),
                       "std": dict(zip(free, ch["samples"].std(0).tolist()))}
    return out


def main(argv=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    from rema.abundance.response import ResponseTable

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", default="all", choices=("all", "deep", "shallow"))
    p.add_argument("--lambda-min", type=float, default=20.0)
    p.add_argument("--response", help="response table (build_response.py)")
    p.add_argument("--free", default=",".join(FREE))
    p.add_argument("--no-ssv", action="store_true", help="Poisson covariance only")
    p.add_argument("--nuts", type=int, default=0, help="NUTS samples (0: none)")
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)
    free = tuple(x for x in a.free.split(",") if x)
    tag = a.tag or f"{a.variant}_lam{a.lambda_min:g}"
    C.OUT.mkdir(parents=True, exist_ok=True)
    dv, zmap, keep = C.data_vector(a.variant, a.lambda_min)
    print(f"{dv.meta['NCLUSTER']} clusters; area per z bin {np.round(dv.area)} deg2")
    print(dv.counts.astype(int))
    res = {"variant": a.variant, "lambda_min": a.lambda_min, "counts": dv.counts.tolist(),
           "area": dv.area.tolist(), "lam_edges": [float(x) for x in dv.lam_edges],
           "z_edges": dv.z_edges.tolist(), "lam_mean": dv.lam_mean.tolist(), "z_mean": dv.z_mean.tolist(),
           "fits": []}
    models = [("noresp", None)]
    if a.response:
        models.append(("resp", ResponseTable.read(a.response)))
        res["response"] = str(a.response)
    for label, resp in models:
        r = fit_one(dv, zmap, keep, free, resp, a.nuts, tag, label, ssv=not a.no_ssv)
        res["fits"].append(r)
        b, e = r["best"], r["fisher_err"]
        print(f"[{label}] Omega_m = {b['Omega_m']:.4f} +- {e['Omega_m']:.4f}, sigma8 = {r['sigma8']:.4f} "
              f"+- {r['sigma8_err']:.4f}; chi2 counts {r['chi2']['counts']:.1f} ({r['chi2']['n_counts']} bins), "
              f"WL {r['chi2']['wl']:.1f} ({r['chi2']['n_wl']}); {r['time']:.0f} s")
    path = C.OUT / f"fit_{tag}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
