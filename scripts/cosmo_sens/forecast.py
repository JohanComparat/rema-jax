"""Forecast for the DR11 counts: constraints, the part of dN/dtheta due to the finder, and the
parameter shifts when the finder's response is ignored.

    python scripts/cosmo_sens/forecast.py --response RESPONSE.fits [--fit FIT.json]
        [--free Omega_m,ln10A_s,...] [--tag NAME]

The fiducial is the best fit without response of ``--fit`` (fit_dr11.py), else the defaults; the
binning, area and covariance are those of the DR11 data vector. The universe is the fiducial;
the catalogue is made by a finder run at a cosmology offset by -Delta from it (Omega_m -+0.02,
-+0.05; w0 -+0.2; the response coefficients of the table, measured around the finder's
Omega_m = 0.3, w0 = -1, are used around this cosmology). Its counts N_resp are fitted by the model
without the response: the shift F^-1 J^T C^-1 [N_resp - N_noresp] (plus the weak-lensing term)
is the bias of ignoring the finder's cosmology dependence. Writes
``$REMA_WORK/cosmo_sens/forecast_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosmo_common as C  # noqa: E402

FREE = ("Omega_m", "ln10A_s", "h", "n_s", "Omega_b", "mor_a", "mor_b", "mor_c", "sigma_int",
        "ln_s0", "s1", "dz_bias")
SHIFTS = [("Omega_m", -0.05), ("Omega_m", -0.02), ("Omega_m", 0.02), ("Omega_m", 0.05), ("w0", -0.2),
          ("w0", 0.2)]


def main(argv=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    from rema.abundance import fisher as FI
    from rema.abundance.counts import CountsModel, CountsSetup
    from rema.abundance.covariance import counts_covariance, sigma2_b_slabs
    from rema.abundance.likelihood import DEFAULTS, Likelihood
    from rema.abundance.response import ResponseTable
    from rema.model.cosmo import _float64

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--response", required=True)
    p.add_argument("--fit", help="fit_dr11.py JSON: its best fit without response is the fiducial")
    p.add_argument("--free", default=",".join(FREE))
    p.add_argument("--tag", default="all")
    a = p.parse_args(argv)
    free = tuple(x for x in a.free.split(",") if x)
    dv, zmap, keep = C.data_vector("all")
    resp = ResponseTable.read(a.response)
    fid = dict(DEFAULTS)
    if a.fit:
        fit = json.loads(Path(a.fit).read_text())
        fid.update(next(f for f in fit["fits"] if f["model"] == "noresp")["best"])
    m0 = CountsModel(CountsSetup.from_data(dv))
    m1 = CountsModel(m0.setup, response=resp, pk=m0.pk)
    L0 = Likelihood(dv, m0, free, fixed=fid, reference=fid)
    pred = L0.predict(fid)
    s2 = sigma2_b_slabs(zmap, dv.z_edges, fid, pk=m0.pk, pix_keep=keep)
    cov = counts_covariance(np.asarray(pred["N"]), np.asarray(pred["bias"]), s2)
    L0 = Likelihood(dv, m0, free, fixed=fid, cov=cov)
    L1 = Likelihood(dv, m1, free, fixed=fid, cov=cov)
    x = L0.x0()
    out = {"fiducial": {k: float(v) for k, v in fid.items()}, "free": list(free), "sigma2_b": s2.tolist()}
    jac = {}
    for name, L in (("noresp", L0), ("resp", L1)):
        jac[name] = FI.jacobians(L, x)
        F = FI.fisher(L, x, jac[name])
        con = FI.constraints(F, free)
        out[name] = {"err": {k: v for k, v in con.items() if k != "_cov"},
                     "sigma8": jac[name]["sigma8_value"], "sigma8_err": FI.sigma8_error(jac[name], con["_cov"]),
                     "F": F.tolist()}
        print(f"[{name}] sigma(Omega_m) = {con['Omega_m']:.4f}, sigma(sigma8) = {out[name]['sigma8_err']:.4f}")
    N0 = np.asarray(pred["N"]).ravel()
    for p_ in ("Omega_m", "w0"):
        if p_ in free:
            k = free.index(p_)
            tot, nor = jac["resp"]["N"][:, k] / N0, jac["noresp"]["N"][:, k] / N0
            out[f"dlnN_d{p_}"] = {"total": tot.reshape(dv.shape).tolist(),
                                  "finder": (tot - nor).reshape(dv.shape).tolist()}
            print(f"d ln N / d {p_}: total\n{np.round(tot.reshape(dv.shape), 3)}\n finder part\n"
                  f"{np.round((tot - nor).reshape(dv.shape), 3)}")
    # d ln N / d theta through the finder for parameters that are not free (e.g. w0).
    for p_, d in (("w0", 0.01),):
        if p_ in free:
            continue
        with _float64():
            up = {**fid, p_: fid[p_] + d}
            dn = {**fid, p_: fid[p_] - d}
            fin = (np.log(np.asarray(L1.predict(up)["N"])) - np.log(np.asarray(L1.predict(dn)["N"]))
                   - np.log(np.asarray(L0.predict(up)["N"])) + np.log(np.asarray(L0.predict(dn)["N"]))) / (2 * d)
        out[f"dlnN_d{p_}_finder"] = fin.tolist()
        print(f"d ln N / d {p_} through the finder\n{np.round(fin, 3)}")
    F0 = np.asarray(out["noresp"]["F"])
    out["shifts"] = []
    import dataclasses

    for p_, d in SHIFTS:
        true = dict(fid)
        # the finder at the truth, but for parameter p_ at truth - d
        offset = dataclasses.replace(resp, fiducial=tuple(float(fid[q]) - (d if q == p_ else 0.0)
                                                          for q in resp.params))
        Ld = Likelihood(dv, CountsModel(m0.setup, response=offset, pk=m0.pk), free, fixed=fid, cov=cov)
        r0, r1 = L0.predict(true), Ld.predict(true)
        dN = (np.asarray(r1["N"]) - np.asarray(r0["N"])).ravel()
        with _float64():
            dwl = np.asarray(Ld.wl_residual(true, r1)) - np.asarray(L0.wl_residual(true, r0))
        sh = FI.shift(L0, F0, jac["noresp"], dN, dwl)
        g = jac["noresp"]["sigma8"]
        row = {"param": p_, "delta": d, "true": float(true[p_]), "finder": float(true[p_] - d),
               "shift": dict(zip(free, sh.tolist())), "sigma8_shift": float(g @ sh),
               "dN_over_N": (dN / N0).reshape(dv.shape).tolist()}
        out["shifts"].append(row)
        print(f"truth - finder {p_} = {d:+.2f}: shift Omega_m {row['shift'].get('Omega_m', 0):+.4f} "
              f"({row['shift'].get('Omega_m', 0) / out['noresp']['err']['Omega_m']:+.2f} sigma), "
              f"sigma8 {row['sigma8_shift']:+.4f} ({row['sigma8_shift'] / out['noresp']['sigma8_err']:+.2f} sigma)")
    C.OUT.mkdir(parents=True, exist_ok=True)
    path = C.OUT / f"forecast_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
