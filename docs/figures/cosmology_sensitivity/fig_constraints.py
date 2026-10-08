"""Omega_m - sigma_8 from the DR11 counts with and without the finder's response (fit_<tag>.json),
and the shifts of the best fit when the response is ignored (forecast_<tag>.json)."""

import json
import os

import numpy as np

from common import BLUE, INK2, MUTED, ORANGE, RESULTS, need, panel_label, plt, record, save

FIT = os.environ.get("COSMO_FIT", "all_cc_sig025")             # fit_dr11.py --tag
FORECAST = os.environ.get("COSMO_FORECAST", "sig025_cc")  # forecast.py --tag
fit_path, fc_path = RESULTS / f"fit_{FIT}.json", RESULTS / f"forecast_{FORECAST}.json"


def sigma8_grad(theta, free):
    """d sigma_8 / d(free parameters) at theta (sigma_8 depends on the cosmology only)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ggah_mod.cosmology import make_pk
    from ggah_mod.cosmology.amplitude import sigma8

    from rema.abundance.counts import cosmology

    pk = make_pk("emu_pk")
    k = jnp.logspace(-4, np.log10(200.0), 512)

    def s8(x):
        th = {**theta, **{p: x[i] for i, p in enumerate(free)}}
        return sigma8(pk.pk(k, 0.0, cosmology(th)), k)

    x = jnp.asarray([theta[p] for p in free], jnp.float64)
    return float(s8(x)), np.asarray(jax.grad(s8)(x))


def ellipse(ax, mean, cov, color, label):
    w, v = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    for nsig, alpha in ((1.52, 1.0), (2.48, 0.5)):           # 68 and 95 % for two parameters
        xy = (v @ (np.sqrt(np.maximum(w, 0))[:, None] * np.stack([np.cos(t), np.sin(t)]))) * nsig
        ax.plot(mean[0] + xy[0], mean[1] + xy[1], color=color, alpha=alpha, label=label if nsig < 2 else None)


if need(fit_path):
    fit = json.loads(fit_path.read_text())
    fig, ax = plt.subplots(figsize=(5, 4.2), constrained_layout=True)
    for f, col in zip(fit["fits"], (BLUE, ORANGE)):
        free = f["free"]
        th = dict(f["best"])
        s8, g = sigma8_grad(th, free)
        cov = np.asarray(f["cov"])
        i = free.index("Omega_m")
        J = np.zeros((2, len(free)))
        J[0, i] = 1.0
        J[1] = g
        ellipse(ax, [th["Omega_m"], s8], J @ cov @ J.T, col,
                "without the finder's response" if f["model"] == "noresp" else "with the response")
        record(f"Om_{f['model']}", th["Omega_m"])
        record(f"Om_err_{f['model']}", f["fisher_err"]["Omega_m"])
        record(f"s8_{f['model']}", s8)
        record(f"s8_err_{f['model']}", float(np.sqrt(g @ cov @ g)))
    ax.set_xlabel("Ω_m")
    ax.set_ylabel("σ_8")
    ax.legend()
    save(fig, "constraints")

fc_paths = [(fc_path, "mass–richness relation free"),
            (RESULTS / f"forecast_{FORECAST}_fixedmor.json", "mass–richness relation fixed")]
fc_paths = [(p, t) for p, t in fc_paths if p.exists()]
if need(fc_path):
    fig, axes = plt.subplots(1, len(fc_paths), figsize=(6 * len(fc_paths), 3.2), constrained_layout=True,
                             squeeze=False)
    for ax, (path, title) in zip(axes[0], fc_paths):
        fc = json.loads(path.read_text())
        sh = fc["shifts"]
        y = np.arange(len(sh))
        eo, es = fc["noresp"]["err"]["Omega_m"], fc["noresp"]["sigma8_err"]
        ax.barh(y - 0.18, [s["shift"]["Omega_m"] / eo for s in sh], height=0.35, color=BLUE, label="Ω_m")
        ax.barh(y + 0.18, [s["sigma8_shift"] / es for s in sh], height=0.35, color=ORANGE, label="σ_8")
        ax.set_yticks(y, [f"{s['param']}: truth − finder = {s['delta']:+g}" for s in sh])
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlabel("shift of the best fit when the response is ignored [σ]")
        ax.set_title(f"{title}: σ(Ω_m) = {eo:.3f}, σ(σ_8) = {es:.3f}")
        ax.legend()
        if path == fc_path:
            for s in sh:
                record(f"shift_Om_{s['param']}{s['delta']:+g}_sigma", s["shift"]["Omega_m"] / eo)
                record(f"shift_s8_{s['param']}{s['delta']:+g}_sigma", s["sigma8_shift"] / es)
            record("forecast_Om_err", eo)
            record("forecast_s8_err", es)
    save(fig, "shifts")
