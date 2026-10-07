"""Covariance of the counts: Poisson and super-sample variance (Hu & Kravtsov 2003).

    C[(i,j), (i',j')] = delta N_ij + delta_jj' sigma_b^2(j) (b N)_ij (b N)_i'j

with b the mean halo bias of a bin and sigma_b^2(j) the variance of the linear density averaged
over the footprint of redshift bin j (the pixels with z_vlim above its upper edge) times the
redshift slab, from :func:`ggah_mod.covariance.supersample.slab_sigma2_b` (exact at low
multipoles, Limber above). Different redshift slabs are taken as independent. The covariance is
computed once, at a reference model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from ..model.cosmo import _float64
from .area import ZvlimMap


def sigma2_b_slabs(zmap: ZvlimMap, z_edges: Sequence[float], theta: Mapping, pk=None,
                   pix_keep=None, ell_max: int | None = None) -> np.ndarray:
    """sigma_b^2 of each redshift bin's volume-limited footprint and slab."""
    from ggah_mod.cosmology import make_pk
    from ggah_mod.covariance.geometry import from_healpix_mask
    from ggah_mod.covariance.supersample import slab_sigma2_b

    from .counts import cosmology

    pk = pk if pk is not None else make_pk("emu_pk")
    ze = np.asarray(z_edges, np.float64)
    keep = np.ones(zmap.pix.size, bool) if pix_keep is None else np.asarray(pix_keep, bool)
    out = np.zeros(ze.size - 1)
    with _float64():
        cosmo = cosmology({k: jnp.asarray(v, jnp.float64) for k, v in theta.items()})
        k = jnp.logspace(-4, np.log10(200.0), 512)

        def fields_at(zs):
            return [SimpleNamespace(k=k, pk_cb=pk.pk_cb(k, z, cosmo)) for z in zs]

        for j in range(ze.size - 1):
            sel = keep & (zmap.zvlim >= ze[j + 1])
            if not sel.any():
                continue
            mask = zmap.select(sel).full_mask()
            geo = from_healpix_mask(mask, name=f"zbin{j}", z_min=float(ze[j]), z_max=float(ze[j + 1]),
                                    ell_max=ell_max, provenance="rema z_vlim map")
            out[j] = float(slab_sigma2_b(geo, cosmo, fields_at, SimpleNamespace(two_halo_spectrum="cb")))
    return out


def counts_covariance(N: np.ndarray, bias: np.ndarray, sigma2_b: np.ndarray | None = None) -> np.ndarray:
    """Covariance of the flattened counts [n_lambda * n_z] (row-major, lambda first)."""
    N, bias = np.asarray(N, np.float64), np.asarray(bias, np.float64)
    nl, nz = N.shape
    C = np.diag(np.maximum(N.ravel(), 1.0))
    if sigma2_b is not None:
        bN = bias * N
        for j in range(nz):
            idx = np.arange(nl) * nz + j
            C[np.ix_(idx, idx)] += sigma2_b[j] * np.outer(bN[:, j], bN[:, j])
    return C
