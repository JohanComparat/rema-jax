# Model tables

Small tables used by the red-sequence model. All are FITS binary tables with a column `Z`.
`python -m rema.data.build` rebuilds them from public inputs (downloaded once from fixed
commits, checked against their SHA-256 and cached in `~/.cache/rema/inputs`, about 16 MB);
`python -m rema.data.build --check` compares the rebuilt tables with these.

| File | Columns | Content |
|---|---|---|
| `mstar_des_z03.fits` | `Z`, `MSTAR` | m*(z) in the DES/DECam z band (default) |
| `mstar_legacy_z_ezgal.fits` | `Z`, `MSTAR` | m*(z) in DECam z of a passive population |
| `colors_bc03_legacy_grizw1.fits` | `Z`, `COLOR[4]` | adjacent colours g−r, r−i, i−z, z−W1 of the same population |

**`mstar_des_z03.fits`**: redMaPPer's `data/mstar/mstar_des_z03.fit` (Rykoff et al. 2014;
0.01 ≤ z ≤ 1.2) resampled with linear interpolation on z = 0.01, 0.02, ..., 1.51, and extrapolated
linearly above z = 1.2 (header `ZEXTRAP`).

**`mstar_legacy_z_ezgal.fits`, `colors_bc03_legacy_grizw1.fits`**: a Bruzual & Charlot (2003)
population, Salpeter IMF, Z = 0.02 (the SSP grid distributed with EzGal, Mancone & Gonzalez 2012),
with exponential star formation (τ = 0.1 Gyr) from z_f = 3, normalised to SDSS i = 17.85 (AB) at
z = 0.2. Magnitudes are computed in `rema.model.sps` (JAX) through the DECam 2014 griz and WISE W1
responses of speclite, at z = 0.01, ..., 1.50, in a flat cosmology with Ω_m = 0.3, h = 0.7. The
exponential population is integrated exactly over the SSP ages, and every redshift is evaluated
directly. These two tables replace earlier ones made with EzGal itself, which interpolated the
magnitudes on a coarser redshift grid; the two versions differ by at most 0.007 mag in m* and
0.022 mag in the colours (median 0.002 mag).

The headers record the builder, the model parameters and the input URLs. m*(z) tables are
interpolated with a natural cubic spline. The colour table seeds the red-sequence calibration (its
absolute zero point is refitted on DR11 data).
