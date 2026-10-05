# Model tables

Small tables used by the red-sequence model. All are FITS binary tables with a column `Z`.
`python -m rema.data.build` rebuilds them from public inputs (downloaded once from fixed
commits, checked against their SHA-256 and cached in `~/.cache/rema/inputs`, about 14 MB);
`python -m rema.data.build --check` compares the rebuilt tables with these. The documentation
page "Model tables" (https://rema-jax.readthedocs.io/en/latest/tables.html) shows them and how
to calibrate with them.

| File | Columns | Content |
|---|---|---|
| `mstar_des_z03.fits` | `Z`, `MSTAR` | m*(z) in the DES/DECam z band (default) |
| `mstar_lsst_{i,r,z}03.fits` | `Z`, `MSTAR` | m*(z) in LSST i, r, z, the same way |
| `mstar_legacy_z_ezgal.fits` | `Z`, `MSTAR` | m*(z) in DECam z of a passive population |
| `colors_bc03_legacy_grizw1.fits` | `Z`, `COLOR[4]` | adjacent colours g−r, r−i, i−z, z−W1 of the same population |
| `mstar_lsst_{u,g,r,i,z,y}_ezgal.fits` | `Z`, `MSTAR` | m*(z) of the same population in each LSST band |
| `colors_bc03_lsst_ugrizy.fits` | `Z`, `COLOR[5]` | u−g, g−r, r−i, i−z, z−y |
| `mstar_euclid_{vis,y,j,h}_ezgal.fits` | `Z`, `MSTAR` | m*(z) in each Euclid band (VIS, NISP Y, J, H) |
| `colors_bc03_euclid_visyjh.fits` | `Z`, `COLOR[3]` | VIS−Y, Y−J, J−H |

**`mstar_des_z03.fits`, `mstar_lsst_{i,r,z}03.fits`**: redMaPPer's `data/mstar/mstar_*03.fit`
(Rykoff et al. 2014; 0.01 ≤ z ≤ 1.2) resampled with linear interpolation on z = 0.01, 0.02, ...,
1.51, and extrapolated linearly above z = 1.2 (header `ZEXTRAP`).

**`mstar_*_ezgal.fits`, `colors_bc03_*.fits`**: a Bruzual & Charlot (2003) population, Salpeter
IMF, Z = 0.02 (the SSP grid distributed with EzGal, Mancone & Gonzalez 2012), with exponential
star formation (τ = 0.1 Gyr) from z_f = 3, normalised to SDSS i = 17.85 (AB) at z = 0.2, in a flat
cosmology with Ω_m = 0.3, h = 0.7. Magnitudes are computed in `rema.model.sps` (JAX) through the
speclite responses of three filter sets:

- `legacy`: DECam 2014 griz and WISE W1, at z = 0.01, ..., 1.50;
- `lsst`: LSST ugrizy from tag 1.9 of lsst/throughputs, with the airmass-1.2 standard
  atmosphere (speclite `lsst2023-*`), at z = 0.01, ..., 2.50;
- `euclid`: Euclid VIS and NISP Y, J, H, end-of-life total throughputs (ESA
  NISP-PHOTO-PASSBANDS-V1; speclite `Euclid-*`), at z = 0.01, ..., 2.50.

The exponential population is integrated exactly over the SSP ages, and every redshift is
evaluated directly. The DECam tables replace earlier ones made with EzGal itself, which
interpolated the magnitudes on a coarser redshift grid; the two versions differ by at most
0.007 mag in m* and 0.022 mag in the colours (median 0.002 mag). The population is 0.5 Gyr old
at z = 2.5: above z ≈ 2 the tables describe a young population rather than an old red sequence.
In LSST i and z the model m* is within 0.09 mag of redMaPPer's up to z = 0.9.

The headers record the builder, the model parameters, the filter set (`FILTSET`, `FILTDESC`), the
response file of each m* table (`RESPONSE`) and the input URLs. m*(z) tables are interpolated
with a natural cubic spline. A colour table seeds the red-sequence calibration (its absolute zero
point is refitted on the data).
