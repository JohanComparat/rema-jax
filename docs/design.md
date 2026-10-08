# rema design notes

rema rewrites the redMaPPer cluster finder as a
JAX package. This document gives the algorithm, the places where rema differs from
redMaPPer v0.7.7 (the `jacobic/redmapper` fork) and why, and the validation
on the development area. The DR11 south production catalogue and its validation against the
literature are on the pages "The DR11 south catalogue at CC-IN2P3" and "redMaPPer blind mode on
DR11".

## 1. Scope

- Data: Legacy Surveys DR11 south. The inputs are the sweeps, the row-matched photo-z sweeps
  (which give `Z_SPEC`) and the randoms.
- Bands: g, r, i, z, with z as the reference band. The code works for any band set, so the
  DR10 grz/z redMaPPer calibration can also drive it.
- Modes: blind cluster finding, scans at given positions (redMaPPer zscan), and spectroscopic
  post-processing (cluster spec-z and velocity dispersion).
- Calibration: of the red sequence, the corrections and the wcen centring model on DR11, done in
  rema (`rema calibrate`).

## 2. Galaxy catalogue (`rema/io/legacy.py`)

The selection is explicit; every cut is in `SurveyConfig`:

- MASKBITS without the rejected bits (bright and medium stars, saturation and all-mask in griz,
  bailout, SGA large galaxies, globular clusters, resolved dwarfs, Magellanic clouds). Objects
  fitted as SGA large galaxies (FITBITS bit 9) are kept, since they can be BCGs.
- `TYPE != PSF` (and not DUP), and NOBS ≥ 1 in every band.
- Reference-band S/N ≥ 5 and REFMAG < m*(z_max) + 2.5.

Fluxes and inverse variances are dereddened with MW_TRANSMISSION. No cut is applied to the other
bands' fluxes: faint or negative g fluxes stay, with their errors. The redMaPPer DR10 input catalogue
dropped any galaxy with flux ≤ 0 or σ_m > 1 in any band. On the DR11 tile that loses 2%, 12% and 30% of the red
galaxies at z = 0.7–0.8, 0.8–0.9 and 0.9–1.0.

`ID = LS_ID_DR11 = RELEASE<<42 | BRICKID<<22 | OBJID`. `ZSPEC` is the photo-z sweep's `Z_SPEC`.
Values outside (0.001, 3), and COSMOS2015 photometric "reference" redshifts, are set to −1.

## 3. Footprint and depth (`rema/sky/maps.py`)

The DR11 randoms carry NOBS, GALDEPTH and MASKBITS at uniform positions. They give, per
NESTED HEALPix pixel (nside 1024):

- `FRACGOOD`: the fraction of randoms passing the galaxies' MASKBITS and NOBS cuts, times the
  geometric coverage of the pixel by the data box;
- `SIGF_<band>`: the median dereddened 1σ flux error of a canonical galaxy,
  σ_f = 1/√GALDEPTH/MW. GALDEPTH is an inverse variance in 1/nanomaggy², not a magnitude;
  the 5σ depth is 22.5 − 2.5 log10(5/√GALDEPTH).

The effective area for a galaxy of true magnitude m is A_eff(m) = Σ_pix A_pix FRACGOOD
Φ((F(m) − max(5σ_f, F_cut))/σ_f). DR10 randoms are not used.

**The strip** (one DR11 randoms file, 66 million points, of which about 8 fall in each pixel):
- 70.9 deg² unmasked of 75 deg²;
- median 5σ galaxy depths (dereddened): g 24.65, r 24.29, i 23.83, z 23.40;
- A_eff for z-band S/N ≥ 5: 70.9 deg² to z = 22, 63.8 at 23.0 and 24.8 at 23.5;
- built in about 1 min, reading the 23 GB file in chunks.

## 4. Red-sequence model (`rema/model/redsequence.py`)

For adjacent colours c_j = m_j − m_{j+1}:

  c_j(z, m) = mean_j(z) + slope_j(z) (m − pivot(z)),  C_int(z) = D R D,  D = diag(σ_j(z)).

Every quantity is a natural cubic spline through redshift nodes. Splines are linear in the node
values (`rema/model/splines.py`) and continue linearly beyond the end nodes. The parameters
are unconstrained:

- log σ;
- R from partial correlations through tanh (canonical partial correlations), so R is a valid
  correlation matrix for any parameter values.

`RSModel.from_redmapper_pars` reads redMaPPer `*_pars.fit` files, resampling colours that have
different node sets onto common nodes. `RSModel.from_template` starts from the BC03 griz
colours in `rema/data`.

m*(z) defaults to redMaPPer's DES z03 table, as for DR10 and DES. The EzGal DECam-z table
differs by 0.07 mag at z = 0.9.

## 5. Photometric likelihood (`rema/model/likelihood.py`)

The default `chisq_mode = "lupt"` compares the asinh magnitudes of the non-reference bands with
those of the model, softening each band at its own 1σ flux error b = σ_f:

  μ(f) = zp − 2.5/ln10 [asinh(f/2b) + ln b],  r_b = μ_b(f_b) − μ_b(F_b),
  F_b = 10^(−0.4 (m_ref + (T c)_b − zp)),  J_b = F_b / sqrt(F_b² + 4b²),
  S = diag((2.5/ln10)² (b² + (ε F_b)²)/(F_b² + 4b²)) + J T C_int Tᵀ J + g gᵀ σ_mref²,  g = J (1 + T s).

Here T maps colours to band offsets from the reference band. ε = 1.5% is a flux-error floor.
The χ² has ncol degrees of freedom whatever the S/N.

A pure flux-space χ² was tried first. It gets the intrinsic scatter (Gaussian in magnitudes)
wrong by up to 10% in χ² for bright galaxies. Asinh magnitudes equal magnitudes at high S/N and
are linear in flux at low S/N.

`chisq_mode = "mag"` reproduces redMaPPer (C = C_int + R diag(σ_m²) Rᵀ + s sᵀ σ_ref²) and is
used for comparisons with DR10. The two agree to 2% at S/N > 30, and simulated red-sequence
galaxies follow χ²₃ in both.

## 6. zred (`rema/core/zred.py`)

On a grid with Δz = 0.005:

  ln D(z) = −χ²/2 + ln φ(m_ref; m*(z), α) + ln[E(z)/E(z_ref)].

The last term is redMaPPer's volume factor, verified against the DR10 pars file.

- Grid points where m_ref is outside [m* − 4, m* + 2.5] are excluded.
- zred is the mean of exp(ln D), with the standard deviation as error (floored at 0.005).
- A 5-point parabola replaces the mean when its vertex lies within 2σ.
- Calibrated corrections (mean offset, slope with magnitude, error scaling) are applied
  iteratively, as in redMaPPer.

## 7. Background (`rema/model/background.py`)

Σ_g(z, χ², m) is the density of galaxies per deg², per magnitude and per unit χ², with χ²
computed at hypothesis z. It is factorised as N(m) P(χ² | z, m):

- N(m) = counts / (A_eff(m) Δm).
- P is a kernel-weighted ratio in magnitude, whose kernel widens until it holds 50 galaxies.
  This removes the Poisson noise of the bright bins.
- Bins are 0.02 in z, 0.5 in χ² and 0.2 mag. Galaxies enter at z only if m < m*(z) + 2.5.

Lookup is trilinear and differentiable.

## 8. Richness (`rema/core/richness.py`)

  u_i = 2π r_i Σ_NFW(r_i) φ(m_i)/lumnorm ρ_χ²(χ²_i),  b_i = 2π r_i Σ_g(z, χ²_i, m_i)/D(z)²,
  p_i = λ N(r_λ) u_i / (λ N(r_λ) u_i + b_i),  pmem_i = p_i θ_i pfree_i θ_r(r_i; r_λ),
  Σ_i pmem_i = λ K(r_λ),  r_λ = r0 (λ/100)^β.

- The projected NFW (r_s = 0.15, flat core r < 0.1 h⁻¹Mpc) and its enclosed integral are closed
  form. Power series in u = 1 − x² remove the removable singularity at x = 1. N(r_λ) is exact
  rather than redMaPPer's degree-5 polynomial fit.
- lumnorm uses Gauss–Legendre quadrature, checked against 2.5/ln10 E1(10^(0.4(m* − maxmag)))
  for α = −1.
- θ_i is the soft cut at maxmag = m* + 1.75 (0.2 L*). The local depth enters only through K, so
  incompleteness is not counted twice.
- The solve evaluates the residual on 32 log-spaced values of λ in [0.5, 2000], brackets the
  root and refines it with 4 safeguarded Newton steps, inside `jax.lax.custom_root`. λ therefore has exact implicit derivatives with respect to every model
  parameter. Tests check this against finite differences (1e-4) and against scipy brentq.

Outputs:

- SCALEVAL = λ/Σ pmem = 1/K, and LAMBDA_E = sqrt((1 − ⟨p⟩) λ S).
- LNLAMLIKE = −Σ pmem − Σ ln(1 − pmem), excluding the central galaxy.
- PCOL = λφρ / (λφρ + π r_λ² Σ_g/D²), within r_λ.

## 9. Mask and depth completeness (`rema/core/maskcorr.py`)

Deterministic quadrature replaces redMaPPer's 100 × 6000 Monte-Carlo "maskgals":

- Points on a polar grid (24 Gauss–Legendre radii × 16 azimuths) around the cluster are looked
  up on the device in the footprint map, with JAX nested `ang2pix`.
- The completeness is FRACGOOD × S, where S = ∫ φ θ_i P_sel dm / ∫ φ θ_i dm is the selected
  fraction at the local depth.
- Its azimuthal mean f(R) gives K(r_λ) as a profile-weighted 1-D integral inside every solver
  step. MASKFRAC = 1 − K_geo(r_λ), with FRACGOOD only.

## 10. Cluster redshift z_λ (`rema/core/zlambda.py`)

redMaPPer's iteration:

- Soft top-70% weights on pcol.
- L(z) = Σ_i w_i ln L_i(z) on a local 21-point grid, evaluated one grid point at a time to save
  memory, with a parabola vertex as the new z.
- Up to 5 iterations, stopping at convergence.

p(z) ∝ exp(L) V(z) is computed on 21 bins over ±4σ of the curvature width; Z_LAMBDA_E is its
standard deviation.

## 11. Centring (`rema/core/centering.py`)

Two methods, chosen by `centering.method`: "bcg", "wcen", or "auto" (wcen when the calibration
holds a fitted wcen model, BCG otherwise; `rema blind/scan --centering` overrides it). Both write
the same columns: up to `maxcen` = 5 candidates in ID_CENT, RA_CENT, DEC_CENT, P_CEN, Q_CEN,
P_SAT, P_FG and P_C, with NCENT_GOOD, Q_MISS and the connectivity W of the centre. Unused slots
have ID −1 and position −400.

- **CenteringBCG** picks the brightest galaxy within r_λ, or within 0.4 h⁻¹Mpc of the input
  position in scan mode, with pmem > 0.8 or |zred − z| < 2 zred_e. A cluster without such a
  galaxy is dropped during percolation, as in redMaPPer. It gives P_CEN[0] = 1.
- **CenteringWcenZred**, ported from redMaPPer:
  - **Candidates:** galaxies within r_λ (and 0.4 h⁻¹Mpc in scan mode) with pfree ≥ 0.5, zred
    χ² < 100, and pmem > 0 or |zred − z| < 5 zred_e.
  - **Three likelihoods per candidate:**
    - central: N(m; m* + Δ0 + Δ1 ln(λ/30), σ_m) · N(zred; z, zred_e) · N(ln w; cen);
    - satellite: the luminosity filter φ(m)/lumnorm · N(zred; z, zred_e) · N(ln w; sat), times
      the λ/S − 1 satellites;
    - foreground: N(ln w; fg) · Σ_zred(zred, m) π r_λ² / D², with Σ_zred the zred background
      (`ZredBkg`, `rema/model/background.py`).

    zred is compared with z itself, as in redMaPPer: its percolation and zscan build the centring
    without the z_λ correction (`run_percolation.py:285`, `run_zscan.py:251`).
  - **Connectivity:** w = ln[Σ p L/d / (Σ p L / r_λ)] over the galaxies with p > 0 within
    r_λ of the candidate, with d softened by 0.05 h⁻¹Mpc and L the luminosity. The ln w
    Gaussians are in the log of this log, as in redMaPPer and its calibration; their widths
    scale as 1/√(min(λ, 100)/S/30).
  - **Probabilities:** P_C = pfree · ucen / (ucen + (λ/S − 1) usat + bcounts), clipped at
    0.99999. The 5 largest are normalised as in redMaPPer: P_CEN and Q_CEN, then P_SAT and P_FG
    split from 1 − P_CEN. The most probable candidate is the centre.
  - **Deviations from redMaPPer:**
    - Q_MISS is stored; redMaPPer computes it but writes 0.
    - With `calib.wcen_niter = 2`, the calibration's likelihood pass (fixed z = z_spec, no first
      pass) centres the zred term of LNCGLIKE on z_spec; redMaPPer's spectroscopic mode centres
      it on the first-pass z_λ.
    - At most 64 candidates enter the pairwise w (exact unless more could reach P_C > 0).
    - A failed zred is never a candidate.
    - The richness state is the consistent one at z_λ of the seed.
  - **LNCGLIKE:** the likelihood pass adds the seed's central log-likelihood
    ln N(m) + ln max(N(zred; zrmod(z), zred_e), 1e−10) + ln N(ln w), at z = the first-pass z_λ,
    to LNLAMLIKE to rank the candidates (LNLIKE), as redMaPPer's likelihood pass
    (`run_likelihoods.py:198-237`). It is computed in log space, and non-finite rows are dropped.
    - zrmod is redMaPPer's z → zred_uncorr mapping, part of the z_λ correction: the median zred
      of central galaxies at raw z_λ = z, fitted by `rema calibrate` (§13).
    - As in redMaPPer, its natural cubic spline is tabulated every 0.002 (`zlambda.py:621-663`)
      and interpolated linearly, with linear extrapolation (`run_likelihoods.py:213-216`). The
      table is a traced leaf of the wcen model.
    - zrmod(z) = z without the mapping: calibration files written before it, and the calibration
      runs, which have no z_λ correction (as redMaPPer's, `calibrate.py:422`).
  - **Calibration (`rema calibrate`, `rema/calib/wcen.py`):**
    - It follows redMaPPer's WcenCalibrator on spectroscopically seeded clusters.
    - The training clusters are BCG-centred (calib_niter = 1, as in the DR10 run).
    - The foreground and satellite ln W distributions come from random points and random
      satellites drawn within the consolidated clusters (MASKFRAC < 0.2, λ/S > 3) whose BCG
      is their own seed. These are re-run at fixed z, with percolation among them.
    - Unlike redMaPPer, these fixed-z runs do not reject a cluster whose z_λ would fail at the
      new centre.
    - Empty or too small samples give undefined parameters, and runs then use BCG centring.
    - The central-magnitude model (Δ0, Δ1, σ_m) and the central ln W are fitted as in
      redMaPPer.
    - `calib.wcen_niter = 2` re-runs, with wcen centring, the clusters seeded on the first
      centres (as redMaPPer's later iterations) and refits the central terms on the P_CEN/P_SAT
      mixture.

## 12. Modes

- **Scan** (`rema/modes/scan.py`):
  - λ(z) and LNLAMLIKE(z) are computed on Δz = 0.005 steps in a fixed 0.5 h⁻¹Mpc aperture.
  - Neighbours are gathered per Δz = 0.1 chunk, keeping only galaxies brighter than that
    chunk's limit.
  - ZMAX is the likelihood peak (ZMAX_EDGE flags a peak on the grid edge).
  - z_λ is then refined with the percolation aperture. The optical centre is chosen within
    0.4 h⁻¹Mpc, and λ and z_λ are recomputed there.
- **Blind** (`rema/modes/blind.py`):
  - **Seeds:** galaxies with zred in range, χ² < 20 and m < m* + 1.75.
  - **First pass:** fixed 0.5 h⁻¹Mpc aperture, z_λ and λ, without z_λ errors and p(z) (as
    in redMaPPer); λ ≥ 3 is kept.
  - **Likelihood pass:** r0 = 1, β = 0.2, giving LNLAMLIKE.
  - **Percolation:**
    - Order: decreasing LNLIKE, ties by seed ID; a candidate is skipped when its seed's
      pfree < 0.5.
    - Step: λ at the seed and the candidate's redshift (one richness evaluation), then z_λ at
      the seed without errors, BCG recentring, and z_λ (with errors and p(z)) and λ at the new
      centre.
    - Rejection, as in redMaPPer's `RunPercolation`: a candidate is dropped, and claims nothing,
      in any of these cases:
      - λ/S < 3 at the seed (checked before any z_λ work);
      - z_λ fails;
      - no central galaxy is found;
      - the new central galaxy has pfree < 0.5;
      - λ/S < 3 at the new centre.

      In the 75 deg² strip, 44,725 candidates were dropped at the seed, 4,359 for z_λ, 1,350
      for a claimed central galaxy and 7,098 at the new centre, against 12,367 accepted.

      Without the central-galaxy check, a candidate seeded on a member of a richer cluster could
      re-centre on that cluster's BCG and survive as a low-λ fragment.
    - Claims: galaxies within R_MASK and brighter than m* + 2.5 get p added to their claimed
      fraction.
    - Batching: the result is that of the sequential loop. Candidate j waits for every
      higher-ranked candidate i whose claims can reach j's read region, that is when
      the seed separation < read(j) + reach(i). Both radii are bounded from twice the
      likelihood-pass λ at z − 0.05, floored at the lower end of the redshift range. The ready
      candidates (no unfinished blocker) cannot affect each other. They are computed together,
      and their claims are applied in any order.
    - The dependency graph is built with dual-tree searches between candidates binned by read
      radius and by reach. It is stored as CSR arrays. For the 138,873 candidates of the
      75 deg² strip that is 40 million pairs, built in 7 s. A single search bounded by the
      largest reach (degrees at z ≈ 0.05) needed more than 50 GB.
  - **Consolidation:** MASKFRAC < 0.2, λ/S ≥ 3, centre in the own box.
  - **Checkpoints:** `rema blind --checkpoint DIR` keeps the first-pass and likelihood results.
    A rerun with the same seeds, model and configuration resumes after them.
- **Regions** (`rema/pipeline.py`, `scripts/slurm/`; see the HPC page). A large run cuts the
  sky into regions:
  - **Own boxes:** runs of whole sweeps in 10° Dec bands, about 100 deg² each, with the polar
    cap as one ring.
  - **Data boxes:** the own box grown by 2°.
  - **Ownership:** a region keeps the clusters whose final centre lies in its own box. Own
    boxes do not overlap, so there are no duplicates.
  - **Buffer:** 2° covers what changes a cluster directly: its aperture, plus the mask radius
    and seed offset of a higher-ranked neighbour, which is 1.6° for λ = 100 at z = 0.05.
    Percolation's full dependency radius is 2.3°, so chains of claims from beyond the buffer
    change clusters near a boundary at second order, as in redMaPPer's tiling.
  - **Inputs:** the galaxies come from per-sweep tables read once. The footprints come from a
    pixel-sorted copy of the randoms, which a region reads only over its data box.
  - **Measured on the 74 deg² strip,** run as three 25 deg² regions and as one region:
    - the same 12,857 clusters;
    - every λ ≥ 5 cluster with the same central galaxy, at any distance from the internal
      boundaries;
    - λ equal to a median relative difference of 3·10⁻⁶.

    15 clusters (0.12%) move to another z_λ solution. All of them had stopped at the
    5-iteration cap of z_λ (Z_LAMBDA_NITER = 5, about 20% of the clusters), where float32 noise
    from a different batching decides the outcome. They lie up to 4.8° from any boundary, so
    this is not a boundary effect.
- **Spectroscopic post-processing** (`rema/modes/specpost.py`):
  - The Clerc et al. (2016) velocity clipping, vectorised in NumPy.
  - Biweight location (c = 6), a 5000 km/s window, then 20 iterations of re-selection at 3σ
    from all spectroscopic members. σ is the gapper estimator below 15 members, the biweight
    scale (c = 9) above.
  - A seeded bootstrap (64 resamples) gives the means and errors.
  - BEST_Z falls back from SPEC_Z_BOOT to CG_SPEC_Z to Z_LAMBDA.
  - Tested identical to a reference recursive implementation for 3 to 80 members.

## 13. Calibration (`rema/calib/`)

An EM version of redMaPPer's calibration, run on DR11 with the `Z_SPEC` of the photo-z sweeps:

1. **Initial model** (`init.py`): the BC03 griz template colours, refined on the spectroscopic
   galaxies brighter than m*(z) + 1.75. In sliding redshift bins, galaxies within 2σ of the
   current model are kept, the median residuals shift the mean colours and the robust widths
   give the scatter. Six iterations shrink the width from 0.15 mag.
2. **E-step** (`driver.py`):
   - Compute zred and the χ² background with the current model.
   - Spectroscopic seeds: z_spec in range, m < m*(z) + 1, χ²(z_spec) < 20.
   - Run the likelihood pass and an exact percolation at fixed z = z_spec (`keepz`), so each
     cluster counts once.
   - The weights are the members' pmem in clusters with λ ≥ 5.
3. **M-step** (`fit.py`): a joint fit of every node, Σ w (χ² + ln det S)/2 plus a
   second-difference smoothness prior, using optax L-BFGS in float64. Pivot magnitudes are the
   members' weighted median reference magnitude.
4. **Corrections** (`corrections.py`):
   - zred: offset and error scaling from spectroscopic members (pmem > 0.5).
   - z_λ: offset and extra scatter from z_λ of the calibration clusters, started at the seed's
     z_spec and compared with it.
   - z → zred_uncorr, for LNCGLIKE (§11), as redMaPPer's `zlambdacal.py:338-343`:
     - Sample: the same clusters, with redMaPPer's cuts λ/S > 3
       (calib_zlambda_minlambda) and MASKFRAC < 0.2, and a valid z_λ and central zred.
     - Fit: the median zred of their central galaxies against their raw z_λ. This is a natural
       cubic spline on the z_λ-correction nodes (0.08 apart,
       calib_zlambda_nodesize), with an L1 cost minimised by L-BFGS-B from the identity
       (MedZFitter).
     - Storage: the ZRED_UNCORR column of the ZLAMBDACORR HDU. A file without it reads as the
       identity.

The calibration file holds the model, the corrections, the background and the configuration.
`rema calibrate --plots DIR` adds diagnostic figures.

## 14. Validation so far

- **Foundations:** cosmology (ggah_mod) agrees with astropy to 2e-4. HEALPix agrees with healpy
  exactly in float64. Splines agree with scipy, and the NFW closed forms with quadrature.
- **zred on the DR11 tile** (DR10 grz model, 1.37M galaxies), for red spec-z galaxies: bias
  −0.004, NMAD 0.019 (0.013–0.026 per redshift bin). Speed: 0.25 s on the RTX 3060 (5M
  galaxies/s), 8.7 s on CPU.
- **Mock clusters** (182 injected into a 200 deg² synthetic field):
  - median λ_out/λ_in = 0.94–1.00;
  - z_λ bias < 0.001 for z ≤ 0.5 and −0.003 at z = 0.75, with scatter 0.002–0.005.
- **DR11 griz development calibration** (3 sweeps, RA 0–5, Dec −15–0, 75 deg²; Z_SPEC in two
  of them). It takes 9.4 min on CPU and uses 34,653 spectroscopic galaxies.
  - Initial model from 8,466 spectroscopic red galaxies, then 2 EM iterations.
  - z_λ against the seeds' spec-z for 1,906 clusters with λ ≥ 5: bias −0.0042, NMAD 0.0088
    (before the z_λ correction fitted from these same clusters).
  - The red sequence agrees with the DR10 grz calibration in r−z (within 0.02 mag at all z) and
    in g−r up to z = 0.6. At z ≥ 0.7, where g is barely detected, g−r differs by up to 0.1.
  - Intrinsic scatter in r−i and i−z is 0.01–0.06 mag.
- **Scan at the 25 ACT DR5 clusters of the strip** (DR11 griz calibration, `rema scan --specpost`,
  72 s on CPU):
  - All 25 within |Δz| < 0.05 of the ACT redshift; z_λ,opt − z_ACT has median −0.0004(1+z) and
    NMAD 0.0033(1+z).
  - 13 clusters get a spectroscopic redshift from ≥ 3 members (σ_v 317–1264 km/s). A flagged
    2240 km/s case falls back to the central galaxy's spec-z.
- **Scan at 19 ACT DR5 clusters** (DR10 grz model, magnitude mode) against the redMaPPer DR10 scan:
  z_λ,opt agrees within 0.02 for every cluster at z < 0.8, mostly within 0.005; λ_opt is
  within 20% for 12 of 15.
- **Blind run on 2.25 deg² (DR10 grz model):** the ACT cluster at (1.530, −2.525) is recovered
  at z_λ = 0.628, λ = 79.5; DR10 had 0.628 and 75.9.
- **Blind run over the 75 deg² strip** (DR11 griz development calibration, no footprint yet,
  `rema blind --specpost` on an RTX 3060):
  - 436,947 seeds gave 140,545 first-pass candidates (99 s), then 138,873 after the likelihood
    pass (66 s).
  - Percolation (527 s, 2,249 rounds) produced 12,367 clusters. It skipped 68,974 seeds that
    were already claimed and rejected 57,532 candidates.
  - The total was 12 min, plus 1.5 min of spectroscopic post-processing.
- **Against redMaPPer DR10** (grz, wcen centring) in the 59 deg² interior, matching within 3′
  and |Δz| < 0.05(1+z):
  - 0.1 < z < 0.7, λ ≥ 20: 85% of the 129 DR10 clusters are recovered and 95% of the 188 rema
    clusters are confirmed. ln(λ/λ_DR10) = +0.04 ± 0.17 (median ± NMAD) and
    Δz/(1+z) = +0.001 ± 0.006.
  - 0.7 < z < 0.9: λ is 30% lower (ln ratio −0.30 ± 0.40). This is DR10's depth correction:
    its SCALEVAL is 1.29 at z = 0.7–0.8 and 1.58 at 0.8–0.9, and rema has none without a
    footprint. Against DR10's uncorrected λ/S, the ln ratios are −0.12 and −0.07.
  - Within 1.5′ only 73% of the DR10 λ ≥ 20 clusters match. Most of the rest are the same
    cluster centred 1.5–3′ away on a different central galaxy (DR10 uses the wcen model, rema
    CenteringBCG).
- **z_λ against the clusters' spectroscopic redshifts** (SPEC_Z_BOOT, λ ≥ 20, ≥ 3 spectroscopic
  members): 65 clusters, bias +0.0003, NMAD 0.0065, no outliers.
- **ACT DR5:** the 24 clusters at z < 0.9 are all detected, 22 within 2′ and 2 at 2.4′ and 2.7′,
  with Δz NMAD 0.0044. The z = 0.968 cluster is outside the blind range; scan mode finds it at
  z = 0.948.
- **CPU against GPU** on the same strip: 12,363 against 12,367 clusters.
  - λ ≥ 20: all 330 have the same central galaxy; |Δλ/λ| is at most 8e-4 (median 2e-6) and
    |Δz| at most 1.5e-4.
  - λ ≥ 5: 99.8% have the same central galaxy.
- **With the footprint** (DR11 randoms, step 2):
  - Calibration on the strip with mask and depth corrections: 2,763 clusters, z_λ against
    z_spec NMAD 0.0100. The wcen parameters agree with the run without a footprint (Δ0 −1.55,
    Δ1 −0.34, σ_m 0.41).
  - Blind run: 12,888 clusters after the MASKFRAC < 0.2 cut. Against DR10 within 3′:
    - 0.1 < z < 0.7, λ ≥ 20: 85% recovered, 96% confirmed, ln(λ/λ_DR10) = +0.14 ± 0.19,
      Δz/(1+z) = +0.002 ± 0.006;
    - 0.7 < z < 0.9: 87% recovered, 90% confirmed, ln ratio −0.21 ± 0.47.
  - z_λ against the clusters' spectroscopic redshifts (92 clusters with λ ≥ 20 and ≥ 3 members):
    bias +0.0017, NMAD 0.0055, no outliers.
  - Scan at the ACT clusters: z_λ NMAD 0.0035 (0.0042 without the footprint), optical centre
    as in redMaPPer DR10 for 80%.
  - Two ACT clusters on the Dec = 0 edge of the strip get MASKFRAC 0.20 and 0.36, because their
    apertures run off the data, and are dropped by the MASKFRAC cut. The DR11 mask bits cover
    only 3–8% of the randoms around them. On the HPC, regions overlap.
  - **λ normalisation:** rema λ is 10–15% above DR10 at z < 0.7. It moved by about +7% when the
    calibration gained the third sweep's 6,000 spectroscopic redshifts. A calibration on 75 deg²
    is noisy. With the production calibration (400 deg²), rema λ is 0.89 times DES Y1 redMaPPer,
    0.90 times SDSS DR8 (decreasing with z as SDSS becomes shallow) and 1.13 times the λ_norm
    of Kluge et al. (2024) (page "redMaPPer blind mode on DR11").
  - **Depth correction:** rema's catalogue reaches S/N 5 in z (5σ depth 23.4), deeper than
    m*+1.75 at z = 0.9. So SCALEVAL stays near 1 where redMaPPer's 10σ-limited counting needs
    1.1–1.6. For the ACT clusters (an SZ selection, independent of both catalogues), rema's λ at
    z > 0.7 is at least DR10's. The −0.2 to −0.3 ln ratio at z > 0.7 against DR10 λ ≥ 20
    clusters comes mostly from selecting on DR10's λ.
- **Percolation rules:** before the redMaPPer rejections were added (see section 12), a mock
  z = 0.6 cluster also produced a λ = 4 fragment re-centred on one of its members.
- **wcen centring:**
  - **Kernel against redMaPPer:** a float64, loop-based numpy port of CenteringWcenZred,
    LNCGLIKE and W (`tests/wcen_reference.py`) serves as the reference. The kernel gives the same
    candidates and centres and probabilities within 4e-6, including the strict thresholds and the
    edge cases. An adversarial review with 16 code mutants found no defect, and the tests now
    catch every mutant.
  - **Calibration on the strip** (626 BCG-centred training clusters; 256 random centres and 438
    random satellites, 42 s):
    - Δ0 = −1.54, Δ1 = −0.35, σ_m = 0.42;
    - ln W of centrals 0.18 ± 0.19, of satellites 0.05 ± 0.26, of the foreground −0.29 ± 0.22;
    - the brightest-satellite model as redMaPPer's Monte Carlo.
  - **Blind run on the strip, wcen against BCG,** with the same calibration and candidates, for
    clusters matched to DR10 within 3′:
    - the central is DR10's for 57% (BCG 54%) at λ_DR10 = 10–20 and 52% (BCG 50%) at λ ≥ 20;
    - DR10's central is among rema's 5 candidates for 83% and 73% of them;
    - at 0.7 < z < 0.9, 50% of the DR10 λ ≥ 20 clusters are within 0.5′ (BCG 43%);
    - P_CEN[0] has median 0.910 and 10th percentile 0.557, against 0.895 and 0.536 in DR10.
  - **Scan at the 25 ACT clusters:** the optical centre is redMaPPer DR10's (wcen) for 80%,
    against 68% with BCG centring.
  - **LNCGLIKE with the z → zred_uncorr mapping** (§11), on the notebook region (9 deg² own box):
    - The mapping, fitted on the strip from 2,586 clusters, has zred_uncorr − z within ±0.011 from
      z = 0.13 to 0.85, −0.018 at 0.05 and −0.024 at 0.93.
    - LNCGLIKE changes by a median of 0.13 and by more than 1 for 3.6% of the candidates. The
      percolation order has a rank correlation of 0.999 with the identity's.
    - The catalogue goes from 1,507 to 1,505 clusters, and 99.3% keep their central galaxy. All 43
      with λ ≥ 20 are unchanged in centre; the 10 that change centre have a median λ of 3.6.
- **Purity at high redshift.** In the strip catalogue, the counts of λ ≥ 5 clusters rise towards
  z_λ = 0.85–0.90. Per comoving volume, this bin holds 1.2–1.3 times the density of
  z_λ = 0.65–0.78.
  - **Not an edge effect:**
    - The percolated candidates cut at 0.85 show no excess below 0.85.
    - A run with `model.zrange` up to 0.95 keeps the excess at 0.85–0.90 (×1.2–1.35). It grows to
      ×1.45 at 0.90–0.95.
  - **Noise detections:** in a null test, the galaxy positions of the notebook region are drawn
    again uniformly over the footprint, with the photometry kept, so no real cluster remains.
    - It yields λ ≥ 5 detections at 0.3 deg⁻² at z < 0.3, 1–2 deg⁻² at 0.45–0.8, 3.3 deg⁻² at
      0.80–0.85 and 4.2 deg⁻² at 0.85–0.90.
    - That is 40–64% of the real counts at z > 0.7, against 12–30% at z < 0.6.
    - For λ ≥ 10 it yields 0.08 deg⁻² at 0.85–0.90, against 2.8 deg⁻² in the data, and none
      below. Its largest λ is 12.
  - **Spectroscopy:** take the clusters with at least two spectroscopic members (26 at
    z_λ > 0.85). Their spectroscopic coherence in excess of chance suggests that about 40% of
    them are real, against 60–67% at z < 0.5.
  - **Conclusion:** down to λ = 5, the catalogue favours completeness, and at z_λ > 0.75 it is
    impure. Use λ ≥ 10 there, or calibrate a z-dependent threshold with null tests over a larger
    area.

## 15. Photo-z cluster finding

The page "Photo-z cluster finding on DR11" (`docs/photoz_finders.rst`) describes the method and
the results; this section lists the code.

- **Photo-z widths** (`rema/model/photoz.py`): s = max(err_scale(m) ZPHOT_STD,
  err_floor (1 + ZPHOT)), `ZPHOT_E` in the region's table; galaxies without a usable photo-z get
  −1 and are never members.
- **Field** (`build_photoz_bkg`): Σ_pz(z, m) = N(m) P(z | m) from the stacked Gaussians of the
  region's galaxies, on a 0.005 grid in z over 0–1.6 and 0.2 mag bins, P kernel-smoothed in
  magnitude as the χ² background. Stored in a `ZredBkg` (bilinear lookup), built per region,
  never written to the calibration.
- **Photo-z filter** (`model.filter: photoz`): `FilterModel` gets the static `filter`,
  `pz_nsig_max`, `pz_mag_max` and the leaf `pzbkg`; `Neighbors` gets `zphot`, `zphot_e`
  (None in red-sequence runs, so the existing kernels and their compilations are unchanged).
  `_filter_terms` dispatches to `_terms_pz` (ρ = p_i(z), Σ_g = Σ_pz(z, m), χ² = x², ln det =
  2 ln s, finite placeholders only) and `_member_lnl_one` to ln p_i(z). `Region.build` skips the
  χ² background, the wcen model and the z_λ correction; BCG centring tests |ZPHOT − z| < 2s.
  `richness.min_lnlamlike` cuts the candidates after the first and the likelihood pass.
  Checked: the red-sequence outputs of a mock region are bit-identical to those of 0.3.2.
- **PSCD** (`rema/pscd/`): `model.py` (template, ⟨1/s⟩ tables, cumulative magnitude
  integrals, NumPy NFW and Schechter: the JAX versions compiled for every new array shape and took
  most of the time), `grid.py` (gnomonic grid, cloud-in-cell painting, overlap-add convolution),
  `detect.py` (cube of S, α, β, γ and the score L = A²α where the S/N passes; extraction from
  the maxima of 32 × 32-pixel tiles, of which only those touched by a cleaning are recomputed;
  cells whose profile is less than 20 % inside the footprint are not searched), `run.py` (region
  driver and outputs). On the 75 deg² strip: 3.5 M galaxies, a 505 × 1512 × 88 cube, built in
  2 min; about 60 ms per detection.
- **Null tests** (`rema/validate/null.py`): seeded permutations within 0.1 mag bins, applied in
  `Region.build` and `run_pscd` before anything else (the colour shuffle also recomputes zred).

## 16. Performance notes

- zred: 1.4M galaxies in 9 s on CPU, 0.25 s on an RTX 3060.
- Device-array pytrees only. A numpy array inside a pytree argument (for example the cosmology
  table) pushes every jitted call onto JAX's slow dispatch path, and so do eager `jnp` calls on
  Python scalars. Host bookkeeping (m*, D_A, magnitude limits) uses numpy versions.
- GPU matrix products: float32 products default to TF32 on NVIDIA GPUs, whose ~1e-3 relative
  error changed χ² enough to alter about 6% of the percolated clusters. Every small matrix
  product in the model uses `precision=HIGHEST`, and CPU and GPU catalogues now agree.
- λ solve: a 32-point log-spaced bracket (one grid point per step, see the compile-time note
  below) plus 4 safeguarded Newton steps, wrapped in `custom_root`.
- z_λ likelihood grid: evaluated in chunks of 7 redshifts (memory O(7K), 3 sequential steps).
- GPU compile time. XLA's GPU reduction emitter unrolls (grid points × K/256) evaluations,
  times a factor from the batch axes, into every thread of a reduction over the K neighbours.
  With the 32-point λ grid at K = 8192 (blind mode), and with the scan kernel's batch of
  positions × redshift steps, one kernel held 1,024 erf evaluations and 35,600 registers.
  NVIDIA's `ptxas` needed more than 10 minutes for each such kernel; its time grows faster
  than linearly with kernel length. The fixes:
  - The λ grid is now evaluated one point per step (`lax.map`): 32 sequential reductions, about
    1% of a z_λ call.
  - The z_λ grid takes 7 points per step up to K = 2048 and fewer above
    (`rema.core.richness.grid_chunk`).

  Cold compiles on an RTX 3060 now take 2.6–2.8 s for richness and 7–8 s for z_λ, at
  B × K = 256 × 512 and 64 × 8192.
- Percolation (`rema/modes/blind.py::percolate`) is exact and runs on the dependency graph of
  the candidates. A candidate depends on the higher-ranked candidates whose claims can reach its
  read region. Ready candidates are computed together, in one call per neighbour-count bucket,
  and accepted in rank order. Their batches are padded to a few sizes: powers of 8 on GPUs,
  where padding rows are almost free, and powers of 2 on CPUs, where they cost as much as real
  rows. On 600 and 1,500 candidates the results equal the sequential loop. Warm timings per
  candidate (600 candidates on 16 deg², all compilations done):

  | device | sequential | batch 64 |
  |---|---|---|
  | CPU (16 threads) | 10.2 ms | 9.0 ms |
  | RTX 3060 | 13.7 ms | 6.7 ms |

- Blind mode over 75 deg² on an RTX 3060:
  - first pass 99 s (437,000 seeds, 0.23 ms each);
  - likelihood pass 66 s;
  - percolation 527 s (139,000 candidates, 3.8 ms each).

  Percolation was 814 s before the cheap seed check skipped z_λ for the 78% of computed
  candidates that fail λ/S ≥ 3.

  On the laptop's CPU (i9-11900H, 8 cores, 16 threads) the same run takes 62 min:
  - first pass 917 s (2.1 ms per seed);
  - likelihood pass 82 s;
  - percolation 2,717 s.

  That is 9× slower than the GPU for the first pass and 5× for percolation. Neighbour arrays
  are now padded to powers of 2 on CPUs instead of powers of 4, which makes CPU percolation
  1.7× faster: 17.3 against 30.1 ms per candidate on 9,915 candidates of the strip, with the
  same clusters. That brings the run to about 45 min. The DR11 south production (18,500 deg²,
  270 regions of 100 deg² with 2° buffers) took 990 task-hours on 5 CPU cores each, and about
  1 h per region on a V100.
- wcen centring in percolation: 839 s against 659 s with BCG for the 166,000 candidates of the
  strip (RTX 3060), so 27% more. The kernel compiles in under a second.
- Scan mode on an RTX 3060: 4,000 positions in 72 s (18 ms each, for 191 redshift steps and two
  z_λ refinements), plus 16 s for the first batch with compilation. The z_λ refinement keeps
  every position's own aperture and magnitude limit. Before, one low-redshift position in a
  batch of 128 could inflate K for the whole batch to tens of thousands and exhaust the GPU
  memory.
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` on small GPUs. The CLI enables a persistent
  compilation cache (`~/.cache/rema/jax`): a run compiles a few dozen programs, each a few
  seconds to compile (z_λ: 7–8 s per shape on a GPU). On an HPC, point `JAX_COMPILATION_CACHE_DIR` at a shared
  directory so that every job reuses the same compiled shapes.
