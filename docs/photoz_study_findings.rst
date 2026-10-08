Photo-z study: all the findings
===============================

The findings of the photo-z cluster-finding study of 2026-10-08, in one place: the results, what
was learned about the two photo-z finders, the issues found and fixed in the code and the
environments on the way, the costs, and what remains open. The method and the figures are on
:doc:`photoz_finders`.

Summary
-------

- At the same false-detection rate (0.3 per deg², purity 97–98 % by the null tests), **PSCD finds
  10–40 % more clusters than the red-sequence finder**; the photo-z filter of rema finds no more.
- The gain is at **z < 0.5**, in poor systems rich in blue members (median red fraction of the
  PSCD members 0.32–0.41, against 0.51–0.56 for the photo-z filter).
- PSCD recovers more of the **low-mass clusters** of Wen & Han (2024): 38–64 % against 13–20 % at
  z < 0.2; above M500 = 3 × 10¹⁴ M☉ every finder recovers 75–100 %.
- **No finder gains at z > 0.6**: fainter than z = 21 the DR11 photo-z scatter by 0.03–0.08 (1 + z),
  with 3–14 % outliers.
- **97–99 % of the photo-z detections are also red-sequence clusters**, mostly of low λ: the
  photo-z finders re-rank the systems, they do not reveal new ones.
- Cluster redshifts are as good as the red sequence's: **NMAD 0.006–0.008 (1 + z)**.
- The red-sequence code path is unchanged: bit-identical outputs on a mock region, and the strip
  rerun matches 0.3.1 cluster by cluster (all 2,734 clusters with λ ≥ 5 matched, scatter of
  ln λ 7 × 10⁻⁶, one centre changed).

Results
-------

Areas: the strip (RA 0–5°, Dec −15–0°, DECaLS; own box 37.4 deg²), region 9 of the production plan
(RA 15–30°, Dec −55 to −45°, DES; 92.0 deg²) and region 82 (RA 140–150°, Dec 5–15°, DECaLS;
92.8 deg²).

Detections at 0.3 false detections per deg²
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each finder thresholded on each of its statistics (threshold, number of clusters in the own box,
purity estimated from the null test):

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - finder, statistic
     - strip
     - region 9 (DES)
     - region 82 (DECaLS)
   * - red sequence, λ
     - 12.7: 490 (0.98)
     - 13.6: 1,063 (0.98)
     - 18.6: 977 (0.97)
   * - red sequence, SNR
     - 4.25: 522 (0.98)
     - 4.34: 1,214 (0.98)
     - 5.12: 1,114 (0.98)
   * - photo-z filter, SNR
     - 5.31: 469 (0.98)
     - 5.86: 1,093 (0.98)
     - 5.86: 941 (0.97)
   * - photo-z filter, λ
     - 31.9: 273 (0.96)
     - 36.5: 618 (0.96)
     - 38.5: 576 (0.95)
   * - photo-z filter z < 22, SNR
     - 5.38: 430 (0.97)
     - 5.91: 997 (0.97)
     - 5.82: 932 (0.97)
   * - PSCD, AMICO SNR
     - 4.13: 659 (0.98)
     - 4.43: 1,232 (0.98)
     - 4.45: 1,128 (0.98)
   * - PSCD, SNR_NOCL
     - 10.6: 728 (0.99)
     - 12.0: 1,462 (0.98)
     - 12.2: 1,230 (0.98)
   * - PSCD, λ*
     - 24.1: 388 (0.97)
     - 26.8: 884 (0.97)
     - 28.5: 780 (0.97)
   * - PSCD z < 22, SNR_NOCL
     - 10.8: 743 (0.99)
     - 12.2: 1,472 (0.98)
     - 12.2: 1,309 (0.98)

- Best statistic of each finder: SNR for the red sequence and the photo-z filter, SNR_NOCL for
  PSCD; with it PSCD has 39–42 %, 20–21 % and 10–18 % more clusters than the red sequence.
- A richness is a weaker detection statistic than a likelihood: λ keeps 6–12 % fewer clusters than
  SNR for the red sequence, and 39–44 % fewer for the photo-z filter.
- The red sequence's λ threshold is 12.7 and 13.6 in the deeper areas but 18.6 in region 82: in
  shallow DECaLS data it makes more false detections at fixed λ (compare the excess of λ ≥ 20
  clusters in shallow areas of the production catalogue).

Redshift distribution
~~~~~~~~~~~~~~~~~~~~~

Clusters at 0.3 false per deg² with each finder's default statistic (λ, SNR, AMICO SNR):

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - finder: z ≤ 0.6 / z > 0.6
     - strip
     - region 9
     - region 82
   * - red sequence
     - 294 / 196
     - 544 / 519
     - 508 / 469
   * - photo-z filter
     - 293 / 176
     - 541 / 552
     - 614 / 327
   * - PSCD
     - 477 / 182
     - 831 / 401
     - 888 / 240
   * - PSCD, z < 22
     - 537 / 117
     - 1,037 / 305
     - 992 / 181

- PSCD has about twice the red sequence's clusters at 0.1 < z < 0.45, and fewer above z = 0.6.
- The photo-z filter is the best photo-z finder at z > 0.6 in DES (552 against 519 for the red
  sequence), not in DECaLS (327 against 469).
- In region 82 the red sequence peaks at z = 0.8–0.9: its known spike of z_λ at 0.82–0.86 and its
  impurity in shallow data, which a single null threshold over the whole redshift range does not
  remove.
- The cut at z = 22 removes a quarter to a third of PSCD's clusters at z > 0.6 and adds a few at
  low z.

External catalogues
~~~~~~~~~~~~~~~~~~~

Fraction recovered at 0.3 false per deg² (counterpart within 1 h⁻¹ Mpc and
\|Δz\|/(1 + z) < 0.05; default statistics):

.. list-table::
   :header-rows: 1
   :widths: 30 10 15 15 15 15

   * - catalogue, area
     - N
     - red seq.
     - photo-z filter
     - PSCD
     - PSCD z < 22
   * - ACT DR6, strip
     - 15
     - 0.93
     - 0.93
     - 0.93
     - 0.93
   * - ACT DR6, region 9
     - 65
     - 0.88
     - 0.91
     - 0.88
     - 0.85
   * - ACT DR6, region 82
     - 92
     - 0.90
     - 0.90
     - 0.87
     - 0.85
   * - SPT-SZ, region 9
     - 14
     - 1.00
     - 1.00
     - 1.00
     - 1.00
   * - eRASS1, region 9
     - 51
     - 0.84
     - 0.84
     - 0.88
     - 0.88
   * - eRASS1, region 82
     - 51
     - 0.80
     - 0.82
     - 0.78
     - 0.78
   * - Wen & Han 2024, strip
     - 2,423
     - 0.18
     - 0.17
     - 0.25
     - 0.24
   * - Wen & Han 2024, region 9
     - 6,719
     - 0.15
     - 0.15
     - 0.17
     - 0.18
   * - Wen & Han 2024, region 82
     - 6,476
     - 0.13
     - 0.13
     - 0.16
     - 0.17

At z < 0.2, PSCD recovers 38–64 % of the Wen & Han clusters and the red sequence 13–20 %.

Overlap and member colours
~~~~~~~~~~~~~~~~~~~~~~~~~~

- 97–99 % of the photo-z finders' clusters have a red-sequence counterpart (λ ≥ 3, same matching).
- The photo-z finders keep 85–92 % of the red-sequence clusters with λ ≥ 20 on the strip and in
  region 9, and 64–69 % in region 82.
- Probability-weighted fraction of the members on the red sequence (χ² < 9 at the cluster
  redshift), median over the clusters: photo-z filter 0.51–0.56 (members brighter than
  m* + 1.75), PSCD 0.32–0.41 (members down to the survey limit).

Cluster redshifts
~~~~~~~~~~~~~~~~~

Against the spectroscopic redshifts of the clusters (velocity clipping of the spectroscopic
members, or the central's redshift):

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - finder
     - strip: NMAD, median (N)
     - region 82: NMAD, median (N)
   * - red sequence
     - 0.0072, +0.0035 (254)
     - (run without spectroscopy)
   * - photo-z filter
     - 0.0057, +0.0035 (260)
     - 0.0072, +0.0050 (663)
   * - photo-z filter, z < 22
     - 0.0058, +0.0034 (246)
     - 0.0071, +0.0049 (674)
   * - PSCD
     - 0.0075, +0.0033 (403)
     - 0.0074, +0.0052 (866)
   * - PSCD, z < 22
     - 0.0066, +0.0033 (405)
     - 0.0070, +0.0052 (945)

In units of 1 + z; no redshift correction is applied to the photo-z finders. Region 9 has no
spectroscopic redshifts in the DR11 photo-z sweeps.

DR11 photo-z widths
~~~~~~~~~~~~~~~~~~~

Robust width of (ZPHOT − ZSPEC)/ZPHOT_STD, the factor applied to Z_PHOT_STD_I, with the scatter,
outlier fraction (\|dz\|/(1 + z) > 0.15) and offset of dz/(1 + z):

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - z-band magnitude
     - strip (40,850)
     - region 82 (78,790)
     - strip NMAD dz, outliers
     - region 82 NMAD dz, outliers, offset
   * - < 19
     - 0.850
     - 0.858
     - 0.012, 0.1 %
     - 0.013, 0.1 %, +0.001
   * - 19–20
     - 0.762
     - 0.746
     - 0.017, 0.4 %
     - 0.019, 0.6 %, +0.0005
   * - 20–21
     - 0.679
     - 0.678
     - 0.020, 0.8 %
     - 0.023, 1.3 %, −0.001
   * - 21–22
     - 0.546
     - 0.564
     - 0.030, 3.3 %
     - 0.045, 7.0 %, −0.003
   * - 22–24
     - 0.586
     - 0.613
     - 0.070, 14 %
     - 0.083, 17 %, −0.030

Z_PHOT_STD overestimates the scatter by a factor 1.2 (bright) to 1.8 (faint); the strip's
calibration holds in region 82.

What was learned about the finders
----------------------------------

The photo-z filter
~~~~~~~~~~~~~~~~~~

- It is far less selective than the red sequence. With redMaPPer's cuts, 59 % of the seeds of a
  9 deg² strip box reach λ ≥ 3, and 49 % with the photo-z shuffled, against 24 % for the
  red-sequence seeds. The percolation then has as many candidates and dependencies on 9 deg² as
  the red-sequence run has on 75 deg².
- A significance floor fixes it. Candidates need SNR = √(2 LNLAMLIKE) ≥ 3 after the first and the
  likelihood pass (``richness.min_lnlamlike: 4.5``). This leaves 9,490 candidates on that box,
  against 2,031 in the null test (204 with SNR > 5, against 3,440). On the strip it leaves 67,081
  candidates, against 176,136 for the red sequence.
- On mocks whose members are half blue, its λ counts the members of every colour (λ = 31–45 for
  input λ = 25–40), where the red-sequence λ holds about half (16–27). On the data this does not
  turn into more detections: at a matched false rate it keeps 10–15 % fewer clusters than the
  red sequence on SNR. Its field (any colour within the photo-z window) is much denser than the
  red sequence's field.
- BCG centring of blue-rich clusters can be 0.05° (0.5 h⁻¹ Mpc at z = 0.3) off the injected centre
  on mocks.

PSCD (the AMICO method)
~~~~~~~~~~~~~~~~~~~~~~~

- The amplitude is unbiased on mocks drawn from the template. For A = 2 at z = 0.3, 0.7 and 0.85 it
  gives A = 1.95 ± 0.05, 2.01 ± 0.08 and 1.85 ± 0.09 (36 clusters each). The scatter of A
  matches σ_A. In the field the pull is −0.17 ± 0.88: the noise is measured on all the galaxies,
  the clusters included.
- Two conditions matter for the amplitude:

  - **The profile needs a core.** With the KiDS template's cusp (core 0.02 h⁻¹ Mpc), A came out
    up to 35 % low at z ≥ 0.5. Smaller pixels did not help; a flat core of 0.1 h⁻¹ Mpc (rema's
    radial filter) did.
  - **The members and the filter need the same magnitude limit.** With members drawn to 22.0 and
    the filter integrated to 22.3, A was 0.95, 0.76 and 0.64 of its input value at z = 0.3, 0.7
    and 0.85. On the data this limit is the local 5σ depth of the footprint.

- AMICO's S/N = A/σ_A is dominated by the cluster's own shot noise. The weights favour a few bright
  members: the effective number of members α²/γ is 8–21 (z = 0.7 to 0.2). A cluster of amplitude A
  then has S/N ≈ √(A α²/γ), about 3–7 for A = 1–2, while it stands 25–50σ above the background
  (SNR_NOCL). At a matched false rate, SNR_NOCL keeps 9–19 % more clusters than AMICO's S/N.
- Extraction with the cleaning: about 60 ms per detection; 12,192 detections on the strip in
  12 min, 28,000–34,000 per region. Cells whose profile lies less than 20 % inside the footprint
  are not searched (at the edges α → 0 made A overflow).

The null tests
~~~~~~~~~~~~~~

- Shuffling the photo-z among galaxies of the same magnitude keeps the angular clustering. Down to
  the extraction thresholds, the null runs therefore keep many detections: 29–38 % as many as the
  real data for the photo-z filter and 56–68 % for PSCD. The thresholds sit where the null curve
  falls steeply: 0.3 false per deg² is reached at SNR 4.1–4.5 for PSCD and 5.3–5.9 for the photo-z
  filter.
- The colour shuffle of the red-sequence null keeps 71 % (strip), 56 % (DES) and 90 % (region 82)
  as many clusters as the real data down to λ = 3. In shallow data most low-λ red-sequence
  detections are chance alignments.

Code and environment issues found and fixed
-------------------------------------------

- **Background area** (fixed in rema). A background built from the galaxy table, i.e. when the
  calibration has none, or for ``rema background``, counted the effective area of the whole
  footprint. Galaxies read over a smaller data box therefore got a background too low by the ratio
  of the areas: 8 times on a 9 deg² box of the 75 deg² strip footprint. The area is now that of the
  footprint's pixels in the data box. Production runs used the calibration's background and were
  not affected.
- **YAML numbers** (fixed). ``1e6`` is a string in YAML 1.1; float keys now read it as a number.
- **JAX compilation of eager calls**. JAX functions called eagerly on arrays of new shapes compile
  every time: 129 s of a 150 s PSCD run. PSCD's profile and luminosity functions are now NumPy
  (equal to the JAX ones to 10⁻¹³ against direct integration).
- **The test venv recipe**. ``pip install "jax==0.10.2" "jaxlib==0.10.2"`` in a venv with
  ``--system-site-packages`` installs nothing, because pip sees the conda jaxlib. The venv must
  install with ``--ignore-installed --no-deps``.

  - With the conda jaxlib, two runs of the same mock region gave 231 and 224 clusters, and 9 tests
    failed.
  - With the PyPI wheels, all 267 tests of 0.3.2 pass, and two runs are bit-identical (88 clusters).

- **JAX at CC-IN2P3**. jax 0.11.2 aborted (``Fatal Python error``) while writing its persistent
  compilation cache on /sps during the test suite; the tests run there without the cache.
- **Bash scripts edited while running**. bash reads a script as it runs it, so a driver edited
  during a run can fail (``unexpected EOF``). Long chains now run from a frozen copy.

Costs
-----

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - run
     - strip (laptop: RTX 3060, 24 cores)
     - regions (CC-IN2P3: V100 or 8 htc cores)
   * - red sequence
     - 41 min, 176,136 candidates
     - 59–60 min, 401,000–643,000 candidates
   * - photo-z filter
     - 16 min, 67,081 candidates
     - 21–25 min, 171,000–197,000 candidates
   * - PSCD
     - 14 min, 12,192 detections, 4.2 GB
     - 14–17 min, 28,000–34,000 detections, 7–9 GB

The whole study at CC-IN2P3 (galaxy tables of the two regions, 20 region runs, 3 strip runs) used
5.2 GB on /sps.

Open questions and next steps
-----------------------------

- Null thresholds per redshift bin. One threshold for the whole range lets the red sequence keep
  its false detections at z > 0.8 in shallow data, and favours it at high z.
- Photo-z distributions from the quantiles (Z_PHOT_L68, U68, L95, U95: split normal), and a model
  of the outliers, instead of Gaussians.
- Purity from mocks with realistic large-scale structure (AMICO's SinFoniA); the null tests miss
  projections of real structures.
- A depth-dependent background for the red sequence: its false detections at fixed λ grow in
  shallow data.
- A photo-z z_λ correction (offsets of +0.003 to +0.005 (1 + z)).
- Larger areas: 14–92 SZ and X-ray clusters per area are small samples.

Where everything is
-------------------

- **Code**:

  - ``rema.model.photoz``, ``rema.model.background.build_photoz_bkg``;
  - the photo-z branch of ``rema.core.richness``, ``rema.core.zlambda`` and ``rema.modes``;
  - ``rema.pscd`` (``rema pscd``), ``rema.validate.null``, ``rema.validate.photoz``;
  - ``rema.validate.compare.match_physical`` and ``null_threshold``;
  - ``rema.config.apply_overrides`` (``--set``).

- **Scripts**: ``scripts/photoz/`` (``photoz_strip.sh``, ``photoz_test.sh``, ``fit_errscale.py``,
  ``photoz_dr11.yaml``, ``pz_filter.yaml``); figures ``docs/figures/photoz_finders/make_all.py``.
- **Runs**:

  - ``~/data/legacysurvey/dr11/south/rema/notebooks/photoz/<area>/<variant>/clusters.fits``, with
    area ``strip``, ``0009`` or ``0082``, and the regions' footprints;
  - at CC-IN2P3, ``/sps/lsst/users/jcomparat01/rema_photoz`` (``runs/<variant>/<region>/``,
    ``strip/<variant>/``, galaxy tables, footprints, logs).

- **Variants**: ``rs_wcen`` (red sequence; the regions' copy is the cosmology study's tier B
  fiducial run), ``rs_bcg``, ``pz_v0``, ``pz_v1`` (photo-z filter; v1 with z < 22), ``pscd_0``,
  ``pscd_1``, and the null tests ``null_rs`` (colours shuffled), ``null_pz``, ``null_pz1``,
  ``null_pscd``, ``null_pscd1`` (photo-z shuffled).
