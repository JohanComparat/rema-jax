Cluster randoms
===============

``rema cluster-randoms`` turns a cluster catalogue into volume-limited samples and their random catalogues.
It takes the catalogue, the footprint map of the DR11 randoms (``rema maps``), the region plans of the merged runs, the DR11 randoms themselves and, for a sample selected by an X-ray or SZ survey, that survey's selection function.
It produces, per sample, one FITS file with the clusters, the weighted random points and the per-cluster diagnostics, ready for the clustering and lensing estimators of ``sum_stat``.
The method is that of redMaPPer (Rykoff et al. 2016, Sects. 3.4 and 3.6; McClintock et al. 2019, Sect. 4.1.3), extended to an external selection.

The method
----------

Every cluster of a sample is copied :math:`n_{\rm samp}` times (100 by default) to random positions where it could have been found.
The positions are DR11 random points that pass the galaxies' cuts (``MASKBITS``, ``NOBS``, :math:`E(B-V)`), lie in the area of the run and, for an externally selected sample, in the external survey's footprint.
A copy of a cluster at redshift :math:`z_\lambda` goes to a point drawn uniformly among those with :math:`z_{\rm vlim} > z_\lambda`, where the catalogue is volume limited at that redshift.

The richness is not re-measured at the new position.
The aperture :math:`r_\lambda = r_0(\lambda/100)^\beta` is laid on the footprint map, and the completeness of the membership solve gives the values the finder would have recorded there:

.. math::

   {\rm SCALEVAL} = 1/K(r_\lambda), \qquad {\rm MASKFRAC} = 1 - K_{\rm geo}(r_\lambda),

with :math:`K` the profile-weighted fraction of the cluster's galaxies that are unmasked and above the S/N cut, and :math:`K_{\rm geo}` its unmasked part (:doc:`design`).
They are computed at ``Z_LAMBDA_RAW``, the redshift before the :math:`z_\lambda` correction, as in the catalogue.
On the 3-sweep validation strip they reproduce the catalogue's MASKFRAC to :math:`2\times10^{-4}` for 99.9% of the 12 888 clusters, and SCALEVAL to the :math:`5\times10^{-3}` precision of the :math:`\lambda` solve.

Copy :math:`j` of cluster :math:`i` passes with probability

.. math::

   q_{ij} = P_{\rm ext}({\rm obs}_i, z_i, x_j)\; \mathbb{1}[{\rm MASKFRAC}_j < 0.2]\; \mathbb{1}[\lambda_i / {\rm SCALEVAL}_j \geq \lambda_{\rm det}],

with :math:`P_{\rm ext} = 1` for a sample selected in richness alone, and the copies share the cluster's weight:

.. math::

   w_{ij} = n_{\rm samp}\, q_{ij} \Big/ \sum_k q_{ik}.

The weights of a cluster's copies add up to :math:`n_{\rm samp}`, so the weighted randoms have exactly the redshift, richness and observable distribution of the sample, and the angular distribution of its selection.
With :math:`q \in \{0, 1\}` this is redMaPPer's :math:`n_{\rm samp}/n_{\rm keep}` (``RandomWeigher``).

Samples selected in X-rays or SZ
--------------------------------

A cluster of an X-ray or SZ sample was selected twice: by the external survey, then by the optical confirmation on DR11.
The DR11 depth maps describe only the second.
The first enters through :math:`P_{\rm ext}`, the probability that the survey detects, at the copy's position, a cluster with the source cluster's measured observables (count rate, :math:`\tilde y_0`, :math:`\xi`); the survey's masks restrict the positions.

The data must not carry selections that the randoms cannot reproduce.
We do not cut on the eRASS1 ``PCONT`` or ``IN_ZVLIM`` columns, nor on other external optical confirmations, which were computed on the DR10 imaging.
A sample is the external detection cut, the rema optical confirmation on DR11 (:doc:`public_catalogues_scan`), and the survey's own masks; the redshift is rema's :math:`z_\lambda`.

A selection is a class registered in :mod:`rema.randoms.selection` with ``in_footprint``, ``own_cut`` and ``p_detect``, named in the sample file.
``cluster-randoms external-join`` first adds to the scan of a public catalogue (:doc:`public_catalogues_scan`) the columns its selection needs, as ``EXT_<NAME>``, by the row of the original catalogue.

.. list-table:: External selections.
   :header-rows: 1
   :widths: 14 40 46

   * - Name
     - Detection probability at the copy's position :math:`x`
     - Inputs
   * - ``act_dr6``
     - :math:`\Phi(\tilde y_{0,i}/\sigma_{\tilde y_0}(x) - q_{\rm cut})`, with ``fixed_y_c`` and the noise of the 2.4 arcmin filter
     - nemo RMS map of DR6 (``DR6_nemo-products_v1.0.tgz``), reduced to HEALPix; ``flags == 0``
   * - ``spt``
     - :math:`\Phi(\sqrt{\zeta_j^2+3} - \xi_{\rm cut})`, :math:`\zeta_j = \sqrt{\xi_i^2-3}\,\gamma_{f_j}/\gamma_{f_i}` (Bocquet et al. 2019, Eqs. 6-7)
     - Field boxes and depths :math:`\gamma_f`
   * - ``psz2``
     - 1 inside the mask (the completeness is averaged over the sky)
     - ``HFI_PCCS_SZ-selfunc-*_R2.08.fits``
   * - ``erass1_approx``
     - :math:`\Phi((c_0 + c_1\ln C_j + c_2\ln\theta_{500} - \ln L_{\rm cut})/s)`, :math:`C_j` the counts at :math:`x`: exposure and absorption moved, flagged APPROX
     - Exposure map, HI4PI, the catalogue (``cluster-randoms erass1-model``)

The eRASS1 model is ours, not the selection function of Clerc et al. (2024), which is not public.
We fitted it on the 10 435 clusters of the primary catalogue with a count rate, a flux and an extent: the eRASS1 exposure is 0.27 times the eRASS:4 one (0.22-0.32 for 90% of the clusters), and the extent likelihood follows :math:`\ln L_{\rm ext} = -1.40 + 1.04\ln C - 0.44\ln\theta_{500}` with a scatter of 0.63, from a fit truncated at :math:`L_{\rm ext} = 3`.
CR500/F500 does not depend on :math:`n_{\rm H}` in the catalogue, so the absorption uses an e-folding column of :math:`1.7\times10^{21}` cm\ :sup:`-2`.

Steps
-----

.. code-block:: bash

   R=/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema
   rema maps --index $IDX --nrand 4 --out footprint_dr11.fits
   rema cluster-randoms domain --runs $R/rema_dr11_v0.2.0_ra0-240 $R/rema_dr11_v0.2.0_ra240-360 --out domain.fits
   rema cluster-randoms vlim-map --footprint footprint_dr11.fits --domain domain.fits \
       --catalog $R/rema_dr11_v0.2.0/clusters_dr11.fits --index $IDX --nrand 4 --out zvlim_l0.2_n5.fits
   rema cluster-randoms copies --clusters $R/rema_dr11_v0.2.0/clusters_dr11.fits --spec lambda.yaml \
       --zvlim zvlim_l0.2_n5.fits --domain domain.fits --index $IDX --nrand 4 --out copies.fits
   rema cluster-randoms place --copies copies.fits --footprint footprint_dr11.fits
   rema cluster-randoms sample --clusters ... --spec lambda.yaml --zvlim ... --domain ... \
       --copies copies.fits --out lambda_vlim.fits
   rema cluster-randoms qa --sample lambda_vlim.fits --zvlim ... --domain ... --out lambda_vlim_qa.json

.. list-table:: Steps of ``rema cluster-randoms``.
   :header-rows: 1
   :widths: 14 86

   * - Step
     - What it does
   * - ``domain``
     - The own boxes of the regions that finished, in every part, with the runs' :math:`|b|` cut. Seams are the edges where boxes of different parts touch; ``--seam-buffer`` removes the area closer to them.
   * - ``vlim-map``
     - :math:`z_{\rm vlim}` per pixel of the footprint: the redshift at which a galaxy of ``--lstar`` :math:`L_*` (0.2) reaches the ``--nsigma`` (5) depth in the reference band, :math:`m_*(z) - 2.5\log_{10}(0.2) = 22.5 - 2.5\log_{10}(5\sigma_f)`. With the randoms, the area of a pixel is the number of good random points of the domain in it over their density.
   * - ``copies``
     - Selects the sample's clusters (domain, :math:`z_\lambda < z_{\rm vlim}` at their own position, bins, cuts, external cut) and copies each to random points of :math:`z_{\rm vlim} > z_\lambda`.
   * - ``place``
     - SCALEVAL, MASKFRAC and :math:`r_\lambda` of every copy, on CPU or GPU.
   * - ``sample``
     - The weights, for the sample of ``--spec``: any sample whose clusters were copied (fewer bins, a higher threshold) reuses the copies.
   * - ``qa``
     - The ratio of the data to the weighted randoms against :math:`z_{\rm vlim}`, the depths, the good fraction, :math:`|b|` and the seam distance, overall and per redshift bin.

The sample file (YAML) holds the bins and cuts of :class:`rema.randoms.samples.SampleSpec`:

.. code-block:: yaml

   name: lambda_vlim
   z_edges: [0.10, 0.20, 0.30, 0.45, 0.60, 0.75]
   obs_edges: [20, 30, 45, 60, 1000]       # bins of `observable` (LAMBDA)
   detection_from_bin: false               # true: LAMBDA / SCALEVAL >= the bin's lower edge
   cuts: []                                # e.g. ["EXT_LIKE >= 6"]
   selection: null                         # e.g. {name: act_dr6, q_cut: 5}
   n_samp: 100

The volume limit
----------------

We use 0.2 :math:`L_*` at 5σ by default: the members that make :math:`\lambda` (brighter than 0.2 :math:`L_*`, S/N ≥ 5 in z) are then all detectable below :math:`z_{\rm vlim}`, and :math:`\lambda` does not depend on the depth there.
Kluge et al. (2024) and the figures of :doc:`redmapper_dr11` use 10σ, which is more conservative; ``--nsigma 10`` gives it.

Data against randoms on DR11
----------------------------

We ran the chain on the laptop with one DR11 randoms file (2 500 points per deg²) and 10 copies per cluster.
The domain covers 18 491 deg²; at 0.2 :math:`L_*` and 5σ the catalogue is volume limited over 99% of it at :math:`z = 0.7` and 96% at :math:`z = 0.8` (at 10σ: 95% at 0.6, 55% at 0.7).
The parent sample (:math:`\lambda \geq 10`, :math:`0.05 \leq z_\lambda < 0.85`, below :math:`z_{\rm vlim}`) has 488 416 clusters; ``place`` took 7.7 min for its 4.9 M copies on ten CPU cores, and 92.5% of the copies have MASKFRAC < 0.2.

We compare the ``lambda_vlim`` sample (371 587 clusters) with its randoms in six bins of the z-band depth, averaged over NSIDE-64 pixels (``qa`` does the same: a template measured per NSIDE-1024 pixel from a few random points is noisy, and the noise makes a trend of its own).
At :math:`0.1 < z < 0.3` the ratio of data to randoms is flat (:math:`\chi^2` = 16 and 19 for 6 bins).
Above, the shallow sky holds more clusters than the deep sky: at :math:`0.45 < z < 0.6` the ratio goes from 1.26 in the shallowest sixth to 0.73 in the deepest, the same in every richness bin, with the 10σ volume limit as with the 5σ one, and within the DECaLS area alone (Dec > −20°: 1.14 to 0.73).
The randoms model the mask and the volume limit; this trend comes from the catalogue, at fixed :math:`\lambda`, and most likely from the :math:`\chi^2` background of the membership filter, which is one for the whole survey while the photometric errors change with the depth.
Until the finder models it, clustering measurements above :math:`z = 0.3` need weights against the depth (e.g. ``sys_mapping``) or a sample of homogeneous depth.

The external selection is not optional for an X-ray sample.
The eRASS1 sample (EXT_LIKE ≥ 6, confirmed by rema on DR11, volume limited, in the Western Galactic hemisphere: 4 723 clusters, 500 copies each) has the same density as its randoms in six bins of the eROSITA exposure to within 0.88-1.11 (:math:`\chi^2` = 35 for 6 bins) with the approximate detection probability; without it, that is with the DR11 selection alone, the ratio goes from 0.49 at the lowest exposure to 2.29 at the highest (:math:`\chi^2` = 1 440).
Against :math:`n_{\rm H}` the :math:`\chi^2` falls from 142 to 22.
``qa --template NAME=FILE:COLUMN`` adds any HEALPix map (exposure, noise) to the checks and reports both.
For ACT DR6, ``cluster-randoms act-selfn`` averages the noise of the reference filter (``stitched_RMSMap_Arnaud_M2e14_z0p4.fits``) over the pixels of the area mask with flags 0, in NSIDE-2048 pixels: 13 109 deg², close to the 12 590 deg² of the flags-0 area of DR6.
At the 8 942 clusters with flags 0, ``fixed_y_c`` over that noise is 1.0149 times ``fixed_SNR`` (68% of them within 0.4%), a factor the command removes.
The ACT DR6 sample (fixed S/N ≥ 5, flags 0, confirmed by rema, volume limited: 3 189 clusters) follows its randoms against the ACT noise to within 0.96-1.07 (:math:`\chi^2` = 5.8 for 6 bins), against :math:`\chi^2` = 310 without the SZ selection; with the Eddington average of the detection probability (``eddington_alpha: 3``) :math:`\chi^2` = 3.2.
ACT's own random catalogues (``DR6_randoms_v1.0.tgz``: mock clusters of nemo with S/N ≥ 5, here at :math:`0.1 \leq z < 0.7` in the DR11 domain) put 27% of the clusters in the sixth of the area with the lowest noise, against 24% for our randoms and 25% for the data: the mocks are not confirmed optically, and the measured :math:`\tilde y_0` overestimates the detection probability in noisy areas (Eddington bias).

For the SPT-SZ sample (238 clusters with :math:`\xi \geq 5` and :math:`z_\lambda \geq 0.25`) the field depths change the randoms little: the depth-template :math:`\chi^2` are 1-13 for 4 bins with and without them.

The PSZ2 sample (union S/N ≥ 6 in the survey mask, confirmed by rema: 309 clusters) needs more than the released selection function, whose completeness is averaged over the sky.
With the mask only, its density relative to the randoms rises with the ecliptic latitude, from 0.71 ± 0.10 below 10.5° to 1.31 ± 0.13 above 39° (:math:`\chi^2` = 15 for 4 bins), and with the Planck hit counts, from 0.63 to 1.45 (:math:`\chi^2` = 28): Planck observed the ecliptic poles longest.
``cluster-randoms psz2-model`` takes the hit counts :math:`H` of the HFI 143 GHz map (``HFI_SkyMap_143_2048_R2.02_full.fits``, averaged over NSIDE-256 pixels, about the scale of the matched filters) as the noise, :math:`\sigma \propto H^{\gamma}`, and the ``psz2`` selection then moves the S/N of a cluster as :math:`{\rm S/N}_i (H_i/H_j)^{\gamma}`.
We use :math:`\gamma = -1/2`, white noise; a regression of the catalogue's own noise proxy, :math:`\ln(Y_{5R500}/{\rm S/N}) = a + \gamma \ln H + \delta \ln Y_{5R500}`, gives :math:`\gamma = -0.129 \pm 0.014`, diluted towards 0 by the scatter of that proxy (it is recorded as ``gamma_fit``).
With :math:`\gamma = -1/2` the ratio against the hit counts is flat (0.94-1.06, :math:`\chi^2` = 0.7 for 4 bins), against the ecliptic latitude too (:math:`\chi^2` = 3.2), and against the optical depth the :math:`\chi^2` falls from 13 to 5.
Values of :math:`\gamma` from -0.13 to -0.7 give :math:`\chi^2` = 18, 10, 5.0, 0.7 and 0.9 against the hit counts.

Copies made once serve any stricter sample: ``sample`` with a :math:`z_{\rm vlim}` map or a domain that differ from those of ``copies`` (a higher depth significance, a seam buffer) drops the copies outside them, and the remaining copies are uniform in the smaller area.
