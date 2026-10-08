"""PSCD: Photo-z Space Cluster Detection.

An implementation of the AMICO matched-filter cluster finder (Bellagamba, Roncarelli, Maturi &
Moscardini 2018, MNRAS 473, 5221, arXiv:1705.03029; with the magnitude-dependent redshift
statistics and the KiDS template of Maturi et al. 2019, MNRAS 485, 498, arXiv:1810.02811). It is
not the AMICO code: the method follows the papers, the implementation is rema's, on the DR11
photo-z (:mod:`rema.model.photoz`). Galaxies of any colour count; no red sequence is used.

The data D(theta, m, z), galaxies smeared by their photo-z distribution p_i(z), are modelled as
A M_c(theta - theta_c, m) q(z_c, z) + N(m, z), where M_c is the cluster template
(:mod:`rema.pscd.model`) and N the field (the stacked photo-z distributions,
:func:`rema.model.background.build_photoz_bkg`). On a 3D grid of positions and redshifts
(:mod:`rema.pscd.grid`) the optimal linear filter gives (Bellagamba et al. 2018, Eqs. 5-12)

    S(theta_c, z_c) = sum_i M_c(theta_i - theta_c, m_i) p_i(z_c) / N(m_i, z_c)
    A = S / alpha - B,   B = beta / alpha,   sigma_A^2 = 1/alpha + A gamma / alpha^2,
    alpha = int M_c^2 q^2 / N,   beta = int M_c,   gamma = int M_c^3 q^3 / N^2,

with the integrals over the unmasked area and down to the local depth, and the likelihood
L = L0 + A^2 alpha (Eq. 14). Detections are extracted one at a time (:mod:`rema.pscd.detect`):
the cell of highest likelihood with S/N = A / sigma_A above the threshold; the membership
probabilities of the galaxies around it (Eq. 24),

    P(i in j) = P_f,i A_j M_j p_i(z_j) / (A_j M_j p_i(z_j) + N(m_i, z_j)),

reduce their field probabilities P_f,i, and their weighted contributions are removed from the
map (the cleaning, Eq. 25) before the next detection. The richness LAMBDA is sum_i P(i in j) and
LAMBDA_STAR the same sum over members brighter than m* + 1.5 within R200 (Maturi et al. 2019).
"""
