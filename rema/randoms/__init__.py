"""Cluster randoms: the selection function of a cluster catalogue as random points.

As redMaPPer (Rykoff et al. 2016, sections 3.4-3.6), every cluster of a sample is copied to
random positions where it could have been found: positions where the survey is volume limited
at the cluster's redshift (the z_vlim map, :mod:`rema.randoms.vlim`) inside the area of the run
(:mod:`rema.randoms.domain`). Its aperture completeness is computed there from the footprint
alone (:mod:`rema.randoms.aperture`; richness is not re-measured), and the copies that would
have passed the catalogue cuts, times the probability that an external survey (X-ray, SZ)
detects the cluster there (:mod:`rema.randoms.selection`), share the cluster's weight
(:mod:`rema.randoms.weigh`). Weighted randoms then have the redshift, richness and observable
distribution of the sample, and the angular distribution of its selection.
"""
