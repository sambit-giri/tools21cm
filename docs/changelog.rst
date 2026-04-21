=========
Changelog
=========

v2.4
----
* ``DistributionDiagnostic`` class family for comparing and diagnosing probability distributions.
* ``SampledDistribution``: diagnose distributions from sample arrays (MCMC chains, Monte Carlo draws, bootstrap replicates); supports importance weights.
* ``GriddedProbabilities``: same interface for distributions on a regular N-D probability grid.
* Corner plots (``corner`` and ``getdist`` backends) with default 68% and 95% contours; mixed-dimensionality overlays supported.
* Forest plots showing 68% and 95% credible intervals across distributions.
* Calibration metrics: Z-score, PIT, bias, RMSE, Mahalanobis distance, coverage.
* ``fftconvolve`` now handles arrays of unequal shape.

v2.3
----
* GPU-accelerated topology: Euler characteristics via PyTorch, with Apple M-chip (MPS) support.
* Radio telescope sensitivity: SEFD tables, SKA-Low Bessel primary beam, UV mapping in Lagrangian space, uniform weighting, spectral-leakage suppression.
* Astrophysical data: fesc LyA constraints, Qin+2025 MAP reionization model, reionization observational constraints.
* Zarr file format support.
* Noise lightcone fixes (double ``jansky_2_kelvin`` call, decreasing-redshift input).
* ``fftconvolve`` moved to dedicated ``fft_functions.py``.

v2.2
----
* Bispectrum and integrated bispectrum estimators.
* Multiple SKA layouts (AA1, AA2, AA*, AA4) with antenna-wise gain modelling.
* UV track simulation speed-up (×10).
* ViteBetti topology with Cython acceleration.
* py21cmfast interface for dark-matter halo retrieval.
* Landy-Szalay correlation function estimator.
* Migrated build system to ``pyproject.toml``; ``scipy`` version compatibility layer.

v2.1
----
* Modules to analyse 21 cm images added.
* Compatible with python 3 only.

v1.1
----
* Sambit Giri and Garrelt Mellema made the package more general purpose in 2016 and renamed it to `tools21cm <https://tools21cm.readthedocs.io/>`_.
* Compatible with python 2 only.
* Post this version, we stop the development of the python 2 version.

v0.1
----
* Initial version of the package was called `c2raytools <https://ttt.astro.su.se/~gmell/c2raytools/build/>`_. 
* It was created by Hannes Jensen and Garrelt Mellema in 2014 to analyse data output from |c2ray|_ and |cubep3m|_ simulations.
* Compatible with python 2 only.

.. |c2ray| replace:: C\ :sup:`2`\RAY
.. _c2ray: https://github.com/garrelt/C2-Ray3Dm

.. |cubep3m| replace:: CUBEP\ :sup:`3`\M
.. _cubep3m: https://wiki.cita.utoronto.ca/index.php/CubePM
