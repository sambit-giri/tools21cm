=========
Tools21cm
=========

.. image:: https://joss.theoj.org/papers/10.21105/joss.02363/status.svg
   :target: https://doi.org/10.21105/joss.02363

A python package for analysing 21-cm signals from the Epoch of Reionization (EoR) and Cosmic Dawn (CD). The source files can be found `here <https://github.com/sambit-giri/tools21cm>`_.

Note: There are some modules in the package that are still under active development. Therefore please contact the authors if you get erronous results.


Package details
===============

The package provides tools to analyse cosmological simulations of EoR and CD. It contains modules to create mock 21-cm observations for current and upcoming radio telescopes, such as LOFAR, MWA and SKA, and to construct statistical measures.

Input
-----

Currently, `Tools21cm` supports the following simulation codes:

* |cubep3m|_: a high performance cosmological N-body code
* |c2ray|_: a numerical method for calculating 3D radiative transfer and for simulating the EoR and CD
* `GRIZZLY <https://arxiv.org/abs/1710.09397>`_: an EoR and CD simulation code based on 1D radiative transfer 
* `21cmFAST <https://21cmfast.readthedocs.io/en/latest/>`_: a semi-numerical cosmological simulation code for the radio 21cm signal
* `Simfast21 <https://github.com/mariogrs/Simfast21>`_: a semi-numerical cosmological simulation code for the radio 21cm signal
* `sem_num <https://arxiv.org/abs/1403.0941>`_: a simple set of codes to semi-numerically simulate HI maps during reionization


.. |c2ray| replace:: C\ :sup:`2`\RAY
.. _c2ray: https://github.com/garrelt/C2-Ray3Dm

.. |cubep3m| replace:: CUBEP\ :sup:`3`\M
.. _cubep3m: https://github.com/jharno/cubep3m

Outputs
-------

There are various manipulation and analysis moduled in `Tools21cm`. 

* Angular coordinates: methods to convert data between physical (cMpc) coordinates and observational (angular-frequency) coordinates

* Bubble Statistics: methods to calcluate the sizes of the regions of interest and estimate the size distributions

* Cosmological calculators: various functions for calculating some cosmological stuff

* Identifying regions: methods to identify regions of interest in images

* Lightcone: methods to construct lightcones

* Power spectrum: contains functions to estimate various two point statistics

* Reading simuation outputs: methods to read simulations outputs

* Smoothing: methods to smooth or reduce resolution of the data to reduce noise

* Point statistics: contains various useful statistical methods

* Temperature: methods to estimate the brightness temperature

For detail documentation and how to use them, see `here <https://tools21cm.readthedocs.io/contents.html>`_.

Under Developement
------------------

* Foreground model: methods to simulate and analyse the foreground signal for 21 cm signal
* Primary beam: methods to simulate the primary beam of radio telescope
* Telescope noise: 
	* simulate the radio telescope observation strategy
	* simulate telescope noise
* Topology: methods to estimate the topology of the region of interest

