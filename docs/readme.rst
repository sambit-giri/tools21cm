=========
Tools21cm
=========

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
* `grizzly <https://arxiv.org/abs/1710.09397>`_: an EoR and CD simulation code based on 1D radiative transfer 


.. |c2ray| replace:: C\ :sup:`2`\RAY
.. _c2ray: https://github.com/garrelt/C2-Ray3Dm

.. |cubep3m| replace:: CUBEP\ :sup:`3`\M
.. _cubep3m: https://github.com/jharno/cubep3m