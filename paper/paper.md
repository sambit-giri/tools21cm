---
title: 'Tools21cm: A python package to analyse the large-scale 21-cm signal from the Epoch of Reionization and Cosmic Dawn'
tags:
  - Python
  - astronomy
  - early universe
  - 21-cm signal
  - reionization
authors:
  - name: Sambit K. Giri
    orcid: 0000-0002-2560-536X
    affiliation: "1, 2" 
  - name: Garrelt Mellema
    orcid: 0000-0002-2512-6748
    affiliation: 1
  - name: Hannes Jensen
    affiliation: 3
affiliations:
 - name: Department of Astronomy and Oskar Klein Centre, Stockholm University, AlbaNova, SE-106 91 Stockholm, Sweden
   index: 1
 - name: Institute for Computational Science, University of Zurich, Winterthurerstrasse 190, CH-8057 Zurich, Switzerland
   index: 2
 - name: Self
   index: 3
date: 8 June 2020
bibliography: paper.bib

---

# Summary

The Cosmic Dawn (CD) and Epoch of Reionization (EoR) are among the least understood epochs from the history of our Universe. Large-scale cosmological simulations are useful to understand this era and help prepare for future observations. `Tools21cm` is a data manipulation and analysis package for studying such cosmological simulations. It is written in the python programming language and designed to be very user-friendly.

The 21-cm signal, produced by the spin-flip transition of neutral hydrogen, is a unique tracer of matter in large-scale structures present during the EoR and CD [e.g. @Furlanetto2006CosmologyUniverse; @Pritchard201221cmCentury]. This signal will be cosmologically redshifted and thus found at low radio frequencies.
Much of the functionality of `Tools21cm` is focused on the construction and analysis of mock 21-cm observations in the context of current and upcoming radio telescopes, such as the Low Frequency Array [LOFAR; @vanHaarlem2013LOFAR:ARray], the Murchison Widefield Array [MWA; @Tingay2013MWA; @Wayth2018MWAdesign], Hydrogen Epoch of Reionization Array [HERA; @deboer2017hydrogen] and the Square Kilometre Array [SKA; @Mellema2013ReionizationArray]. `Tools21cm` post-processes cosmological simulation data to create mock 21-cm observations.

Radio telescopes typically observe the redshifted 21-cm signal over a range of frequencies and therefore produce 21-cm images at different redshifts. A sequence of such images from different redshifts is known as a tomographic data set and is three dimensional. `Tools21cm` can construct such tomographic data sets from simulation snapshots [@Datta2012Light-coneSpectrum; @Giri2018BubbleTomography]. When constructing these data sets, it can also add the impact of peculiar velocities, leading to an effect known as redshift space distortions [e.g. @Jensen2013ProbingDistortions; @Jensen2016TheMeasurements; @Giri2018BubbleTomography]. See @giri2019tomographic for a detailed description of tomographic 21-cm data sets.

`Tools21cm` also includes tools to calculate a wide range of statistical quantities from simulated 21-cm data sets. These include one-point statistics, such as the global or sky-averaged signal as a function of frequency, as well the variance, skewness and kurtosis [e.g. @Ross2017SimulatingDawn; @Ross2019EvaluatingDawn]. It can also characterise the spatial fluctuations in the signal through spherically and cylindrically averaged power spectra [@Jensen2013ProbingDistortions; @Ross2017SimulatingDawn; @Giri2019NeutralTomography] and position dependent power spectra [@Giri2019Position-dependentReionization].
It also has the capability to find interesting features, such as ionized regions, in (tomographic) image data [@Giri2018OptimalObservations; @Giri2019NeutralTomography] and from these to derive statistical quantities, such as size distributions [@Giri2018BubbleTomography] and topological quantities such as the Euler characteristic [@Giri2019NeutralTomography]. Such statistical characterisations of the data are required when comparing observations with simulations using a Bayesian inference framework in order to derive constraints on model parameters [e.g. @Greig201521CMMC:Signal].


# Acknowledgements

We acknowledge the contributions and feedback from Keri Dixon, Hannah Ross, Raghunath Ghara, Ilian Iliev, Catherine Watkinson and Michele Bianco. We are also thankful for support by Swedish Research Council grant 2016-03581.

# References
