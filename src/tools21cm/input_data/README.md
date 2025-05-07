# SegUNet weights
Weights of the trained neural network, SegUNet, for HII identification in noisy 21-cm images (segmentation). See [Bianco et al. (2021)](https://arxiv.org/abs/2102.06713) for details.

- For coeaval noisy cubes: segunet_02-10T23-52-36_128slice_ep56.h5
- For noisy lightcones: segunet_03-11T12-02-05_128slice_ep35.h5

# SKA-Low Antenna Configuration
The locations of SKAO's low-frequency component (SKA-Low) at different construction stages. See [this link](https://www.skao.int/en/explore/construction-journey) for more information about these stages.

# System Equivalent Flux Density (SEFD) tables
Sensitivity A/T [m²/K] of a single station as a function of frequency. Tables calculated from the [SKAO webtool](http://skalowsensitivitybackup-env.eba-daehsrjt.ap-southeast-2.elasticbeanstalk.com/sensitivity_radec_vs_freq/).

AAVS2 antenna are the (tree like) dipole antenna of SKA-Low station, while the EDA2, as for its predecessor EDA1, consists of 256 MWA bowtie dipoles. 

- Reference Paper: [Sokolowski et al. (2022)](https://arxiv.org/abs/2204.05873)
- Github repository conatining the code of the reference paper: [marcinsokolowski/station_beam](https://github.com/marcinsokolowski/station_beam/tree/master)
- The code that calculates the sensivity: [sensitivity_db.py](https://github.com/marcinsokolowski/station_beam/blob/master/python/sensitivity_db.py)
