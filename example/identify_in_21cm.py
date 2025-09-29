import numpy as np
import matplotlib.pyplot as plt
import tools21cm as t2c

### Setting the simulation environment
t2c.set_sim_constants(244)

### Reading files
xfrac_filename = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/244Mpc/244Mpc_f2_0_250/results/xfrac3d_6.418.bin'
dens_filename  = '/disk/dawn-1/sgiri/simulations/244Mpc/coarser_densities/nc250/6.418n_all.dat'

xfrac = t2c.read_c2ray_files(xfrac_filename, file_type='xfrac') # Ionization fraction file
neut  = 1 - xfrac 						# Neutral fraction file

dens  = t2c.read_c2ray_files(dens_filename, file_type='dens')   # Density file

### Redshift from filename.....It can be manually given also
z = float(xfrac_filename.split('_')[-1].split('.b')[0])

### Making 21-cm coeval cube
dt = t2c.calc_dt(xfrac, dens, z)

### Smoothing neutral field to SKA resolution
smt_dt = t2c.smooth_coeval(dt, z)

### Generating the binary field from 21-cm signal using KMeans
bin_xf_sim = t2c.threshold_kmeans_3cluster(dt, upper_lim=True, n_jobs=5)
bin_nf_sim = 1. - bin_xf_sim		# The neutral binary field at Sim-Res

bin_xf_smt = t2c.threshold_kmeans_3cluster(smt_dt, upper_lim=True, n_jobs=5)
bin_nf_smt = 1. - bin_xf_smt            # The neutral binary field at Sim-Res


