import numpy as np
import matplotlib.pyplot as plt
import tools21cm as t2c

### Setting the simulation environment
t2c.set_sim_constants(244)

### Reading files
xfrac_filename = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/244Mpc/244Mpc_f2_0_250/results/xfrac3d_6.450.bin'

xfrac = t2c.read_c2ray_files(xfrac_filename, file_type='xfrac') # Ionization fraction file
neut  = 1 - xfrac 						# Neutral fraction file

### Redshift from filename.....It can be manually given also
z = float(xfrac_filename.split('_')[-1].split('.b')[0])

### Smoothing neutral field to SKA resolution
smt_neut = t2c.smooth_coeval(neut, z)

### Generating the binary fields
xth = 0.5                          # The fixed threshold to identify the regions of interest
bin_neut = neut > xth 
bin_smt  = smt_neut > xth

### Looking at the slices
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.subplots_adjust(left=0.07, bottom=0.06, right=0.90, top=0.96, wspace=0.01, hspace=0.15)
im00 = axes[0,0].imshow(neut[:,:,125], vmin=0, vmax=1)
im01 = axes[0,1].imshow(smt_neut[:,:,125], vmin=0, vmax=1)
im10 = axes[1,0].imshow(bin_neut[:,:,125], vmin=0, vmax=1)
im11 = axes[1,1].imshow(bin_smt[:,:,125], vmin=0, vmax=1)

cax = fig.add_axes([0.91, 0.08, 0.02, 0.88])
cbar = fig.colorbar(im00, cax=cax)
cbar.set_label('x$_\mathrm{HI}$')

plt.show()


#################### Size statistics

### MFP
rs, dn, r_p             = t2c.mfp(bin_neut, boxsize=t2c.conv.LB)
rs_smt, dn_smt, r_p_smt = t2c.mfp(bin_smt, boxsize=t2c.conv.LB)

### FOF
mp, sz         = t2c.fof(bin_neut) # This gives a list of sizes
mp_smt, sz_smt = t2c.fof(bin_smt)  # We have to convert them into the probability distribution

volume_resolution = t2c.conv.LB**3/neut.shape[0]**3
vs, vdn, dm             = t2c.plot_fof_sizes(sz*volume_resolution, bins=30)
vs_smt, vdn_smt, dm_smt = t2c.plot_fof_sizes(sz_smt*volume_resolution, bins=30)


### Ploting the BSDs
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(6.0, 12.0,forward=True)
fig.subplots_adjust(left=0.14, bottom=0.06, right=0.96, top=0.96, wspace=0.01, hspace=0.25)

axes[0].set_title('MFP-BSD')
axes[0].semilogx(rs, dn, c='b', label='SimRes')
axes[0].semilogx(rs_smt, dn_smt, '--', c='b', label='Smooth')
axes[0].set_xlim(1.25,110)
axes[0].set_xlabel('R (Mpc)')
axes[0].set_ylabel('R$\\frac{\mathbf{dp}}{\mathbf{dR}}$')
axes[0].legend(loc=2)
axes[1].set_title('FOF-BSD')
axes[1].loglog(vs, vdn, c='r', label='SimRes', linestyle='steps')
axes[1].loglog(vs_smt, vdn_smt, c='r', label='Smooth', linestyle='steps--')
axes[1].set_ylim(max(dm,dm_smt),1)
axes[1].set_xlim(volume_resolution,3e6)
axes[1].set_xlabel('V (Mpc$^3$)')
axes[1].set_ylabel('V$^2\\frac{\mathbf{dp}}{\mathbf{dV}}$ (Mpc)')
axes[1].legend(loc=2)
plt.show()


