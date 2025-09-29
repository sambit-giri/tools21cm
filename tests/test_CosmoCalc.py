import numpy as np 
import tools21cm as t2c 
from astropy.cosmology import FlatLambdaCDM

def test_calculators_Planck18():
	# Planck 2018
	Om0 = 0.315
	Ob0 = 0.044
	H0 = 67.4
	s8 = 0.811
	ns = 0.965

	t2c.set_hubble_h(H0/100)  
	t2c.set_omega_matter(Om0)
	## t2c.set_omega_lambda(1.0-Om0)
	t2c.set_omega_baryon(Ob0)
	t2c.set_sigma_8(s8)
	t2c.set_ns(ns)

	assert t2c.const.h==(H0/100)
	assert t2c.const.Omega0==Om0 
	assert t2c.const.OmegaB==Ob0
	assert t2c.const.sigma_8==s8
	assert t2c.const.n_s==ns

	cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
	cdist = lambda z: cosmo.comoving_distance(z).to('Mpc').value
	ldist = lambda z: cosmo.luminosity_distance(z).to('Mpc').value

	assert np.abs(t2c.M_to_Tvir(1e8, 9.)/10334-1)<0.001
	assert np.abs(t2c.Tvir_to_M(1e4, 9.)/95191424-1)<0.001
	assert np.abs(t2c.angular_size(100, 9.)/21.93-1)<0.001
	assert np.abs(t2c.angular_size_comoving(1000, 9.)/(np.rad2deg(1000/cdist(9.)))-1)<0.001
	assert np.abs(t2c.c_to_p(100, 9.)/10-1)<0.001
	assert np.abs(1e4/cdist(t2c.cdist_to_z(1e4))-1)<0.001 # np.abs(t2c.cdist_to_z(1e4)/11.86-1)<0.01
	assert np.abs(t2c.z_to_cdist(9.)/cdist(9)-1)<0.001
	assert np.abs(t2c.deg_to_cdist(2., 9.)/(cdist(9.)*np.deg2rad(2))-1)<0.001
	assert np.abs(t2c.luminosity_distance(9.)/ldist(9.)-1)<0.001
	assert np.abs(t2c.nu_to_cdist(500)/cdist(t2c.nu_to_z(500))-1)<0.001
	assert np.abs(t2c.nu_to_wavel(50)/6-1)<0.001
	assert np.abs(t2c.nu_to_z(50)/27.4-1)<0.001
	assert np.abs(t2c.p_to_c(100, 9)/1000-1)<0.001
	assert np.abs(t2c.z_to_nu(9.)/142.0-1)<0.001

def test_calculators_WMAP7():
	# WMAP7
	Om0 = 0.27
	Ob0 = 0.044
	H0 = 70.0
	s8 = 0.80
	ns = 0.96

	t2c.set_hubble_h(H0/100)  
	t2c.set_omega_matter(Om0)
	## t2c.set_omega_lambda(1.0-Om0)
	t2c.set_omega_baryon(Ob0)
	t2c.set_sigma_8(s8)
	t2c.set_ns(ns)

	assert t2c.const.h==(H0/100)
	assert t2c.const.Omega0==Om0 
	assert t2c.const.OmegaB==Ob0
	assert t2c.const.sigma_8==s8
	assert t2c.const.n_s==ns

	assert np.round(t2c.M_to_Tvir(1e8, 9))==10094.0
	assert np.round(t2c.Tvir_to_M(1e4, 9))==98606776.0
	assert np.round(t2c.angular_size(100, 9))==22.0
	assert np.round(t2c.angular_size_comoving(1000, 9.))==6.0
	assert np.round(t2c.c_to_p(100, 9.))==10.0
	assert np.round(t2c.cdist_to_z(1e4), decimals=1)==10.9
	assert np.round(t2c.z_to_cdist(9.))==9567.0
	assert np.round(t2c.deg_to_cdist(2., 9.))==334.0
	assert np.round(t2c.luminosity_distance(9.))==95668.0
	assert np.round(t2c.nu_to_cdist(500))==5073.0
	assert np.round(t2c.nu_to_wavel(50))==6.0
	assert np.round(t2c.nu_to_z(50))==27.0
	assert t2c.p_to_c(100, 9)==1000
	assert np.round(t2c.z_to_nu(9.))==142.0