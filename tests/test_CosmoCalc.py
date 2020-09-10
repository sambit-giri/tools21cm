import numpy as np 
import tools21cm as t2c 

def test_calculators():
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