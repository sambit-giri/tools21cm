import numpy as np 
import tools21cm as t2c 

box_dims = 200
dims   = [128,128,128]
gauss  = np.random.normal(loc=0., scale=1., size=dims)
kbins  = 10
mubins = 2

def test_cross_power_spectrum_1d():
	'''
	With this test, cross_power_spectrum_nd and radial_average are also test.
	'''
	pp, kk = t2c.cross_power_spectrum_1d(gauss, gauss, kbins=kbins, box_dims=box_dims)
	slope  = (np.log10(pp*kk**3/2/np.pi**2)[kbins-3]-np.log10(pp*kk**3/2/np.pi**2)[3])/(np.log10(kk)[kbins-3]-np.log10(kk)[3])
	assert np.abs(slope-3)<=0.1 

def test_cross_power_spectrum_mu():
	'''
	With this test, cross_power_spectrum_nd and mu_binning are also test.
	'''
	pp, mm, kk = t2c.cross_power_spectrum_mu(gauss, gauss, kbins=kbins, mubins=mubins, box_dims=box_dims)
	slope  = (np.log10(pp[0,:]*kk**3/2/np.pi**2)[kbins-3]-np.log10(pp[0,:]*kk**3/2/np.pi**2)[3])/(np.log10(kk)[kbins-3]-np.log10(kk)[3])
	assert np.abs(slope-3)<=0.1 

def test_power_spectrum_1d():
	'''
	With this test, power_spectrum_nd and radial_average are also test.
	'''
	pp, kk = t2c.power_spectrum_1d(gauss, kbins=kbins, box_dims=box_dims)
	slope  = (np.log10(pp*kk**3/2/np.pi**2)[kbins-3]-np.log10(pp*kk**3/2/np.pi**2)[3])/(np.log10(kk)[kbins-3]-np.log10(kk)[3])
	assert np.abs(slope-3)<=0.1 

def test_dimensionless_ps():
	'''
	With this test, power_spectrum_nd and radial_average are also test.
	'''
	dd, kk = t2c.dimensionless_ps(gauss, kbins=kbins, box_dims=box_dims)
	slope  = (np.log10(dd)[kbins-3]-np.log10(dd)[3])/(np.log10(kk)[kbins-3]-np.log10(kk)[3])
	assert np.abs(slope-3)<=0.25

def test_power_spectrum_mu():
	'''
	With this test, power_spectrum_nd and mu_binning are also test.
	'''
	pp, mm, kk = t2c.power_spectrum_mu(gauss, kbins=kbins, mubins=mubins, box_dims=box_dims)
	slope  = (np.log10(pp[0,:]*kk**3/2/np.pi**2)[kbins-3]-np.log10(pp[0,:]*kk**3/2/np.pi**2)[3])/(np.log10(kk)[kbins-3]-np.log10(kk)[3])
	assert np.abs(slope-3)<=0.1



