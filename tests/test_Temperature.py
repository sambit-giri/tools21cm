import numpy as np 
from skimage.morphology import ball
import tools21cm as t2c 

rad  = 5
xHII = ball(rad)
rho  = np.ones_like(xHII)*3.968722e-31
z    = 7.059

def test_calc_dt():
	dt = t2c.calc_dt(xHII, rho, z)
	assert dt.max()-23.35746076<0.1
	assert dt.min()==0

def test_calc_dt_full():
	dt = t2c.calc_dt_full(xHII, rho, t2c.Tcmb0*(1+z)*np.ones_like(xHII), z) 
	assert np.all(dt<0.1)

def test_calc_dt_lightcone():
	dt = t2c.calc_dt_lightcone(xHII, rho, z)
	test = True
	for i in range(1,dt.shape[2]):
		if dt[0,0,i]<dt[0,0,i]: test = False
	assert test 

def test_mean_dt():
	assert np.abs(t2c.mean_dt(z)-23.833022132850783)<0.1

