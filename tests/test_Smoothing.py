import numpy as np 
import tools21cm as t2c 

data = np.zeros((9,9,9))
data[4,4,4] = 1

def test_smooth_coeval():
	'''
	With this, smooth_coeval_gauss, smooth_coeval_tophat, gauss_kernel and tophat_kernel
	are also tested.
	'''
	smt = t2c.smooth_coeval(data, 9, box_size_mpc=90)
	assert smt[4,4,4]<1

def test_interpolate2d():
	sl = data[:,:,4]
	out1 = t2c.interpolate2d(sl, np.array([4]), np.array([4.5]), order=1).squeeze()
	out2 = t2c.interpolate2d(sl, np.array([4]), np.array([4]), order=1).squeeze()
	assert out1==0.5
	assert out2==1.0

def test_interpolate3d():
	out1 = t2c.interpolate3d(data, np.array([4]), np.array([4]), np.array([4.5]), order=1).squeeze()
	out2 = t2c.interpolate3d(data, np.array([4]), np.array([4]), np.array([4]), order=1).squeeze()
	assert out1==0.5
	assert out2==1.0


def test_tophat_kernel_3d():
	kernel = t2c.tophat_kernel_3d(5,5)
	assert np.all((kernel-1/5**3)<0.001)

