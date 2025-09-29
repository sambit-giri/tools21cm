import numpy as np 
import tools21cm as t2c

data1 = np.ones((10,10,10))
data2 = np.random.normal(0,1,size=(10,10,10))
data3 = np.array([np.ones((10,10))*i for i in range(10)])

def test_apply_func_along_los():
	assert np.all(t2c.apply_func_along_los(data1, np.mean, 2)==1)
	assert np.all(t2c.apply_func_along_los(data3, np.mean, 0)==np.arange(10))
	assert np.all(t2c.apply_func_along_los(data3, np.std, 0)==0)

def test_skewness():
	assert t2c.skewness(data2)/data2.std()**2<=1
	assert t2c.skewness(data3)==0

def test_mass_weighted_mean_xi():
	assert t2c.mass_weighted_mean_xi(data1, data3)==1
	assert t2c.mass_weighted_mean_xi(data3, data1*2)==data3.mean()
	assert np.abs(t2c.mass_weighted_mean_xi(data3, data3)-(data3**2).mean()/data3.mean())<1

def test_signal_overdensity():
	assert np.all(t2c.signal_overdensity(data1, 2)==0)

def test_subtract_mean_signal():
	assert np.all(t2c.subtract_mean_signal(data3, 0)==0)



