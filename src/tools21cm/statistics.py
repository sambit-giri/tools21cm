'''
This file contains various useful statistical methods
'''

import numpy as np
#from lightcone import _get_slice

def skewness(x):
	''' 
	Calculate the skewness of an array.
	Note that IDL calculates the skewness in a slightly different way than Python. 
	This routine uses the IDL definition. 
	
	Parameters:
		x (ndarray): The array containing the input data.
		
	Returns:
		The skewness.
	
	'''
	mx = np.mean(x)
	n = np.size(x)
	xdiff = x-mx
	#return (sum(xdiff**3)/n)/((sum(xdiff**2)/n)**(3./2.)) #This is how SciPy does it
	return (np.sum(xdiff**3)/n)/((np.sum(xdiff**2)/(n-1))**(3./2.))

def kurtosis(x):
        ''' 
        Calculate the kurtosis of an array.
        It uses the definition given in Ross et al. (2017).
        
        Parameters:
                x (ndarray): The array containing the input data
                
        Returns:
                The kurtosis.

        '''
        mx = np.mean(x)
        n = np.size(x)
        xdiff = x-mx
        #return (sum(xdiff**3)/n)/((sum(xdiff**2)/n)**(3./2.)) #This is how SciPy does it
        return (np.sum(xdiff**4)/n)/((np.sum(xdiff**2)/(n-1))**(2.))


def mass_weighted_mean_xi(xi, rho):
	''' Calculate the mass-weighted mean ionization fraction.
	
	Parameters:
		xi (ndarray): the ionized fraction
		rho (ndarray): the density (arbitrary units)
		
	Returns:
		The mean mass-weighted ionized fraction.
	
	 '''
	xi = xi.astype('float64')
	rho = rho.astype('float64')
	return np.mean(xi*rho)/np.mean(rho)


def subtract_mean_signal(signal, los_axis=2):
	'''
	Subtract the mean of the signal along the los axis. 
	
	Parameters:
		signal (ndarray): the signal to subtract the mean from
		los_axis (int): the line-of-sight axis (Default: 2)
			
	Returns:
		The signal with the mean subtracted
		
	TODO:vectorize 
	'''
	signal_out = signal.copy()
	
	for i in range(signal.shape[los_axis]):
		if los_axis in [0,-3]:
			signal_out[i,:,:] -= signal[i,:,:].mean()
		if los_axis in [1,-2]:
			signal_out[:,i,:] -= signal[:,i,:].mean()
		if los_axis in [2,-1]:
			signal_out[:,:,i] -= signal[:,:,i].mean()

	return signal_out

                                                               
def signal_overdensity(signal, los_axis):
	'''
	Divide by the mean of the signal along the los axis and subtract one.
	
	Parameters:
		signal (ndarray): the signal to subtract the mean from
		los_axis (int): the line-of-sight axis
			
	Returns:
		The signal with the mean subtracted
		
	TODO:vectorize 
	'''
	signal_out = signal.copy()
	
	for i in range(signal.shape[los_axis]):
		if los_axis == 0:
			signal_out[i,:,:] /= signal[i,:,:].mean()
		if los_axis == 1:
			signal_out[:,i,:] /= signal[:,i,:].mean()
		if los_axis == 2:
			signal_out[:,:,i] /= signal[:,:,i].mean()

	return signal_out - 1.


def apply_func_along_los(signal, func, los_axis):
	'''
	Apply a function, such as np.var() or np.mean(), along
	the line-of-sight axis of a signal on a 
	per-slice basis.
	
	Parameters:
		signal (ndarray): the signal
		func (callable): the function to apply
		los_axis (int): the line-of-sight axis
		
	Returns:
		An array of length signal.shape[los_axis]
		
		
	Example:
		Calculate the variance of a lightcone along the 
		line-of-sight:
		
		>>> lightcone = t2c.read_cbin('my_lightcone.cbin')
		>>> dT_var = t2c.apply_func_along_los(lightcone, np.var, 2)
		
	'''
	assert los_axis >= 0 and los_axis < len(signal.shape)
	output = np.zeros(signal.shape[los_axis])
	
	for i in range(len(output)):
		signal_slice = _get_slice(signal, i, los_axis)
		output[i] = func(signal_slice)
		
	return output

def _get_slice(data, idx, los_axis, slice_depth=1):
    '''
    Slice a data cube along a given axis. For internal use.
    '''
    assert len(data.shape) == 3 or len(data.shape) == 4
    assert los_axis >= 0 and los_axis < 3
    
    idx1 = idx
    idx2 = idx1+slice_depth

    if len(data.shape) == 3: #scalar field
        if los_axis == 0:
            return np.squeeze(data[idx1:idx2,:,:])
        elif los_axis == 1:
            return np.squeeze(data[:,idx1:idx2,:])
        return np.squeeze(data[:,:,idx1:idx2])
    else: #Vector field
        if los_axis == 0:
            return np.squeeze(data[:,idx1:idx2,:,:])
        elif los_axis == 1:
            return np.squeeze(data[:,:,idx1:idx2,:])
        return np.squeeze(data[:,:,:,idx1:idx2])