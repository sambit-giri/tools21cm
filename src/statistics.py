#This file contains various useful statistical methods

import numpy as np
from lightcone import _get_slice

def skewness(x):
	''' 
	Calculate the skewness of an array.
	Note that IDL calculates the skewness in a slightly different way than Python. 
	This routine uses the IDL definition. 
	
	Parameters:
		* x (numpy array): The array containing the input data
		
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
        Calculate the skewness of an array.
        Note that IDL calculates the skewness in a slightly different way than Python. 
        This routine uses the IDL definition. 
        
        Parameters:
                * x (numpy array): The array containing the input data
                
        Returns:
                The skewness.

        '''
        mx = np.mean(x)
        n = np.size(x)
        xdiff = x-mx
        #return (sum(xdiff**3)/n)/((sum(xdiff**2)/n)**(3./2.)) #This is how SciPy does it
        return (np.sum(xdiff**4)/n)/((np.sum(xdiff**2)/(n-1))**(2.))


def mass_weighted_mean_xi(xi, rho):
	''' Calculate the mass-weighted mean ionization fraction.
	
	Parameters:
		* xi (numpy array): the ionized fraction
		* rho (numpy array): the density (arbitrary units)
		
	Returns:
		The mean mass-weighted ionized fraction.
	
	 '''
	xi = xi.astype('float64')
	rho = rho.astype('float64')
	return np.mean(xi*rho)/np.mean(rho)


def subtract_mean_signal(signal, los_axis):
	'''
	Subtract the mean of the signal along the los axis. 
	
	Parameters:
		* signal (numpy array): the signal to subtract the mean from
		* los_axis (integer): the line-of-sight axis
			
	Returns:
		The signal with the mean subtracted
		
	TODO:vectorize 
	'''
	signal_out = signal.copy()
	
	for i in range(signal.shape[los_axis]):
		if los_axis == 0:
			signal_out[i,:,:] -= signal[i,:,:].mean()
		if los_axis == 1:
			signal_out[:,i,:] -= signal[:,i,:].mean()
		if los_axis == 2:
			signal_out[:,:,i] -= signal[:,:,i].mean()

	return signal_out

                                                               
def signal_overdensity(signal, los_axis):
	'''
	Divide by the mean of the signal along the los axis and subtract one.
	
	Parameters:
		* signal (numpy array): the signal to subtract the mean from
		* los_axis (integer): the line-of-sight axis
			
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
		* signal (numpy array): the signal
		* func (callable): the function to apply
		* los_axis (integer): the line-of-sight axis
		
	Returns:
		An array of length signal.shape[los_axis]
		
		
	Example:
		Calculate the variance of a lightcone along the 
		line-of-sight:
		
		>>> lightcone = c2t.read_cbin('my_lightcone.cbin')
		>>> dT_var = c2t.apply_func_along_los(lightcone, np.var, 2)
		
	'''
	assert los_axis >= 0 and los_axis < len(signal.shape)
	output = np.zeros(signal.shape[los_axis])
	
	for i in range(len(output)):
		signal_slice = _get_slice(signal, i, los_axis)
		output[i] = func(signal_slice)
		
	return output


