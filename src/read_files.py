import numpy as np
import c2raytools as c2t 

def _load_binary_data(filename, dtype=np.float32): 
	""" 
	We assume that the data was written 
	with write_binary_data() (little endian). 
	""" 
	f = open(filename, "rb") 
	data = f.read() 
	f.close() 
	_data = np.fromstring(data, dtype) 
	if sys.byteorder == 'big':
		_data = _data.byteswap()
	return _data 

def read_21cmfast_files(filename):
	"""
	Parameter
	---------
	filename: Give the filename of the 21cmFAST output files.

	Return
	---------
	Numpy array
	"""
	bla = _load_binary_data(filename)
	dim = round(bla.size**0.333333)
	return bla.reshape(dim,dim,dim)

def read_c2ray_files(filename, file_type='xfrac', density_file=None):
	"""
	Parameter
	---------
	filename    : Give the filename of the C2Ray output files.
	file_type   : The file of file being read has to be mentioned. The options are 
	               'xfrac', 'dens', 'vel'. (Default: 'xfrac')
	density_file: This is necessary if file_type=='vel'. (Default: None)

	Return
	---------
	Numpy array
	"""
	assert file_type in ['xfrac', 'dens', 'vel']
	if file_type=='xfrac': out = c2t.XfracFile(filename).xi
	elif file_type=='dens': 
		dens, dtype = c2t.helper_functions.get_data_and_type(filename)
		out = dens.astype(np.float64)
	elif file_type=='vel':
		assert density_file is not None
		dfile = c2t.density_file.DensityFile(density_file)
		vel_file = c2t.vel_file.VelocityFile(filename)
		out = vel_file.get_kms_from_density(dfile)
	return out

def read_grizzly_files(filename):
	"""
	Parameter
	---------
	filename: Give the filename of the GRIZZLY xfrac files.

	Return
	---------
	Numpy array
	"""
	return c2t.XfracFile(filename).xi

	
