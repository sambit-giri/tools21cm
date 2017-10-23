import numpy as np
import c2raytools as c2t
import glob

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


def coeval_21cm_c2ray(xfrac_dir, dens_dir, z, interpolation='linear', mean_subtract=False):
	"""
	xfrac_dir     : Give the path that contains the xfrac-files.
	dens_dir      : Give the path that contains the density-files.
	z	      : Redshift.
	interpolation : This is used when the xfrac cube at that redshift is not available.
	"""
	xfrac = coeval_xfrac_c2ray(xfrac_dir, z, interpolation=interpolation)
	dens  = coeval_dens_c2ray(dens_dir, z)
	dt    = c2t.calc_dt(xfrac, dens, z=z)
	if mean_subtract: return dt-dt.mean()
	else: return dt


def coeval_xfrac_c2ray(xfrac_dir, z, interpolation='linear'):
	"""
	xfrac_dir     : Give the path that contains the xfrac-files.
	z	      : Redshift.
	interpolation : This is used when the coveal cube at that redshift is not available.
	"""
	if not interpolation in ['linear']: #, 'step', 'sigmoid', 'step_cell'
		raise ValueError('Unknown interpolation type: %s' % interpolation)
	xfrac_files = glob.glob(xfrac_dir + '/xfrac3d_*.bin')
	xfrac_zs = None
	xfrac_zs = c2t.lightcone._get_file_redshifts(xfrac_zs, xfrac_files)
	if z in xfrac_zs:
		xfrac = c2t.XfracFile(xfrac_files[np.argwhere(z==xfrac_zs)]).xi
	else:
		z_l = xfrac_zs[xfrac_zs<z].max()
		z_h = xfrac_zs[xfrac_zs>z].min()
		xfrac_l = c2t.XfracFile(xfrac_files[xfrac_zs[xfrac_zs<z].argmax()]).xi
		xfrac_h = c2t.XfracFile(xfrac_files[xfrac_zs[xfrac_zs>z].argmin()]).xi
		xfrac = xfrac_h + (xfrac_l-xfrac_h)*(z-z_h)/(z_l-z_h)
		print "The xfrac cube has been interpolated using", interpolation, "interpolation."
		print "CAUTION: This data should be used with care."
	return xfrac

def coeval_dens_c2ray(dens_dir, z):
	"""
	dens_dir      : Give the path that contains the density-files.
	z	      : Redshift.
	"""
	dens_files  = glob.glob(dens_dir + '/*n_all.dat')
	dens_zs  = None
	dens_zs  = c2t.lightcone._get_file_redshifts(dens_zs, dens_files)
	if z in dens_zs:
		dens, dtype = c2t.helper_functions.get_data_and_type(dens_files[np.argwhere(z==dens_zs)])
	else:
		#z_l = dens_zs[dens_zs<z].max()
		z_h = dens_zs[dens_zs>z].min()
		#dens_l,dtype = c2t.helper_functions.get_data_and_type(dens_files[dens_zs[dens_zs<z].argmax()])
		dens_h, dtype = c2t.helper_functions.get_data_and_type(dens_files[dens_zs[dens_zs>z].argmin()])
		#dens = dens_h + (dens_l-dens_h)*(z-z_h)/(z_l-z_h)
		dens = dens_h*(z/z_h)**3
		#print "The density cube has been interpolated using", interpolation, "interpolation."
		print "The density has been scaled from the density at the previous time."
	return dens.astype(np.float64)
	
