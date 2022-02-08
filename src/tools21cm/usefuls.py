import numpy as np
import sys
from . import conv, const, helper_functions, lightcone
from . import temperature as tm
from . import density_file, vel_file, xfrac_file
import glob

def loading_verbose(string):
	msg = ("Completed: " + string )
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()

def loading_msg(msg):
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()

def Mgrid_2_Msolar(M):
	return M*(conv.M_grid*const.solar_masses_per_gram)

def Msolar_2_Mgrid(M):
	return M/(conv.M_grid*const.solar_masses_per_gram)

def get_zs_list(xfrac_dir, file_type='/xfrac3d_*.bin'):
	"""
	xfrac_dir: Provide the directory whic contains all the data with redshift values in the name.
	file_type: Give the filename used to list it in Unix.
		   Example- xfrac files:   '/xfrac3d_*.bin' (Default)
			    density files: '/*n_all.dat'
	return   
	------
	The list of redshift values in the increasing order.
	"""
	xfrac_files = glob.glob(xfrac_dir + file_type)	
	xfrac_zs = None
	xfrac_zs = lightcone._get_file_redshifts(xfrac_zs, xfrac_files)
	return np.sort(xfrac_zs)

			
def coeval_21cm(xfrac_dir, dens_dir, z, interpolation='linear', mean_subtract=False):
	"""
	xfrac_dir     : Give the path that contains the xfrac-files.
	dens_dir      : Give the path that contains the density-files.
	z	      : Redshift.
	interpolation : This is used when the coveal cube at that redshift is not available.
	"""
	xfrac = coeval_xfrac(xfrac_dir, z, interpolation=interpolation)
	dens  = coeval_dens(dens_dir, z, interpolation=interpolation)
	dt    = tm.calc_dt(xfrac, dens, z=z)
	if mean_subtract: return dt-dt.mean()
	else: return dt


def coeval_xfrac(xfrac_dir, z, interpolation='linear'):
	"""
	xfrac_dir     : Give the path that contains the xfrac-files.
	z	      : Redshift.
	interpolation : This is used when the coveal cube at that redshift is not available.
	"""
	if not interpolation in ['linear']: #, 'step', 'sigmoid', 'step_cell'
		raise ValueError('Unknown interpolation type: %s' % interpolation)
	xfrac_files = glob.glob(xfrac_dir + '/xfrac3d_*.bin')
	xfrac_zs = None
	xfrac_zs = lightcone._get_file_redshifts(xfrac_zs, xfrac_files)
	if z in xfrac_zs:
		xfrac = xfrac_file.XfracFile(xfrac_files[np.argwhere(z==xfrac_zs)]).xi
	else:
		z_l = xfrac_zs[xfrac_zs<z].max()
		z_h = xfrac_zs[xfrac_zs>z].min()
		xfrac_l = xfrac_file.XfracFile(xfrac_files[xfrac_zs[xfrac_zs<z].argmax()]).xi
		xfrac_h = xfrac_file.XfracFile(xfrac_files[xfrac_zs[xfrac_zs>z].argmin()]).xi
		xfrac = xfrac_h + (xfrac_l-xfrac_h)*(z-z_h)/(z_l-z_h)
		print("The xfrac cube has been interpolated using", interpolation, "interpolation.")
	return xfrac

def coeval_dens(dens_dir, z, interpolation='linear'):
	"""
	dens_dir      : Give the path that contains the density-files.
	z	      : Redshift.
	interpolation : This is used when the coveal cube at that redshift is not available.
	"""
	if not interpolation in ['linear']: #, 'step', 'sigmoid', 'step_cell'
		raise ValueError('Unknown interpolation type: %s' % interpolation)
	dens_files  = glob.glob(dens_dir + '/*n_all.dat')
	dens_zs  = None
	dens_zs  = lightcone._get_file_redshifts(dens_zs, dens_files)
	if z in dens_zs:
		dens, dtype = helper_functions.get_data_and_type(dens_files[np.argwhere(z==dens_zs)])
	else:
		z_l = dens_zs[dens_zs<z].max()
		z_h = dens_zs[dens_zs>z].min()
		dens_l, dtype = helper_functions.get_data_and_type(dens_files[dens_zs[dens_zs<z].argmax()])
		dens_h, dtype = helper_functions.get_data_and_type(dens_files[dens_zs[dens_zs>z].argmin()])
		dens = dens_h + (dens_l-dens_h)*(z-z_h)/(z_l-z_h)
		print("The density cube has been interpolated using", interpolation, "interpolation.")
	return dens.astype(np.float64)

def coeval_overdens(dens_dir, z, interpolation='linear'):
	"""
	dens_dir      : Give the path that contains the density-files.
	z	      : Redshift.
	interpolation : This is used when the coveal cube at that redshift is not available.
	"""
	dens     = coeval_dens(dens_dir, z, interpolation=interpolation)
	overdens = dens/dens.mean(dtype=np.float64) - 1. 
	return overdens

def coeval_vel(dens_dir, vel_dir, z, interpolation='linear'):
	"""
	vel_dir       : Give the path that contains the velocity-files.
	z	      : Redshift.
	interpolation : This is used when the coveal cube at that redshift is not available.
	"""
	if not interpolation in ['linear']: #, 'step', 'sigmoid', 'step_cell'
		raise ValueError('Unknown interpolation type: %s' % interpolation)
	dens_files = glob.glob(dens_dir + '/*n_all.dat')
	vel_files  = glob.glob(vel_dir + '/*v_all.dat')
	vel_zs     = None
	vel_zs     = lightcone._get_file_redshifts(vel_zs, vel_files)
	def get_vel(vel_file, dens_file):
		dfile = density_file.DensityFile(dens_file)
		vel_file = vel_file.VelocityFile(vel_file)
		vel = vel_file.get_kms_from_density(dfile)
		return vel
	if z in vel_zs:
		vel = get_vel(vel_files[np.argwhere(z==vel_zs)],dens_files[np.argwhere(z==vel_zs)])
	else:
		z_l = vel_zs[vel_zs<z].max()
		z_h = vel_zs[vel_zs>z].min()
		vel_l = get_vel(vel_files[vel_zs[vel_zs<z].argmax()],dens_files[vel_zs[vel_zs<z].argmax()])
		vel_h = get_vel(vel_files[vel_zs[vel_zs>z].argmin()],dens_files[vel_zs[vel_zs>z].argmin()])
		vel = vel_h + (vel_l-vel_h)*(z-z_h)/(z_l-z_h)
		print("The velocity cube has been interpolated using", interpolation, "interpolation.")
	return vel

