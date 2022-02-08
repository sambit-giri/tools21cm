import numpy as np
from scipy.interpolate import interp1d, interp2d
from .usefuls import *
import os, glob
import time, timeit

lam_lya   = 1215.                #Ang
lam_HII   = 912.                 #Ang
f_esc_lya = 1			 #Assumed
dlam      = 1                    #Ang
h_planck  = 6.626e-34            #m^2 kg / s
c_light   = 3e8                  #m/s
nu_lya    = c_light/1215.67e-10  #Hz
E_lya     = 10.19884             #eV
m2parsec  = 3.24e-17
yr2sec    = 3.154e7

def _read_halofile(filename, z=-1):
	if filename.split('-')[-1] == 'coarsest_sources_used_wfgamma.dat': src = np.loadtxt(filename, skiprows=1)
	else: src = np.loadtxt(filename+'/'+str(z)+'coarsest_sources_used_wfgamma.dat', skiprows=1)
	if z<0: z = float(filename.split('-')[-2].split('/')[-1])
	return src, z

def Lya_coupling_coeff(ncells, boxsize, sourcelist, z=-1, SED=None, lum=None, lam=None):
	assert SED is not None or (lum is not None and lam is not None)
	if SED is not None and type(SED) == str: lum, lam = np.loadtxt(SED)
	elif SED is not None: lum, lam = SED
	if (lum is not None and lam is not None):
		if type(lum)==str: lum = np.loadtxt(lum)   # Energy/time/mass
		if type(lam)==str: lam = np.loadtxt(lam)   # angstrom
	sed_func = interp1d(lam, lum, kind='cubic')
	xc_cube  = np.zeros((ncells, ncells, ncells))
	src, z   = _read_halofile(sourcelist, z=z)
	n_src    = src.shape[0]
	for s in xrange(n_src):
		source_pos, mass = src[s,:-1]-np.ones(3), src[s,-1]
		na_cube  = one_source(ncells, boxsize, source_pos, mass, sed_func)
		xc_cube += na_cube
		loading_verbose(str((s+1)*100/n_src)+'%')
	return xc_cube
	

def one_source(z, ncells, boxsize, source_pos, mass, sed_func):
	start = timeit.default_timer()
	i,j,k    = source_pos
	xx,yy,zz = np.mgrid[0:ncells,0:ncells,0:ncells]
	rr2 = ((xx-i)**2 + (yy-j)**2 + (zz-k)**2)*boxsize**2/ncells**2
	rr  = np.sqrt(rr2)
	zss = cm.cdist_to_z(rr+cm.z_to_cdist(z))
	rp2 = rr2/(1.0+zss)**2
	lms = lam_lya*(1+z)/(1+zss)
	eng = sed_func(lms)*mass*dlam/(4*np.pi*rp2)
	eng[lms<=lam_HII] = 0.
	n_a = eng/h_planck/nu_lya
	stop  = timeit.default_timer()
	print(stop-start)
	return n_a

def Lya_coupling_coeff_MPI(ncells, boxsize, sourcelist_loc, z=-1, SED_loc=None, n_jobs=5, lum=None, lam=None):
	assert type(sourcelist_loc) == str
	assert SED_loc is not None or (lum is not None and lam is not None)
	if z<0: z = float(sourcelist_loc.split('-')[-2].split('/')[-1])
	if (lum is not None and lam is not None): 
		sed = np.vstack((lam,lum))
		dummy_SED_name = 'dummy_SED'+str(time.time())+'.dat'
		np.savetxt(dummy_SED_name, sed.T)
		SED_loc = dummy_SED_name
	flname   = 'dummy_xc_cube'+str(time.time())+'.npy'
	run_prog = 'mpiexec -n '+str(n_jobs)+' python apply_mpi_lya_c.py '+str(ncells)+' '+str(boxsize)+' '+str(sourcelist_loc)+' '+str(SED_loc)+' '+str(z)+' '+flname
	os.system(run_prog)
	if SED_loc == dummy_SED_name: os.remove(dummy_SED_name)
	xc_cube = np.load(flname)
	os.remove(flname)
	return xc_cube

def Lya_coupling_coeff_(ncells, boxsize, sourcelist, z=-1, SED=None, lum=None, lam=None):
	assert SED is not None or (lum is not None and lam is not None)
	if SED is not None and type(SED) == str: lum, lam = np.loadtxt(SED)
	elif SED is not None: lum, lam = SED
	if (lum is not None and lam is not None):
		if type(lum)==str: lum = np.loadtxt(lum)   # Energy/time/mass
		if type(lam)==str: lam = np.loadtxt(lam)   # angstrom
	sed_func = interp1d(lam, lum, kind='cubic')
	rr  = 10**np.linspace(-2,np.log10(boxsize*1.8),1000)
	zs  = cm.cdist_to_z(rr+cm.z_to_cdist(z))
	lms = lam_lya*(1+z)/(1+zs)
	zs_HII = zs[lms<=lam_HII].min()
	rp  = rr/(1.0+zs)
	eng = sed_func(lms)*dlam/(4*np.pi*rp**2)
	eng_func = interp1d(zs, eng, kind='cubic')
	src, z   = _read_halofile(sourcelist, z=z)
	n_src    = src.shape[0]
	masses   = Mgrid_2_Msolar(src[:,3])               # in solar mass
	xc_cube  = np.zeros((ncells, ncells, ncells))
	xx,yy,zz = np.mgrid[0:ncells,0:ncells,0:ncells]
	for s in xrange(n_src):
		source_pos = src[s,:-1]-np.ones(3)
		na_cube  = one_source(xx, yy, zz, ncells, boxsize, source_pos, masses[p], eng_func, zs_HII)
		xc_cube += na_cube
		loading_verbose(str((s+1)*100/n_src)+'%')
	return xc_cube

def one_source_(xx, yy, zz, z, ncells, boxsize, source_pos, mass, eng_func, zs_HII):
	start = timeit.default_timer()
	i,j,k = source_pos
	rr2 = ((xx-i)**2 + (yy-j)**2 + (zz-k)**2)*boxsize**2/ncells**2
	rr  = np.sqrt(rr2); rr[rr==0] = 0.1
	zss = cm.cdist_to_z(rr+cm.z_to_cdist(z))
	eng = eng_func(zss)
	eng[zss>=zs_HII] = 0.
	n_a = eng*mass/E_lya
	stop  = timeit.default_timer()
	print(stop-start)
	return n_a

def Lya_coupling_coeff_useprofile(z, ncells, boxsize, source_dir, source_file='*-coarsest_sources_used_wfgamma.dat', SED=None, lum=None, lam=None, tstep=11.5):
	assert SED is not None or (lum is not None and lam is not None)
	if SED is not None and type(SED) == str: lam, lum = np.loadtxt(SED).T
	elif SED is not None: lum, lam = SED
	if (lum is not None and lam is not None):
		if type(lum)==str: lum = np.loadtxt(lum)   # Energy/time/mass
		if type(lam)==str: lam = np.loadtxt(lam)   # angstrom
	sourcelists = np.array(glob.glob(source_dir+source_file))
	source_zs   = np.array([float(ss.split('/')[-1].split('-')[0]) for ss in sourcelists])
	if not(z in source_zs): 
		z = source_zs[np.abs(source_zs-z).argmin()]
		print("The nearest sourcelist in the given directory is of z =%.3f"%z)
	z_max   = z+(lam_lya/lam_HII - 1.)
	src_zs  = source_zs[source_zs>=z]
	src_zs  = src_zs[src_zs<=z_max]
	r_light = c_light*tstep*yr2sec*m2parsec
	for zi in src_zs:
		if zi==z: zl = -1
		else: zl = source_zs[source_zs<zi].max()
		engs    = _get_profile(z, lam, lum, boxsize, ncells, r_light*(1+zl), r_light*(1+zi))
		sources = np.loadtxt(sourcelists[source_zs==zi][0], skiprows=1)
		xc_cube = _get_xc(sources, engs, ncells)
	return xc_cube

def _get_xc(sources, engs, ncells, mass_axis=3, time_taken=False):
	start = timeit.default_timer()
	masses  = Mgrid_2_Msolar(sources[:,mass_axis])
	xc_cube = np.zeros((ncells, ncells, ncells))
	n_src   = sources.shape[0]
	engs   *= mass/E_lya
	for s in xrange(n_src):
		i,j,k = sources[s,:3]-np.ones(3)
		mass  = masses[s]
		eng_  = engs[(ncells-i):(2*ncells-i),(ncells-j):(2*ncells-j),(ncells-k):(2*ncells-k)]
		xc_cube += eng_
	stop  = timeit.default_timer()
	if time_taken: print("%d seconds"%(stop-start))
	return xc_cube
	

def _get_profile(z, lam, lum, boxsize, ncells, r_min, r_max):
	sed_func = interp1d(lam, lum, kind='cubic')
	rs = 10**np.linspace(-2,np.log10(boxsize*1.8),1000)
	zs  = cm.cdist_to_z(rs+cm.z_to_cdist(z))
	lms = lam_lya*(1+z)/(1+zs)
	rp  = rs/(1.0+zs)
	rs_HII = rs[lms<=lam_HII].min()
	eng = sed_func(lms)*dlam/(4*np.pi*rp**2)
	xx,yy,zz = np.mgrid[-ncells:ncells,-ncells:ncells,-ncells:ncells]
	rr2 = (xx**2 + yy**2 + zz**2)*boxsize**2/ncells**2
	rr  = np.sqrt(rr2); rr[rr==0] = 0.1
	eng_func = interp1d(rs, eng, kind='cubic')
	engs     = eng_func(rr)
	engs[rr>=rs_HII] = 0.
	engs[rr<=r_min]  = 0.
	engs[rr>=r_max]  = 0.
	eng0 = np.roll(engs, ncells, axis=0)
	eng1 = np.roll(engs, ncells, axis=1)
	eng2 = np.roll(engs, ncells, axis=2)
	engs = engs+eng0+eng1+eng2
	return engs


	
	
