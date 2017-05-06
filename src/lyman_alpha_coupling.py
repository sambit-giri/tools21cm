import numpy as np
import c2raytools as c2t
from scipy.interpolate import interp1d, interp2d
from usefuls import *
import os
import time, timeit

lam_lya   = 1215.                #Ang
lam_HII   = 912.                 #Ang
f_esc_lya = 1			 #Assumed
dlam      = 1                    #Ang
h_planck  = 6.626e-34            #m^2 kg / s
c_light   = 3e8                  #m/s
nu_lya    = c_light/1215.67e-10  #Hz
E_lya     = 10.19884             #eV

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
	zss = c2t.cdist_to_z(rr+c2t.z_to_cdist(z))
	rp2 = rr2/(1.0+zss)**2
	lms = lam_lya*(1+z)/(1+zss)
	eng = sed_func(lms)*mass*dlam/(4*np.pi*rp2)
	eng[lms<=lam_HII] = 0.
	n_a = eng/h_planck/nu_lya
	stop  = timeit.default_timer()
	print stop-start
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
	zs  = c2t.cdist_to_z(rr+c2t.z_to_cdist(z))
	zs_HII = zs[lms<=lam_HII].min()
	lms = lam_lya*(1+z)/(1+zs)
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
	zss = c2t.cdist_to_z(rr+c2t.z_to_cdist(z))
	eng = eng_func(zss)
	eng[zss>=zs_HII] = 0.
	n_a = eng*mass/E_lya
	stop  = timeit.default_timer()
	print stop-start
	return n_a




	
	
