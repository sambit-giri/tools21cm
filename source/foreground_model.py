import numpy as np
import c2raytools as c2t
from telescope_functions import jansky_2_kelvin, from_antenna_config

def galactic_synch_fg(z, ncells, tele_file):
	X  = np.random.normal(size=(ncells, ncells))
	Y  = np.random.normal(size=(ncells, ncells))
	nu = c2t.z_to_nu(z)
	nu_s,A150,beta_,a_syn,Da_syn = 150,513,2.34,2.8,0.1
	Nbase, N_ant = from_antenna_config(filename, z)
	C_syn = A150*(1000/l)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
	return 0

def extragalactic_pointsource_fg(z, ncells, boxsize, S_max=100):
	nu = c2t.z_to_nu(z)
	fg = np.zeros((ncells,ncells))
	dS = 0.01
	Ss = np.linspace(0.1, S_max, dS)
	solid_angle = boxsize**2/c2t.z_to_cdist(z)**2
	N  = int(10**3.75*np.trapz(Ss**(-1.6), x=Ss, dx=dS)*solid_angle)
	x,y = np.random.random_integers(0, high=ncells, size=(2,N))
	alpha_ps = 0.7+0.1*np.random.random(size=N)
	nu_s, S_s = 150, S_max
	S_nu = S_s*(nu/nu_s)**(-alpha_ps)
	for i,j,a in x,y,S_nu: fg[i,j] = a
	return jansky_2_kelvin(fg, z, boxsize=boxsize, ncells=ncells)
	
	
	
	 
	
