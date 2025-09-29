import numpy as np
import sys
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from scipy.interpolate import interp1d
#import c2raytools as c2t
from . import cosmo as cm

lam_lya   = 1215.                #Ang
lam_HII   = 912.                 #Ang
f_esc_lya = 1			 #Assumed
dlam      = 1                    #Ang
h_planck  = 6.626e-34            #m^2 kg / s
c_light   = 3e8                  #m/s
nu_lya    = c_light/1215.67e-10  #Hz

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ncells  = float(sys.argv[1])
boxsize = float(sys.argv[2])
sourcelist_loc = sys.argv[3]
SED_loc = sys.argv[4]
z       = float(sys.argv[5])
flname  = sys.argv[6]

src      = np.loadtxt(sourcelist_loc, skiprows=1)
n_src    = src.shape[0]
lam, lum = np.loadtxt(SED_loc).T
sed_func = interp1d(lam, lum, kind='cubic')

def one_source(ncells, boxsize, source_pos, mass, sed_func):
	i,j,k    = source_pos
	xx,yy,zz = np.mgrid[0:ncells,0:ncells,0:ncells]
	rr2 = ((xx-i)**2 + (yy-j)**2 + (zz-k)**2)*boxsize**2/ncells**2
	rr  = np.sqrt(rr2)
	zss = cm.cdist_to_z(rr)
	lms = lam_lya/(1+zss)
	eng = sed_func(lms)*mass*dlam/(4*np.pi*rr2)
	eng[lms<=lam_HII] = 0
	n_a = eng/h_planck/nu_lya
	return n_a

local_n = n_src/size
count   = 0
xc_cube = np.zeros((ncells, ncells, ncells))
while(count < local_n):
	s       = local_n*rank+count
	count  += 1
	source_pos, mass = src[s,:-1]-np.ones(3), src[s,-1]
	na_cube  = one_source(ncells, boxsize, source_pos, mass, sed_func)
	xc_cube += na_cube
	if rank+1 == size:
		#print s+1, "sources or", size*count*100/n_src, "% done"
		print('%d sources or %.1f \% done'%(s+1,size*count*100/n_src))

remain = n_src%size
if rank==1 and remain!=0:
	cc = 0
	while(cc < n_src%size):
		cc += 1
		source_pos, mass = src[-cc,:-1]-np.ones(3), src[s,-1]
		na_cube  = one_source(ncells, boxsize, source_pos, mass, sed_func)
		xc_cube += na_cube
		#print n_src-remain+cc, "sources or 100 % done"
		print('sources or 100 \% done'%(n_src-remain+cc))
	
if rank == 0:
	for i in range(1, size):
		recv_buffer = np.empty(local_n, dtype=np.int)
		comm.Recv(recv_buffer, ANY_SOURCE)
		xc_cube += recv_buffer
else:
	# all other process send their result
	comm.Send(np.array(xc_cube))

if comm.rank == 0:
	np.save(flname, xc_cube)

	
