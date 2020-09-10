import numpy as np
from glob import glob
from . import const
from . import conv
from .helper_functions import print_msg

class NbodyParticle:
	'''
	A simple struct to hold info about a single halo
	'''
	def __init__(self):
		self.pos = [0.0, 0.0, 0.0]		# Position in grid points
		self.vel = [0.0, 0.0, 0.0]		# Velocity in simulation units
		self.pid = 0					# Particle ID


class NbodyFile:
	'''
	A CubeP3M Nbody file.
	
	Use the read_from_file method to load a Nbody file, or 
	pass the filespath to the constructor.
	
	Some useful attributes of this class are:
	
	nrpart (int): total number of particles
	a (float): the scale factor of the file
	z (float): the redshift of the file
	'''
	
	def __init__(self, filespath=None, z=None, node=None, alt_format=False, pid_file=True):
		'''
		Initialize the file. If filespath is given, read data. Otherwise,
		do nothing.
		
		Parameters:
			filespath = None (string): the path to the nodes directories containing the xv.dat files.
			z = None (float) : redshift value
			node = None (float) : if specified will return only the output of the specified node
			alt_format = False (bool): whether to use an alternative-style file format.

		Returns:
			Nothing
		'''
		self.nbody = []
		self.filespath = filespath
		self.z = z
		self.node = node
		self.alt_format = alt_format
		self.pid_file = pid_file

		if z == None:
			raise NameError('Redshift value not specified, please define.')

		if self.filespath:
			self.filespath += '/' if self.filespath[-1] != '/' else ''
			self.npart = self.get_npart()
			if node != None:
				self.read_from_file(node=self.node)
		else:
			raise NameError('Files path not specified, please define.')


	def _get_header(self, f):
		'''
		Read header for xv.dat and PID.dat 
		'''

		np_local_xv = np.fromfile(f, count=1, dtype='int32')[0]

		self.a, t, tau  = np.fromfile(f, count=3, dtype='float32')
		assert round(1./self.a-1, 2) == round(self.z, 2)

		nts = np.fromfile(f, count=1, dtype='int32')[0]
		dt_f_acc, dt_pp_acc, dt_c_acc  = np.fromfile(f, count=3, dtype='float32')
		cur_checkpoint, cur_proj, cur_halo = np.fromfile(f, count=3, dtype='int32')
		mass_p = np.fromfile(f, count=1, dtype='float32')[0]

		return np_local_xv


	def get_npart(self):
		filesname_xv = ['%snode%d/%.3fxv%d.dat' %(self.filespath, i, self.z, i) for i in range(len(glob(self.filespath+'node*')))]
		
		npart = 0
		for fn_xv in filesname_xv:
			fxv = open(fn_xv, 'rb')
			np_local_xv = self._get_header(fxv)
			npart += np_local_xv
			fxv.close()

		return npart


	def read_from_file(self, node=None):
		'''
		Read data from file.
		'''
		if node != None:
			self.node = node
		else:
			self.node = None

		# if else statement to read halo file for one redshift and one node or all togheter
		if(self.node == None):
			print('Reading file at z=%.3f for all nodes...' %self.z)
			filesname_xv = ['%snode%d/%.3fxv%d.dat' %(self.filespath, i, self.z, i) for i in range(len(glob(self.filespath+'node*')))]
			filesname_pid  = ['%snode%d/%.3fPID%d.dat' %(self.filespath, i, self.z, i) for i in range(len(glob(self.filespath+'node*')))]
		else:
			print('Reading file %.3fxv%d.dat...' %(self.z, self.node))
			filesname_xv = ['%snode%d/%.3fxv%d.dat' %(self.filespath, self.node, self.z, self.node)]
			filesname_pid = ['%snode%d/%.3fPID%d.dat' %(self.filespath, self.node, self.z, self.node)]

		npart_xv = 0
		npart_pid = 0
		for fn_xv, fn_pid in zip(filesname_xv, filesname_pid):
			fxv = open(fn_xv, 'rb')
			np_local_xv = self._get_header(fxv)
			npart_xv += np_local_xv
			pos_node = np.fromfile(fxv, count=np_local_xv*6, dtype='float32').reshape((np_local_xv,6), order='F')
			fxv.close()

			if(self.pid_file):
				fpid = open(fn_pid, 'rb')
				np_local_pid = self._get_header(fpid)
				npart_pid += np_local_pid

				# read PID file and return pid array
				pid_node = np.fromfile(fpid, count=np_local_pid, dtype='int64')
				fpid.close()
			
			if(self.pid_file): assert np_local_pid == np_local_xv
			
			for i in range(0,np_local_xv):
				nbody = NbodyParticle()
				
				nbody.pos = pos_node[i,:3] #* conv.LB / float(conv.nbox_fine)		# in Mpc
				nbody.vel = pos_node[i,3:] #* conv.velconvert(self.z)				# km/s
				
				if(self.pid_file): nbody.pid = pid_node[i]
				self.nbody.append(nbody)

		if(self.pid_file): assert npart_pid == npart_xv
		self.npart = npart_xv

		print_msg( '...done')


