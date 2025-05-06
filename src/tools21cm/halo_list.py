import numpy as np
from glob import glob
from time import time
from tqdm import tqdm
from .helper_functions import print_msg
from . import const
from . import conv

class Halo:
	'''
	A simple struct to hold info about a single halo
	'''
	def __init__(self):
		self.pos    = [0.0, 0.0, 0.0]	# Position in grid points
		self.pos_cm = [0.0, 0.0, 0.0]	# Center of mass position in grid points
		self.vel = [0.0, 0.0, 0.0]		# Velocity in simulation units
		self.l   = [0.0, 0.0, 0.0]		# Angular momentum in simulation units
		self.vel_disp = 0.0				# Velocity dispersion in simulation units
		self.r  = 0.0					# Virial radius in grid units
		self.m  = 0.0					# Grid mass
		self.mp = 0						# Number of particles


class HaloRockstar:
	'''
	A class that holds information about a large number of halos, as read from Rockstar halo list file.
	Contains methods to select halos based on different criteria. This file is very slow if you need to read a large number of halos.
	'''
	def __init__(self, filename=None, mass_def='vir', max_select_number=-1, startline = 0):
		'''
		Initialize the object. If filename is given, read the file. Otherwise, do nothing.
		'''
		self.halos = []
		self.mass_def = mass_def

		if filename:
			self.read_from_file(filename, max_select_number)

	def read_from_file(self, filename, max_select_number=-1):
		'''
		Read a Rockstar halo list.
		
		Parameters:
			filename (string): The file to read from
			max_select_number = -1 (int): The max number of halos to read. If -1, there is no limit.
		Returns:
			True if all the halos were read. False otherwise.
		'''

		print_msg('Reading Rockstar Halo Catalog %s...' % filename)
		self.filename = filename
		
		import fileinput
		from astropy import units as U

		#Read the file line by line, since it's large
		for linenumber, line in tqdm(enumerate(fileinput.input(filename))):
			if(linenumber == 0):
				# Store the variable from the file header
				header = line.split()
				if(self.filename[self.filename.rfind('.')+1:] == 'list'):
					idx_pos = header.index('X')
					idx_vel = header.index('VX')
					idx_l = header.index('JX')
					idx_vrms = header.index('Vrms')
					idx_r = header.index('R'+self.mass_def)
					idx_m = header.index('M'+self.mass_def)
					lnr_cosm = 2
					lnr_part = 5
					lnr_vals = 15
				elif(self.filename[self.filename.rfind('.')+1:] == 'ascii'):
					idx_pos = header.index('x')
					idx_vel = header.index('vx')
					idx_l = header.index('Jx')
					idx_vrms = header.index('vrms')
					idx_r = header.index('r'+self.mass_def)
					idx_m = header.index('m'+self.mass_def)
					lnr_cosm = 3
					lnr_part = 6
					lnr_vals = 19
				else:
					ValueError('ERROR: wrong file format (must be .list or .ascii).')
			elif(linenumber == 1):
				# Store the redshift from the file header
				a = float(line.split()[-1])
				self.z = 1./a - 1.
			elif(linenumber == lnr_cosm):
				# Store cosmology quanity from the file header
				cosm = line.split()
				self.Om, self.Ol, self.h = float(cosm[2][:-1]), float(cosm[5][:-1]), float(cosm[-1])
			elif(linenumber == lnr_part):
				# Store particle mass from the file header
				self.part_mass = float(line.split()[2]) #* U.Msun/self.h
			elif(linenumber > lnr_vals):
				vals = line.split()
				
				#Create a halo and add it to the list
				if(len(self.halos) > max_select_number):
					halo = Halo()
					halo.pos = np.array(vals[idx_pos:idx_pos+3]).astype(float) #* U.Mpc/self.h
					halo.vel = np.array(vals[idx_vel:idx_vel+3]).astype(float) #* U.km/U.s
					halo.pos_cm = halo.pos
					halo.l = np.array(vals[idx_l:idx_l+3]).astype(float) #* U.Msun/self.h*U.Mpc/self.h*U.km/U.s
					halo.vel_disp = float(vals[idx_vrms]) #*U.km/U.s
					halo.r = float(vals[idx_r]) #* U.kpc/self.h
					halo.m = float(vals[idx_m]) #* U.Msun/self.h
					halo.mp = int(round(halo.m / self.part_mass, 0))
					self.halos.append(halo)
				else:
					break
		
		fileinput.close()
		return True
	
	def get(self, var=None):
		if(var == 'm'):
			data = np.array([halo.m for halo in tqdm(self.halos)])
		elif(var == 'r'):
			data = np.array([halo.r for halo in tqdm(self.halos)])
		elif(var == 'pos'):
			data = np.array([halo.pos for halo in tqdm(self.halos)])
		elif(var == 'vel'):
			data = np.array([halo.vel for halo in tqdm(self.halos)])
		elif(var == 'vel_disp'):
			data = np.array([halo.vel_disp for halo in tqdm(self.halos)])
		elif(var == 'l'):
			data = np.array([halo.l for halo in tqdm(self.halos)])
		elif(var == 'pos_cm'):
			data = np.array([halo.pos_cm for halo in tqdm(self.halos)])
		return data


class HaloCubeP3MFull:
	'''
	Adapted from HaloList
	A class that holds information about a large number of halos, as read from a 
	cube3pm halo list file - the full catalogue.
	Contains methods to select halos based on different criteria. This file is very slow
	if you need to read a large number of halos.
	'''
	def __init__(self, filename=None, box_len=None, output_len=None, min_select_mass=0.0, max_select_mass=None, max_select_number=-1, startline = 0):
		'''
		Initialize the object. If filename is given, read the file. Otherwise,
		do nothing.
		
		Parameters:
				* filename = None (string): The file to read from
				* box_len = None (float) : The size of the box in cMpc/h units
				* output_len = None (int or float) : This can be either the mesh grid size (for unit in grid) or the box size in cMpc/h units
				* min_select_mass = 0.0 (float): The lower threshold mass in solar masses.
						Only halos above this mass will be read.
				* max_select_mass = None (float): The upper threshold mass in solar masses.
						Only halos below this mass will be read. If None, there is no limit.
				* max_select_number = -1 (int): The max number of halos to read. If -1, there
						is no limit.
				* startline = 0 (int): The line in the file where reading will start.
		Returns:
				Nothing
		'''
		self.halos = []

		if box_len == None:
			ValueError('need to define the refined number of particles from the N-body simulation: n_box')
		else:
			conv.set_sim_constants(boxsize_cMpc=box_len)
			self.box_len = float(box_len)
			self.nbox_fine = conv.nbox_fine 

		if output_len != None:
			self.box_len = float(output_len)

		if filename:
			self.z = float(filename[filename.rfind('/')+1:filename.rfind('halo')])
			self.read_from_file(filename, min_select_mass, max_select_mass, max_select_number, startline)

	def read_from_file(self,filename, min_select_mass = 0.0, max_select_mass = None, max_select_number=-1, startline=0):
		'''
		Read a halo list.
		
		Parameters:
				* filename (string): The file to read from
				* min_select_mass = 0.0 (float): The lower threshold mass in solar masses.
						Only halos above this mass will be read.
				* max_select_mass = None (float): The upper threshold mass in solar masses.
						Only halos below this mass will be read. If None, there is no limit.
				* max_select_number = -1 (int): The max number of halos to read. If -1, there
						is no limit.
				* startline = 0 (int): The line in the file where reading will start.
		Returns:
				True if all the halos were read. False otherwise.
		'''

		self.halos = []
		# self.halo_pos = []

		print_msg('Reading halo file %s...' % filename)
		self.filename = filename
		import fileinput
		#Store the redshift from the filename
		import os.path
		name = os.path.split(filename)[1]
		self.z = float(name.split('halo')[0])

		# Read the file line by line, since it's large
		linenumber = 1
		skipped_lines = 0
		tstart = time()
		
		min_select_grid_mass = min_select_mass/(conv.M_grid*const.solar_masses_per_gram)
		if max_select_mass:
				print_msg('Max_select_mass: %g' % max_select_mass)
				max_select_grid_mass = max_select_mass/(conv.M_grid*const.solar_masses_per_gram)

		for line in tqdm(fileinput.input(filename)):
				if linenumber < startline: #If you want to read from a particular line
						linenumber += 1
						continue
				if max_select_number >= 0 and len(self.halos) >= max_select_number:
						fileinput.close()
						return False
				if linenumber % 100000 == 0:
						print_msg(f'{linenumber} lines read in {(time()-tstart):.2f} s' )
				linenumber += 1
				# print('LINE:', line)
				vals = line.split()
				if len(vals)==1:
						skipped_lines += 1
						# print('skipping this line')
						continue
				# print('VALS:', vals)
				grid_mass = float(vals[-3])

				# Create a halo and add it to the list
				if grid_mass > min_select_grid_mass and (max_select_mass == None or grid_mass < max_select_grid_mass):
						halo = Halo()
						# The following lines used the map() function to convert
						# parts of the vals list into floats before putting them
						# into an array. In Python 3 map() returns an iterable,
						# not a list, so changed this to a list operation.
						# GM/200601

						halo.pos    = self._reposition(np.array([float(i) for i in vals[:3]]) * self.box_len / self.nbox_fine) # same units as box_len
						halo.pos_cm = self._reposition(np.array([float(i) for i in vals[3:6]]) * self.box_len / self.nbox_fine) # same units as box_len
						halo.vel = conv.gridvel_to_kms(gridvel=np.array([float(i) for i in vals[6:9]]), z=self.z)	# km/s
						halo.l   = 10**(conv.gridmass_to_msol(conv.gridpos_to_mpc(conv.gridvel_to_kms(gridvel=np.array([float(i) for i in vals[9:12]]), z=self.z)))) * const.h * const.h # Msun/h * Mpc/h * km/s
						halo.vel_disp = conv.gridvel_to_kms(gridvel=np.array([float(i) for i in vals[12:15]]), z=self.z)	# km/s
						halo.m  = 10**(conv.gridmass_to_msol(float(vals[16]))) * const.h # Msun/h
						halo.mp = float(vals[17])
						halo_mass_1 = float(vals[18])
						var_x = np.array([float(i) for i in vals[19:22]])
						halo.solar_masses = grid_mass*conv.M_grid*const.solar_masses_per_gram
						self.halos.append(halo)

				# if linenumber==50:
				#         break
		fileinput.close()
			
		print(f'Total number of lines read: {linenumber-1}')
		print(f'Number of lines skipped   : {skipped_lines}')
		print(f'Total time elasped        : {(time()-tstart):.2f} s')
		return True

	def get(self, var=None):
		if(var == 'm'):
			data = np.array([halo.m for halo in tqdm(self.halos)])
		elif(var == 'pos'):
			data = np.array([halo.pos for halo in tqdm(self.halos)])
		elif(var == 'vel'):
			data = np.array([halo.vel for halo in tqdm(self.halos)])
		elif(var == 'vel_disp'):
			data = np.array([halo.vel_disp for halo in tqdm(self.halos)])
		elif(var == 'l'):
			data = np.array([halo.l for halo in tqdm(self.halos)])
		elif(var == 'pos_cm'):
			data = np.array([halo.pos_cm for halo in tqdm(self.halos)])
		return data

	def _reposition(self, pos):
		for i, val in enumerate(pos):
			if(val < 0):
				pos[i] = self.box_len + val
			elif(val > self.box_len):
				pos[i] = val - self.box_len
		return pos

class HaloCubeP3M:
	'''
	A CubeP3M Halo cataloge files have the following structure:

		Column 1-3:		hpos(:) (halo position (cells))
		Column 4,5:		mass_vir, mass_odc (mass calculated on the grid (in grid masses))
		Column 6,7:		r_vir, r_odc (halo radius, virial and overdensity based)
		Column 8-11:	x_mean(:) (centre of mass position)
		Column 11-14:	v_mean(:) (bulk velocity)
		Column 15-18:	l_CM(:) (angular momentum)
		Column 19-21:	v2_wrt_halo(:) (velocity dispersion)
		Column 21-23:	var_x(:) (shape-related quantity(?))
		Column 17 :		pid_halo
	
	Some useful attributes of this class are:

		nhalo (int)	: total number of haloes
		z (float)	: the redshift of the file
		a (float)	: the scale factor of the file
		mass_def (string): either 'vir' or 'odc' for viral or overdensity shperical cutoff
		pid_flag (bool): to indicate if particle IDs are stored in xv files (50 8-byte int + 50*6 4-byte float)
		cosm_unit (bool): condition to convert from grid units to cosmological (comoving) units (i.e: Mpc/h, Msun/h, etc)
	'''
	
	def __init__(self, filespath=None, z=None, node=None, mass_def='vir', pid_flag=True, cosm_unit=True):
		'''
		Initialize the file. If filespath is given, read data. Otherwise, do nothing.
		'''

		self.halos = []
		if not z:
			raise NameError('Redshift value not specified, please define.')

		if filespath:
			filespath += '/' if filespath[-1] != '/' else ''
			self.read_from_file(filespath, z, node, mass_def, pid_flag, cosm_unit)
		else:
			raise NameError('Files path not specified, please define.')

	def _get_header(self, file):
		# Internal use. Read header for xv.dat and PID.dat
		nhalo = np.fromfile(file, count=1, dtype='int32')[0]
		halo_vir, halo_odc = np.fromfile(file, count=2, dtype='float32')
		return nhalo, halo_vir, halo_odc


	def _reposition(self, pos):
		Lbox = conv.boxsize
		new_pos = pos.copy()

		for i, val in enumerate(pos):
			if(val < 0):
				new_pos[i] = Lbox + val
			elif(val > Lbox):
				new_pos[i] = val - Lbox
		
		return new_pos
	

	def read_from_file(self, filespath, z, node, mass_def, pid_flag, cosm_unit):
		'''
		Read Cube3PM halo catalog from file.
		
		Parameters:
			filespath (string): the path to the nodes directories containing the xv.dat files.
			z = None (float) : redshift value.
			node = None (float) : if specified will return only the output of the specified node
			mass_def = 'vir' (string) : the mass devinition used, can be 'vir' (viral mass) or 'odc' (overdensity)
			pid_flag = True (bool): whether to use the PID-style file format.

		Returns:
			Nothing
		'''

		self.filespath = filespath
		self.z = z

		# if else statement to read halo file for one redshift and one node or all togheter
		if(node == None):
			print_msg('Reading Cube3PM Halo Catalog from all nodes...')
			filesname = ['%snode%d/%.3fhalo%d.dat' %(filespath, i, self.z, i) for i in range(len(glob(filespath+'node*')))]
		else:
			print_msg('Reading Cube3PM Halo Catalog from node = %d...' %node)
			filesname = ['%snode%d/%.3fhalo%d.dat' %(filespath, node, self.z, node)]

		self.nhalo = 0

		for fn in filesname:
			f = open(fn, 'rb')
			nhalo_node, halo_vir, halo_odc = self._get_header(f)				
			self.nhalo += nhalo_node
			
			for i in range(nhalo_node):
				halo = Halo()
				halo.pos = np.fromfile(f, count=3, dtype='float32')		# grid length
				mass_vir, mass_odc, r_vir, r_odc = np.fromfile(f, count=4, dtype='float32')
				
				if(mass_def == 'vir'):
					halo.m = mass_vir # Msun/h
					halo.r = r_vir # Mpc/h
				else:
					halo.m = mass_odc # Msun/h
					halo.r = r_odc # Mpc/h
				
				halo.mp = 0		# TODO: DOUBLE CHEKC THIS QUANITTY
				halo.pos_cm = np.fromfile(f, count=3, dtype='float32')
				halo.vel = np.fromfile(f, count=3, dtype='float32')
				halo.l = np.fromfile(f, count=3, dtype='float32')
				halo.vel_disp = np.linalg.norm(np.fromfile(f, count=3, dtype='float32'))
				var_x = np.fromfile(f, count=3, dtype='float32')	# shape-related quantity(?)
				
				if(pid_flag):
					pid_halo_node = np.fromfile(f, count=50, dtype='int64')
					xv_halo_node = np.fromfile(f, count=50*6, dtype='float32').reshape((50,6), order='C')
				else:
					_ = np.fromfile(f, count=6, dtype='float32')

				if(cosm_unit):
					halo.pos = self._reposition(pos=conv.gridpos_to_mpc(halo.pos) * const.h)	# Mpc/h
					halo.m = 10**(conv.gridmass_to_msol(halo.m)) * const.h 		# Msun/h
					halo.r = conv.gridpos_to_mpc(halo.r) * const.h 				# Mpc/h
					halo.pos_cm = self._reposition(pos=conv.gridpos_to_mpc(halo.pos_cm) * const.h)	# Mpc/h
					halo.vel = conv.gridvel_to_kms(gridvel=halo.vel, z=self.z)	# km/s
					halo.l = 10**(conv.gridmass_to_msol(conv.gridpos_to_mpc(conv.gridvel_to_kms(gridvel=halo.l, z=self.z)))) * const.h * const.h # Msun/h * Mpc/h * km/s
					halo.vel_disp = conv.gridvel_to_kms(gridvel=halo.vel_disp, z=self.z)	# km/s

				self.halos.append(halo)
		return True

	def get(self, var=None):
		if(var == 'm'):
			data = np.array([halo.m for halo in tqdm(self.halos)])
		elif(var == 'r'):
			data = np.array([halo.r for halo in tqdm(self.halos)])
		elif(var == 'pos'):
			data = np.array([halo.pos for halo in tqdm(self.halos)])
		elif(var == 'vel'):
			data = np.array([halo.vel for halo in tqdm(self.halos)])
		elif(var == 'vel_disp'):
			data = np.array([halo.vel_disp for halo in tqdm(self.halos)])
		elif(var == 'l'):
			data = np.array([halo.l for halo in tqdm(self.halos)])
		elif(var == 'pos_cm'):
			data = np.array([halo.pos_cm for halo in tqdm(self.halos)])
		return data




class HaloList:
	'''
	A class that holds information about a large number of halos, as read from a 
	halo list file.
	Contains methods to select halos based on different criteria. This file is very slow
	if you need to read a large number of halos.
	
	TODO: write a better implementation of this class.
	'''
	def __init__(self, filename=None, min_select_mass = 0.0, max_select_mass = None, 
			max_select_number=-1, startline = 0):
		'''
		Initialize the object. If filename is given, read the file. Otherwise,
		do nothing.
		
		Parameters:
			* filename = None (string): The file to read from
			* min_select_mass = 0.0 (float): The lower threshold mass in solar masses.
				Only halos above this mass will be read.
			* max_select_mass = None (float): The upper threshold mass in solar masses.
				Only halos below this mass will be read. If None, there is no limit.
			* max_select_number = -1 (int): The max number of halos to read. If -1, there
				is no limit.
			* startline = 0 (int): The line in the file where reading will start.
		Returns:
			Nothing
		'''
		self.halos = []

		if filename:
			self.read_from_file(filename, min_select_mass, max_select_mass, max_select_number, 
					startline)

	def read_from_file(self,filename, min_select_mass = 0.0, max_select_mass = None, max_select_number=-1, 
			startline=0):
		'''
		Read a halo list.
		
		Parameters:
			* filename (string): The file to read from
			* min_select_mass = 0.0 (float): The lower threshold mass in solar masses.
				Only halos above this mass will be read.
			* max_select_mass = None (float): The upper threshold mass in solar masses.
				Only halos below this mass will be read. If None, there is no limit.
			* max_select_number = -1 (int): The max number of halos to read. If -1, there
				is no limit.
			* startline = 0 (int): The line in the file where reading will start.
		Returns:
			True if all the halos were read. False otherwise.
		'''

		self.halos = []

		print_msg('Reading halo file %s...' % filename)
		self.filename = filename
		import fileinput

		#Store the redshift from the filename
		import os.path
		name = os.path.split(filename)[1]
		self.z = float(name.split('halo')[0])

		#Read the file line by line, since it's large
		linenumber = 1
		min_select_grid_mass = min_select_mass/(conv.M_grid*const.solar_masses_per_gram)
		if max_select_mass:
			print_msg('Max_select_mass: %g' % max_select_mass)
			max_select_grid_mass = max_select_mass/(conv.M_grid*const.solar_masses_per_gram)

		for line in tqdm(fileinput.input(filename)):
			if linenumber < startline: #If you want to read from a particular line
				linenumber += 1
				continue
			if max_select_number >= 0 and len(self.halos) >= max_select_number:
				fileinput.close()
				return False
			if linenumber % 100000 == 0:
				print_msg('Read %d lines' % linenumber)
			linenumber += 1

			vals = line.split()			
			grid_mass = float(vals[-3])

			#Create a halo and add it to the list
			if grid_mass > min_select_grid_mass and (max_select_mass == None or grid_mass < max_select_grid_mass):
				halo = Halo()
                                # The following lines used the map() function to convert
                                # parts of the vals list into floats before putting them
                                # into an array. In Python 3 map() returns an iterable,
                                # not a list, so changed this to a list operation.
                                # GM/200601
				halo.pos = np.array([float(i) for i in vals[:3]])
				halo.pos_cm = np.array([float(i) for i in vals[3:6]])
				halo.vel = np.array([float(i) for i in vals[6:9]])
				halo.l = np.array([float(i) for i in vals[9:12]])
				halo.vel_disp = np.array([float(i) for i in vals[12:15]])
				halo.m = float(vals[16])
				halo.mp = float(vals[17])
				halo_mass_1 = float(vals[18])
				var_x = np.array([float(i) for i in vals[19:22]])
				halo.solar_masses = grid_mass*conv.M_grid*const.solar_masses_per_gram
				self.halos.append(halo)

		fileinput.close()

		return True
