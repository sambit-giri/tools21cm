import numpy as np
from . import const
from . import conv
from .helper_functions import print_msg

class DensityFile:
	'''
	A CubeP3M density file.

	Use the read_from_file method to load a density file, or 
	pass the filename to the constructor.

	Attributes:
		raw_density (numpy array): the density in simulation units
		cgs_density (numpy array): the baryonic density in g/cm^3
		z (float): the redshift of the file (-1 if it couldn't be determined from the file name)

	'''

	def __init__(self, filename = None, old_format = False):
		'''
		Initialize the file. If filename is given, read data. Otherwise,
		do nothing.
		
		Parameters:
			filename = None (string): the file to read from.
			old_format = False (bool): whether to use the old-style file format.

		Returns:
			Nothing
		'''
		if filename:
			self.read_from_file(filename, old_format)

	def read_from_file(self, filename, old_format = False):

		print_msg('Reading density file:%s ...' % filename)
		self.filename = filename
		#Read raw data from density file
		f = open(filename, 'rb')

		if old_format:
			self.mesh_x = 203
			self.mesh_y = 203
			self.mesh_z = 203
		else:
			temp_mesh = np.fromfile(f,count=3,dtype='int32')
			self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh

		self.raw_density = np.fromfile(f, dtype='float32')
		self.raw_density = self.raw_density.reshape((self.mesh_x, self.mesh_y, self.mesh_z), order='F')
		
		f.close()

		#Convert to g/cm^3 (comoving)
		conv_factor = const.rho_crit_0*(float(self.mesh_x)/float(conv.nbox_fine))**3*const.OmegaB
		self.cgs_density = self.raw_density*conv_factor
		print_msg('Mean density: %g' % np.mean(self.cgs_density.astype('float64')))
		print_msg('Critical matter density: %g' % (const.rho_crit_0*const.OmegaB))

		#Store the redshift from the filename
		try:
			import os.path
			name = os.path.split(filename)[1]
			if old_format:
				self.z = float(name[:5])
			else:
				self.z = float(name.split('n_')[0])
		except:
			print_msg('Could not determine redshift from file name')
			self.z = -1
		print_msg( '...done')

	def write_to_file(self, filename, raw_density=None, cgs_density=None, old_format=False):
		'''
		Write data to file in the same format as it is read.
		
		Parameters:
			filename (string): the file to write to.
			raw_density (numpy array): the density in simulation units
			cgs_density (numpy array): the baryonic density in g/cm^3
			old_format = False (bool): whether to use the old-style file format.

		Returns:
			Nothing
		'''
		assert raw_density is not None or cgs_density is not None, 'Provide data (raw_density or cgs_density) to write.'

		if raw_density is not None:
			self.raw_density = raw_density
			self.mesh_x, self.mesh_y, self.mesh_z = self.raw_density.shape
			conv_factor = const.rho_crit_0*(float(self.mesh_x)/float(conv.nbox_fine))**3*const.OmegaB
			self.cgs_density = self.raw_density*conv_factor
		if cgs_density is not None:
			self.cgs_density = cgs_density
			self.mesh_x, self.mesh_y, self.mesh_z = self.cgs_density.shape
			conv_factor = const.rho_crit_0*(float(self.mesh_x)/float(conv.nbox_fine))**3*const.OmegaB
			self.raw_density = cgs_density/self.conv_factor
		else:
			pass 

		print_msg(f'Writing density file: {filename}...')
		self.filename = filename

		try:
			with open(filename, 'wb') as f:
				# Write mesh dimensions only if not old_format
				if not old_format:
					# Ensure mesh dimensions are set; use self.raw_density.shape if available
					if self.raw_density.shape:
						self.mesh_x, self.mesh_y, self.mesh_z = self.raw_density.shape
					else:
						print_msg("Warning: 'raw_density' attribute does not have a valid shape. Writing with current mesh dimensions.")

					temp_mesh = np.array([self.mesh_x, self.mesh_y, self.mesh_z], dtype=np.int32)
					temp_mesh.tofile(f)
				
				# Write the raw density data (always float32)
				# Flatten the data array and write it.
				# Use 'F' order to match Fortran-style reshaping on read.
				self.raw_density.astype(np.float32).T.flatten().tofile(f)
				
			print_msg('...done')

		except Exception as e:
			print_msg(f"Error writing density file: {e}")
			

class ClumpingFile:
	'''
	A CubeP3M clumping factor file.

	Use the read_from_file method to load a clumping factor file, or 
	pass the filename to the constructor.

	Some useful attributes of this class are:

	* raw_clumping (numpy array): the clumping factor in simulation units
	* z (float): the redshift of the file (-1 if it couldn't be determined from the file name)

	'''

	def __init__(self, filename=None):
		'''
		Initialize the file. If filename is given, read data. Otherwise,
		do nothing.

		Parameters:
			* filename = None (string): the file to read from.
		Returns:
			Nothing
		'''
		if filename:
			self.read_from_file(filename)

	def read_from_file(self, filename):

		print('Reading clumping factor file: %s ...' %
				(filename.rpartition("/")[-1]))
		self.filename = filename
		# Read raw data from clumping factor file
		f = open(filename, 'rb')
		temp_mesh = np.fromfile(f, count=3, dtype='int32')
		self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh
		self.raw_clumping = np.fromfile(f, dtype='float32')
		self.raw_clumping = self.raw_clumping.reshape((self.mesh_x, self.mesh_y, self.mesh_z), order='F')

		f.close()
		print('Average raw clumping:\t\t%.3e' % np.mean(self.raw_clumping))
		print('Min and Max clumping:\t%.3e  %.3e' % (np.min(self.raw_clumping), np.max(self.raw_clumping)))

		# Store the redshift from the filename
		try:
			import os.path
			name = os.path.split(filename)[1]
			self.z = float(name.split('c_')[0])
		except:
			print('Could not determine redshift from file name')
			self.z = -1
		print('...done')

	def write_to_file(self, filename, raw_clumping=None):
		'''
		Write data to file in the same format as it is read.
		
		Parameters:
			filename (string): the file to write to.
			raw_clumping (numpy array): the clumping factor in simulation units
		Returns:
			Nothing
		'''
		self.raw_clumping = raw_clumping
		if self.raw_clumping is None:
			print_msg("Error: No raw clumping data to write. Please load a clumping file first or set the 'raw_clumping' attribute.")
			return

		print_msg(f'Writing clumping factor file: {filename}...')
		self.filename = filename

		try:
			with open(filename, 'wb') as f:
				# Ensure mesh dimensions are set; use self.raw_clumping.shape if available
				if self.raw_clumping.shape:
					self.mesh_x, self.mesh_y, self.mesh_z = self.raw_clumping.shape
				else:
					print_msg("Warning: 'raw_clumping' attribute does not have a valid shape. Writing with current mesh dimensions.")

				# Write mesh dimensions (3 int32 values)
				temp_mesh = np.array([self.mesh_x, self.mesh_y, self.mesh_z], dtype=np.int32)
				temp_mesh.tofile(f)
				
				# Write the raw clumping data (always float32)
				# Flatten the data array and write it.
				# Use 'F' order to match Fortran-style reshaping on read.
				self.raw_clumping.astype(np.float32).T.flatten().tofile(f)
				
			print_msg('...done')

		except Exception as e:
			print_msg(f"Error writing clumping file: {e}")