from . import const
import numpy as np
import os
from . import density_file as df
from .helper_functions import print_msg 

class XfracFile:
	'''
	A C2Ray xfrac file.

	Use the read_from_file method to load an xfrac file, or 
	pass the filename to the constructor.

	Attributes:
		xi (numpy array): the ionized fraction
		z (float): the redshift of the file (-1 if it couldn't be determined from the file name)

	'''
	def __init__(self, filename = None, old_format=False, neutral=False, binary_format=False):
		'''
		Initialize the file. If filename is given, read data. Otherwise,
		do nothing.
		
		Parameters:
			filename = None (string): the file to read from.
			old_format = False (bool): whether to use the old-style 
				file format.
		Returns:
			Nothing
		'''
		if filename:
			self.read_from_file(filename, old_format, neutral=neutral, binary_format=binary_format)

	def read_from_file(self, filename, old_format=False, neutral=False, binary_format=False):
		'''
		Read data from file.
		
		Parameters:
			filename (string): the file to read from.
			old_format = False (bool): whether to use the old-style (32 bits)
				file format.
						neutral = False (bool): whether the content is the neutral or ionized fraction
						binary_format = False (bool): whether the file is in Fortran unformatted or binary (no record separators) format 
		Returns:
			Nothing
		'''
		print_msg('Reading xfrac file:%s...' % filename)
		self.filename = filename

		f = open(filename, 'rb')
		if binary_format:
			temp_mesh = np.fromfile(f, count=3, dtype='int32')
			self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh #[0:2]
		else:
			temp_mesh = np.fromfile(f, count=6, dtype='int32')
			self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh[1:4]

		if old_format:
			self.xi = np.fromfile(f, dtype='float32')
		else:
			self.xi = np.fromfile(f, dtype='float64')
		self.xi = self.xi.reshape((self.mesh_x, self.mesh_y, self.mesh_z), order='F')

		if neutral:
			self.xi = 1.0-self.xi

		f.close()
		print_msg('...done')

		#Store the redshift from the filename
		import os.path
		try:
			name = os.path.split(filename)[1]
			self.z = float(name.split('_')[1][:-4])
		except:
			print_msg('Could not determine redshift from file name')
			self.z = -1

	def write_to_file(self, filename, xi, old_format=False, neutral=False, binary_format=False):
		'''
		Write data to file.
		
		Parameters:
			filename (string): the file to write to.
			xi (numpy array): the ionized fraction
			old_format = False (bool): whether to use the old-style (32 bits)
				file format.
			neutral = False (bool): whether to write the neutral or ionized fraction
			binary_format = False (bool): whether the file is in Fortran unformatted or binary (no record separators) format 
		Returns:
			Nothing
		'''
		self.xi = xi

		print_msg('Writing xfrac file:%s...' % filename)
		self.filename = filename

		f = open(filename, 'wb')

		# Determine data type for xi based on old_format
		dtype_xi = 'float32' if old_format else 'float64'

		# Determine mesh dimensions
		# If xi is already a numpy array, its shape will give the mesh dimensions
		if self.xi.shape:
			self.mesh_x, self.mesh_y, self.mesh_z = self.xi.shape
		else:
			print_msg("Warning: xi attribute is not shaped. Assuming 0,0,0 for mesh dimensions.")
			self.mesh_x, self.mesh_y, self.mesh_z = 0, 0, 0 # Fallback

		# Prepare mesh dimensions for writing
		if binary_format:
			temp_mesh = np.array([self.mesh_x, self.mesh_y, self.mesh_z], dtype='int32')
			temp_mesh.tofile(f)
		else:
			# For Fortran unformatted, usually there are record markers.
			# In C2Ray's old format, it seems to write 6 int32 values where [1:4] are the mesh.
			# We'll replicate this if possible, otherwise use a placeholder.
			# The original read_from_file reads 6, and uses 1:4. Let's write the same.
			# Assuming the other values are not critical or can be default (e.g., 0).
			temp_mesh = np.zeros(6, dtype='int32')
			temp_mesh[1] = self.mesh_x
			temp_mesh[2] = self.mesh_y
			temp_mesh[3] = self.mesh_z
			temp_mesh.tofile(f)

		# Prepare data for writing (handle neutral option)
		data_to_write = self.xi
		if neutral:
			data_to_write = 1.0 - self.xi

		# Reshape to 1D and write
		data_to_write.astype(dtype_xi).T.flatten().tofile(f) # Transpose and flatten with Fortran order

		f.close()
		print_msg('...done')

