from . import const, conv
from .helper_functions import print_msg
from . import density_file as df
import numpy as np

class VelocityFile:
	'''
	A CubeP3M velocity/momentum file.
	
	Use the read_from_file method to load a density file, or 
	pass the filename to the constructor.
	
	Some useful attributes of this class are:
	
	* raw_velocity (numpy array): the velocity in simulation units
	* z (float): the redshift of the file (-1 if it couldn't be determined from the file name)
	
	To get the velocity in km/s, use get_kms_from_density
	
	'''
	
	def __init__(self, filename = None):
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
		'''
		Read data from file. Sets the instance variables
		self.raw_velocity and self.kmsrho8
		
		Parameters:
			* filename (string): the file to read from.
		Returns:
			Nothing
		'''
		print_msg('Reading velocity file: %s...' % filename)
		self.filename = filename

		#Read raw data from velocity file
		f = open(filename, 'rb')
		temp_mesh = np.fromfile(f, count=3, dtype='int32')
		self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh
		self.raw_velocity = np.fromfile(f, dtype='float32').astype('float64')
		f.close()
		self.raw_velocity = self.raw_velocity.reshape((3, self.mesh_x, self.mesh_y, self.mesh_z), order='F')

		#Store the redshift from the filename
		try:
			import os.path
			name = os.path.split(filename)[1]
			self.z = float(name.split('v_')[0])
		except:
			print_msg('Could not determine redshift from file name')
			self.z = -1

		#Convert to kms/s*(rho/8)
		self.kmsrho8 = self.raw_velocity*conv.velconvert(z = self.z)


		print_msg('...done')

	def get_kms_from_density(self, density):
		''' 
		Get the velocity in kms. Since the file stores
		momentum rather than velocity, we need the density for this.
		
		Parameters:
			* density (string, DensityFile object or numpy array): the density
				or a file to read the density from.
				
		Returns:
			A numpy array with the same dimensions as the simulation box,
			containing the velocity in km/s.
		'''

		if isinstance(density,str):
			dfile = df.DensityFile(density)
			density = dfile.raw_density
		elif isinstance(density, df.DensityFile):
			density = density.raw_density

		return self.kmsrho8/(density/8)




