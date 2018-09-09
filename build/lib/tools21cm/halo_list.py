import numpy as np
from helper_functions import print_msg
import const
import conv

#A simple struct to hold info about single halo
class Halo:
	'''
	A simple struct to hold info about a single halo
	'''
	def __init__(self):
		self.pos = (0.0, 0.0, 0.0) #Position in grid points
		self.pos_cm = (0.0, 0.0, 0.0) #Center of mass position in grid points
		self.vel = (0.0, 0.0, 0.0) #Velocity in simulation units
		self.l = (0.0, 0.0, 0.0) #Angular momentum in simulation units
		self.vel_disp = 0.0 #Velocity dispersion in simulation units
		self.r = 0.0 #Virial radius in grid units
		self.m = 0.0 #Grid mass
		self.mp = 0 #Number of particles
		self.solar_masses = 0.0 #Mass in solar masses


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

		for line in fileinput.input(filename):
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
				halo.pos = np.array(map(float, vals[:3]))
				halo.pos_cm = np.array(map(float, vals[3:6]))
				halo.vel = np.array(map(float, vals[6:9]))
				halo.l = np.array(map(float, vals[9:12]))
				halo.vel_disp = float(vals[12])
				halo.r = float(vals[13])
				halo.m = float(vals[14])
				halo.mp = float(vals[15])
				halo.solar_masses = grid_mass*conv.M_grid*const.solar_masses_per_gram
				self.halos.append(halo)

		fileinput.close()

		return True

			

