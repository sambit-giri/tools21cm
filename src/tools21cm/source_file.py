from . import const
from . import conv
import numpy as np
from .helper_functions import print_msg

class SourceFile:
    '''
    A C2Ray Source file.
    
    Use the read_from_file method to load an source file, or 
    pass the filename to the constructor.
    
    Attributes:
        sources_list (numpy array): the list of sources
        sources_coeval (numpy array): the sources gridded into a ceoval cube
        z (float): the redshift of the file (-1 if it couldn't be determined from the file name)
    '''
    def __init__(self, filename, mass='hm', old_format=False):
        '''
        Initialize the file. If filename is given, read data. Otherwise,
        do nothing.
        
        Parameters:
          filename (string): the file to read from.
          mass = 'hm' (string) : which mass-range sources from C2Ray. Either 'hm' or 'lm', correspoding to high-mass or low-mass atomically cooling halos
          old_format = False (bool): whether to use the old-style 
        file format.
        Returns:
          Nothing
        '''
        self.mass = mass

        if filename:
            self.read_from_file(filename, old_format)


    def read_from_file(self, filename, old_format=False):
        '''
        Read data from file.
            
        Parameters:
            filename (string): the file to read from.
            old_format = False (bool): whether to use the old-style (32 bits)
                file format.
        Returns:
            Nothing
        '''
        print_msg('Reading Source file:%s...' % filename)
        self.filename = filename
        with open(filename, 'rb') as f:
            lines = f.readlines()
            nr_src = int(lines[0].split()[0])

        if(self.mass == 'hm'):
            idx_mass = 3
            self.sources_list = np.zeros((nr_src, 4))
        elif(self.mass  == 'lm'):
            idx_mass = 4
            self.sources_list = np.zeros((nr_src, 4))
        else:
            idx_mass = [3, 4]
            self.sources_list = np.zeros((nr_src, 5))
            pass
    
        # loop over list
        for i in range(nr_src):
            idx = 2*i+1
            text = lines[idx].split()
            self.sources_list[i,:3] = np.array(text[:3], dtype=int)-1
            self.sources_list[i,3:] = np.power(10, conv.gridmass_to_msol(float(text[idx_mass]))) # Msun

        print_msg('...done')

        #Store the redshift from the filename
        try:
            self.z = float(filename[filename.rfind('/')+1:filename.rfind('-coarsest')])
        except:
            print_msg('Could not determine redshift from file name')
            self.z = -1

    def grid_sources(self, mesh):
        self.sources_coeval = np.zeros((mesh, mesh, mesh))
        
        # loop over list
        for n in range(self.sources_list.shape[0]):
            i, j, k = self.sources_list[n,:3].astype(int)
            self.sources_coeval[i,j,k] = self.sources_list[n,3]  # Msun
        
        print_msg('...done')