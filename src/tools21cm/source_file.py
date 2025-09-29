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

    def write_to_file(self, filename, sources_list=None, old_format=False):
        '''
        Write source data to a file in the C2Ray Source file format.
        
        Parameters:
            filename (string): The file path to write to.
            sources_list (numpy array): the list of sources
            old_format = False (bool): Currently not used for writing, as the source file format
                                        doesn't typically vary with 'old_format' in the same way
                                        as xfrac or density files. Kept for consistency.
        Returns:
            Nothing
        '''
        self.sources_list = sources_list
        if self.sources_list is None or self.sources_list.shape[0] == 0:
            print_msg("Error: No sources_list data to write. Please load a source file first or set the 'sources_list' attribute.")
            return

        print_msg(f'Writing Source file: {filename}...')
        
        try:
            with open(filename, 'w') as f: # Open in text write mode
                nr_src = self.sources_list.shape[0]
                # Write the first line: number of sources and placeholder (e.g., total mass, typically 0.0)
                # The read method only uses the first number, so 0.0 is a safe placeholder for the second value.
                f.write(f"{nr_src} 0.0\n") 

                # Determine which column contains the mass for writing
                # This depends on how sources_list was populated (by read_from_file or manually).
                # The original file format has coordinates as 1-indexed integers and log10 of grid mass.
                
                # Default to 'hm' column if self.mass is 'hm' or not specified,
                # otherwise 'lm' column if self.mass is 'lm'.
                # If self.sources_list has 5 columns, it means both hm and lm are stored.
                # We need to decide which one to write back to the file format.
                # Assuming the original file structure implies a single mass value per source line.
                # Let's prioritize writing the 'hm' mass if available (index 3 for 5-col array).

                for i in range(nr_src):
                    # Convert coordinates back to 1-indexed integers for writing
                    x, y, z = self.sources_list[i,:3].astype(int) + 1 
                    
                    # Get the mass value(s) in Msun
                    current_mass_msun = None
                    if self.sources_list.shape[1] == 5: # (x, y, z, hm_mass, lm_mass)
                        # Decide which mass to write back: if self.mass is specified use that, else default to hm
                        if self.mass == 'lm':
                            current_mass_msun = self.sources_list[i,4] # Use lm_mass
                        else: # self.mass is 'hm' or default
                            current_mass_msun = self.sources_list[i,3] # Use hm_mass
                    elif self.sources_list.shape[1] == 4: # (x, y, z, mass) for hm or lm
                        current_mass_msun = self.sources_list[i,3]
                    else:
                        raise ValueError("Unexpected sources_list shape. Expected 4 or 5 columns.")

                    # Convert Msun back to log10 of grid mass as expected by the file format
                    # Ensure it's not log10(0) if mass is 0, to avoid errors.
                    log10_grid_mass = conv.gridmass_to_msol(current_mass_msun) if current_mass_msun > 0 else -99.0 # Use a placeholder for zero mass

                    # Write each source line: x y z log10(grid_mass)
                    # The original file format has two lines per source, the first is coordinates and log10 grid mass
                    # and the second is empty. Let's replicate this based on the original parsing logic that does `lines[2*i+1]`
                    f.write(f"{x} {y} {z} {log10_grid_mass:.6f}\n") # Using .6f for precision
                    f.write("\n") # Empty line as per typical C2Ray source file format

            print_msg('...done')

        except Exception as e:
            print_msg(f"Error writing Source file: {e}")