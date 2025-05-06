#This file contains various helper routines

import numpy as np
from scipy.interpolate import interp1d
from . import const
import glob, os, time
import os.path
from numpy import array, asarray, roll
from numpy.linalg import matrix_rank
from scipy.fftpack import fft, ifft, fftn, ifftn
from numpy.fft import rfftn, irfftn
from math import ceil, floor

try:
        import numexpr as ne
        numexpr_available = True
except:
        numexpr_available = False


def real_to_fourier_amplitude_phase(data):
        data_fft = fftn(data)
        a, b     = np.real(data_fft), np.imag(data_fft)
        R        = np.sqrt(a**2+b**2)
        phi      = np.arctan(b/a)
        return R, phi

def get_source_redshifts(source_dir, z_low = None, z_high = None, bracket=False):
        ''' 
        Make a list of the redshifts of all the xfrac files in a directory.
        
        Parameters:
                * xfrac_dir (string): the directory to look in
                * z_low = None (float): the minimum redshift to include (if given)
                * z_high = None (float): the maximum redshift to include (if given)
                * bracket = False (bool): if true, also include the redshifts on the
                        lower side of z_low and the higher side of z_high
         
        Returns: 
                The redhifts of the files (numpy array of floats) '''

        source_files = glob.glob(os.path.join(source_dir,'*-coarsest_wsubgrid_sources.dat'))

        redshifts = []
        for f in source_files:
                try:
                        z = float(f[f.rfind('/')+1:f.rfind('-coarsest_wsubgrid_sources')])
                        redshifts.append(z)
                except: 
                        pass

        return _get_redshifts_in_range(redshifts, z_low, z_high, bracket)


def get_xfrac_redshifts(xfrac_dir, z_low = None, z_high = None, bracket=False):
        ''' 
        Make a list of the redshifts of all the xfrac files in a directory.
        
        Parameters:
                * xfrac_dir (string): the directory to look in
                * z_low = None (float): the minimum redshift to include (if given)
                * z_high = None (float): the maximum redshift to include (if given)
                * bracket = False (bool): if true, also include the redshifts on the
                        lower side of z_low and the higher side of z_high
         
        Returns: 
                The redhifts of the files (numpy array of floats) '''

        xfrac_files = glob.glob(os.path.join(xfrac_dir,'xfrac*.bin'))

        redshifts = []
        for f in xfrac_files:
                try:
                        z = float(f.split('_')[-1][:-4])
                        redshifts.append(z)
                except: 
                        pass

        return _get_redshifts_in_range(redshifts, z_low, z_high, bracket)


def get_dens_redshifts(dens_dir, z_low=None, z_high=None, bracket=False):
        ''' 
        Make a list of the redshifts of all the density files in a directory.
        
        Parameters:
                * dens_dir (string): the directory to look in
                * z_low = None (float): the minimum redshift to include (if given)
                * z_high = None (float): the maximum redshift to include (if given)
                * bracket = False (bool): if true, also include the redshifts on the
                        lower side of z_low and the higher side of z_high
         
        Returns: 
                The redhifts of the files (numpy array of floats) '''

        dens_files = glob.glob(os.path.join(dens_dir,'*n_all.dat'))

        redshifts = []
        for f in dens_files:
                try:
                        z = float(os.path.split(f)[1].split('n_')[0])
                        redshifts.append(z)
                except:
                        pass
        
        return _get_redshifts_in_range(redshifts, z_low, z_high, bracket)


def _get_redshifts_in_range(redshifts, z_low, z_high, bracket):
        '''
        Filter out redshifts outside of range. For internal use.
        '''
        redshifts = np.array(redshifts)
        redshifts.sort()
        if bracket:
                if z_low < redshifts.min() or z_high > redshifts.max():
                        raise Exception('No redshifts to bracket range.')
                z_low = redshifts[redshifts <= z_low][-1]
                z_high = redshifts[redshifts >= z_high][0]
        if z_low == None:
                z_low = redshifts.min()-1
        if z_high == None:
                z_high = redshifts.max()+1
        idx = (redshifts >= z_low)*(redshifts <= z_high)
        redshifts = redshifts[idx]

        return np.array(redshifts)


def print_msg(message, print_time=True):
        ''' Print a message if verbose is true '''
        if verbose:
                if print_time:
                        timestr = time.strftime('%Y/%m/%d %H:%M:%S')
                        message = '%s --- %s' % (timestr, message)
                print(message)
                

def flt_comp(x,y, epsilon=0.0001):
        ''' Compare two floats, return true of difference is < epsilon '''
        return abs(x-y) < epsilon


def get_interpolated_array(in_array, new_len, kind='nearest'):
        ''' Get a higher-res version of an array.
        
        Parameters:
                * in_array (numpy array): the array to upscale
                * new_len (integer): the new length of the array
                * kind = 'nearest' (string): the type of interpolation to use
                
        Returns:
                The upscaled array. 
        ''' 

        old_len = len(in_array)
        func = interp1d(np.linspace(0,1,old_len), in_array, kind=kind)
        out_array = func(np.linspace(0,1,new_len))
        return out_array


def read_cbin(filename, bits=32, order='C', dimensions=3, records=False):
        ''' Read a binary file with three inital integers (a cbin file).
        
        Parameters:
                * filename (string): the filename to read from
                * bits = 32 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                        for C style ordering, or 'F' for fortran style.
                * dimensions (int): the number of dimensions of the data (default:3)
                * records (boolean): does the file contain record separators?
                        
        Returns:
                The data as a three dimensional numpy array.
        '''

        assert(bits ==32 or bits==64)

        f = open(filename)
        
        print_msg('Reading cbin file: %s' % filename)

        counter=dimensions+3 if records else dimensions 
        header = np.fromfile(f, count=counter, dtype='int32')
        if records: temp_mesh=header[1:4]
        else: temp_mesh=header[0:3]
        
        datatype = np.float32 if bits == 32 else np.float64
        data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
        data = data.reshape(temp_mesh, order=order)
        f.close()
        return data


def read_raw_binary(filename, bits=64, order='C'):
        ''' Read a raw binary file with no mesh info. The mesh
        is assumed to be cubic.
        
        Parameters:
                * filename (string): the filename to read from
                * bits = 64 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                        for C style ordering, or 'F' for fortran style.
                        
        Returns:
                The data as a three dimensional numpy array.
        '''

        assert(bits ==32 or bits==64)

        f = open(filename)
        
        print_msg('Reading raw binary file: %s' % filename)

        datatype = np.float32 if bits == 32 else np.float64
        data = np.fromfile(f, dtype=datatype)
        n = round(len(data)**(1./3.))
        print_msg('Mesh size appears to be: %d' % n)
        data = data.reshape((n, n, n), order=order)
        return data


def save_raw_binary(filename, data, bits=64, order='C'):
        ''' Save a raw binary file with no mesh info.
        
        Parameters:
                * filename (string): the filename to read from
                * data (numpy array): the data to save
                * bits = 64 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                        for C style ordering, or 'F' for fortran style.
                        
        Returns:
                The data as a three dimensional numpy array.
        '''
        data = data.flatten(order=order)
        datatype = np.float32 if bits == 32 else np.float64
        data = data.astype(datatype)
        data.tofile(filename)
        

def save_cbin(filename, data, bits=32, order='C'):
        ''' Save a binary file with three inital integers (a cbin file).
        
        Parameters:
                * filename (string): the filename to save to
                * data (numpy array): the data to save
                * bits = 32 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                        for C style ordering, or 'F' for fortran style.
                        
        Returns:
                Nothing
        '''
        print_msg('Saving cbin file: %s' % filename)
        assert(bits ==32 or bits==64)
        f = open(filename, 'wb')
        mesh = np.array(data.shape).astype('int32')
        mesh.tofile(f)
        datatype = (np.float32 if bits==32 else np.float64)
        data.flatten(order=order).astype(datatype).tofile(f)
        f.close()
        
        
def read_fits(filename, header=True):
        '''
        Read a fits file and return the data as a numpy array
        
        Parameters:
                * filename (string): the fits file to read
                * header = True    : If True. the header is also returned
        Returns:
                numpy array containing the data
        '''
        try:
                import pyfits as pf
                hdulist = pf.open(filename)
                header = hdulist[0].header
                data = hdulist[0].data.astype('float64')
        except:
                from astropy.io import fits
                fits_image_filename = fits.util.get_testdata_filepath(filename)
                hdul = fits.open(fits_image_filename)
                header = hdul[0].header
                data = hdul[1].data.astype('float64')

        if header: return data, header
        else: return data


def save_fits(data, filename, header=None):
        '''
        Save data as a fits file. The data can be a file object,
        a file to read or a pure data array.
        
        Parameters:
                * indata (XfracFile, DensityFile, string or numpy array): the data to save
                * filename (string): the file to save to
                * header = None    : header added to the fits file.
                
        Returns:
                Nothing
        
        '''
        
        save_data, datatype = get_data_and_type(data)

        try:
                import pyfits as pf
                if type(header)==pf.header.Header: pf.writeto(filename, save_data, header)
                else: pf.writeto(filename, save_data)
        except:
                from astropy.io import fits
                hdu = pf.PrimaryHDU(save_data.astype('float64'))
                hdulist = pf.HDUList([hdu])
                hdulist.writeto(filename, clobber=True)

        

def determine_filetype(filename):
        '''
        Try to figure out what type of data is in filename.
        
        Parameters:
                * filename (string): the filename. May include the full
                        path.
                
        Returns:
                A string with the data type. Possible values are:
                'xfrac', 'density', 'velocity', 'cbin', 'unknown'
                
        '''
        
        filename = os.path.basename(filename)
        
        if 'xfrac3d' in filename:
                return 'xfrac'
        elif 'n_all' in filename:
                return 'density'
        elif 'Temper' in filename:
                return 'temper'
        elif 'v_all' in filename:
                return 'velocity'
        elif '.cbin' in filename:
                return 'cbin'
        elif 'dbt' in filename:
                return 'dbt'
        return 'unknown'


def get_data_and_type(indata, cbin_bits=32, cbin_order='c', raw_density=False):
        '''
        Extract the actual data from an object (which may
        be a file object or a filename to be read), and
        determine what type of data it is.
        
        Parameters:
                * indata (XfracFile, DensityFile, string or numpy array): the data
                * cbin_bits (integer): the number of bits to use if indata is a cbin file
                * cbin_order (string): the order of the data in indata if it's a cbin file
                * raw_density (bool): if this is true, and the data is a 
                        density file, the raw (simulation units) density will be returned
                        instead of the density in cgs units
                
        Returns:
                * A tuple with (outdata, type), where outdata is a numpy array 
                containing the actual data and type is a string with the type 
                of data. Possible values for type are 'xfrac', 'density', 'cbin'
                and 'unknown'
                
        '''
        import tools21cm.density_file
        import tools21cm.xfrac_file
        import tools21cm.temper_file

        if isinstance(indata, tools21cm.xfrac_file.XfracFile):
                return indata.xi, 'xfrac'
        elif isinstance(indata, tools21cm.temper_file.TemperFile):
                return indata.temper, 'temper'
        elif isinstance(indata, tools21cm.density_file.DensityFile):
                if raw_density:
                        return indata.raw_density, 'density'
                else:
                        return indata.cgs_density, 'density'
        elif isinstance(indata, str):
                filetype = determine_filetype(indata)
                if filetype == 'xfrac':
                        return get_data_and_type(tools21cm.xfrac_file.XfracFile(indata))
                elif filetype == 'temper':
                        return get_data_and_type(tools21cm.temper_file.TemperFile(indata))
                elif filetype == 'density':
                        return get_data_and_type(tools21cm.density_file.DensityFile(indata))
                elif filetype == 'cbin':
                        return read_cbin(indata, bits=cbin_bits, order=cbin_order), 'cbin'
                elif filetype == 'dbt':
                        return np.load(indata),'dbt'
                else:
                        raise Exception('Unknown file type')
        elif isinstance(indata, np.ndarray):
                return indata, 'unknown'
        raise Exception('Could not determine type of data')

def save_data(savefile, data, filetype=None, **kwargs):
        if filetype is None: filetype = ''
        if '.npy' in savefile[-5:] or filetype.lower() in ['npy', 'python_pickle']:
                np.save(savefile, data)
        elif '.pkl' in savefile[-5:] or filetype.lower() in ['pkl','pickle']:
                import pickle
                pickle.dump(data, open(savefile, 'wb'))
        elif '.cbin' in savefile[-5:] or filetype.lower()=='cbin':
                save_cbin(savefile, data, bits=kwargs.get('bits',32), order=kwargs.get('order','C'))
        elif '.fits' in savefile[-5:] or filetype in ['fits']:
                save_fits(data, savefile, header=kwargs.get('header'))
        elif '.bin' in savefile[-5:] or filetype in ['bin', 'binary']:
                save_raw_binary(savefile, data, bits=kwargs.get('bits',64), order=kwargs.get('order','C'))
        else:
                print('Unknown filetype.')
                return False 
        return True
        

def get_mesh_size(filename):
        '''
        Read only the first three integers that specify the mesh size of a file.
        
        Parameters:
                * filename (string): the file to read from. can be an xfrac file,
                        a density file or a cbin file.
                        
        Returns:
                (mx,my,mz) tuple
        '''
        datatype = determine_filetype(filename)
        f = open(filename, 'rb')
        if datatype == 'xfrac':
                temp_mesh = np.fromfile(f, count=6, dtype='int32')
                mesh_size = temp_mesh[1:4]
        elif datatype == 'temper':
                temp_mesh = np.fromfile(f,count=3,dtype='int32')
                mesh_size = temp_mesh[1:4]
        elif datatype == 'density':
                mesh_size = np.fromfile(f,count=3,dtype='int32')
        elif datatype == 'cbin':
                mesh_size = np.fromfile(f,count=3,dtype='int32')
        elif datatype == 'dbt':
                mesh_size = np.array([250,250,250])
        else:
                raise Exception('Could not determine mesh for filetype %s' % datatype)
        f.close()
        return mesh_size
                

def outputify(output):
        '''
        If given a list with only one element, return the element
        If given a standard python list or tuple, make it into
        a numpy array.
        
        Parameters:
                output (any scalar or list-like): the output to process
                
        Returns:
                The output in the correct format.
        '''
        
        if hasattr(output, '__iter__'): #List-like
                if len(output) == 1:
                        return output[0]
                elif not type(output) == np.ndarray:
                        return np.array(output)
        return output
                
                
def determine_redshift_from_filename(filename):
        '''
        Try to find the redshift hidden in the filename.
        If there are many sequences of numbers in the filename
        this method will guess that the longest sequence is the
        redshift.
        
        Parameters:
                * filename (string): the filename to analyze
                
        Returns:
                * redshift (float) 
                If no redshift could be found, return -1
        '''
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        
        number_strs = [] #Will contain all sequences of numbers
        last_was_char = True
        for s in filename:
                if s.isdigit() or s == '.':
                        if last_was_char:
                                number_strs.append([])
                        last_was_char = False
                        number_strs[-1].append(s)
                else:
                        last_was_char = True
        
        longest_idx = 0
        for i in range(len(number_strs)):
                if len(number_strs[i]) > len(number_strs[longest_idx]):
                        longest_idx = i
                number_strs[i] = ''.join(number_strs[i])
                
        if len(number_strs) == 0:
                return -1
                
        return float(number_strs[longest_idx])


def find_idx(ar, values):
        '''
        Find the (fractional) indices of values in an array.
        If values contains values outside of the range of ar,
        these will be clamped to the bounds of ar.
        
        Parameters:
                * ar (numpy array): the array to search through
                * values (numpy array or float): the value(s) to look for
                
        Returns:
                The indices of the values. Can be in-between integer indices
        '''
        
        #Make input into an array and clamp outliers
        values = np.atleast_1d(values)
        values[values > ar.max()] = ar.max()
        values[values < ar.min()] = ar.min()
        
        #Make sure that the values are monotonically increasing/decreasing
        dv = ar[1:]-ar[:-1]
        if (not np.all(dv >= 0.)) and (not np.all(dv <= 0.)):
                raise ValueError('Array must be monotonically increasing or decreasing')
        
        #If decreasing, reverse
        if np.all(dv <= 0.):
                ar = ar[::-1]
                decreasing = True
        else:
                decreasing = False
        
        #Calculate indices
        integer_part = np.searchsorted(ar, values)-1
        x1 = ar[integer_part]
        x2 = ar[integer_part+1]
        fractional_part = (values-x1)/(x2-x1)
        out = integer_part+fractional_part
        if decreasing:
                out = len(ar)-1-out
        
        return outputify(out)


def get_eval():
        '''
        Evaluate an expression using numexpr if
        available. For internal use.
        ''' 
        if numexpr_available:
                return ne.evaluate
        return eval
                        

verbose = False
def set_verbose(_verbose):
        '''
        Turn on or off verbose mode.
        
        Parameters:
                * verb (bool): whether or not to be verbose
                
        Returns:
                Nothing
        '''
        global verbose
        verbose = _verbose

def fftconvolve(in1, in2):
    """Convolve two N-dimensional arrays using FFT.

    This is a modified version of the scipy.signal.fftconvolve.
    The new feature is derived from the fftconvolve algorithm used in the IDL package.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    """
    in1 = asarray(in1)
    in2 = asarray(in2)

    #if matrix_rank(in1) == matrix_rank(in2) == 0:  # scalar inputs
    #    return in1 * in2
    #elif not in1.ndim == in2.ndim:
    if not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same rank")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex128) or
                      np.issubdtype(in2.dtype, np.complex128))

    fsize = s1

    fslice = tuple([slice(0, int(sz)) for sz in fsize])
    if not complex_result:
        ret = irfftn(rfftn(in1, fsize) *
                     rfftn(in2, fsize), fsize)[fslice].copy()
        ret = ret.real
    else:
        ret = ifftn(fftn(in1, fsize) * fftn(in2, fsize))[fslice].copy()

    # Shift the axes back 
    shift = np.floor(fsize*0.5).astype(int)
    list_of_axes = tuple(np.arange(0, shift.size))
    ret = roll(ret, -shift, axis=list_of_axes)
    return ret

def combined_mean_variance(means, variances, sample_sizes=None):
    '''
    Estimate the combined mean and variance of multiple datasets with different means, variances, and sample sizes.

    Parameters
    ----------
    means : array-like, list, or dict
        Means of individual datasets. If a dict, values are used.

    variances : array-like, list, or dict
        Variances of individual datasets. If a dict, values are used.

    sample_sizes : array-like, list, dict, or None, optional
        Sample sizes for each dataset. If None, all sample sizes are assumed to be 1.

    Returns
    -------
    mean_comb : float or numpy array
        Combined mean of the datasets.

    var_comb : float or numpy array
        Combined variance of the datasets.

    Notes
    -----
    - The formula used to compute the combined mean is:
      mean_comb = (Σ(sample_sizes[i] * means[i])) / Σ(sample_sizes[i])

    - The formula for combined variance accounts for both the internal variance of each dataset 
      and the spread of the dataset means:
      var_comb = (Σ((sample_sizes[i] - 1) * variances[i] + sample_sizes[i] * (means[i] - mean_comb)^2)) / (Σ(sample_sizes[i]) - 1)

    Example
    -------
    >>> means = [2.0, 3.0, 4.0]
    >>> variances = [0.5, 0.7, 0.6]
    >>> sample_sizes = [10, 15, 20]
    >>> combined_mean_variance(means, variances, sample_sizes)
    (3.2, 0.6666666666666666)
    '''
    # Convert inputs to numpy arrays for easier manipulation
    if isinstance(means, list) or isinstance(means, dict):
        means = np.array(list(means.values()) if isinstance(means, dict) else means)

    if isinstance(variances, list) or isinstance(variances, dict):
        variances = np.array(list(variances.values()) if isinstance(variances, dict) else variances)

    if sample_sizes is None:
        sample_sizes = np.ones(means.shape)
    elif isinstance(sample_sizes, list) or isinstance(sample_sizes, dict):
        sample_sizes = np.array(list(sample_sizes.values()) if isinstance(sample_sizes, dict) else sample_sizes)

    # Calculate combined mean
    mean_comb = np.sum(sample_sizes * means, axis=0) / np.sum(sample_sizes, axis=0)

    # Calculate combined variance
    var_comb = np.sum((sample_sizes - 1) * variances + sample_sizes * (means - mean_comb)**2, axis=0) / (np.sum(sample_sizes, axis=0) - 1)

    return mean_comb, var_comb
