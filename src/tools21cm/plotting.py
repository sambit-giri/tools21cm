import numpy as np
from . import xfrac_file
from . import density_file
from . import conv
from .helper_functions import get_data_and_type

def plot_slice(data, los_axis = 0, slice_num = 0, logscale = False, **kwargs):
    '''
    Plot a slice through a data cube. This function will produce a nicely
    formatted image plot with the correct units on the axes.
    
    Parameters:
        * data (XfracFile, DensityFile, string or numpy array): the data to 
            plot. The function will try to determine what type of data it's 
            been given. 
        * los_axis = 0 (integer): the line of sight axis. Must be 0,1 or 2
        * slice_num = 0 (integer): the point along los_axis where the slice
            will be taken.
        * logscale = False (bool): whether to plot the logarithm of the data
            
    Kwargs:
        All kwargs are sent to matplotlib's imshow function. This can be used to,
        for instance, change the colormap.
        
    Returns:
        Nothing.
        
    Example (plot an xfrac file with a custom color map):
        >>> xfile = t2c.XfracFile('xfrac3d_8.515.bin')
        >>> t2c.plot_slice(xfile, cmap = pl.cmap.hot)
    '''
    
    import pylab as pl
    
    #Determine data type
    plot_data, datatype = get_data_and_type(data)
    
    #Take care of different LOS axes
    assert (los_axis == 0 or los_axis == 1 or los_axis == 2)
    if los_axis == 0:
        get_slice = lambda data, i : data[i,:,:]
    elif los_axis == 1:
        get_slice = lambda data, i : data[:,i,:]
    else:
        get_slice = lambda data, i : data[:,:,i]
    
    data_slice = get_slice(plot_data, slice_num)
    ext = [0, conv.LB, 0, conv.LB]
    if (logscale):
        data_slice = np.log10(data_slice)

    #Plot
    pl.imshow(data_slice, extent=ext, **kwargs)
    cbar = pl.colorbar()
    pl.xlabel(r'$\mathrm{cMpc}$')
    pl.ylabel(r'$\mathrm{cMpc}$')
    
    #Make redshift string
    try:
        z_str = '$z = %.2f$' % data.z
    except Exception:
        z_str = ''
    
    #Set labels etc
    if datatype == 'xfrac':
        if (logscale):
            cbar.set_label(r'$\log_{10} x_i$')
        else:
            cbar.set_label(r'$x_i$')
        pl.title('Ionized fraction, %s' % z_str)
    elif datatype == 'density':
        if (logscale):
            cbar.set_label(r'$\log_{10} \\rho \; \mathrm{[g \; cm^{-3}]}$')
        else:
            cbar.set_label(r'$\\rho \; \mathrm{[g \; cm^{-3}]}$')
        pl.title('Density, %s' % z_str)
        
def plot_hist(data, logscale = False, **kwargs):
    '''
    Plot a histogram of the data in a data cube.
    
    Parameters:
        * data (XfracFile, DensityFile, string or numpy array): the data to 
            plot. The function will try to determine what type of data it's 
            been given. 
        * logscale = False (bool): whether to plot the logarithm of the data
            
    Kwargs:
        All kwargs are sent to matplotlib's hist function. Here, you can specify,
        for example, the bins keyword
        
    Returns:
        Nothing.
    '''

    import pylab as pl
    
    #Determine data type
    plot_data, datatype = get_data_and_type(data)
    
    #Fix bins
    if datatype == 'xfrac' and not 'bins' in kwargs.keys():
        kwargs['bins'] = np.linspace(0,1,30)
    else: 
        kwargs['bins'] = 30
        
    #Plot
    if not 'histtype' in kwargs.keys():
        kwargs['histtype'] = 'step'
    if not 'color' in kwargs.keys():
        kwargs['color'] = 'k'
    
    pl.hist(plot_data.flatten(), log = logscale, **kwargs)
        
    #Labels
    if datatype == 'xfrac':
        pl.xlabel('$x_i$')
    elif datatype == 'density':
        pl.xlabel(r'$\\rho \; \mathrm{[g \; cm^{-3}]}$')
        
if __name__ == '__main__':
    import tools21cm as t2c
    import pylab as pl
    
    t2c.set_verbose(True)
    
    pl.figure()
    
    dfilename = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/coarser_densities/nc256_halos_removed/6.905n_all.dat'
    xfilename = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/114Mpc_f2_10S_256/results_ranger/xfrac3d_8.958.bin'
    
    dfile = t2c.DensityFile(dfilename)
#    plot_slice(dfile, los_axis=1, logscale=True, cmap=pl.cm.hot)
#    ax2 = pl.subplot(1,2,2)
#    plot_slice(xfilename)
    plot_slice(t2c.XfracFile(xfilename))
    pl.show()
    
    
    
    
