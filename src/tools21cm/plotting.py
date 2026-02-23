import numpy as np
from . import xfrac_file
from . import density_file
from . import conv
from .helper_functions import get_data_and_type

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors as mcolors

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

def print_chain_stats(samples_dict, weights_dict, param_names):
    """Calculates and prints Mean +/- 1-sigma for each chain."""
    print(f"{'Chain':<15} | {'Parameter':<15} | {'Mean':<10} | {'1-sigma (68%)'}")
    print("-" * 65)
    
    for name, samples in samples_dict.items():
        weights = weights_dict[name] if weights_dict else np.ones(len(samples))
        # Normalize weights
        weights /= np.sum(weights)
        
        for i, p_name in enumerate(param_names):
            data = samples[:, i]
            
            # Weighted Mean
            mean = np.average(data, weights=weights)
            
            # Weighted Quantiles (16th and 84th for 1-sigma)
            # Sort data to find percentiles
            idx = np.argsort(data)
            sorted_data = data[idx]
            sorted_weights = weights[idx]
            cumulative_weights = np.cumsum(sorted_weights)
            
            low, high = np.interp([0.16, 0.84], cumulative_weights, sorted_data)
            plus = high - mean
            minus = mean - low
            
            label = p_name if p_name else f"Param {i}"
            print(f"{name:<15} | {label:<15} | {mean:>10.4f} | +{plus:.4f} / -{minus:.4f}")
        print("-" * 65)

def plot_triangle(samples_dict, weights_dict=None, 
                       fig=None, backend='corner',
                       levels=[0.68, 0.95], 
                       param_range=None, bins=20,
                       smooth=0.75, smooth1d=0.75,
                       colors=None, param_names=None,
                       truths=None, truth_color='k',
                       show_stats=True,
                       **kwargs):
    """
    Standardized interface for MCMC triangle plots.
    Supports: 'corner', 'getdist', 'chainconsumer'
    """
    names = list(samples_dict.keys())
    print(f"Backend: {backend}")
    
    if show_stats:
        print_chain_stats(samples_dict, weights_dict, param_names)

    # --- 1. CORNER.PY BACKEND ---
    if backend.lower() == 'corner':
        import corner
        for i, name in enumerate(names):
            color = colors[name] if isinstance(colors, dict) else colors
            fig = corner.corner(
                samples_dict[name], 
                weights=weights_dict[name] if weights_dict else None,
                bins=bins, fig=fig, color=color,
                labels=param_names if i == 0 else None,
                range=param_range, levels=levels,
                smooth=smooth, smooth1d=smooth1d,
                truths=truths if i == 0 else None,
                truth_color=truth_color,
                plot_datapoints=False, 
                plot_density=False, 
                **kwargs
            )
        if kwargs.get('show_legend', True):
            # Add legend
            handles = [mlines.Line2D([], [], color=(colors[n] if isinstance(colors, dict) else colors), label=n) for n in names]
            bbox_to_anchor = kwargs.get('bbox_to_anchor', (0.95, 0.95))
            legend_fontsize = kwargs.get('legend_fontsize', 14)
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)
        return fig

    # --- 2. GETDIST BACKEND ---
    elif backend.lower() == 'getdist':
        from getdist import plots, MCSamples
        custom_settings = {
            'smooth_scale_1D': smooth1d, 
            'smooth_scale_2D': smooth,
        }
        internal_names  = [f"p{i}" for i,pn in enumerate(param_names)]
        internal_labels = [pn.replace('$','') for pn in param_names]
        internal_ranges = {pn:list(param_range[i]) for i,pn in enumerate(internal_names)}
        print(internal_ranges)
        mcs = []
        for name in names:
            w = weights_dict[name] if weights_dict else None
            s = MCSamples(samples=samples_dict[name], weights=w, 
                          label=name,
                          names=internal_names, 
                          labels=internal_labels, 
                          ranges=internal_ranges,
                          settings=custom_settings,
                          )
            mcs.append(s)
        
        g = plots.get_subplot_plotter()
        # GetDist uses 'markers' for the truth lines in the 1D/2D plots
        g.triangle_plot(mcs, filled=True, levels=levels,
                        colors=[colors[n] for n in names] if isinstance(colors, dict) else None,
                        legend_labels=names, markers=truths, **kwargs)
        return g

    # --- 3. CHAINCONSUMER BACKEND ---
    elif backend.lower() == 'chainconsumer':
        from chainconsumer import Chain, ChainConsumer, Truth, PlotConfig
        import pandas as pd
        c = ChainConsumer()
        for name in names:
            color = colors[name] if isinstance(colors, dict) else colors
            df = pd.DataFrame(samples_dict[name], columns=param_names)
            c.add_chain(Chain(samples=df, weights=weights_dict[name] if weights_dict else None,
                              name=name, color=mcolors.to_hex(color)))
        
        # Mapping the truths list to a Truth object
        if truths is not None:
            # Create a dictionary of {param_name: value}
            truth_dict = {p: v for p, v in zip(param_names, truths)}
            c.add_truth(Truth(location=truth_dict, color=truth_color))
        
        extents = truth_dict = {p: v for p, v in zip(param_names, param_range)}
        c.set_plot_config(PlotConfig(bins=bins, extents=extents, smooth=smooth))
        return c.plotter.plot(**kwargs)
        
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
    
    
    
    
