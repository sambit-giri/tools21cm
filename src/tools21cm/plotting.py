from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors as mcolors
from scipy.linalg import inv
from scipy.stats import entropy
import pandas as pd

# External dependencies for corner and mcmc plots
try:
    import corner
except ImportError:
    corner = None

try:
    from getdist import plots, MCSamples
except ImportError:
    plots, MCSamples = None, None

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
        
class DistributionDiagnostic(ABC):
    """
    Base class for diagnosing and comparing probability distributions.

    Attributes:
        backend (str): 'corner' or 'getdist' for multidimensional plots.
        true_values (list): Ground truth values for computing diagnostic metrics.
        param_labels (list): LaTeX labels for the parameters (e.g., [r'\Omega_m']).
        distributions (dict): Dictionary storing distribution data and stats.
    """
    _METRIC_LABELS = {
        'Z':           r'$Z_p = |\mu_p - \theta_{\mathrm{truth},p}|\,/\,\sigma_p$',
        'PIT':         r'$F_p(\theta_{\mathrm{truth},p})$',
        'Bias':        r'$\tilde{\theta}_p - \theta_{\mathrm{truth},p}$',
        'CI68':        r'$\Delta_{68,p}$',
        'Mahalanobis': r'$D_M$',
        'KL':          r'$D_{KL}$ (bits)',
        'RMSE':        r'RMSE',
        'Cover_68':    'Cover 68%',
        'Cover_95':    'Cover 95%',
    }
    _PER_PARAM = {'Z', 'PIT', 'Bias', 'CI68'}
    _IDEAL_VALUES = {
        'Z': 0.0, 'PIT': 0.5, 'Bias': 0.0, 'Mahalanobis': 1.0,
        'KL': 0.0, 'RMSE': 0.0
    }

    def __init__(self, backend='corner', true_values=None, param_labels=None):
        self.backend = backend.lower()
        self.true_values = true_values
        self.param_labels = param_labels  # Can be None; will be generated dynamically if needed
        self.distributions = {}

        # Priority on C0-C9 for clarity, then tab20 for density
        self.fallback_colors = [f'C{i}' for i in range(10)] + \
                               [plt.get_cmap('tab20')(i) for i in range(20)]

    def _get_default_param_labels(self, num_params):
        """Generates labels like \theta_1, \theta_2... if param_labels is None or too short."""
        if self.param_labels is None:
            return [r"\theta_{%d}" % (i+1) for i in range(num_params)]
        if len(self.param_labels) < num_params:
            extended = list(self.param_labels)
            for i in range(len(self.param_labels), num_params):
                extended.append(r"\theta_{%d}" % (i+1))
            return extended
        return self.param_labels

    def _get_distribution_label(self, label):
        """Returns provided label or generates 'Distribution N'."""
        if label is not None:
            return label
        return "Distribution %d" % (len(self.distributions) + 1)

    @abstractmethod
    def add_distribution(self, data, label=None, color=None):
        """Must be implemented by subclasses."""

    def _calculate_base_metrics(self, points, weights):
        """Common metric calculation for weighted samples."""
        weights_norm = weights / np.sum(weights)
        num_params = points.shape[1]

        means = np.average(points, axis=0, weights=weights_norm)
        cov = np.cov(points.T, aweights=weights_norm)
        sigmas = np.sqrt(np.diag(cov))

        cis = []
        for p in range(num_params):
            data_p = points[:, p]
            idx = np.argsort(data_p)
            sorted_data = data_p[idx]
            sorted_weights = weights_norm[idx]
            cum_weights = np.cumsum(sorted_weights)

            def quantile(q, _cw=cum_weights, _sd=sorted_data): return float(np.interp(q, _cw, _sd))
            def cdf_at(val, _cw=cum_weights, _sd=sorted_data): return float(np.interp(val, _sd, _cw))

            cis.append({
                'median': quantile(0.500),
                'lo1': quantile(0.160), 'hi1': quantile(0.840),
                'lo2': quantile(0.025), 'hi2': quantile(0.975),
                'cdf_at': cdf_at,
            })

        metrics = {'means': means, 'sigmas': sigmas, 'cov': cov, 'cis': cis}

        if self.true_values is not None:
            # Handle variable parameter counts
            tv_slice = np.asarray(self.true_values)[:num_params]
            metrics['z_scores'] = np.abs(means - tv_slice) / sigmas
            delta = means - tv_slice
            metrics['rmse'] = np.sqrt(np.mean(delta**2))

            try:
                metrics['mahalanobis'] = float(np.sqrt(delta @ inv(cov) @ delta))
            except:
                metrics['mahalanobis'] = np.nan

            metrics['pit'] = np.array([cis[p]['cdf_at'](tv_slice[p]) for p in range(len(tv_slice))])
            metrics['cover_68'] = np.array([cis[p]['lo1'] <= tv_slice[p] <= cis[p]['hi1'] for p in range(len(tv_slice))])
            metrics['cover_95'] = np.array([cis[p]['lo2'] <= tv_slice[p] <= cis[p]['hi2'] for p in range(len(tv_slice))])

            # Info gain proxy (relative to unit volume)
            try:
                metrics['entropy'] = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov))
            except:
                metrics['entropy'] = np.nan

        return metrics

    def plot_corner(self, levels=None, **kwargs):
        """Corner/triangle plot for all added distributions.

        Args:
            levels: Contour levels as probability fractions. Default: [0.68, 0.95].
            **kwargs: Passed to the backend (corner or getdist).
        """
        if not self.distributions:
            raise ValueError("No distributions added.")
        if levels is None:
            levels = [0.68, 0.95]

        if self.backend == 'corner':
            return self._plot_corner_backend(levels=levels, **kwargs)
        elif self.backend == 'getdist':
            return self._plot_getdist_backend(levels=levels, **kwargs)

    def _plot_corner_backend(self, levels=None, **kwargs):
        if corner is None: raise ImportError("Please install 'corner'.")
        if levels is None:
            levels = [0.68, 0.95]
        fig = None
        handle_map = {}  # orig insertion index → legend handle

        max_params = max([p['points'].shape[1] for p in self.distributions.values()])
        full_labels = [f"${l}$" for l in self._get_default_param_labels(max_params)]

        # Render full-dim distributions first so the base figure exists before transplanting
        ordered = sorted(
            enumerate(self.distributions.items()),
            key=lambda x: -x[1][1]['points'].shape[1]
        )

        for orig_i, (name, dist) in ordered:
            color = dist['color'] or self.fallback_colors[orig_i % len(self.fallback_colors)]
            k = dist['points'].shape[1]

            if k == max_params:
                opts = {
                    'labels': full_labels, 'color': color, 'levels': levels,
                    'fill_contours': True, 'plot_datapoints': False, 'fig': fig
                }
                opts.update(kwargs)
                fig = corner.corner(dist['points'], weights=dist['weights'], **opts)
            else:
                # Render on a temporary figure then transplant artists into the
                # top-left k×k sub-panels of the main figure — no fake data added
                extra_kw = {kk: vv for kk, vv in kwargs.items()
                            if kk not in ('fig', 'labels', 'color', 'levels',
                                          'fill_contours', 'plot_datapoints')}
                temp_fig = corner.corner(
                    dist['points'], weights=dist['weights'],
                    labels=full_labels[:k], color=color, levels=levels,
                    fill_contours=True, plot_datapoints=False, **extra_kw
                )
                temp_axarr = np.array(temp_fig.axes).reshape(k, k)
                main_axarr = np.array(fig.axes).reshape(max_params, max_params)
                for row in range(k):
                    for col in range(k):
                        src = temp_axarr[row, col]
                        dst = main_axarr[row, col]
                        for coll in list(src.collections):
                            coll.remove()
                            dst.add_collection(coll)
                            coll.set_transform(dst.transData)
                        if row == col:
                            for ln in list(src.lines):
                                ln.remove()
                                dst.add_line(ln)
                plt.close(temp_fig)

            handle_map[orig_i] = mlines.Line2D([], [], color=color, label=name, lw=2)

        if self.true_values:
            corner.overplot_lines(fig, self.true_values[:max_params],
                                  color="gray", ls="--", alpha=0.5)

        handles = [handle_map[i] for i in sorted(handle_map)]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.95))
        return fig

    def _plot_getdist_backend(self, levels=None, **kwargs):
        if plots is None: raise ImportError("Please install 'getdist'.")
        if levels is None:
            levels = [0.68, 0.95]
        samples_list = []

        max_params = max([p['points'].shape[1] for p in self.distributions.values()])
        full_labels = self._get_default_param_labels(max_params)

        colors = []
        for i, (name, dist) in enumerate(self.distributions.items()):
            k = dist['points'].shape[1]
            p_names = [f"p{j}" for j in range(k)]
            s = MCSamples(samples=dist['points'], weights=dist['weights'],
                          names=p_names, labels=full_labels[:k], label=name)
            samples_list.append(s)
            colors.append(dist['color'] or self.fallback_colors[i % len(self.fallback_colors)])

        line_args = [{'color': c} for c in colors]

        g = plots.get_subplot_plotter()
        g.triangle_plot(samples_list, filled=True, contour_levels=levels,
                        colors=colors, line_args=line_args,
                        markers=self.true_values, **kwargs)
        return g

    def plot_forest(self):
        num_dist = len(self.distributions)
        max_params = max([p['points'].shape[1] for p in self.distributions.values()])
        labels = self._get_default_param_labels(max_params)

        fig, axes = plt.subplots(1, max_params, figsize=(max_params*4, num_dist * 0.4 + 2), sharey=True)
        if max_params == 1: axes = [axes]

        for p in range(max_params):
            ax = axes[p]
            for i, (name, dist) in enumerate(self.distributions.items()):
                if p >= dist['points'].shape[1]: continue

                ci = dist['stats']['cis'][p]
                color = dist['color'] or self.fallback_colors[i % len(self.fallback_colors)]
                ax.errorbar(ci['median'], i, xerr=[[ci['median'] - ci['lo2']], [ci['hi2'] - ci['median']]],
                            fmt='none', color=color, lw=1, alpha=0.3)
                ax.errorbar(ci['median'], i, xerr=[[ci['median'] - ci['lo1']], [ci['hi1'] - ci['median']]],
                            fmt='o', color=color, lw=3)

            if self.true_values and p < len(self.true_values):
                ax.axvline(self.true_values[p], color='gray', ls='--', alpha=0.6)
            ax.set_xlabel(f'${labels[p]}$')
            if p == 0:
                ax.set_yticks(range(num_dist))
                ax.set_yticklabels(list(self.distributions.keys()))
            ax.invert_yaxis()
        plt.tight_layout()
        return fig


class GriddedProbabilities(DistributionDiagnostic):
    """Diagnoses distributions defined on a regular N-D probability grid."""
    def __init__(self, coords_1d=None, **kwargs):
        super().__init__(**kwargs)
        self.coords_1d = coords_1d if coords_1d is not None else np.linspace(0, 1, 100)

    def add_distribution(self, grid, label=None, color=None):
        label = self._get_distribution_label(label)
        ndim = grid.ndim
        axes_coords = [self.coords_1d] * ndim
        mesh = np.meshgrid(*axes_coords, indexing='ij')
        points = np.vstack([m.flatten() for m in mesh]).T
        weights = grid.flatten()

        self.distributions[label] = {
            'grid': grid, 'points': points, 'weights': weights,
            'color': color,
            'stats': self._calculate_base_metrics(points, weights)
        }


class SampledDistribution(DistributionDiagnostic):
    """Diagnoses distributions represented as samples (e.g. MCMC chains, Monte Carlo draws)."""
    def add_distribution(self, samples, label=None, weights=None, color=None):
        label = self._get_distribution_label(label)
        if weights is None:
            weights = np.ones(len(samples))

        self.distributions[label] = {
            'points': samples, 'weights': weights,
            'color': color,
            'stats': self._calculate_base_metrics(samples, weights)
        }

