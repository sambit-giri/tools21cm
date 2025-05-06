import numpy as np
import matplotlib.pyplot as plt 
import os, sys 

class ReionizationHistory:
    def __init__(self):
        self.load_default_data()

    def struct_data_into_dict(self, z1, zl1, zh1, x1, u1, l1):
        return {
            'z_mean' : z1[x1.astype(bool)], 
            'z_l'    : zl1[x1.astype(bool)],
            'z_h'    : zh1[x1.astype(bool)],
            'xn_mean': x1[x1.astype(bool)],
            'xn_upp' : u1[x1.astype(bool)],
            'xn_low' : l1[x1.astype(bool)],
            'z_low_limit' : z1[~x1.astype(bool)*~u1.astype(bool)],
            'zl_low_limit' : zl1[~x1.astype(bool)*~u1.astype(bool)],
            'zh_low_limit' : zh1[~x1.astype(bool)*~u1.astype(bool)],
            'xn_low_limit': l1[~x1.astype(bool)*~u1.astype(bool)],
            'z_upp_limit' : z1[~x1.astype(bool)*~l1.astype(bool)],
            'zl_upp_limit' : zl1[~x1.astype(bool)*~l1.astype(bool)],
            'zh_upp_limit' : zh1[~x1.astype(bool)*~l1.astype(bool)],
            'xn_upp_limit': u1[~x1.astype(bool)*~l1.astype(bool)],
            }

    def append_constaints_data(self, data_dict, name='new'):
        key_list = ['z_mean', 'z_l', 'z_h', 
                'z_upp_limit', 'zl_upp_limit', 'zh_upp_limit'
                'z_low_limit', 'zl_low_limit', 'zh_low_limit', 
                'xn_mean', 'xn_upp', 'xn_low', 
                'xn_low_limit', 'xn_upp_limit']
        assert np.all([ke in key_list for ke in data_dict.keys()])
        self.constraint_data['new'] = data_dict
        print(f'The data dictionary added to constraint_data with {name} as key.')
        return None

    def load_default_data(self):
        # (McGreer et al. 2015)
        z1  = np.array([5.605, 5.903, 6.076])
        zl1 = np.array([5.605, 5.903, 6.076])
        zh1 = np.array([5.605, 5.903, 6.076])
        x1 = np.array([False,False,False])
        u1 = np.array([0.038,0.060,0.382])
        l1 = np.array([False,False,False])
        Lya_DarkFraction = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # (Mason et al. 2018a; Hoag et al. 2019; Mason et al. 2019)
        z1  = np.array([7.024,7.509,7.999,10.600])
        zl1 = np.array([6.507,6.992,7.510,10.401])
        zh1 = np.array([7.445,7.984,8.475,10.795])
        x1 = np.array([0.588, 0.878,False,False])
        u1 = np.array([0.699,0.981,False,0.880])
        l1 = np.array([0.443,0.832,0.757,False])
        Lya_EW = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # (Ouchi et al. 2010; Greig et al. 2016)
        z1  = np.array([6.604])
        zl1 = np.array([6.604])
        zh1 = np.array([6.604])
        x1 = np.array([False])
        u1 = np.array([0.501])
        l1 = np.array([False])
        Lya_clustering = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # (Davies et al. 2018)
        z1  = np.array([7.093,7.540])
        zl1 = np.array([7.093,7.540])
        zh1 = np.array([7.093,7.540])
        x1 = np.array([0.482,0.601])
        u1 = np.array([0.740,0.797])
        l1 = np.array([0.222,0.371])
        QSO_damping = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # Bruton S., Lin Y.-H., Scarlata C., Hayes M. J., 2023, ApJ, 949, L40
        Bruton2023 = {
            'Lya_DarkFraction': Lya_DarkFraction,
            'Lya_EW'          : Lya_EW,
            'Lya_clustering'  : Lya_clustering,
            'QSO_damping'     : QSO_damping,
        }

        # Morishita T., et al., 2023, ApJ, 947, L24
        # Jones G. C., et al., 2023, arXiv e-prints, p. arXiv:2306.02471
        # Bolan P., et al., 2022, MNRAS, 517, 3263
        z1  = np.array([7.000,7.880,6.700,7.600])
        zl1 = np.array([5.600,7.880,6.500,7.000])
        zh1 = np.array([7.500,7.880,6.900,8.200])
        x1 = np.array([0.700,False,False,0.830])
        u1 = np.array([0.900,False,0.440,0.940])
        l1 = np.array([0.500,0.450,False,0.620])
        Lya_EW = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # Hsiao T. Y.-Y., et al., 2023, arXiv e-prints, p. arXiv:2305.03042
        # Umeda H., Ouchi M., Nakajima K., Harikane Y., Ono Y., Xu Y., Isobe Y., Zhang Y., 2023, arXiv e-prints, p. arXiv:2306.00487
        z1  = np.array([10.170,7.120,7.440,8.280,10.28])
        zl1 = np.array([10.170,7.040,7.200,7.840,8.880])
        zh1 = np.array([10.170,7.180,7.780,8.690,11.40])
        x1 = np.array([False,0.540,0.690,0.920,0.940])
        u1 = np.array([False,0.670,0.990,1.000,1.000])
        l1 = np.array([0.900,0.000,0.310,0.360,0.530])
        Gal_damping = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # Jin X., et al., 2023, ApJ, 942, 59
        z1  = np.array([6.100,6.300,6.500,6.700])
        zl1 = np.array([6.100,6.300,6.500,6.700])
        zh1 = np.array([6.100,6.300,6.500,6.700])
        x1 = np.array([False,False,False,False])
        u1 = np.array([0.69+0.06,0.79+0.04,0.87+0.03,0.94+0.06])
        l1 = np.array([False,False,False,False])
        Lya_DarkFraction = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        # Wang F., et al., 2020, ApJ, 896, 23
        # Greig B., Mesinger A., Davies F. B., Wang F., Yang J., Hennawi J. F., 2022, MNRAS, 512, 5390
        z1  = np.array([7.000,7.090,7.540,7.290])
        zl1 = np.array([7.000,7.090,7.540,7.290])
        zh1 = np.array([7.000,7.090,7.540,7.290])
        x1 = np.array([0.700,0.440,0.310,0.490])
        u1 = np.array([0.70+0.20,0.44+0.23,0.31+0.18,0.49+0.11])
        l1 = np.array([0.70-0.23,0.44-0.24,0.31-0.19,0.49-0.11])
        QSO_damping = self.struct_data_into_dict(z1, zl1, zh1, x1, u1, l1)

        Keating2024 = {
            'Lya_DarkFraction': Lya_DarkFraction,
            'Lya_EW'          : Lya_EW,
            #'Lya_clustering' : Lya_clustering,
            'QSO_damping'     : QSO_damping,
            'Gal_damping'     : Gal_damping,
        }
        self.constraint_data = {
            'Bruton2023': Bruton2023, 
            'Keating2024': Keating2024,
            }
        return None
    
    def ax_plot(self, ax, data_dict, **kwargs):
        label = kwargs.get('label')
        ls = kwargs.get('ls', ' ')
        marker = kwargs.get('marker')
        markersize = kwargs.get('markersize')
        color = kwargs.get('color', 'Grey')
        yerr  = kwargs.get('yerr', 0.06) 
        xerr  = kwargs.get('xerr', 0.0) 
        try:
            # Constraint measurements
            xx = data_dict['z_mean'] 
            xl = xx-data_dict['z_l']
            xh = data_dict['z_h']-xx
            yy = data_dict['xn_mean'] 
            yl = yy-data_dict['xn_low'] 
            yu = data_dict['xn_upp']-yy
            ax.errorbar(xx, yy, xerr=[xl,xh], yerr=[yl,yu],
                ls=ls, marker=marker, markersize=markersize, 
                color=color, label=label)
            label = None
        except:
            pass
        try:
            # Upper limit
            xx = data_dict['z_upp_limit'] 
            xl = xx-data_dict['zl_upp_limit']
            xh = data_dict['zh_upp_limit']-xx
            yy = data_dict['xn_upp_limit'] 
            ax.errorbar(xx, yy, xerr=[xl,xh], yerr=yerr, uplims=True,
                ls=ls, marker=marker, markersize=markersize, 
                color=color, label=label)
            label = None
        except:
            pass
        try:
            # Lower limit
            xx = data_dict['z_low_limit'] 
            xl = xx-data_dict['zl_low_limit']
            xh = data_dict['zh_low_limit']-xx
            yy = data_dict['xn_low_limit'] 
            ax.errorbar(xx, yy, xerr=[xl,xh], yerr=0.06, lolims=True,
                ls=ls, marker=marker, markersize=markersize, 
                color=color, label=label)
            label = None
        except:
            pass
        return ax

def plot_with_error(ax, data_dict, **kwargs):
    ax = kwargs.get('ax')
    if ax is None: fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    label = kwargs.get('label')
    ls = kwargs.get('ls', ' ')
    marker = kwargs.get('marker')
    markersize = kwargs.get('markersize')
    color = kwargs.get('color', 'Grey')
    yerr  = kwargs.get('yerr', 0.06) 
    xerr  = kwargs.get('xerr', 0.0) 
    key_def = kwargs.get('key_def')
    if key_def is None: 
        key_def = {
            'xmean': 'xmean', 'xlow': 'xlow', 'xupp': 'xupp',
            'ymean': 'ymean', 'ylow': 'ylow', 'yupp': 'yupp',
            'xmean_upp_limit': 'xmean_upp_limit', 'y_upp_limit': 'y_upp_limit', 
            'xlow_upp_limit': 'xlow_upp_limit', 'xupp_upp_limit': 'xupp_upp_limit',
            'xmean_low_limit': 'xmean_low_limit', 'y_low_limit': 'y_low_limit', 
            'xlow_low_limit': 'xlow_low_limit', 'xupp_low_limit': 'xupp_low_limit',
        }
    try:
        # Constraint measurements
        xx = data_dict[key_def['xmean']] 
        xl = xx-data_dict[key_def['xlow']]
        xh = data_dict[key_def['xupp']]-xx
        yy = data_dict[key_def['ymean']] 
        yl = yy-data_dict[key_def['ylow']] 
        yu = data_dict[key_def['yupp']]-yy
        ax.errorbar(xx, yy, xerr=[xl,xh], yerr=[yl,yu],
            ls=ls, marker=marker, markersize=markersize, 
            color=color, label=label)
        label = None
    except:
        pass
    try:
        # Upper limit
        xx = data_dict[key_def['xmean_upp_limit']] 
        xl = xx-data_dict[key_def['xlow_upp_limit']]
        xh = data_dict[key_def['xupp_upp_limit']]-xx
        yy = data_dict[key_def['ymean_upp_limit']] 
        ax.errorbar(xx, yy, xerr=[xl,xh], yerr=yerr, uplims=True,
            ls=ls, marker=marker, markersize=markersize, 
            color=color, label=label)
        label = None
    except:
        pass
    try:
        # Lower limit
        xx = data_dict[key_def['xmean_low_limit']] 
        xl = xx-data_dict[key_def['xlow_low_limit']]
        xh = data_dict[key_def['xupp_low_limit']]-xx
        yy = data_dict[key_def['ymean_low_limit']] 
        ax.errorbar(xx, yy, xerr=[xl,xh], yerr=0.06, lolims=True,
            ls=ls, marker=marker, markersize=markersize, 
            color=color, label=label)
        label = None
    except:
        pass
    return ax
    
def compare_reion_hist(file_dict, saveplot=False):
    '''
    Compare simulation reionization history with available data.

    Parameters:
    - file_dict (dict): A dictionary of filenames or numpy array (column 0: redshift, column -1: mean ionization).
                        The keys as assumed to be the model names used in the plot legend.
    - saveplot (str, optional): The plot will be saved if a filename is provided. 
                                Default is False.

    Returns:
    - matplotlib.figure.Figure: The generated matplotlib figure.
    '''
    constraint  = ReionizationHistory()
    ax_plot     = constraint.ax_plot
    Bruton2023  = constraint.constraint_data['Bruton2023']
    Keating2024 = constraint.constraint_data['Keating2024']

    fig, ax = plt.subplots(1,1,figsize=(7,6))
    ax = ax_plot(ax, Bruton2023['Lya_DarkFraction'], marker='o', label=r'Ly$\alpha$ dark pixel fraction')
    ax = ax_plot(ax, Keating2024['Lya_DarkFraction'], marker='o', label=None)
    ax = ax_plot(ax, Bruton2023['Lya_EW'], marker='*', markersize=10, label=r'Ly$\alpha$ equivalent width')
    ax = ax_plot(ax, Keating2024['Lya_EW'], marker='*', markersize=10, label=None)
    ax = ax_plot(ax, Bruton2023['Lya_clustering'], marker='D', label=r'Ly$\alpha$ emitter clustering')
    ax = ax_plot(ax, Bruton2023['QSO_damping'], marker='s', label=r'Quasar damping wing')
    ax = ax_plot(ax, Keating2024['QSO_damping'], marker='s', label=None)
    ax = ax_plot(ax, Keating2024['Gal_damping'], marker='p', label=r'Galaxy damping wing')
    for ii,ke in enumerate(file_dict.keys()):
        pyc2ray_log = np.loadtxt(file_dict[ke]) if isinstance(file_dict[ke],str) else file_dict[ke]
        ax.plot(pyc2ray_log[:,0], 1-pyc2ray_log[:,-1], lw=3, c=f'C{ii}', label=f'{ke}')
    ax.grid()
    ax.axis([4.45,12.25,-0.1,1.1])
    ax.set_xlabel('$z$', fontsize=16)
    ax.set_ylabel('$x_\mathrm{HI}$', fontsize=16)
    ax.legend(loc=4)
    plt.tight_layout()
    if saveplot: plt.savefig('saveplot')
    plt.show()
    
    return fig

if __name__ == "__main__":

    if len(sys.argv)<2:
        print('Usage: python {} <Photoncount filename1> [Photoncount filename2] ...'.format(sys.argv[0]))
        sys.exit(1)

    constraint = ReionizationHistory()
    ax_plot = constraint.ax_plot
    Bruton2023 = constraint.constraint_data['Bruton2023']
    Keating2024 = constraint.constraint_data['Keating2024']

    fig, ax = plt.subplots(1,1,figsize=(7,6))
    ax = ax_plot(ax, Bruton2023['Lya_DarkFraction'], marker='o', label=r'Ly$\alpha$ dark pixel fraction')
    ax = ax_plot(ax, Keating2024['Lya_DarkFraction'], marker='o', label=None)
    ax = ax_plot(ax, Bruton2023['Lya_EW'], marker='*', markersize=10, label=r'Ly$\alpha$ equivalent width')
    ax = ax_plot(ax, Keating2024['Lya_EW'], marker='*', markersize=10, label=None)
    ax = ax_plot(ax, Bruton2023['Lya_clustering'], marker='D', label=r'Ly$\alpha$ emitter clustering')
    ax = ax_plot(ax, Bruton2023['QSO_damping'], marker='s', label=r'Quasar damping wing')
    ax = ax_plot(ax, Keating2024['QSO_damping'], marker='s', label=None)
    ax = ax_plot(ax, Keating2024['Gal_damping'], marker='p', label=r'Galaxy damping wing')
    for ii,ff in enumerate(sys.argv[1:]):
        pyc2ray_log = np.loadtxt(ff)
        ax.plot(pyc2ray_log[:,0], 1-pyc2ray_log[:,-1], lw=3, label=f'Model {ii+1}')
    ax.grid()
    ax.axis([4.45,12.25,-0.1,1.1])
    ax.set_xlabel('$z$', fontsize=16)
    ax.set_ylabel('$x_\mathrm{HI}$', fontsize=16)
    ax.legend(loc=4)
    plt.tight_layout()
    # plt.savefig('reion_hist.pdf')
    plt.show()