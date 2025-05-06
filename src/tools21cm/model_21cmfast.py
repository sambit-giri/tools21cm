import numpy as np
from time import time 

def _is_py21cmfast_installed():
    try:
        import py21cmfast as p21c
        return p21c
    except:
        print('Install py21cmfast (https://21cmfast.readthedocs.io/) to model reionization with 21cmFAST.')
        return None

def run_21cmfast_init(
                    user_params={"HII_DIM":32, "DIM":32*3, "BOX_LEN":128, "USE_INTERPOLATION_TABLES": True},
                    cosmo_params={"OMb":0.049, "OMm":0.31, "POWER_INDEX":0.96, "SIGMA_8":0.83, "hlittle":0.67},
                    write=False,
                    direc=None,
                    random_seed=42,
                    regenerate=False,
                    verbose=True,
                    **global_kwargs):
    p21c = _is_py21cmfast_installed()
    if p21c is None: return None
    tstart = time()
    if verbose: print('Creating initial conditions...')
    ic = p21c.initial_conditions(
                    user_params=user_params,
                    cosmo_params=cosmo_params,
                    write=write,
                    direc=direc,
                    random_seed=random_seed,
                    regenerate=regenerate,
                    **global_kwargs)
    if verbose: print(f'...done in {time()-tstart:.1f} s')
    return ic 
    
def run_21cmfast_matter(redshift,
                        init_box=None,
                        user_params={"HII_DIM":32, "DIM":32*3, "BOX_LEN":128, "USE_INTERPOLATION_TABLES": True},
                        cosmo_params={"OMb":0.049, "OMm":0.31, "POWER_INDEX":0.96, "SIGMA_8":0.83, "hlittle":0.67},
                        random_seed=42,
                        regenerate=False,
                        write=False,
                        direc=None,
                        verbose=True,
                        **global_kwargs):
    p21c = _is_py21cmfast_installed()
    if p21c is None: return None
    tstart = time()
    if init_box is None:
        if verbose: print('Creating initial conditions...')
        init_box = p21c.initial_conditions(
                        user_params=user_params,
                        cosmo_params=cosmo_params,
                        write=write,
                        direc=direc,
                        random_seed=random_seed,
                        regenerate=regenerate,
                        **global_kwargs)
        if verbose: print(f'...done in {time()-tstart:.1f} s')
    if verbose: print('Creating matter distribution...')
    fieldz = lambda z: p21c.perturb_field(redshift=z,
                        init_boxes=init_box,
                        **global_kwargs)
    if verbose: print(f'...done in {time()-tstart:.1f} s')
    out = {zi: fieldz(zi) for ii,zi in enumerate(redshift)}
    return out

def run_21cmfast_coeval(redshift,
                        user_params={"HII_DIM":32, "DIM":32*3, "BOX_LEN":128, "USE_INTERPOLATION_TABLES": True},
                        cosmo_params={"OMb":0.049, "OMm":0.31, "POWER_INDEX":0.96, "SIGMA_8":0.83, "hlittle":0.67},
                        astro_params={"F_STAR10":np.log10(0.05), "ALPHA_STAR":0.5, "F_ESC10":np.log10(0.1), "ALPHA_ESC":-0.5, "t_STAR":0.5, "M_TURN":8.7, "R_BUBBLE_MAX":15, "L_X":40},
                        flag_options={"USE_HALO_FIELD":False, "USE_MASS_DEPENDENT_ZETA":True, "INHOMO_RECO":False, "PHOTON_CONS":False},
                        regenerate=False,
                        write=False,
                        direc=None,
                        init_box=None,
                        perturb=None,
                        use_interp_perturb_field=False,
                        pt_halos=False,
                        random_seed=42,
                        verbose=True,
                        **global_kwargs):
    p21c = _is_py21cmfast_installed()
    if p21c is None: return None
    tstart = time()
    if isinstance(redshift,(int,float)): redshift = [float(redshift)]
    if isinstance(redshift,(np.ndarray)): redshift = list(redshift)
    if verbose: print('Modelling reionization...')
    coevals = p21c.run_coeval(redshift=redshift,
                        user_params=user_params,
                        cosmo_params=cosmo_params,
                        astro_params=astro_params,
                        flag_options=flag_options,
                        regenerate=regenerate,
                        write=write,
                        direc=direc,
                        init_box=init_box,
                        perturb=perturb,
                        use_interp_perturb_field=use_interp_perturb_field,
                        pt_halos=pt_halos,
                        random_seed=random_seed,
                        **global_kwargs)
    if verbose: print(f'...done in {time()-tstart:.1f} s')
    out = {zi: coevals[ii] for ii,zi in enumerate(redshift)}
    return out


