import numpy as np 
import astropy
from astropy import units

def _set_cosmo(cosmo):
    try:
        hlittle = cosmo.h
    except:
        cosmo_dict = {
            'PLANCK13': astropy.cosmology.Planck13,
            'PLANCK15': astropy.cosmology.Planck15,
            'PLANCK18': astropy.cosmology.Planck18,
            'WMAP1': astropy.cosmology.WMAP1,
            'WMAP3': astropy.cosmology.WMAP3,
            'WMAP5': astropy.cosmology.WMAP5,
            'WMAP7': astropy.cosmology.WMAP7,
            'WMAP9': astropy.cosmology.WMAP9, 
            }
        if cosmo.upper() in cosmo_dict.keys():
            cosmo = cosmo_dict[cosmo.upper()]
        else:
            print('{} cosmology not implemented and, therefore, using the default Planck18.')
            cosmo = astropy.cosmology.Planck18
        # hlittle = cosmo.h
    return cosmo 

def ParticleMass_to_ParticleNumber(Lbox, particle_mass, cosmo='Planck18'):
    cosmo = _set_cosmo(cosmo)
    if not isinstance(particle_mass, units.Quantity): 
        particle_mass *= units.Msun 
        print('As the particle mass provided is not using Astropy units, Msun is assumed here.')
    if not isinstance(Lbox, units.Quantity): 
        Lbox *= units.Mpc 
        print('As the box length provided is not using Astropy units, Mpc is assumed here.')
    particle_number = (np.cbrt(cosmo.critical_density0*cosmo.Om0/particle_mass)*Lbox).to('')
    print('Particles | along each direction = {} | total = {}'.format(particle_number,particle_number**3))
    return particle_number.value

def halo_mass_to_particle_number(Lbox, halo_mass, particles_in_halo, cosmo='Planck18'):
    particle_mass = halo_mass/particles_in_halo
    return particle_number_to_particle_mass(Lbox, particle_mass, cosmo=cosmo)

def particle_number_to_particle_mass(Lbox, particle_each_direction=None, particle_total=None, cosmo='Planck18'):
    particle_number = particle_each_direction if particle_each_direction is not None else np.cbrt(particle_total)
    cosmo = _set_cosmo(cosmo)
    if not isinstance(Lbox, units.Quantity): 
        Lbox *= units.Mpc 
        print('As the box length provided is not using Astropy units, Mpc is assumed here.')
    particle_mass = cosmo.critical_density0*cosmo.Om0*(Lbox/particle_number)**3
    return particle_mass.to('Msun')
