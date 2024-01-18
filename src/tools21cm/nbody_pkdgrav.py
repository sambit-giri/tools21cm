"""
Scripts to use outputs from the pkdgrav (currently version 3) 
N-body simulations.
"""
import numpy as np 
from scipy.interpolate import splev, splrep
import pandas as pd

class ReaderPkdgrav3:
    def __init__(self, box_len, nGrid, 
            Omega_m=0.31, rho_c=2.77536627e11, verbose=True):
        self.box_len = box_len
        self.nGrid   = nGrid
        self.Omega_m = Omega_m
        self.rho_c   = rho_c 
        self.verbose = verbose

        self.fof_data = None
        self.pk_data  = None

    def dLightSpeedSim(self,dMpcUnit):
        """
        Find the speed of light in simulation units.

        c[Mpc/Gyr] = c[cm/s] * Julian Year[s] / pc[cm] * 1000 
        c_sim = c[Mpc/Gyr] * (x Gyrs/ 1 sim time) * ( 1 sim length/Boxsize (Mpc))
        x = 1/sqrt(4.498*h*h*2.776e-4)

        # Alternative version (Doug's version):
        # Cosmological coordinates
        # G     = 4.30172e-9 Mpc/M. (km/s)^2
        # rho_c = 3 H^2 / (8 pi G)
        # c     = 299792.458 km/s
        #
        # c_sim = c[km/s] * sqrt(Lbox / (G * rho_c * Lbox^3))
        #       = c[km/s] * sqrt(8 pi / (3 H^2 Lbox^2) )
        #       = c[km/s] * sqrt(8 pi / 3) / Lbox / H
        #       = c[km/s] * sqrt(8 pi / 3) / Lbox / h / 100
        # dMpcUnit given in Mpc/h gives:
        #       = 299792.458 * sqrt(8 pi / 3) / 100 / dMpcUnit
        #       = 8677.2079486362706 / dMpcUnit

        Parameters:
            dMpcUnit (float): The simulation length unit in h^-1 Mpc.

        Returns:
            float: The speed of light in simulation units.
        """
        return 8677.2079486362706 / dMpcUnit
    
    def read_particles(self, filename):
        tipsy = open(filename,'rb')
        header_type = np.dtype([('time', '>f8'),('N', '>i4'), ('Dims', '>i4'), ('Ngas', '>i4'), ('Ndark', '>i4'), ('Nstar', '>i4'), ('pad', '>i4')])
        gas_type  = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
                            ('rho','>f4'), ('temp','>f4'), ('hsmooth','>f4'), ('metals','>f4'), ('phi','>f4')])
        dark_type = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
                            ('eps','>f4'), ('phi','>f4')])
        star_type  = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
                            ('metals','>f4'), ('tform','>f4'), ('eps','>f4'), ('phi','>f4')])
        
        header = np.fromfile(tipsy,dtype=header_type,count=1)
        header = dict(zip(header_type.names,header[0]))
        gas  = np.fromfile(tipsy,dtype=gas_type,count=header['Ngas'])
        gas  = pd.DataFrame(gas,columns=gas.dtype.names)
        dark = np.fromfile(tipsy,dtype=dark_type,count=header['Ndark'])
        dark = pd.DataFrame(dark,columns=dark.dtype.names)
        star = np.fromfile(tipsy,dtype=star_type,count=header['Nstar'])
        star = pd.DataFrame(star,columns=star.dtype.names)
        tipsy.close()
        
        data = {}
        data['header'] = header
        data['gas']  = gas
        data['dark'] = dark
        data['star'] = star
        return data
    
class HaloCataloguePkdgrav3(ReaderPkdgrav3):

    def __init__(self, box_len, nGrid, 
            Omega_m=0.31, rho_c=2.77536627e11, verbose=True):
        super().__init__(box_len, nGrid, Omega_m, rho_c, verbose)

    def read_fof_data(self, filename, z=None, dtype=None):
        '''
        Read the FOF data.

        Parameters:
        - filename (str): The name of the data file.
        - z (float, optional): Redshift of the data. Defaults to None, which goes to 0.

        Returns:
        - numpy.ndarray: A structured array.
        '''
        BOX  = self.box_len
        GRID = self.nGrid
        OMEGA_MAT = self.Omega_m
        rho_c = self.rho_c

        dMassFac = rho_c * BOX**3
        dVelFac = 299792.458 / self.dLightSpeedSim(BOX)
        if z is not None:
            if isinstance(z, str): z = float(z)
            dVelFac *= (1 + z) ** 1
        else:
            print('Assuming the redshift to be 0, which affects the following quantities: dVelFac')

        self.dMassFac = dMassFac
        self.dVelFac  = dVelFac

        if dtype is None:
            dtype = np.dtype([
                ('rPot', '<f4', (3,)),
                ('minPot', '<f4'),
                ('rcen', '<f4', (3,)),
                ('rcom', '<f4', (3,)),
                ('vcom', '<f4', (3,)),
                ('angular', '<f4', (3,)),
                ('inertia', '<f4', (6,)),
                ('sigma', '<f4'),
                ('rMax', '<f4'),
                ('fMass', '<f4'),
                ('fEnvironDensity0', '<f4'),
                ('fEnvironDensity1', '<f4'),
                ('rHalf', '<f4'),
                # ('nBH', '<i4'),
                # ('nStar', '<i4'),
                # ('nGas', '<i4'),
                # ('nDM', '<i4'),
                # ('iGlobalGid', '<u8')
            ])

        data = np.fromfile(filename, dtype=dtype)
        self.fof_data_dtype = dtype
        self.fof_data = data
        if self.verbose: 
            print('The data FoF halo data read.')
        return data

    def array_fof_data(self, dtype=float):
        '''
        Create a numpy array of the data.

        Parameters:
        - dtype (type, optional): The data type of the array elements. Defaults to float.

        Returns:
        - numpy.ndarray: An array of data containing [mass, pos_x, pos_y, pos_z, particle_ID].
        '''
        BOX  = self.box_len
        data = self.fof_data 

        if data is not None:
            dMassFac = self.dMassFac
            dVelFac  = self.dVelFac

            str_data = []
            for g in data:
                pos = [(g['rPot'][j] + g['rcom'][j]) * BOX for j in range(3)]
                vel = [dVelFac * g['vcom'][j] for j in range(3)]
                fMass = dMassFac * g['fMass']
                if dtype in ['str','string',str]: 
                    # line = f"{fMass} {pos[0]:20.14f} {pos[1]:20.14f} {pos[2]:20.14f} {g['iGlobalGid']}"
                    line = f"{fMass} {pos[0]:20.14f} {pos[1]:20.14f} {pos[2]:20.14f}"
                else:
                    # line = np.array([fMass,pos[0],pos[1],pos[2],g['iGlobalGid']]).astype(dtype)
                    line = np.array([fMass,pos[0],pos[1],pos[2]]).astype(dtype)
                # print(line)
                str_data.append(line)
            str_data = np.array(str_data)
            if self.verbose:
                print('Returned array contains mass, pos_x, pos_y, pos_z.')
            return str_data 
        else:
            print('Data unread. Use the read_fof_data attribute.')
            return None

    def save_fof_data(self, savefile):
        str_data = self.array_fof_data(dtype='str')            
        np.savetxt(savefile, str_data)
        if self.verbose:
            print(f'The FoF halo data saved as {savefile}')

class PowerSpectrumPkdgrav3(ReaderPkdgrav3):

    def __init__(self, box_len, nGrid, 
            Omega_m=0.31, rho_c=2.77536627e11, verbose=True):
        super().__init__(box_len, nGrid, Omega_m, rho_c, verbose)

    def read_pk_data(self, filename, ks=None, window_size=None):
        pk_data = {}
        rd = np.loadtxt(filename)
        kk, pp = rd[:,0], rd[:,1]
        pk_data['k'] = kk 
        pk_data['P'] = pp 

        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines() if line.startswith('#')]
        pk_data['header'] = []
        for line in lines:
            if 'z=' in line:
                pk_data['z'] = float(line.split('z=')[-1])
            pk_data['header'].append(line)
        
        self.pk_data = pk_data

        if ks is not None:
            tck = splrep(np.log10(kk), np.log10(pp))
            pp  = 10**splev(np.log10(ks), tck)
            kk  = ks
        if window_size is not None:
            # Apply a simple moving average (adjust the window size as needed)
            data = pd.DataFrame({'x': np.log10(kk), 'y': np.log10(pp)})
            data['y_smoothed'] = data['y'].rolling(window=window_size, min_periods=1).mean()
            pp = 10**data['y_smoothed']
        return kk, pp
