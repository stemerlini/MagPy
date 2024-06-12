import numpy as np
from scipy.interpolate import interp2d
import pkg_resources

IEAA_DATA_PATH = pkg_resources.resource_filename('MagPy.Ionisation',\
 'iaea_ionisation_tables/')


class IaeaTable:
    ''' Class to generate a function which relates average ionisation to plasma 
    density and temperature. Function interpolates over lookup tables of 
    average ionisation which were produced using the nLTE code FLYCHK. They 
    were downloaded from the database hosted at [1]. 
    
    The database includes the following caveat about the validity of tabulated 
    data:
        FLYCHK calculations are known to give better results for highly ionized
        plasmas and intermediate electron densities.
    
    Cite as:
        FLYCHK: Generalized population kinetics and spectral model for rapid 
        spectroscopic analysis for all elements, H.-K. Chung, M.H. Chen, 
        W.L. Morgan, Y. Ralchenko and R.W. Lee, 
        High Energy Density Physics, Volume 1, Issue 1, December 2005
    
    Data exists for all elements with atomic numbers in the range 1 (Hydrogen) 
    through to 79 (Gold). 
    
    ************************************************
    |  Access the generated function by calling    |
    |      self.model(Te, ne)                      |
    |  where:                                      |
    |      Te = electron temperature [eV]          |
    |      ne = electron density [cm^-3]           |
    ************************************************
        
    [1] - https://www-amdis.iaea.org/FLYCHK/ZBAR/csd014.php
    '''
    extension='.zvd'
    def __init__(self, AtomicNumber):
        ''' Description of arguments:
        1) AtomicNumber - atomic number of desired element (in the range [1,79]
            inclusive).
        '''
        dpath = IEAA_DATA_PATH + 'default/' + str(AtomicNumber) + self.extension
        ne = []
        Z = []
        Te = []
        with open(dpath, 'r') as f:
            lines = list(f)
            i = 0
            while(i<637):
                i += 1
                line = lines[i]
                ne.append( float(line[10:17]) ) 
                i += 11
                j=0
                Z_row = []
                while(j<36):
                    line = lines[i+j]
                    s=line.strip()
                    TT, ZZ = s.split()
                    Te.append( float(TT) )
                    Z_row.append( float(ZZ) )
                    j += 1
                Z.append(np.array(Z_row))
                i += 37
        self.Z = np.array(Z)
        self.ne = np.array(ne)
        self.Te = np.array(Te[0:36])
        self.model = interp2d(self.Te, self.ne, self.Z)


class IaeaTableMod:
    ''' Class to generate a function which relates average ionisation to plasma 
    density and temperature. Function interpolates over lookup tables of 
    average ionisation which were produced using the nLTE code FLYCHK.
    Tables generated to have a better curve resolution.
    
    ************************************************
    |  Access the generated function by calling    |
    |      self.model(Te, ne)                      |
    |  where:                                      |
    |      Te = electron temperature [eV]          |
    |      ne = electron density [cm^-3]           |
    ************************************************
        
    '''
    extension='.dat'
    def __init__(self, AtomicNumber, Model):
        ''' Description of arguments:
        1) AtomicNumber - atomic number of desired element (in the range [1,79]
            inclusive).
        2) Model - ss: non-lte steady state, lte: lte steady state. 'ss', 'lte'
        '''

        if Model == 'ss':
            dpath = IEAA_DATA_PATH + 'ss/'  + str(AtomicNumber) + self.extension
        elif Model == 'lte':
            dpath = IEAA_DATA_PATH + 'lte/' + str(AtomicNumber) + self.extension
        ne = []
        Z = []
        Te = []
        with open(dpath, 'r') as f:
            lines = list(f)
            i = 0
            while(i<45):
                i += 3
                line = lines[i]
                ne.append( float(line[8:14]) ) 
                i += 2
                j=0
                Z_row = []
                while(j < 20):
                    line = lines[i+j]
                    s=line.strip()
                    TT, ZZ = s.split()
                    Te.append( float(TT) )
                    Z_row.append( float(ZZ) )
                    j += 1
                Z.append(np.array(Z_row))
                i += 20
        self.Z = np.array(Z)
        self.ne = np.array(ne)
        self.Te = np.array(Te[0:20])
        self.model = interp2d(self.Te, self.ne, self.Z)