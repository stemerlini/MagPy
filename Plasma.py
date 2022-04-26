# Plasma Class - Create a plasma and calculate the relative parameters

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import scipy.constants as cons

class plasma:
    def __init__(self, A, ne, Te, Ti, V, B):
        '''
        Initialise a Plasma Object given the following parameters:
        Args:
            example:
            ---------------------------------------------------------------------------------------
            al_flow = {'A':27, 'Te': Te, 'Ti': Ti, 'V':4e6, 'L': 1, 'B': 5}
            al=Plasma(**al_flow)
            ---------------------------------------------------------------------------------------
            A:      ion mass in nucleon masses
            ne:     Electron Density in cm^-3
            Te      electron temperature in eV
            V:      velocity in cm/s
            L:      length scale in cm.
            Zsrc:   Charge State Model source, 'exp' or 'FLY'
            B:      Magnetic Field [Tesla]

        '''
        self.A              =   A                                           # Atomic mass weight                                [gr/mol]
        self.ne             =   ne                                          # Electron Density [cm^-3]
        self.Te             =   Te                                          # Electron Temperature                               [Kg/m3]
        self.Ti             =   Ti                                          # Ion Temperature                                   [eV]
        self.V              =   V                                           # Bulk Velocity                                     [cm/s]
        self.B              =   B                                           # Magnetic Field                                    [Tesla T]

        # Estimate Ionisation Charge State - Z - from Tabled Values
        Z_mod               =   self.ZTe()
        self.Z              =   Z_mod(self.Te)                                         # Charge State for a given Te
        # Density
        self.density        =    self.ne * self.A * cons.m_p * 1e3 / self.Z            # Mass Density 
        # Ion density
        self.ni             =   self.ne/self.Z                                         # Ion Density                                        [cm^-3]
        # Calculate Coulomb Log
        self.col_log_ei     =   self.CoulombLog()
        # Calculate Plasma Parameters
        self.params()

    def ZTe(self):
        """
        Method to return ZTe relation from FLY tabled values

        Requires:
        - Atomic Mass weight, A
        - .zvd tabled files

        return: function Z = z_mod(Te)
        """
        A        =  self.A
        root     =  './iaea_ionisation_tables/' 
        filepath =  root + self.A + '.zvd'
        T_e, Z =  np.genfromtxt(filepath, delimiter = '  ', usecols = [0,1], unpack = True)
        Z_mod = interp1d(T_e, Z)
        return Z_mod

    def CoulombLog(self):
        """
        method to calculate Coulomb Log
        # I need to add different cases
        """
        self.col_log_ei = 23-np.log(self.ne**0.5*self.Z*self.Te**-1.5)                                  # see NRL formulary pg 34

    def speed(self):
        """
        Method to calculate main Speeds:
        - Electron Thermal speed
        - Ions Thermal speed
        - Sound Speed
        - Alfven Speed

        [Using SI units, Kelvin, m, Kg, s]

        """

        # Scientific Constant
        m_e             =      cons.m_e                                                                 # Electron Mass          [Kg]
        m_i             =      self.A*cons.m_u                                                          # Ion Mass               [Kg]
        e               =      cons.e                                                                   # Elemental Charge       [C]
        mu_0            =      cons.mu_0                                                                # Vacuum Permeability    [N A^-2]
        epsilon_0       =      cons.epsilon_0                                                           # Vacuum Permittivity    [F m^-1]
        kb              =      cons.k                                                                   # Boltzmann Constant     [J K^-1]
        c               =      cons.c                                                                   # Light Speed            [m s^-1]
        
        T_e             =    self.Te*e/kb                                                               # Electron Temperature      [K]
        T_i             =    self.Ti*e/kb                                                               # Ion Temperature           [K]
        n_e             =    self.ne * 1e6                                                              # Electron Density, SI      [m^-3]
        n_i             =    n_e/self.Z                                                                 # Ion Density, SI           [m^-3]

        self.V_te       =    np.sqrt(kb*T_e/m_e)                                                        # Electron Thermal Speed    [m s^-1]                          
        self.V_ti       =    np.sqrt(kb*T_i/m_i)                                                        # Ion Thermal Speed         [m s^-1]    
        self.V_S        =    np.sqrt(kb*(self.Z*T_e+T_i)/m_i)                                           # Sound Speed               [m s^-1]
        self.V_A        =    np.sqrt(self.B**2/(mu_0*n_i*m_i))                                          # Alfven Speed              [m s^-1]

    def frequency(self):
        """
        Method to calculate main plasma frequencies
        [Using SI units, Kelvin, m, Kg, s]
        """
        m_e             =      cons.m_e                                                                 # Electron Mass          [Kg]
        m_i             =      self.A*cons.m_u                                                          # Ion Mass               [Kg]
        e               =      cons.e                                                                   # Elemental Charge       [C]
        epsilon_0       =      cons.epsilon_0                                                           # Vacuum Permittivity    [F m^-1]

        n_e             =    self.ne * 1e6                                                              # Electron Density, SI                            [m^-3]
        n_i             =    n_e/self.Z                                                                 # Ion Density, SI                                 [m^-3]

        self.om_ce      =    e*self.B/m_e                                                               # Electron Cyclotron frequency                    [rad s^-1]
        self.om_ci      =    self.Z*e*self.B/m_i                                                        # Ion Cyclotron frequency                         [rad s^-1]
        self.om_pe      =    np.sqrt(e**2*n_e/epsilon_0*m_e)                                            # Electron Plasma Frequency                       [rad s^-1]
        self.om_pi      =    np.sqrt(self.Z**2*e**2*n_i/(epsilon_0*m_i))                                # Ion Plasma Frequency                            [rad s^-1]

        # Collision Rate
        """Using CGS Units, eV, cm, g, s"""
        self.nu_ei      =    2.91e-6*self.Z*self.ne*self.col_log_ei*self.Te**-1.5                       # Collision Frequency: Electrons - Ions           [1/s] ref. NRL FUNDAMENTAL PLASMA PARAMETERS chapter
        self.nu_ie      =    4.80e-8*self.Z**4*self.A**-0.5*self.ni*self.col_log_ei*self.Ti**-1.5       # Collision Frequency: Ions - Electrons           [1/s]

    def lenghtScale(self):
        """
        Method to calculate main lenght scales

        [Using SI units Kelvin, m, Kg, s]

        """

        e               =      cons.e                                                                   # Elemental Charge       [C]
        epsilon_0       =      cons.epsilon_0                                                           # Vacuum Permittivity    [F m^-1]
        kb              =      cons.k                                                                   # Boltzmann Constant     [J K^-1]

        T_e             =    self.Te*e/kb                                                               # electron Temperature, kelvin                    [K]
        T_i             =    self.Ti*e/kb                                                               # ion Temperature, kelvin                         [K]
        n_e             =    self.ne * 1e6                                                              # Electron Density, SI                            [m^-3]

        self.la_de      =    np.sqrt(epsilon_0*kb*T_e/(n_e*e**2))                                       # Debye length                                    [m]      
        self.delta_i    =    c/self.om_pi                                                               # ion inertial length (ion skin depth)            [m]
        self.delta_e    =    c/self.om_pe                                                               # electron inertial length (electron skin depth)  [m]
        
        if self.B != 0:
            self.rho_i  =    self.V_ti/self.om_ci                                                       # Ion Larmor Radius                               [m]
            self.rho_e  =    self.V_te/self.om_ce                                                       # Electron Larmor Radius                          [m]
            self.rho_e  *=  1e2
            self.rho_i  *=  1e2
        
        self.mfp_e      =    self.V_te/self.nu_ei                                                       # electron mean-free-path                         [m]
        self.mfp_i      =    self.V_ti/self.nu_ie                                                       # Ion mean-free-path
        
        """ Convert to CGS units """
        self.la_de      *=  1e2     # [m] --> [cm]
        self.delta_e    *=  1e2     # [m] --> [cm]
        self.delta_i    *=  1e2     # [m] --> [cm]
        self.mfp_i      *=  1e2     # [m] --> [cm]
        self.mfp_e      *=  1e2     # [m] --> [cm]

    def viscosity(self):
        # Viscosity
        """
        Method to calculate Plasma Viscosity (Ryutov 99)
        Using CGS Units, eV, cm, g, s
        """
        self.visc       =    2e19*(self.Ti**2.5)/(self.col_log_ei*self.A**0.5*self.Z**3*self.ne)       # [cm^2 s^-1]

    def resistivity(self):
        """
        Method to calculate plasma resistivity and relative resistive scale
        [Using SI units Kelvin, m, Kg, s]
        """

        m_e             =      cons.m_e                                                                 # Electron Mass          [Kg]
        e               =      cons.e                                                                   # Elemental Charge       [C]
        mu_0            =      cons.mu_0                                                                # Vacuum Permeability    [N A^-2]

        n_e             =    self.ne * 1e6                         # Electron Density      [m-3]

        self.sigma      =    n_e*e**2/(m_e*self.nu_ei)             # Conductivity          [s kg^-3 m^-3 C^-3]
        self.Dm         =    1/(self.sigma*mu_0)                   # Magnetic Diffusivity  [m^2 s^-1]
        self.eta        =    self.Dm*mu_0                          # Resistivity           [s kg^-3 m^-3 C^-3]^-1

        """ Convert to CGS units """
        self.Dm         =    self.Dm*1e4                            #  [m^2 s^-1] --> [cm^2 s^-1]
        self.Leta       =    self.Dm/self.V                         # Electric Resistive scale   [cm]

    def pressure(self):
        """ 
        Method to calculate Pressures:
            -   Magnetic Pressure
            -   Thermal Pressure
            -   Ram Pressure
        
        [Using SI units Kelvin, m, Kg, s] 
        """

        kb              =    cons.k                                   # Boltzmann Constant     [J K^-1]
        mu_0            =    cons.mu_0                                # Vacuum Permeability    [N A^-2]
        e               =    cons.e                                   # Elemental Charge       [C]

        n_e             =    self.ne * 1e6                            # Electron Density       [m^-3]
        n_i             =    n_e/self.Z                               # Ion Density, SI        [m^-3]

        V               =    self.V*1e-2                              # Bulk Speed              [cm/s] --> [m/s]
        T_e             =    self.Te*e/kb                             # Electron Temperature    [K]
        T_i             =    self.Ti*e/kb                             # Ion Temperature         [K]

        self.P_B        =    self.B**2/(2*mu_0)                       # Magnetic Pressure       [N m^-2]
        self.P_th       =    n_i*kb*(self.Z*T_e+T_i)                  # Thermal Pressure        [N m^-2]
        self.P_ram      =    n_i*m_i*V**2                             # Ram Pressure            [N m^-2]

    def dimensionless(self, l = 1):
        """
        Calculate main dimensionless parameters given a characteristic spatial length
        input:
            - l: Characteristic Spatial Length
        [CGS]
        """
        self.l          =    l                                                       # Length [cm]
        self.HallNumber =    self.delta_i / (self.l*1e-2)                            # Hall Number     
        self.Re         =    self.l*self.V / self.visc                               # Reynolds Number                                 
        self.Re_m       =    self.l*self.V / self.Dm                                 # Magnetic Reynolds Number        
        self.beta_th    =    self.P_th / self.P_B                                    # Thermal Beta
        self.beta_ram   =    self.P_ram / self.P_B                                   # Dynamic Beta
        self.M_S        =    self.V*1e-2 / self.V_S                                  # Sonic Mach Number
        self.M_A        =    self.V*1e-2 / self.V_A                                  # Alvenic Mach Number
        self.M_SA       =    self.V*1e-2 / np.sqrt(self.V_A**2 + self.V_S**2)        # Magnitosonic Mach Number

    def printParams(self):

        #useful function tht really should be built in....rounds to n sig figs
        round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1)) 
        
        # Create print list
        electrondensity     =    'Electron Density         =    '   +  str(np.format_float_scientific(self.ne, precision = 1, exp_digits=2))         + ' [cm^-3]'
        ioninertiallength   =    'Ion Inertial Length      =    '   +  str(np.format_float_scientific(self.delta_i, precision = 1, exp_digits=2))    + ' [cm]'
        ionlarmorradius     =    'Ion Larmor Radius        =    '   +  str(np.format_float_scientific(self.rho_i, precision = 1, exp_digits=2))      + ' [cm]'
        ionplasmafrequency  =    'Ion Plasma Frequency     =    '   +  str(np.format_float_scientific(self.om_pi, precision = 1, exp_digits=2))      + ' [rad s^-1]'
        ionmeanfreepath     =    'Ion Mean-Free-Path       =    '   +  str(np.format_float_scientific(self.mfp_i, precision = 1, exp_digits=2))      + ' [cm]'
        elemeanfreepath     =    'Electron Mean-Free-Path  =    '   +  str(np.format_float_scientific(self.mfp_e, precision = 1, exp_digits=2))      + ' [cm]'
        collog              =    'Coulomb Logaritm         =    '   +  str(np.format_float_scientific(self.col_log_ei, precision = 1, exp_digits=2))  

        magneticdiff        =    'Magnetic Diffusivity     =    '   +  str(np.format_float_scientific(self.Dm, precision = 1, exp_digits=2))       + ' [cm^2/s]'
        resistivescale      =    'Resistive Scale          =    '   +  str(np.format_float_scientific(self.Leta, precision = 1, exp_digits=2))       + ' [cm]'
        sonicmachnumber     =    'Sonic Mach Number        =    '   +  str(np.format_float_scientific(self.M_S, precision = 1, exp_digits=2))
        magneticmachnumber  =    'Alfven Mach Number       =    '   +  str(np.format_float_scientific(self.M_A, precision = 1, exp_digits=2))
        msonicmachnumber    =    'Magnetosonic Mach Number =    '   +  str(np.format_float_scientific(self.M_SA, precision = 1, exp_digits=2))

        viscositykinem      =    'kinematic viscosity      =    '   +  str(np.format_float_scientific(self.visc, precision = 1, exp_digits=2))
        reynoldsnumber      =    'Reynolds Number          =    '   +  str(np.format_float_scientific(self.Re, precision = 1, exp_digits=2))
        mareynoldsnumber    =    'Magnetic Reynolds Number =    '   +  str(np.format_float_scientific(self.Re_m, precision = 1, exp_digits=2))
        mabeta              =    'Thermal Beta             =    '   +  str(np.format_float_scientific(self.beta_th, precision = 1, exp_digits=2))
        rambeta             =    'Dynamic Beta             =    '   +  str(np.format_float_scientific(self.beta_ram, precision = 1, exp_digits=2))


        electrontemperature =    'Electron Temperature     =    '   +  str(np.format_float_scientific(self.Te, precision = 1, exp_digits=2))        + ' [eV]'
        iontemperature      =    'Ion Temperature          =    '   +  str(np.format_float_scientific(self.Ti, precision = 1, exp_digits=2))        + ' [eV]'
        chargestate         =    'Charge State - Z         =    '   +  str(self.Z)

        txtstr              =   electrondensity + '\n' + ioninertiallength + '\n' + ionlarmorradius + '\n' + ionmeanfreepath + '\n' + elemeanfreepath + '\n' + collog + '\n' + ionplasmafrequency + '\n' + resistivescale + '\n' + sonicmachnumber + '\n' +  magneticmachnumber + '\n' + msonicmachnumber + '\n' + viscositykinem + '\n' + reynoldsnumber + '\n' + mareynoldsnumber + '\n' + mabeta + '\n' + rambeta + '\n' + electrontemperature + '\n' + iontemperature + '\n' + chargestate + '\n' + magneticdiff
        
        print(txtstr)