# Plasma Class - Create a plasma and calculate the relative parameters

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import scipy.constants as cons
from MagPy.Ionisation import IaeaTable, IaeaTableMod

class Plasma:
    def __init__(self, A, ANum, ne, Te, Ti, V, B, Z = None):
        '''
        Initialise a Plasma Object given the following parameters:
        Args:
            example:
            ---------------------------------------------------------------------------------------
            al_flow = {'A':27, 'ANum': 14, 'ne':1e18, 'Te': Te, 'Ti': Ti, 'V':4e6, 'B': 5, 'Z': Z}
            al=Plasma(**al_flow)
            ---------------------------------------------------------------------------------------
            A:      ion mass in nucleon masses
            ANum:   Atomic Number
            ne:     Electron Density in cm^-3
            Te      electron temperature in eV
            V:      velocity in cm/s
            B:      Magnetic Field [Tesla]
            Z:      if not provided use z_model based on Te and Ne (specify 'lte' or 'ss' to use the custom tables)

        '''
        self.A              =   A                                           # Atomic mass weight                                [gr/mol]
        self.ANum           =   ANum                                        # Atomic Number                                 
        self.ne             =   ne                                          # Electron Density [cm^-3]
        self.Te             =   Te                                          # Electron Temperature                               [Kg/m3]
        self.Ti             =   Ti                                          # Ion Temperature                                   [eV]
        self.V              =   V                                           # Bulk Velocity                                     [cm/s]
        self.B              =   B                                           # Magnetic Field                                    [Tesla T]
        
        if Z is None:
            # Estimate Ionisation Charge State - Z - from Tabled Values
            Z_mod               =   IaeaTable(self.ANum)
            if np.isscalar(self.Te) == True:
                self.Z        =   Z_mod.model(self.Te, self.ne)                   # Charge State for a given Te
            else:
                self.Z = []
                for i in range(len(self.Te)):
                    z = Z_mod.model(self.Te[i], self.ne[i])
                    self.Z.extend(z)
                self.Z = np.array(self.Z)
        elif Z == 'lte':
            Z_mod               =   IaeaTableMod(self.ANum, 'lte')
            if np.isscalar(self.Te) == True:
                self.Z        =   Z_mod.model(self.Te, self.ne)                   # Charge State for a given Te
            else:
                self.Z = []
                for i in range(len(self.Te)):
                    z = Z_mod.model(self.Te[i], self.ne[i])
                    self.Z.extend(z)
                self.Z = np.array(self.Z)
        elif Z == 'ss':
            Z_mod               =   IaeaTableMod(self.ANum, 'ss')
            if np.isscalar(self.Te) == True:
                self.Z        =   Z_mod.model(self.Te, self.ne)                   # Charge State for a given Te
            else:
                self.Z = []
                for i in range(len(self.Te)):
                    z = Z_mod.model(self.Te[i], self.ne[i])
                    self.Z.extend(z)
                self.Z = np.array(self.Z)
        else:
            self.Z = Z
        
        # -----------------------------------------------------------------
        # Density
        self.density        =    self.ne * self.A * cons.m_p * 1e3 / self.Z     # Mass Density   [gr/cm3]
        # Ion density
        self.ni             =   self.ne/self.Z                                  # Ion Density                                        [cm^-3]
        # Calculate Coulomb Log
        self.col_log_ei = self.CoulombLog()

        # Parameters
        self.speed()
        self.frequency()
        self.lengthScale()
        self.viscosity()
        self.resistivity()
        self.pressure()
        self.dimensionless()
        self.timing()
        self.conductivity()

    def CoulombLog(self):
        """
        method to calculate Coulomb Log:
        - Formulas taken from NRL formulary pg 34
        """

        m_e             =      cons.m_e                                                                 # Electron Mass          [Kg]
        m_i             =      self.A*cons.m_u

        X = self.Ti * (m_e/m_i)
        Y = 10*self.Z**2

        if np.isscalar(X) == False: 
            log_ei     =      []
            for i in range(len(X)):
                if ((self.Te[i] > X[i]) and (self.Te[i] < Y[i])): 
                    collog = 23-np.log(self.ne[i]**0.5*self.Z[i]*self.Te[i]**-1.5)  
                    log_ei.append(collog)
                elif (X[i] < Y[i]) and (Y[i] < self.Te[i]):
                    collog = 24-np.log(self.ne[i]**0.5*self.Te[i]**-1)
                    log_ei.append(collog)
                elif (self.Te[i] < X[i]*self.Z[i]):
                    collog = 30-np.log(self.ni[i]**0.5*self.Z[i]**2*self.Te[i]**-1.5/self.A[i])
                    log_ei.append(collog)
            log_ei = np.array(log_ei)
        else:
            if ((self.Te > X) and (self.Te < Y)): 
                log_ei = 23-np.log(self.ne**0.5*self.Z*self.Te**-1.5)  # see NRL formulary pg 34
            elif (X < Y) and (Y < self.Te):
                log_ei = 24-np.log(self.ne**0.5*self.Te**-1)
            elif (self.Te < X*self.Z):
                log_ei = 30-np.log(self.ni**0.5*self.Z**2*self.Te**-1.5/self.A)
       
        return log_ei

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

    def lengthScale(self):
        """
        Method to calculate main lenght scales

        [Using SI units Kelvin, m, Kg, s]

        """

        e               =      cons.e                                                                   # Elemental Charge       [C]
        epsilon_0       =      cons.epsilon_0                                                           # Vacuum Permittivity    [F m^-1]
        kb              =      cons.k                                                                   # Boltzmann Constant     [J K^-1]
        c               =      cons.c                                                                   # Light Speed            [m s^-1]

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
        else: 
            self.rho_e  = np.nan
            self.rho_i  = np.nan
        
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
        # self.visc       =    5e-6 * self.A**0.5 * self.Ti**2.5 / (self.Z**4 * self.density)        # [cm^2 s^-1] Rayleigh-Taylor in finaly structured medium, Ryuton 1996
        self.Lvisc      =    self.visc/self.V               # Viscous Length Scale  [cm] 

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
        m_i             =    self.A*cons.m_u                          # Ion Mass               [Kg]
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
        
        if self.B != 0:
            self.beta_th    =    self.P_th / self.P_B                                    # Thermal Beta
            self.beta_ram   =    self.P_ram / self.P_B                                   # Dynamic Beta
            self.M_A        =    self.V*1e-2 / self.V_A                                  # Alvenic Mach Number
        else:
            self.M_A        =   np.nan
            self.beta_ram   =   np.nan
            self.beta_th    =   np.nan

        self.M_S        =    self.V*1e-2 / self.V_S                                  # Sonic Mach Number
        self.M_SA       =    self.V*1e-2 / np.sqrt(self.V_A**2 + self.V_S**2)        # Magnitosonic Mach Number

    def timing(self):
        m_e             =      cons.m_e * 1e3                 # Electron Mass          [g]
        m_i             =      self.A*cons.m_u *1e3           # Ion Mass               [g]
        e               =      cons.e                         # Elemental Charge       [C]

        # equilibration time
        ni_ei = 1.8e-19 * (m_e * m_i)**0.5*self.Z**2*self.ne*self.col_log_ei / (m_e*self.Ti + m_i*self.Te)**1.5     # [s^-1]
        self.tau_eq  =  1/ni_ei     # Second [s]
        
        #collisional time ions and electrons
        self.tau_ei = 1/self.nu_ei
        self.tau_ie = 1/self.nu_ie

    def conductivity(self):
        # Thermal conductivity
        m_e             =      cons.m_e * 1e3                 # Electron Mass          [g]
        m_i             =      self.A*cons.m_u *1e3           # Ion Mass               [g]
        e               =      cons.e                         # Elemental Charge       [C]

        def ThermalCoefficient(Z):
            ## function retrieved from spreadsheet Thermal Conductivity 
            par     =   3.2132*Z**0.5697
            perp    =   4.6211*Z**-0.191
            return par, perp

        a_par, a_perp   = ThermalCoefficient(self.Z)
        
        self.xi_i_par        =   1.6e-12 * a_par * (self.ni * self.Ti * self.tau_ie / m_i)
        self.Dth_i_par   =   self.xi_i_par / self.ne
        
        if self.B == 0:
            self.xi_i_perp     =   np.nan
            self.Dth_i_perp    =   np.nan
            
        else:
            self.xi_i_perp     =   1.6e-12 * a_perp * (self.ni * self.Ti / ( self.om_ci**2 * self.tau_ie * m_i))
            self.Dth_i_perp    =   self.xi_i_perp / self.ne

        self.xi_e_par        =   1.6e-12 * a_par * (self.ne * self.Te * self.tau_ei / m_e)
        self.Dth_e_par       =   self.xi_e_par / self.ne


        if self.B == 0:
            self.xi_e_perp      =    np.nan
            self.Dth_e_perp     =    np.nan

        else: 
            self.xi_e_perp      =   1.6e-12 * a_perp * (self.ne * self.Te / ( self.om_ce**2 * self.tau_ei * m_e * 1e3 ))
            self.Dth_e_perp    =   self.xi_e_perp / self.ne
        

    def params(self):

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