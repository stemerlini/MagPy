# Plasma Class - Create a plasma and calculate the relative parameters

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import scipy.constants as cons
from MagPy.Ionisation import IaeaTable, IaeaTableMod

class Plasma:
    def __init__(self, A, ANum, ne, Te, Ti, V, B, Z = None, gamma = None):
        '''
        Initialise a Plasma Object given the following parameters:
        Args:
            example:
            ---------------------------------------------------------------------------------------
            al_flow = {'A':27, 'ANum': 14, 'ne':1e18, 'Te': Te, 'Ti': Ti, 'V':4e6, 'B': 5, 'Z': Z, gamma: 5/3}
            al=Plasma(**al_flow)
            ---------------------------------------------------------------------------------------
            A:      ion mass in nucleon masses
            ANum:   Atomic Number
            ne:     Electron Density in cm^-3
            Te      electron temperature in eV
            V:      velocity in cm/s
            B:      Magnetic Field [Tesla]
            Z:      if not provided use z_model based on Te and Ne (specify 'lte' or 'ss' to use the custom tables)
            gamma:  polytropic gamma value, if not specified gamma = 1

        '''
        self.A              =   A                                           # Atomic mass weight                                [gr/mol]
        self.ANum           =   ANum                                        # Atomic Number                                 
        self.ne             =   ne                                          # Electron Density [cm^-3]
        self.Te             =   Te                                          # Electron Temperature                              
        self.Ti             =   Ti                                          # Ion Temperature                                   [eV]
        self.V              =   V                                           # Bulk Velocity                                     [cm/s]
        self.B              =   B                                           # Magnetic Field                                    [Tesla T]
        
        if gamma is None:
            self.gamma          =   1
        else:
            self.gamma          =   gamma

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
        self.CoulombLog()
        # Parameters
        self.speed()
        self.frequency()
        self.lengthScale()
        self.viscosity()
        self.resistivity()
        self.pressure()
        self.timing()
        self.dimensionless()
        self.thermal_conductivityEH()
        # self.thermalconductivity()

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
            self.col_log_ei = np.array(log_ei)
        else:
            if ((self.Te > X) and (self.Te < Y)): 
                self.col_log_ei = 23-np.log(self.ne**0.5*self.Z*self.Te**-1.5)  # see NRL formulary pg 34
            elif (X < Y) and (Y < self.Te):
                self.col_log_ei = 24-np.log(self.ne**0.5*self.Te**-1)
            elif (self.Te < X*self.Z):
                self.col_log_ei = 30-np.log(self.ni**0.5*self.Z**2*self.Te**-1.5/self.A)
    

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
        self.V_S        =    np.sqrt(self.gamma * kb*(self.Z*T_e+T_i)/m_i)                              # Sound Speed               [m s^-1]
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
        self.om_pe      =    np.sqrt(e**2*n_e/(epsilon_0*m_e))                                          # Electron Plasma Frequency                       [rad s^-1]
        self.om_pi      =    np.sqrt(self.Z**2*e**2*n_i/(epsilon_0*m_i))                                # Ion Plasma Frequency                            [rad s^-1]

        # Collision Rate
        """Using CGS Units, eV, cm, g, s"""
        self.nu_ei      =    2.91e-6*self.Z*self.ne*self.col_log_ei*self.Te**-1.5                       # Collision Frequency: Electrons - Ions           [1/s] ref. NRL FUNDAMENTAL PLASMA PARAMETERS chapter does not include Z - refer to Braginskii
        self.nu_ie      =    4.80e-8*self.Z**4*self.A**-0.5*self.ni*self.col_log_ei*self.Ti**-1.5       # Collision Frequency: Ions - Electrons           [1/s] taken from near Maxwellian formulas

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
        
        if np.nonzero(self.B):
            self.rho_i  =    self.V_ti/self.om_ci                                                       # Ion Larmor Radius                               [m]
            self.rho_e  =    self.V_te/self.om_ce                                                       # Electron Larmor Radius                          [m]
            self.rho_e  *=  1e2
            self.rho_i  *=  1e2
        else: 
            self.rho_e  = np.nan
            self.rho_i  = np.nan
        
        self.mfp_e      =    self.V_te/self.nu_ei                                                       # thermal electron mean-free-path                         [m]
        self.mfp_i      =    self.V_ti/self.nu_ie                                                       # thermal Ion mean-free-path
        self.mfp_ii     =    self.V * 1e-2/self.nu_ie                                                          # velocity ion-ion mean-free-path (shock thickness)
        self.mfp_ee     =    self.V * 1e-2/self.nu_ei                                                          # velocity electron mean-free-path (shock thickness)
        """ Convert to CGS units """
        self.la_de      *=  1e2     # [m] --> [cm]
        self.delta_e    *=  1e2     # [m] --> [cm]
        self.delta_i    *=  1e2     # [m] --> [cm]
        self.mfp_i      *=  1e2     # [m] --> [cm]
        self.mfp_e      *=  1e2     # [m] --> [cm]
        self.mfp_ii     *=  1e2     # [m] --> [cm]                                                        
        self.mfp_ee     *=  1e2     # [m] --> [cm]

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

        self.sigma      =    n_e*e**2/(m_e*self.nu_ei)             # Electric Conductivity          [s kg^-3 m^-3 C^-3]
        self.Dm         =    1/(self.sigma*mu_0)                   # Magnetic Diffusivity  [m^2 s^-1]
        self.eta        =    self.Dm*mu_0                          # Electric Resistivity           [s kg^-3 m^-3 C^-3]^-1

        """ Convert to CGS units """
        self.Dm         =    self.Dm*1e4                            #  [m^2 s^-1] --> [cm^2 s^-1]
        self.Leta       =    self.Dm / self.V                       # Electric Resistive scale   [cm]

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
        
        if np.nonzero(self.B):
            self.beta_th    =    self.P_th / self.P_B                                    # Thermal Beta
            self.beta_ram   =    self.P_ram / self.P_B                                   # Dynamic Beta
            self.M_A        =    self.V*1e-2 / self.V_A                                  # Alvenic Mach Number
        else:
            self.M_A        =   np.nan
            self.beta_ram   =   np.nan
            self.beta_th    =   np.nan

        self.M_S        =    self.V*1e-2 / self.V_S                                  # Sonic Mach Number
        self.M_SA       =    self.V*1e-2 / np.sqrt(self.V_A**2 + self.V_S**2)        # Magnitosonic Mach Number
        self.omega_t_e  =    self.om_ce / self.nu_ei                                # omega tau electron
        self.omega_t_i  =    self.om_ci / self.nu_ie                                # omega tau ions


    def timing(self):
        m_e             =      cons.m_e * 1e3                 # Electron Mass          [g]
        m_i             =      self.A*cons.m_u *1e3           # Ion Mass               [g]
        e               =      cons.e                         # Elemental Charge       [C]
        

        # equilibration time
        self.ni_ei      =  1.8e-19 * (m_e * m_i)**0.5*self.Z**2*self.ne*self.col_log_ei / (m_e*self.Ti + m_i*self.Te)**1.5     # [s^-1]
        self.ni_ie      =  1.8e-19 * (m_i * m_e)**0.5*self.Z**2*self.ni*self.col_log_ei / (m_i*self.Te + m_e*self.Ti)**1.5     # [s^-1]
        self.tau_eq_ei  =  1/self.ni_ei     # Second [s]
        self.tau_eq_ie  =  1/self.ni_ie     # Second [s]
        
        #collisional time ions and electrons
        self.tau_ei = 1/self.nu_ei
        self.tau_ie = 1/self.nu_ie

    def thermal_conductivityEH(self):
        # Thermal conductivity - Epperlein_Haines 1985 (More accurate) - only electrons
        # Coefficients in the following table were computed assuming fully ionised plasma so (Atomic Number == Z_bar)
        # NB In our case, it might be more appropriate to use z_bar instead of atomic number!
        
        def near_ANum(Anum, Anum_arr):
            idx = np.argmin(np.abs(Anum_arr - Anum))
            # print('Calculating transport using Anum = {}'.format(Anum_arr[idx]))
            return idx

        ## Heat transport
        Anum_arr = np.array([1,2,3,4,5,6,7,8,10,12,14,60, 100])
        g0 = np.array([3.2, 4.93, 6.12, 7.00, 7.68,8.23, 8.69, 9.07, 9.67, 10.1, 10.5, 12.7, 13.58])
        gp0 = np.array([6.2, 9.3, 10.2, 9.1, 8.6, 8.6, 8.8, 7.9, 7.4, 7.3, 7.1, 6.4, 6.21])
        gp1 = np.array([4.7, 4.0, 3.7, 3.6, 3.5, 3.5, 3.5, 3.4, 3.4, 3.4, 3.4, 3.27, 3.25])
        cp0 = np.array([1.9, 1.9, 1.7, 1.3, 1.1, 1.0, 1.0, 0.9, 0.7, 0.7, 0.7, 0.5, 0.5])
        cp1 = np.array([2.3, 3.8, 4.8, 4.6, 4.6, 4.8, 5.2, 4.7, 4.6, 4.7, 4.6, 4.7, 4.8])
        cp2 = np.array([5.4, 7.8, 8.9, 8.8, 8.8, 9.0, 9.2, 8.8, 8.7, 8.7, 8.7, 8.5, 8.5])

        def kc_par(Anum, Anum_arr):
            idx = near_ANum(Anum, Anum_arr)
            return g0[idx]

        def kc_perp(chi, Anum, Anum_arr):
            idx = near_ANum(Anum, Anum_arr)
            return (gp1[idx] * chi + gp0[idx])/(chi**3 + cp2[idx] * chi**2 + cp1[idx] * chi +cp0[idx])

        """
        This function is in MKS.
        """
        m_e         =    cons.m_e           # Electron Mass          [Kg]
        e           =    cons.e             # Elemental Charge       [C]
        kb          =    cons.k             # Boltzmann Constant     [J K^-1]
        T_e         =    self.Te*e/kb       # Electron Temperature    [K]
        n_e         =    self.ne * 1e6      # Electron Density       [m^-3]    
        Anum        =    self.ANum          # Atomic mass
        chi         =    self.omega_t_e     # omega tau electron

        self.k_conv      =  kb * n_e * T_e/(m_e * self.nu_ei) # NRL p 37, 1/(m s)
        self.kc_perp     =  kc_perp(chi, self.Z, Anum_arr)
        self.kc_par      =  kc_par(self.Z, Anum_arr)
        self.par_to_perp =  self.kc_par/self.kc_perp

        self.k_perp      =  self.kc_perp * self.k_conv 
        self.k_par       =  self.kc_par * self.k_conv
        
        self.C_p         =  5/2*(n_e * (1 + 1/self.Z))  # heat capacity of electrons and ions
        self.Dth_perp    =  self.k_perp/self.C_p        # m^2/s
        self.Dth_par     =  self.k_par/self.C_p         # m^2/s



    def thermalconductivity(self):
        # Coefficient Braginskii
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
        self.Dth_i_par       =   self.xi_i_par / self.ne
        
        if np.nonzero(self.B) is False:
            self.xi_i_perp     =   np.nan
            self.Dth_i_perp    =   np.nan
        else:
            self.xi_i_perp     =   1.6e-12 * a_perp * (self.ni * self.Ti / ( self.om_ci**2 * self.tau_ie * m_i))
            self.Dth_i_perp    =   self.xi_i_perp / self.ne

        self.xi_e_par        =   1.6e-12 * a_par * (self.ne * self.Te * self.tau_ei / m_e)
        self.Dth_e_par       =   self.xi_e_par / self.ne

        if np.nonzero(self.B) is False:
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
        omegatau_i          =    'Magnatisation Ions (omega_tau_i) = ' + str(np.format_float_scientific(self.omega_t_i, precision = 1, exp_digits=2))       
        omegatau_e          =    'Magnatisation Electrons (omega_tau_e) = ' + str(np.format_float_scientific(self.omega_t_e, precision = 1, exp_digits=2))

        viscositykinem      =    'kinematic viscosity      =    '   +  str(np.format_float_scientific(self.visc, precision = 1, exp_digits=2))
        reynoldsnumber      =    'Reynolds Number          =    '   +  str(np.format_float_scientific(self.Re, precision = 1, exp_digits=2))
        mareynoldsnumber    =    'Magnetic Reynolds Number =    '   +  str(np.format_float_scientific(self.Re_m, precision = 1, exp_digits=2))
        mabeta              =    'Thermal Beta             =    '   +  str(np.format_float_scientific(self.beta_th, precision = 1, exp_digits=2))
        rambeta             =    'Dynamic Beta             =    '   +  str(np.format_float_scientific(self.beta_ram, precision = 1, exp_digits=2))

        
        electrontemperature =    'Electron Temperature     =    '   +  str(np.format_float_scientific(self.Te, precision = 1, exp_digits=2))        + ' [eV]'
        iontemperature      =    'Ion Temperature          =    '   +  str(np.format_float_scientific(self.Ti, precision = 1, exp_digits=2))        + ' [eV]'
        chargestate         =    'Charge State - Z         =    '   +  str(self.Z)

        txtstr              =   electrondensity + '\n' + ioninertiallength + '\n' + ionlarmorradius + '\n' + ionmeanfreepath + '\n' + elemeanfreepath + '\n' + collog + '\n' + ionplasmafrequency + '\n' + resistivescale + '\n' + sonicmachnumber + '\n' +  magneticmachnumber + '\n' + msonicmachnumber + '\n' + viscositykinem + '\n' + omegatau_i + '\n' + omegatau_e + '\n' + reynoldsnumber + '\n' + mareynoldsnumber + '\n' + mabeta + '\n' + rambeta + '\n' + electrontemperature + '\n' + iontemperature + '\n' + chargestate + '\n' + magneticdiff
        
        print(txtstr)