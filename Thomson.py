import numpy as np
from numpy import sqrt
import scipy.constants
import scipy.special

m_e=scipy.constants.m_e
m_p=scipy.constants.m_p
e=scipy.constants.e
c=scipy.constants.c
epsilon_0=scipy.constants.epsilon_0
exp=np.exp
Il=scipy.special.iv #modified bessel function of first kind




def S_k_omega(l, l0, theta, A, T_e, T_i, n_e, Z, \
    v_fi=0, v_fe=0):
    '''
    Returns a normalised spectral density function.
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temperatures
    Electron velocity is with respect to ion velocity
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in nm
    Theta is angle between k_in and k_s in degrees
    A i atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    V_fi and V_fe in m/s
    '''
    
    lambda_in = l0*1e-9
    lambda_range = l*1e-9
    
    #physical parameters
    pi=np.pi
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5
    
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    x_e=(omega/k - (v_fe+v_fi))/a 
    x_i=(omega/k-v_fi)/b
    lambda_De=7.43*(T_e/n_e)**0.5 #Debeye length in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) #ion susceptibility
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))#electron susceptibility
    epsilon=1+chi_e+chi_i#dielectric function
    fe0=1/(sqrt(pi)*a)*np.exp(-x_e**2)#electron Maxwellian function
    fi0=1/(sqrt(pi)*b)*np.exp(-x_i**2)#ion Maxwellian
    Skw=2*pi/k*(abs(1-chi_e/epsilon)**2*fe0+Z*abs(chi_e/epsilon)**2*fi0)
    S_norm = Skw/Skw.max() #normalise the spectrum
    return S_norm, alpha
    
