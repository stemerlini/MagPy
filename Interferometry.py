import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from copy import copy


# The following code is taken from Thomas Varnish (twv17@ic.ac.uk), it was 
# written as part of work carried out during a UROP on MAGPIE in 2020.
# This version was "forked" from a notebook hosted at:
# https://github.com/jdhare/magpie_tools

def gaussian_beam(x, y, x0, y0, A, B, F, c, alpha, ravel_output=False):
    alpha = np.radians(alpha)
    xc, yc = x - x0, y - y0
    xr = xc * np.cos(alpha) + yc * np.sin(alpha)
    yr = xc * np.sin(alpha) - yc * np.cos(alpha)
    z = F * np.exp(-(np.power(xr/A, 2) + np.power(yr/B, 2))) + c
    if ravel_output is True:
        return z.ravel()
    else:
        return z

# Synthetic interferogram using cosine
def interferogram(z, traced_both=True):
    if traced_both:
        return np.cos(z * np.pi) ** 2
    else:
        return np.cos(z * np.pi * 2) ** 2

def mask_like(img, masked):
    mask = np.ma.masked_invalid(masked)
    masked_array = np.ma.masked_array(data=img, mask=np.ma.getmask(mask), \
    fill_value=np.nan)
    return masked_array.filled(np.nan)
        
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
def nice_colorbar(ax, size="5%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    return cax

class Reconstruction:
    '''
        Class to reconstruct a background interferogram by fitting a specified 
        function to zero fringes in a shot phase map.
    '''
    def __init__(self, function, initial_params, dont_vary=[]):
        self.function = function
        self.initial_params = initial_params
        
        self.model = Model(self.function, independent_vars=['x', 'y'], \
        nan_policy="omit", kws={"ravel_output": True})
        
        self.params = self.model.make_params()
        
        for p in self.initial_params:
            if p == "alpha":
                self.params[p].set(self.initial_params[p], min=0, max=90, \
                vary=("alpha" not in dont_vary))
            else:
                self.params[p].set(self.initial_params[p], \
                vary=(p not in dont_vary))
            
    def _flattened_function(self, **kwargs):
        return self.function(**kwargs).ravel()        
    
    def fit(self, X, Y, shot, masked_shot, max_nfev=None, xtol=None, \
    ftol=None):        
                
        fit_kws = {}
        if xtol is not None:  # Relative error in the approximate solution.
            fit_kws["xtol"] = xtol
        if ftol is not None:  # Relative error  in the sum of squares.
            fit_kws["ftol"] = ftol
        
        # Run the fitting routine
        self.fit_result = self.model.fit(masked_shot.ravel(), self.params, 
                                         x=X.ravel(), y=Y.ravel(), 
                                         max_nfev=int(max_nfev),
                                         fit_kws=fit_kws)
        print(self.fit_result.fit_report())
        
        # Generate our full result from the best parameter values
        fitted = self.function(X, Y, **self.fit_result.best_values)
        masked_fit = mask_like(img=fitted, masked=shot)
        
        return masked_fit, fitted, self.fit_result.best_values