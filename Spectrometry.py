import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import pkg_resources

FIBER_KEY_PATH = pkg_resources.resource_filename('MagPy.Spectrometry',\
 'fiber_keys/')

class Spectrometer:
    @staticmethod
    def sliceSpectrum(array):
        '''1st row of ASCII file from  Andor spectrometer contains 
        wavelength in nm. This method transposes the input data
        and returns wavelength and spatially resolved spectrum as 
        separate arrays.  
        '''
        a = array.transpose()
        l = a[0]
        np.delete(a, 0)
        return a, l
    
    def __init__(self, path):
        ''' Initialises the spectrometer class. Argument is path to ASCII
        file from Andor spectrometer.     
        '''
        rawim = np.genfromtxt(path)
        self.im, self.l = self.sliceSpectrum(rawim)
    
    def get_index(self, wavelength):
        ''' Returns the index of the row which is closest to the specified 
        wavelength.  
        '''
        i = np.searchsorted(self.l, wavelength, side='left')
        return i
        
    def imshow(self, ax, **kwargs):
        ''' Wraps the pbcolormesh method from pyplot to render the spectrum 
        with wavelength on the x-axis. There's a glitch in this method where 
        the image is flipped up/down, when compared to the structure in the 
        datafile from the spectrograph. I am okay with this as this is the same
        way data is displayed in Solis. Be aware self.im needs to be flipped 
        with np.slice(axis=0) before slicing it up though.
        '''
        h,w = self.im.shape
        X = np.tile(self.l, (h, 1))
        y = range(h, 0, -1)
        Y = np.tile(y, (w,1)).transpose()
        ax.pcolormesh(X, Y, self.im, shading='nearest', **kwargs)
        ax.invert_yaxis()
    
class Bundle:
    ''' Class to handle spectra which are obtained by imaging a 
    linear array (bundle) of optical fibers onto the slit of the spectrograph.  
    '''
    def __init__(self, num_fibers=28, spacing=19, offset=8, disc_rows=5., \
    l0=530., l1=535., dark_wavelength = 540., dark_width= 1., \
    fib_key='DefaultFiberKey'):
        ''' Initialises the Bundle class. Description of arguments:
        1) num_fibers - number of fibers in the array
        2) spacing - spacing [in px] between fibers
        3) offset - spatial coordinate [in px] of the 1st fiber's upper limit
        3) disc rows - number of px to crop from the top & bottom edge of each 
            fiber
        4) l0 - leftmost region of spectra to carry forwards into further 
            analysis
        5) l1 - rightmost region of spectra to carry forwards into further 
            analysis
        6) dark_wavelength - leftmost edge of vertical strip of pixels used 
            to calculate dark count
        7) dark_width - width of the vertical strip of pixels used to calculate
            dark count
        8) Path to csv file containing list of fibers in the order they appear 
        on the CCD (Format is 'number, bundle'. There are no headings)
        '''
        self.num = num_fibers
        self.sp = spacing
        self.off = offset
        self.dis = disc_rows
        self.l0 = l0
        self.l1 = l1
        self.dl = dark_wavelength
        self.dw = dark_width
        fname = FIBER_KEY_PATH + fib_key + '.csv'
        self.f_key = self.read_key_file(fname)
    
    @staticmethod   
    def read_key_file(fname):
        ''' Read data from fiber key file and return as a list.
        '''
        key = []
        with open(fname, 'r') as f:
            for line in f:
                s = line.strip()
                num, bundle = s.split(',')
                num = int(num)
                key.append([bundle, num])
        return key
    
    def get_fiber_y_bounds(self):
        ''' Calculate the upper and lower y-coordinates of each fiber in 
        the spectrum. Returns 2 arguments:
        1) y0 - list of the uppermost ordinates of each fiber
        2) y1 - the lowermost ordinates of each fiver
        '''
        y0s = []
        y1s = []
        for i in range(self.num):
            y0 = self.off + self.dis + i*self.sp
            y1 = self.off - self.dis + (i+1)*self.sp
            
            y0 = int( round(y0) )
            y1 = int( round(y1) )
            
            y0s.append(y0)
            y1s.append(y1)
        return y0s, y1s
    
    def get_fiber_bundle_number(self, index):
        ''' Returns the bundle and key from the fiber at the position
        specified by index.
        '''
        bundle, num = self.f_key[index]
        return bundle, num
        
    def get_fiber_name(self, index):
        ''' Returns a string of the form 'BundleNumber' (i.e. 'A7') for the 
        fiber at the position specified by index.
        '''
        bundle, num = self.get_fiber_bundle_number(index)
        s = str(num) + bundle 
        return s
    
    @staticmethod
    def draw_rect(ax, x0, x1, y0, y1, c='r'):
        ''' Draws a rectangle to axis provided as an input. Description of 
        arguments:
        1) ax - axis to draw to
        2) x0... y1 - the ordinates which bound the rectangle
        3) c - the color of the rectangle
        '''
        tr = np.array((x0, y0))
        tl = np.array((x1, y0))
        bl = np.array((x1, y1))
        br = np.array((x0, y1))
        xy = np.array([tr, tl, bl, br])
        
        rect1 = Polygon(
            xy = xy,
            closed=True,
            color=c,
            fill=True,
            alpha=0.1,
        )
        rect2 = Polygon(
            xy = xy,
            closed=True,
            color='r',
            fill=False,
            alpha=1.0,
            ec='k',
            lw=0.75
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        
    def get_dark_bounds(self):
        ''' Get the bounds of the strip which is used to calculate
        dark count.
        '''
        x0 = self.dl
        x1 = self.dl + self.dw
        y0 = self.off
        y1 = self.off + self.num*self.sp
        return x0, x1, y0, y1

    def draw_split(self, spectrometer, vmin = 800, vmax=6000):
        ''' Draws details of the way data from the spectrograph is split, 
        to aid in debugging/analysis. 
        '''
        fig, axes = plt.subplots(1,2, figsize = (15,15), sharey=True)
        ax1, ax2 = axes
        spectrometer.imshow(ax1, cmap='Greys', vmin=vmin, vmax=vmax)       
        profile = spectrometer.im.sum(1)
        profile = 2*((profile / profile.max()) - 0.5)
        y_ord = range( len(profile), 0, -1 )
        ax2.plot(profile, y_ord, c='k')
        
        y0, y1 = self.get_fiber_y_bounds()
        y0 = np.array(y0)
        y1 = np.array(y1)
        Y = (y0+y1)/2
        for y in Y:
            ax2.axhline(y, 0, 1 , c='r', ls='--', lw=1)

        i = 0
        while(i<len(y0)):
            self.draw_rect(ax1, self.l0, self.l1, y0[i], y1[i])
            s = self.get_fiber_name(i)
            ax2.text(1.05, Y[i], s, c='r')
            i += 1
        x0, x1, y0, y1 = self.get_dark_bounds()
        self.draw_rect(ax1,  x0, x1, y0, y1, c='b')
        ax1.set_xlabel(r'$\lambda \; [nm]$')
        ax1.set_ylabel(r'Position [Px]')
        ax2.set_xlabel(r'Intensity [Arb]')
        
    def get_image_with_dark_count_subtracted(self, spectrometer):
        ''' Returns image from spectrometer with the dark count subtracted.
        Dark count is allowed to vary along the spatial axis and is calculated 
        from the spectral average of the region bounded by self.dl and 
        self.dl+self.dw. Depending on the spectral region taken to obtain 
        background count, this might also contain a contribution from plasma 
        self emission. Provided there's not much spectral variation in self 
        emission, this is okay -- but perhaps this method should be renamed?
        ''' 
        x0 = spectrometer.get_index(self.dl)
        x1 = spectrometer.get_index(self.dl + self.dw)
        dark_reg = spectrometer.im[:, x0:x1]
        dark_cnt = np.average(dark_reg, axis=1)
        sub_im = spectrometer.im.transpose() - dark_cnt.transpose()
        sub_im = np.clip(sub_im, a_min=0., a_max=None) # Set floor of array to
        return sub_im.transpose()                      # 0 to avoid -ive signal 
        
    def split(self, sh_spectrometer, bg_spectrometer):
        ''' Splits images from the 2 spectrometer objects into strips for each
        fiber, subtracts dark count and initialises the fiber objects, which 
        are stored as a dict which is a member var of self.
        '''
        y0, y1 = self.get_fiber_y_bounds()
        sx0 = sh_spectrometer.get_index(self.l0)
        sx1 = sh_spectrometer.get_index(self.l1)
        bx0 = bg_spectrometer.get_index(self.l0)
        bx1 = bg_spectrometer.get_index(self.l1)
        sim = self.get_image_with_dark_count_subtracted(sh_spectrometer)
        bim = self.get_image_with_dark_count_subtracted(bg_spectrometer)
        sim = np.flip(sim, 0) # See Spectrometer.imshow 
        bim = np.flip(bim, 0) # for explanation of this flip
        
        self.fibers = {}
        i = 0
        while(i<len(y0)):
            n = self.get_fiber_name(i)
            sh_im = sim[y0[i]:y1[i], sx0:sx1]
            bk_im = bim[y0[i]:y1[i], bx0:bx1]
            sl = sh_spectrometer.l[sx0:sx1]
            bl = bg_spectrometer.l[bx0:bx1]
            f = Spectrum(sh_im, bk_im, sl, bl, n)
            self.fibers[n]=f
            i += 1
        
class Spectrum:
    def __init__(self, sh_im, bk_im, sh_l, bk_l, name):

        ''' Initialises a fiber class. sh_im/bk_im are a (2D) slice of the 
        image from the CCD corresponding to an individual fiber. l is a (1D)
        slice of the wavelength array which aligns with im. There are separate 
        wavelength arrays for shot / background to account for the potential 
        of a different centre wavelength having been set in Solis. 
        
        Calculates intensity and sd in data from image using a line intagrated method
        along the spatial axis of the spectrum. Errors are calculated by simply taking 
        the squared root of the intensity.

        '''
        self.s_im = sh_im
        self.b_im = bk_im
        self.s_l = sh_l
        self.b_l = bk_l
        self.n = name

        self.s_y, self.s_yerr = self.intensity_and_std(self.s_im, 0)
        self.b_y, self.b_yerr = self.intensity_and_std(self.b_im, 0)
        self.response_params = None #Call self.fit_response to populate

        ## old version -- weigthed avarage intensity calculation
        # s_w = np.sum(self.s_im, 1)
        # s_w /= s_w.mean()
        # s_im_n = self.s_im / s_w[:, np.newaxis]
        # b_w = np.sum(self.b_im, 1)
        # b_w /= b_w.mean()
        # b_im_n = self.b_im / b_w[:, np.newaxis]
        # self.s_y, self.s_yerr = self.weighted_avg_and_std(s_im_n, s_w, 0)
        # self.b_y, self.b_yerr = self.weighted_avg_and_std(b_im_n, b_w, 0)
        # self.response_params = None #Call self.fit_response to populate
    
    @staticmethod
    def intensity_and_std(array, axis):
        """ Method to calculate TS Spectrum Intensity + relative error """
        mean = np.sum(array, axis = axis)
        sd   = np.sqrt(mean)
        return mean, sd

    @staticmethod
    def weighted_avg_and_std(array, weights, axis):
        ''' Method ripped straight from the answer by Eric O Lebigot here:
        stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        '''
        mean = np.average(array, weights=weights, axis=axis)
        var = np.average((array - mean)**2, weights=weights, axis=axis)
        sd = np.sqrt(var)
        return mean, sd
        
    @staticmethod
    def voigt_response(l, l0, sigma, A):
        ''' Returns a shifted voigt profile.
        '''
        dl = l-l0
        gamma=sigma
        y = voigt_profile(dl, sigma, gamma)
        y /= y.max()
        return y*A 
    
    def fit_response(self, approx_probe_wavelength=532.):
        ''' Fit a voigt profile to the background spectrum.
        '''
        y = self.b_y / self.b_y.max()
        
        p_opt, p_cov = curve_fit(self.voigt_response, self.b_l, y, \
            p0=[approx_probe_wavelength, 0.01, 1.])
        l0, s, A = p_opt
        self.response_params = {
            'l0': l0,
            'sigma': s,
            'gamma': s,
            'amp': A,
        }
        
    def get_response(self, l):
        ''' Returns an array representing the fitted response function, centred 
        on lambda_0. Takes one input, an array of wavelengths in nm.
        '''
        l0 = self.response_params['l0']
        s = self.response_params['sigma']
        g = self.response_params['gamma']
        A = self.response_params['amp']
        R = self.voigt_response(l, l0, s, A)
        return R
        
    def get_irf(self, l, w=5.):
        ''' Returns an array representing instrument response, in an array 
        which is symmetric about its centre point. Description of arguments:
        l - An array of wavelengths [nm]. This is used to get the interval
            between grid points (the response function does not live on this 
            grid)
        w - half-width of the spectral grid on which the response function is 
            represented (as a number of FWHMs for the Voigt)
        '''
        dl = l[1]-l[0]
        W = self.get_FWHM_of_response()*w
        grid=np.arange(-W/2., W/2., dl)
        l0 = self.response_params['l0']
        s = self.response_params['sigma']
        g = self.response_params['gamma']
        A = self.response_params['amp']
        R = self.voigt_response(grid, 0., s, A)
        norm_const = trapz(R, grid)
        return R / norm_const
        
    def get_FWHM_of_response(self):
        ''' Returns the FWHM (in nm) of the fitted response. Method taken from: 
        https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
        '''
        s = self.response_params['sigma']
        g = self.response_params['gamma']
        fg = 2.35482*s
        fl = 2.*g
        return 0.5346*fl + np.sqrt(0.2166*fl**2 + fg**2)
   
    def setup_convolution_grid(self, w_grid=0.1, w_kernal=5.):
        ''' Setup a spectral grid and a kernel to be stored internally to self, 
        and used for calculating the convolution of instrument response with 
        a simulated spectrum. Description of inputs:
            w_grid - spacing between grid points as a fraction of the FWHM of
                the fitted instrument response. Value should be <= 1. Smaller
                values are more accurate but convolutions will be slower.
            w_kernal - the width of the computed kernel for convolution, as a 
                multiple of the FWHM of the instrument response. Value should 
                be >= 1. Larger values will be more accurate but slower. 
        '''
        l0 = self.s_l[0]
        l1 = self.s_l[-1]
        dl = self.s_l[1]-l0
        W = w_grid*self.get_FWHM_of_response()
        self.subgrid_scale = int(np.ceil(dl/W)) #Number of grid 
        points_in_grid = self.subgrid_scale * self.s_l.shape[0]
        self.l = np.linspace(l0, l1, points_in_grid)
        self.kernal = self.get_irf(self.l, w_kernal)
        self.y_se = np.ones_like(self.l) #Used for adding const self emission
        self.y_stray = self.get_response(self.l)
        self.y_stray /= self.y_stray.max()
        
    def convolve_with_response(self, f, l):
        ''' Convolve the array specified as the 1st argument with fitted
        spectral response. The spectral grid is specified as the 2nd argument.
        '''
        kernal = self.get_irf(l)
        return convolve1d(f, kernal, mode='constant')
    
    def convolve_fixed_grid(self, f):    
        ''' Convolve the array specified as the 1st argument with the fitted 
        spectral response. The input array should live on the spectral grid 
        which is setup by the self.setup_convolution_grid method.
        '''
        convolved = convolve1d(f, self.kernal, mode='constant')
        return convolved
        
    def downsample_to_data_grid(self, f):
        ''' Downsamples an array expressed on the convolution grid to one
        expressed on the data grid.
        '''
        
        f_downsample = interp1d(self.l,f)
        return f_downsample(self.s_l)
        
    def get_probe_wavelength(self):
        ''' returns the probe wavelength. 
        '''
        return self.response_params['l0']