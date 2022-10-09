# Implement Refractometry class - derived Image class
## work in progress

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.transform as sk_t
from skimage.measure import profile_line
from matplotlib.patches import Polygon
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from lmfit.models import GaussianModel, ConstantModel




class Refractometer:
    def __init__(self, image, scale_x, scale_phi, multiply_by=1, rotate=0):
        """

        Initialises the refractometer class. 
        Arguments are:
         1) image - greyscale image represented as numpy array
         2) rotate - rotation to be applied to image in degrees
         3) scale_x - image scale in pixels per mm
         4) scale_ϕ - image scale in pixels per mrad

        """
        # y axis and z axis must be rescaled based on tow different px to mm values

        self.im         =   sk_t.rotate(image, rotate, resize=False)
        self.im         *=  multiply_by
        self.sc         =   scale_x
        self.scale_phi  =   scale_phi
        self.o          =   np.array([0., 0.])
        self.shape      =   image.shape
        self.r          =   rotate

    def px_to_mm(self, p_px):
        '''Calculates position of point in mm, given position in px'''
        h = self.shape[0]
        p = np.array(p_px, dtype=np.float64)
        p *= np.array([1., -1.])    #Convert handedness 
        p += np.array([0., h])      #Translate origin to BL corner
        p[0] = p[0] / self.sc
        p[1] = p[1] / self.scale_phi
        p -= self.o
        return p
    
    def mm_to_px(self, p_mm):
        '''Calculates position of point in px, given position in mm'''
        h = self.shape[0]
        p = np.array(p_mm)
        p += self.o
        p[0] = p[0] * self.sc
        p[1] = p[1] * self.scale_phi
        p *= np.array([1., -1.]) #Convert handedness 
        p += np.array([0., h]) #Translate origin to TR corner
        return np.array(p, dtype=np.int64)

    def set_origin(self, p_px):
        '''Sets origin of image from value in pixels'''
        self.o  = np.array([0., 0.])
        p_mm = self.px_to_mm(p_px)
        self.o = p_mm
        self.o_px = p_px
        
    def get_origin(self):
        '''Returns the position of the origin in pixels'''
        o = np.array([0., 0.])
        o_px = self.mm_to_px(o)
        return o_px

class Signal:
    def __init__(self, num = 28, spacing=100, offset=0, disc_rows=0, l0=500., l1=2000., dark_wavelength = 0., dark_width= 100.):
        
        ''' Initialises the Signal class. Description of arguments:

            1) spacing          -   spacing [in px] between segments
            2) offset           -   spatial coordinate [in px] of the 1st segment
            3) disc rows        -   number of px to crop from the top & bottom edge of each segment
            4) l0               -   leftmost region of spectra to carry forwards into further analysis
            5) l1               -   rightmost region of spectra to carry forwards into further analysis
            6) dark_wavelength  -   leftmost edge of vertical strip of pixels used to calculate dark count
            7) dark_width       -   width of the vertical strip of pixels used to calculate dark count
        '''

        self.num = num
        self.sp = spacing
        self.off = offset
        self.dis = disc_rows
        self.l0 = l0
        self.l1 = l1
        self.dl = dark_wavelength
        self.dw = dark_width

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

    def get_dark_bounds(self):
        ''' Get the bounds of the strip which is used to calculate
        dark count.
        '''
        x0 = self.dl
        x1 = self.dl + self.dw
        y0 = self.off
        y1 = self.off + self.num*self.sp
        return x0, x1, y0, y1
    
    def get_image_with_dark_count_subtracted(self, spectrometer):
        ''' Returns image from spectrometer with the dark count subtracted.
        Dark count is allowed to vary along the spatial axis and is calculated 
        from the spectral average of the region bounded by self.dl and 
        self.dl+self.dw. Depending on the spectral region taken to obtain 
        background count, this might also contain a contribution from plasma 
        self emission. Provided there's not much spectral variation in self 
        emission, this is okay -- but perhaps this method should be renamed?
        ''' 
        x0 = self.dl
        x1 = self.dl + self.dw
        dark_reg = spectrometer.im[:, x0:x1]
        dark_cnt = np.average(dark_reg, axis=1)
        sub_im = spectrometer.im.transpose() - dark_cnt.transpose()
        sub_im = np.clip(sub_im, a_min=0., a_max=None) # Set floor of array to
        return sub_im.transpose()                      # 0 to avoid -ive signal 

    def draw_spectrum(self, sh, bk, vmin = 400, vmax=900):
        ''' Draws details of the way data from the refractometer, 
        '''

        self.sh = sh
        self.bk = bk

        fig, axCenter = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(.05,.1,.95,.95)

        divider = make_axes_locatable(axCenter)
        axvert = divider.append_axes('right', size='30%', pad=0.5)
        axhoriz = divider.append_axes('top', size='20%', pad=0.25)

        axCenter.imshow(self.sh.im, cmap='Greys', extent = [0,self.sh.shape[1], self.sh.shape[0], 0], vmin=vmin, vmax=vmax)
        rect = plt.Rectangle((0,-12),-0.8, 24, linewidth = 2, edgecolor = 'black', hatch = "///", facecolor='white')
        axCenter.add_patch(rect)
        
        # Sum vertically
        profile_vert  =     self.sh.im.sum(0)
        profile_vert  =     (profile_vert - profile_vert.min() ) / (profile_vert.max() - profile_vert.min())
        response_vert =     self.bk.im.sum(0)
        response_vert =     (response_vert - response_vert.min() ) / (response_vert.max() - response_vert.min())
        xvert         =     range(0, len(profile_vert), 1)
        
        axhoriz.plot(xvert, profile_vert, c = 'k')
        axhoriz.plot(xvert, response_vert, c = 'green', alpha = 0.4)

        axhoriz.set_ylim([0,1])
        axhoriz.set_xlim([0,self.sh.shape[1]])

        # Sum horizontally
        profile_hor     =   self.sh.im.sum(1)
        profile_hor     =   (profile_hor - profile_hor.min() ) / (profile_hor.max() - profile_hor.min())
        response_hor    =   self.bk.im.sum(1)
        response_hor    =   (response_hor - response_hor.min() ) / (response_hor.max() - response_hor.min())
        yhoriz          =   range(len(profile_hor), 0 , -1)

        axvert.plot(profile_hor, yhoriz, c = 'k', zorder = 2)
        
        axvert.plot(response_hor, yhoriz, c = 'green', alpha = 0.2)
        axvert.fill_between(response_hor, yhoriz, alpha=0.2)


        axvert.set_xlim([0,1])
        axvert.set_ylim([self.sh.shape[0], 0])

        axhoriz.margins(x=0)
        axvert.margins(y=0)

        # -------------------------------------------------

        y0, y1 = self.get_fiber_y_bounds()
        y0 = np.array(y0)
        y1 = np.array(y1)
        Y = (y0+y1)/2
        for y in Y:
            axhoriz.axvline(y, 0, 1 , c='r', ls='--', lw=1)

        i = 0
        while(i<len(y0)):
            # self.draw_rect(axCenter, y0[i], y1[i], self.l0, self.l1)
            axhoriz.text(Y[i], 0.8, str(i), c='r', fontsize = 'small')
            i += 1
        
        # x0, x1, y0, y1 = self.get_dark_bounds()
        # self.draw_rect(axCenter, y0, y1, x0, x1, c='b')

        axCenter.set_ylabel(r'Spectrum [Px]')
        axCenter.set_xlabel(r'Position [Px]')
        axCenter.set_xlim([0,self.sh.shape[1]])
        axCenter.set_ylim([self.sh.shape[0],0])
        axvert.set_xlabel(r'Intensity [Arb]')

    def split(self, sh_refractometer, bk_refractometer, darkCount = False):

        ''' Splits images from the objects into strips for each
        segment, subtracts dark count and initialises the segment objects, which 
        are stored as a dict which is a member var of self.
        '''

        y0, y1  =    self.get_fiber_y_bounds()
        sx0     =    self.l0
        sx1     =    self.l1
        bx0     =    self.l0
        bx1     =    self.l1
        if darkCount:
            sim     =    self.get_image_with_dark_count_subtracted(sh_refractometer)
            bim     =    self.get_image_with_dark_count_subtracted(bk_refractometer)
        else:
            sim     =    sh_refractometer.im
            bim     =    bk_refractometer.im

        # sim = np.flip(sim, 0) # See Spectrometer.imshow 
        # bim = np.flip(bim, 0) # for explanation of this flip
        
        self.segments = {}
        i = 0
        while(i<len(y0)-1):
            n = str(i)
            sh_im = sim[sx0:sx1, y0[i]:y1[i]]
            bk_im = bim[bx0:bx1, y0[i]:y1[i]]
            sh_l  = np.linspace(sx0, sx1, len(sh_im[:,1]))
            bk_l  = np.linspace(bx0, bx1, len(bk_im[:,1]))
            sh_l_rad  = (sh_l - (sh_l.max() + sh_l.min()) / 2) / sh_refractometer.scale_phi
            bk_l_rad  = (bk_l - (bk_l.max() + bk_l.min()) / 2) / bk_refractometer.scale_phi
            f     = Spectrum(sh_im, bk_im, sh_l, bk_l, sh_l_rad, bk_l_rad, n)
            self.segments[n]=f
            i += 1

class Spectrum:
    def __init__(self, sh_im, bk_im, s_l, b_l, s_l_rad, b_l_rad, name):

        ''' Initialises a fiber class. sh_im/bk_im are a (2D) slice of the 
        image from the CCD corresponding to an individual fiber. l is a (1D)
        slice of the wavelength array which aligns with im. There are separate 
        wavelength arrays for shot / background to account for the potential 
        of a different centre wavelength having been set in Solis. 
        
        Calculates intensity and sd in data from image using a line intagrated method
        along the spatial axis of the spectrum. Errors are calculated by simply taking 
        the squared root of the intensity.

        '''
        self.s_im       =   sh_im
        self.b_im       =   bk_im
        self.s_l        =   s_l
        self.b_l        =   b_l
        self.s_l_rad    =   s_l_rad
        self.b_l_rad    =   b_l_rad
        self.n          =   name

        self.s_y, self.s_yerr = self.intensity_and_std(self.s_im, 1)
        self.b_y, self.b_yerr = self.intensity_and_std(self.b_im, 1)
        self.response_params = None #Call self.fit_response to populate
        self.spectrum_params = None #Call self.fit_spectrum to populate


    @staticmethod
    def intensity_and_std(array, axis):
        """ Method to calculate TS Spectrum Intensity + relative error """
        mean = np.sum(array, axis = axis)
        mean_norm = (mean - mean.min() ) / (mean.max() - mean.min())
        sd   = np.sqrt(mean_norm)
        return mean_norm, sd

    @staticmethod
    def voigt_response(l, l0, sigma, A):
        ''' Returns a shifted voigt profile.
        '''
        dl = l-l0
        gamma=sigma
        y = voigt_profile(dl, sigma, gamma)
        y /= y.max()
        return y*A 
    
    def fit_response(self, approx_probe_angle=0.):
        ''' Fit a voigt profile to the background spectrum.
        '''
        y = self.b_y / self.b_y.max()
        
        p_opt, p_cov = curve_fit(self.voigt_response, self.b_l, y, \
            p0=[approx_probe_angle, 1., 0.01])
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
        l0 =  self.response_params['l0']
        s = self.response_params['sigma']
        g = self.response_params['gamma']
        A = self.response_params['amp']
        R = self.voigt_response(l, l0, s, A)
        return R
    
    def fit_spectrum(self, approx_probe_angle=0.):
        ''' Fit a Gaussian profile to the spectrum.
        '''
        mod=GaussianModel()+ConstantModel()
        mod.make_params()
        mod.set_param_hint('mean', value = approx_probe_angle)
        mod.set_param_hint('sigma', value=0.01)
        mod.set_param_hint('amplitude', value=100)
        y = self.s_y / self.s_y.max()
        res=mod.fit(y, x=self.s_l, nan_policy='omit')
        return res.best_fit