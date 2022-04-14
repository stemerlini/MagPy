# Implement Refractometry class - derived Image class
## work in progress

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk_t
from skimage.measure import profile_line

class Refractometer:
    def __init__(self, image, scale_x, scale_phi, rotate=None):
        """
        Initialises the refractometer class. 
        Arguments are:
         1) image - greyscale image represented as numpy array
         2) rotate - rotation to be applied to image in degrees
         3) scale_x - image scale in pixels per mm
         4) scale_ϕ - image scale in pixels per mrad
        """
        # y axis and z axis must be rescaled based on tow different px to mm values

        self.im      =   sk_t.rotate(image, rotate, resize=False)
        self.sc      =   scale_x
        self.scale_phi =   scale_phi
        self.o       =   np.array([0., 0.])
        self.shape   =   image.shape
        self.r       =   rotate
    
    def px_to_mm(self, p_px):
        '''Calculates position of point in mm, given position in px'''
        h = self.shape[0]
        p = np.array(p_px, dtype=np.float64)
        p *= np.array([1., -1.]) #Convert handedness 
        p += np.array([0., h]) #Translate origin to BL corner
        p /= self.sc
        p -= self.o
        return p
    
    def mm_to_px(self, p_mm):
        '''Calculates position of point in px, given position in mm'''
        h = self.shape[0]
        p = np.array(p_mm)
        p += self.o
        p *= self.sc
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
    
    def imshow(self, ax, **kwargs):
        ''' Wraps the pbcolormesh method from pyplot to render the spectrum 
        with wavelength on the x-axis.
        '''
        h,w = self.im.shape
        X = np.tile(self.l, (h, 1))
        y = range(h, 0, -1)
        Y = np.tile(y, (w,1)).transpose()
        ax.pcolormesh(X, Y, self.im, shading='nearest', **kwargs)
        ax.invert_yaxis()

class Signal:
    def __init__(self, dark_width = 1., dark_wavelength = 540):
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
    
    def get_dark_bounds(self):
        ''' Get the bounds of the strip which is used to calculate
        dark count.
        '''
        x0 = self.dl
        x1 = self.dl + self.dw
        y0 = self.off
        y1 = self.off + self.num*self.sp
        return x0, x1, y0, y1

    def draw_spectrum(self, refractometer, vmin = 800, vmax=6000):
        ''' Draws details of the way data from the refractometer, 
        '''
        fig, axes = plt.subplots(1,2, figsize = (15,15), sharey=True)
        ax1, ax2 = axes
        refractometer.imshow(ax1, cmap='Greys', vmin=vmin, vmax=vmax)       
        profile = refractometer.im.sum(1)
        profile = 2*((profile / profile.max()) - 0.5)
        y_ord = range( len(profile), 0, -1 )
        ax2.plot(profile, y_ord, c='k')
        
        x0, x1, y0, y1 = self.get_dark_bounds()
        self.draw_rect(ax1,  x0, x1, y0, y1, c='b')
        ax1.set_xlabel(r'$\lambda \; [nm]$')
        ax1.set_ylabel(r'Position [Px]')
        ax2.set_xlabel(r'Intensity [Arb]')
    
    def get_image_with_dark_count_subtracted(self, refractometer):
        ''' Returns image from refractometer with the dark count subtracted.
        Dark count is allowed to vary along the spatial axis and is calculated 
        from the spectral average of the region bounded by self.dl and 
        self.dl+self.dw.
        ''' 
        x0 = refractometer.get_index(self.dl)
        x1 = refractometer.get_index(self.dl + self.dw)
        dark_reg = refractometer.im[:, x0:x1]
        dark_cnt = np.average(dark_reg, axis=1)
        sub_im = refractometer.im.transpose() - dark_cnt.transpose()
        sub_im = np.clip(sub_im, a_min=0., a_max=None) # Set floor of array to
        return sub_im.transpose()