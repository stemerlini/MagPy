import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk_t


class Image:
    '''Initialize Image class. Parameters:
    1) image - greyscale image represented as numpy array
    2) rotation - rotation to be applied to image in degrees
    3) pxpermm - image scale in pixels per mm'''
            
    def __init__(self, image, rotate, pxpermm):
        self.im = sk_t.rotate(image, rotate, resize=True)
        self.sc = pxpermm
        self.o = np.array([0., 0.])
        
    def px_to_mm(self, p_px):
        '''Calculates position of point in mm, given position in px'''
        h = self.im.shape[0]
        p = np.array(p_px, dtype=np.float64)
        p *= np.array([1., -1.]) #Convert handedness 
        p += np.array([0., h]) #Translate origin to BL corner
        p /= self.sc
        p -= self.o
        return p
    def mm_to_px(self, p_mm):
        '''Calculates position of point in px, given position in mm'''
        h = self.im.shape[0]
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
        
    def get_origin(self):
        '''Returns the position of the origin in pixels'''
        o = np.array([0., 0.])
        o_px = self.mm_to_px(o)
        return o_px
    
    def plot_mm(self, ax, **kwargs):
        '''Plot image with axes in physical units. kwargs are passed
        to plt.imshow method.'''
        x0 = 0
        x1 = self.im.shape[1]
        y0 = 0
        y1 = self.im.shape[0]
        x0, y0 = self.px_to_mm([x0, y0])
        x1, y1 = self.px_to_mm([x1, y1])
        extent = [x0, x1, y1, y0]
        ax.imshow(self.im, extent=extent, **kwargs)
        
    def plot_px(self, ax, **kwargs):
        '''Plot image with axes pixels. kwargs are passed
        to plt.imshow method.'''
        ax.imshow(self.im, **kwargs)