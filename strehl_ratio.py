# Author: Eric Shore
# Purpose: To calculate the Strehl Ratio of an AO System
import numpy as np
from scipy.special import j1


# Function for determining the (normalized) maximum intensity for the actual PSF
def actual(data, centerY, centerX, noise_lim):

    I_real_temp = []
    for y in range(centerY-noise_lim, centerY+noise_lim+1):
        for x in range(centerX-noise_lim, centerX+noise_lim+1):
            I_real_temp.append(data[y][x])

    I_real = np.array(I_real_temp)
    cp = data[centerY][centerX]/float(np.sum(I_real))
    return [cp, 0]


# Function for determining the expected PSF if the system were diffraction limited
def ideal(diametro, wavelength, fuoco, pixel_size, noise_lim):
    # subpixel scale
    xscale = 10
    yscale = 10
    # Noise limited threshlod of PSF
    x_room = noise_lim
    y_room = noise_lim
    # Create arrays for X and Y positions of subpixels
    x = np.arange(-(x_room+0.5)*xscale, (x_room+.5)*xscale) * \
        pixel_size/xscale  # in m from central position
    y = np.arange(-(y_room+0.5)*yscale, (y_room+.5)*yscale) * \
        pixel_size/yscale  # in m from central poisition
    # determine expected PSF
    # array to hold distances from centre of PSF
    r = np.zeros([len(x), len(y)])
    u = np.zeros(np.shape(r))  # array to hold Bessel Function variable
    I = np.zeros(np.shape(r))  # array to hold intensities

    # loop through every pixel
    for i in range(len(x)):
        for j in range(len(y)):
            # determine distance to centre
            r[i, j] = (x[i]**2+y[j]**2)**0.5
            # determine intensity at every pixel
            if r[i, j] == 0:  # if at centre set I to 1
                I[i, j] = 1.
            else:  # otherwise calculate I
                # Maca il 2*pi perchè utilizzo il diametro e non il raggio
                u[i, j] = np.pi*diametro/wavelength * \
                    np.arcsin(r[i, j]/(r[i, j]**2+fuoco**2)**0.5)
                I[i, j] = (2*j1(u[i, j])/(u[i, j]))**2
    # normalize I
    I /= np.sum(I)

    # caclulate Intensity at every pixel (from intensity at every subpixel)
    I2 = np.zeros([round(len(x)/xscale), round(len(y)/yscale)])
    # loop through every pixel
    for i in range(round(len(x)/xscale)):
        for j in range(round(len(y)/yscale)):
            # intensity just the sum of all intensities in every subpixel
            I2[i, j] = np.sum(I[xscale*i:xscale*(i+1), yscale*j:yscale*(j+1)])
    cp = np.amax(I2)
    return [cp, I2]
