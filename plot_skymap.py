# %%
import numpy as np
from scipy import interpolate
from astrotools import skymap, healpytools as hpt


# %%
def plot_skymap(theta, phi, data, nside=64, **kwargs):
    """
    plot skymap using given data on given coordinates

    Args:
        theta (1d array): theta coordinate, -pi/2 to pi/2
        phi (1d array): phi coordinate, -pi to pi
        data (2d array): data to be plotted. data[i, j] is the value on (theta[i], phi[j]).
        nside (int, optional): HealPix parameter nside. Defaults to 64.
        kwargs: keyword args passing to astrotools.skymap.heatmap
    """
    data_func = interpolate.interp2d(phi, theta, data)
    data_hp = np.hstack([data_func(*hpt.pix2ang(nside, i))
                         for i in range(hpt.nside2npix(nside))])
    return skymap.heatmap(data_hp, **kwargs)

# %%
