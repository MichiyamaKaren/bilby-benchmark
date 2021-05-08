# %%
import numpy as np

from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_symmetric_mass_ratio
from bilby.core.utils import solar_mass as M_sun, gravitational_constant as G, speed_of_light as c

from typing import Callable, List


def SPA_time_shift(frequency_array, mass_1, mass_2, mode=(2, 2), to_order=8):
    '''
    Calculate t(f) of stationary phase approximation(SPA) for a special spherical harmonic mode of GW.
    See (A12) in Niu, arXiv:1910.10592

    Args
        frequency_array: (1d array): frequency array, unit:Hz
        mass_1, mass_2 (float): masses of the compact binary, unit: solar mass
        mode (Tuple[int]): harmonic mode of GW, default to be (2, 2)
        to_order (int): specifying to which order the SPA takes, 1-8, default to be 8

    Returns
        1d array: time array t(f)
    '''
    gamma = 0.5772
    total_mass = (mass_1 + mass_2) * M_sun
    chirp_mass = component_masses_to_chirp_mass(mass_1, mass_2) * M_sun
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)

    l, m = mode
    f = 2 * frequency_array / m
    v = (G * total_mass * np.pi * frequency_array / c**3)**(1 / 3)
    v = v.astype('float64')

    tau = [1,
           0,
           743 / 252 + 11 / 3 * eta,
           -32 / 5 * np.pi,
           3058673 / 508032 + 5429 / 504 * eta + 617 / 72 * eta**2,
           (-7729 / 252 + 13 / 3 * eta) * np.pi,
           -10052469856691 / 23471078400 + 128 / 3 * np.pi**2 + 6848 / 105 * gamma + (3147553127 / 3048192 - 451 / 12 * np.pi**2) * eta -
           15211 / 1728 * eta**2 + 25565 / 1296 *
           eta**3 + 3424 / 105 * np.log(16 * v**2),
           (-15419335 / 127008 - 75703 / 756 * eta + 14809 / 378 * eta**2) * np.pi]

    t_shift = -5/256 * (G * chirp_mass)**(-5 / 3) * c**5 * (np.pi * f)**(-8 / 3) * \
        sum([tau_i * v**i for i, tau_i in enumerate(tau[:to_order])])
    t_shift = t_shift.astype('float64')
    return t_shift


def polarization_tensor(theta: float, phi: float, psi: float, mode: str) -> np.ndarray:
    """
    Calculate polarization tensor in an arbitary frame. 

    Args:
        theta (float): spherical coordinate theta of the source
        phi (float): spherical coordinate phi of the source
        psi (float): polarization angle
        mode (str): polarization mode, 'plus' or 'cross'

    Returns:
        2d array: polarization tensor
    """
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta)
                  * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    if mode.lower() == 'plus':
        return np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)


def time_delay(delta_r: np.ndarray, theta: float, phi: float) -> float:
    omega = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
    return np.dot(omega, delta_r) / c


def serialize(serializeable_paras: List[str]) -> Callable[[Callable], Callable]:
    """
    Generator of decorator which serialize certain parameters of a function, i.e.,
    for parameters which should be a single number, serializing it means generating a function
    which can recieve iterable of this parameters and returns a numpy array,
    whos elements are corresponding to each value of the serialized parameter.
    Notes that serialized parameters should be passed by keyword argument.

    For example, f is a function which has 3 parameters `a`, `b`, and `c`.
    f1 = serialize(['a', 'b'])(f)
    f1(a=[1,2,3],b=[1,2,3],c=0) gives np.array([f(1,1,0), f(2,2,0), f(3,3,0)])
    f1(a=[1,2,3],b=0,c=0) gives np.array([f(1,0,0), f(2,0,0), f(3,0,0)])

    Args:
        serializeable_paras (List[str]): name of parameters to be seialized

    Returns:
        Callable[[Callable], Callable]: decorator
    """    
    def decorator(func: Callable) -> Callable:
        def serialized_func(*args, **kwargs):
            serializing_paras = [
                para for para in serializeable_paras if para in kwargs and hasattr(kwargs[para], '__iter__')]
            if serializing_paras:
                para_values = zip(*[kwargs.pop(para) for para in serializing_paras])
                results = []
                for *value_i, in para_values:
                    kwargs.update(
                        {para: value for para, value in zip(serializing_paras, value_i)})
                    results.append(func(*args, **kwargs))
                return np.array(results)
            else:
                return func(*args, **kwargs)
        return serialized_func
    return decorator

# %%
