# %%
from typing import Callable, List
import numpy as np


def SPA_time_shift(frequency_array, mass_1, mass_2, mode=(2, 2), to_order=8):
    '''
    h(t) = F_+(t)*h_+(t) + F_x(t)*h_x(t)
    Fourier Transfrom, we get
    h(f) = F_+(f)*h_+(f) + F_x(f)*h_x(f)
    However, antenna response in f domain F(f) is hard to obtain, so we use stationary phase approximation(SPA),
        changing F(f) into F(t(f)) which is a function in time domain.
    This function calculates the t(f) for a special spherical harmonic mode of GW.
    See (A12) in Niu, arXiv:1910.10592

    Parameters
    -------
    f: array_like
        frequency array, unit:Hz
    tc: float
        geocent time, unit:s
    m1, m2: float
        masses of the compact binary, unit: solar mass
    mode: Iterable of int, length 2, default to be (2, 2)
        harmonic mode of GW
    to_order: int, 1-8, default to be 8
        there are 8 items of SPA, tau0 to tau7 (see the reference), this parameter 

    Returns
    -------
    array_like: time array to be added on geocent time, corresponding to `f`
    '''
    G = 6.67e-11
    c = 299792458
    gamma = 0.5772
    M_sun = 2e30
    mass_1 *= M_sun
    mass_2 *= M_sun
    M = mass_1 + mass_2
    eta = mass_1 * mass_2 / M**2
    M_c = eta**0.6 * M

    l, m = mode
    f = 2 * frequency_array / m
    v = (G * M * np.pi * frequency_array / c**3)**(1 / 3)
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

    t_shift = -5/256 * (G * M_c)**(-5 / 3) * c**5 * (np.pi * f)**(-8 / 3) * \
        sum([tau_i * v**i for i, tau_i in enumerate(tau[:to_order])])
    t_shift = t_shift.astype('float64')
    return t_shift


def polarization_tensor_in_arbitary_frame(theta: float, phi: float, psi: float, mode: str) -> np.ndarray:
    """
    calculate polarization tensor in arbitary frame, using spherical coordinate of the source. 

    Args:
        theta (float): spherical coordinate theta
        phi (float): spherical coordinate phi
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
    speed_of_light = 299792458
    omega = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi), np.cos(theta)])
    return np.dot(omega, delta_r) / speed_of_light


def serialize(serializeable_paras: List[str]) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        def serialized_func(*args, **kwargs):
            serializing_paras = [
                para for para in serializeable_paras if para in kwargs and hasattr(kwargs[para], '__iter__')]
            if serializing_paras:
                para_values = zip(*[kwargs.pop(para)
                                    for para in serializing_paras])
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
