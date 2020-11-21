import numpy as np


def t_SPA(f, tc, m1, m2, mode=(2, 2), to_order=8):
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
    array_like: time array corresponding to `f`
    '''
    G = 6.67e-11
    c = 299792458
    gamma = 0.5772
    m1 = m1 * 2e30
    m2 = m2 * 2e30
    M = m1 + m2
    eta = m1 * m2 / M**2
    M_c = eta**0.6 * M
    v = (G * M * np.pi * f / c**3)**(1 / 3)
    v = v.astype('float64')

    l, m = mode
    f = 2 * f / m

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

    t = tc - 5 / 256 * (G * M_c)**(-5 / 3) * c**5 * (np.pi * f)**(-8 / 3) * \
        sum([tau_i * v**i for i, tau_i in enumerate(tau[:to_order])])
    t = t.astype('float64')
    return t

