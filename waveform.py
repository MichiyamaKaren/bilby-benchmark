import numpy as np

from pycbc.waveform import get_fd_waveform


def GR_waveform(farray, mass_1, mass_2,
                phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, **kwargs):
    flow = 1e-4
    fhigh = farray[-1]
    deltaf = farray[1] - farray[0]

    approximant = kwargs.get('waveform_approximant', 'IMRPhenomPv2')
    mode_array = kwargs.get('mode_array', [2, 2])

    hp, hc = get_fd_waveform(
        approximant=approximant,
        mass1=mass_1, mass2=mass_2,
        distance=luminosity_distance,
        inclination=iota, coa_phase=phase,
        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
        delta_f=deltaf, f_lower=flow, f_final=fhigh,
        mode_array=mode_array)

    hp = np.array([hp.at_frequency(f) for f in farray])
    hc = np.array([hc.at_frequency(f) for f in farray])
    return dict(plus=hp, cross=hc)
