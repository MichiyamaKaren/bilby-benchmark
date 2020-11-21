# %%
import numpy as np

import bilby
from bilby.gw import utils as gwutils
from .utils import t_SPA
from functools import partial

from .typing import HarmonicMode, ModeArray
from .typing import Interferometer, WaveformGenerator
from typing import Any, Callable, Optional, Dict, Tuple, List

# %%
def _t_map(func: Callable[[float], Any], t_array: np.ndarray, default_generator: Callable[[], Any]):
    result = []
    for ti in t_array:
        try:
            result.append(func(ti))
        except RuntimeError:
            result.append(default_generator())
    return np.array(result)


def single_mode_response(ifo: Interferometer, waveform_polarizations: Dict[str, np.ndarray],
                         ra: float, dec: float, psi: float, t: np.ndarray):
    signal = {}
    for polarization in ['plus', 'cross']:
        def wrapped_pt(time):
            return gwutils.get_polarization_tensor(ra, dec, time, psi, polarization)
        polarization_tensor = _t_map(
            wrapped_pt, t, default_generator=lambda: np.zeros((3, 3)))
        F = np.einsum('aij,ij->a', polarization_tensor,
                      ifo.geometry.detector_tensor)
        signal[polarization] = waveform_polarizations[polarization] * F
    return sum(signal.values())


def get_detector_response(ifo: Interferometer, mode_array: ModeArray,
                          generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                          parameters: Dict,
                          parameter_conversion: Optional[Callable[[Dict], Tuple[Dict, List]]] = None) -> np.ndarray:
    parameters = parameters.copy()
    if parameter_conversion is None:
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    parameters, _ = parameter_conversion(parameters)
    m1 = parameters['mass_1']
    m2 = parameters['mass_2']
    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']
    tc = parameters['geocent_time']

    signal_mode = []
    for mode in mode_array:
        t = t_SPA(ifo.frequency_array, tc, m1, m2)
        t = t.astype('float64')
        waveform_polarizations = generator_from_mode(
            mode).frequency_domain_strain(parameters)

        signal = single_mode_response(
            ifo, waveform_polarizations, ra, dec, psi, t)
        partial_time_delay = partial(ifo.time_delay_from_geocenter, ra, dec)
        dt = _t_map(partial_time_delay, t, default_generator=lambda: np.nan)[
            ifo.strain_data.frequency_mask]

        signal *= ifo.strain_data.frequency_mask

        dt += tc - ifo.strain_data.start_time
        signal[ifo.strain_data.frequency_mask] = signal[ifo.strain_data.frequency_mask] * np.exp(
            -1j * 2 * np.pi * dt * ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask])
        signal_mode.append(signal)
    return sum(signal_mode)
