# %%
import numpy as np

from .response import get_detector_response

from .typing import HarmonicMode, ModeArray
from .typing import Ifo, Interferometer, TriangularInterferometer, WaveformGenerator
from typing import Dict, Callable, Iterable, Union, Any

# %%
def single_ifo_SNR(ifo: Interferometer, mode_array: ModeArray,
                   generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                   parameters: Dict):
    signal = get_detector_response(
        ifo, mode_array, generator_from_mode, parameters)
    return np.sqrt(ifo.optimal_snr_squared(signal)).real


def SNR(ifo: Ifo, mode_array: ModeArray,
        generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
        parameters: Dict, **kwargs):
    parameters = parameters.copy()
    parameters.update(kwargs)
    if isinstance(ifo, Interferometer):
        return single_ifo_SNR(ifo, mode_array, generator_from_mode, parameters)
    elif isinstance(ifo, TriangularInterferometer):
        return np.sqrt(sum([single_ifo_SNR(ifo_i, mode_array, generator_from_mode, parameters)**2 for ifo_i in ifo])).real

# %%

def single_ifo_SNR_stationary(ifo: Interferometer, waveform_generator: WaveformGenerator,
                              parameters: Dict):
    waveform = waveform_generator.frequency_domain_strain(parameters)
    signal=ifo.get_detector_response(waveform, parameters)
    return np.sqrt(ifo.optimal_snr_squared(signal)).real


def SNR_stationary(ifo: Ifo, waveform_generator: WaveformGenerator,
                   parameters: Dict, **kwargs):
    parameters = parameters.copy()
    parameters.update(kwargs)
    if isinstance(ifo, Interferometer):
        return single_ifo_SNR_stationary(ifo, waveform_generator, parameters)
    elif isinstance(ifo, TriangularInterferometer):
        return np.sqrt(sum([single_ifo_SNR_stationary(ifo_i, waveform_generator, parameters)**2 for ifo_i in ifo])).real
