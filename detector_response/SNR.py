# %%
import numpy as np

import bilby
from .response import get_detector_response

from .typing import HarmonicMode, ModeArray
from .typing import Interferometer, TriangularInterferometer, WaveformGenerator
from typing import Dict, Callable, Iterable, Union, Any

# %%
def single_ifo_SNR(ifo: Interferometer, mode_array: ModeArray,
                   generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                   parameters: Dict):
    signal = get_detector_response(
        ifo, mode_array, generator_from_mode, parameters)
    return np.sqrt(ifo.optimal_snr_squared(signal)).real


def SNR(ifo: Union[Interferometer, TriangularInterferometer],
        mode_array: ModeArray,
        generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
        parameters: Dict, **kwargs):
    parameters = parameters.copy()
    parameters.update(kwargs)
    if isinstance(ifo, Interferometer):
        return single_ifo_SNR(ifo, mode_array, generator_from_mode, parameters)
    elif isinstance(ifo, TriangularInterferometer):
        return np.sqrt(sum([single_ifo_SNR(ifo_i, mode_array, generator_from_mode, parameters)**2 for ifo_i in ifo])).real
