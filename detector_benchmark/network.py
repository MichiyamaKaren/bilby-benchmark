import numpy as np

from .expand_ifo import ExpandedInterferometer

from typing import List, Dict, Callable, Optional, Iterable, Union
from .typing import WaveformGenerator, Interferometer, TriangularInterferometer, HarmonicMode, ModeArray

SingleIfo = Union[Interferometer, TriangularInterferometer, ExpandedInterferometer]

class Network:
    def __init__(self, name: str, ifos: List[SingleIfo]) -> None:
        self.name = name
        self.ifos: List[ExpandedInterferometer] = []
        for ifo in ifos:
            self.append_ifo(ifo)

    def append_ifo(self, ifo: SingleIfo) -> None:
        if isinstance(ifo, Interferometer):
            self.ifos.append(ExpandedInterferometer.from_ifo_instance(ifo))
        elif isinstance(ifo, TriangularInterferometer):
            self.ifos += [
                ExpandedInterferometer.from_ifo_instance(ifo_i) for ifo_i in ifo]
        else:
            self.ifos.append(ifo)

    def set_strain_data_from_power_spectral_densities(self, sampling_frequency, duration, start_time=0):
        for ifo in self.ifos:
            ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency,
                                                            duration=duration,
                                                            start_time=start_time)

    def get_optimal_snr_stationary(self, parameters: Dict, waveform_generator: WaveformGenerator) -> float:
        waveform = waveform_generator.frequency_domain_strain(parameters)
        snr_squares = []
        for ifo in self.ifos:
            signal = ifo.get_detector_response(waveform, parameters)
            snr_squares.append(ifo.optimal_snr_squared(signal))
        return np.sqrt(sum(snr_squares)).real

    def get_optimal_snr_rotating(self, parameters: Dict,
                                 waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                                 mode_array: ModeArray) -> float:
        snr_squares = []
        for ifo in self.ifos:
            signal = ifo.get_detector_response_rotating(parameters, waveform_generator_from_mode, mode_array)
            snr_squares.append(ifo.optimal_snr_squared(signal))
        return np.sqrt(sum(snr_squares)).real
    
    def fisher_matrix_stationary(
        self, parameters: Dict, derivate_parameters: List[str], waveform_generator: Optional[WaveformGenerator] = None,
        convert_log: Optional[Iterable[str]] = None, convert_cos: Optional[Iterable[str]] = None,
        step=1e-7) -> np.ndarray:

        fishers = [ifo.fisher_matrix_stationary(
            parameters, derivate_parameters, waveform_generator,
            convert_log=convert_log, convert_cos=convert_cos, step=step)
            for ifo in self.ifos]
        return sum(fishers)
    
    def fisher_matrix_rotating(
        self, parameters: Dict, derivate_parameters: List[str],
        waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
        mode_array: ModeArray,
        convert_log: Optional[Iterable[str]] = None, convert_cos: Optional[Iterable[str]] = None,
        step=1e-7) -> np.ndarray:

        fishers = [ifo.fisher_matrix_rotating(
            parameters, derivate_parameters, waveform_generator_from_mode, mode_array,
            convert_log=convert_log, convert_cos=convert_cos, step=step)
            for ifo in self.ifos]
        return sum(fishers)
