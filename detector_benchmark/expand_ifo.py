import numpy as np
from .derivate import complex_gradient, convert_derivate_log, convert_derivate_cos

import lal
import bilby
from bilby.gw.detector import Interferometer
from bilby.core.utils import ra_dec_to_theta_phi
from . import utils

from typing import Callable, Optional, Iterable, List, Dict
from .typing import WaveformGenerator, HarmonicMode, ModeArray


class ExpandedInterferometer(Interferometer):
    @classmethod
    def from_ifo_instance(cls, ifo: Interferometer):
        return cls(ifo.name, ifo.power_spectral_density, ifo.minimum_frequency, ifo.maximum_frequency, ifo.length, ifo.latitude, ifo.longitude,
                   ifo.elevation, ifo.xarm_azimuth, ifo.yarm_azimuth, xarm_tilt=ifo.xarm_tilt, yarm_tilt=ifo.yarm_tilt, calibration_model=ifo.calibration_model)

    @utils.serialize(serializeable_paras=['sideral_time'])
    def antenna_response(self, ra: float, dec: float, psi: float, mode: str, sideral_time: float) -> float:
        theta, phi = ra_dec_to_theta_phi(ra, dec, sideral_time)
        polarization_tensor = utils.polarization_tensor(theta, phi, psi, mode)
        return np.einsum('ij,ij->', self.detector_tensor, polarization_tensor)

    def get_detector_response(self, waveform_polarizations, parameters, sideral_time_shift=0):
        signal = {}
        gmst = lal.GreenwichMeanSiderealTime(parameters['geocent_time']) % (2*np.pi)
        sideral_time = gmst + sideral_time_shift
        for mode in waveform_polarizations.keys():
            det_response = self.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['psi'],
                mode, sideral_time=sideral_time)

            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values())

        signal_ifo *= self.strain_data.frequency_mask

        #dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
        time_delay = self.time_delay_from_geocenter(ra=parameters['ra'], dec=parameters['dec'], sideral_time=sideral_time)
        if isinstance(time_delay, np.ndarray):
            time_delay = time_delay[self.strain_data.frequency_mask]
        signal_ifo[self.strain_data.frequency_mask] = signal_ifo[self.strain_data.frequency_mask] * np.exp(
            -1j * 2 * np.pi * parameters['geocent_time'] * self.strain_data.frequency_array[self.strain_data.frequency_mask])
        signal_ifo[self.strain_data.frequency_mask] = signal_ifo[self.strain_data.frequency_mask] * np.exp(
            -1j * 2 * np.pi * time_delay * self.strain_data.frequency_array[self.strain_data.frequency_mask])

        return signal_ifo

    def get_detector_response_rotating(self, parameters: Dict,
                                       waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                                       mode_array: ModeArray):
        sideral_day = 86164.0905
        single_mode_signals = []
        for harmonic_mode in mode_array:
            generator = waveform_generator_from_mode(harmonic_mode)
            waveform = generator.frequency_domain_strain(parameters)
            time_shift = utils.SPA_time_shift(
                frequency_array=generator.frequency_array,
                mass_1=generator.parameters['mass_1'], mass_2=generator.parameters['mass_2'],
                mode=harmonic_mode)
            sideral_time_shift = time_shift * 2*np.pi / sideral_day
            single_mode_signals.append(self.get_detector_response(
                waveform_polarizations=waveform, parameters=parameters, sideral_time_shift=sideral_time_shift))
        return sum(single_mode_signals)

    @utils.serialize(serializeable_paras=['sideral_time'])
    def time_delay_from_geocenter(self, ra: float, dec: float, sideral_time: float) -> float:
        theta, phi = ra_dec_to_theta_phi(ra, dec, sideral_time)
        return utils.time_delay(self.vertex, theta, phi)

    def _signal_func_stationary(self, waveform_generator: WaveformGenerator,
                                derivate_parameters: List[str], fixed_parameters: Dict):
        def signal_func(parameters_vector):
            parameters = fixed_parameters.copy()
            parameters.update({para: value for para, value in zip(derivate_parameters, parameters_vector)})
            waveform = waveform_generator.frequency_domain_strain(parameters)
            return self.get_detector_response(waveform, parameters)
        return signal_func

    def _signal_func_rotating(self, waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                              mode_array: ModeArray, derivate_parameters: List[str], fixed_parameters: Dict):
        def signal_func(parameters_vector):
            waveform_parameters = fixed_parameters.copy()
            derivate_parameters = {para: value for para,
                                value in zip(derivate_parameters, parameters_vector)}
            waveform_parameters.update(derivate_parameters)
            return self.get_detector_response_rotating(waveform_parameters, waveform_generator_from_mode, mode_array)
        return signal_func

    def signal_inner_product(self, signal1, signal2):
        return bilby.gw.utils.noise_weighted_inner_product(
            signal1[self.frequency_mask], signal2[self.frequency_mask],
            power_spectral_density=self.power_spectral_density_array[self.frequency_mask],
            duration=self.duration).real
    
    def _signal_derivate_to_fisher_matrix(
        self, parameters: Dict, derivate_parameters: List[str], signal_derivate: np.ndarray,
        convert_log: Optional[Iterable[int]] = None, convert_cos: Optional[Iterable[int]] = None) -> np.ndarray:

        name_to_index_map = {name: i for i, name in enumerate(derivate_parameters)}
        if convert_log is not None:
            for para in convert_log:
                signal_derivate[name_to_index_map[para]] = convert_derivate_log(
                    parameters[para], signal_derivate[name_to_index_map[para]])
        if convert_cos is not None:
            for para in convert_cos:
                signal_derivate[name_to_index_map[para]] = convert_derivate_cos(
                    parameters[para], signal_derivate[name_to_index_map[para]])

        ndim = len(derivate_parameters)
        fisher_matrix = np.zeros((ndim, ndim))
        for i in range(ndim):
            for j in range(i+1):
                fisher_matrix[i, j] = self.signal_inner_product(signal_derivate[i], signal_derivate[j])
                fisher_matrix[j, i] = fisher_matrix[i, j]
        return fisher_matrix

    def fisher_matrix_stationary(
        self, parameters: Dict, derivate_parameters: List[str], waveform_generator: WaveformGenerator,
        convert_log: Optional[Iterable[str]] = None, convert_cos: Optional[Iterable[str]] = None,
        step=1e-7) -> np.ndarray:

        fixed_parameters = {
            key: value for key, value in parameters.items() if key not in derivate_parameters}

        parameters_vector = [parameters[para] for para in derivate_parameters]
        signal_func = self._signal_func_stationary(waveform_generator, derivate_parameters, fixed_parameters)
        signal_derivate = complex_gradient(signal_func, parameters_vector, step=step)

        return self._signal_derivate_to_fisher_matrix(parameters, derivate_parameters, signal_derivate, convert_log, convert_cos)

    def fisher_matrix_rotating(
        self, parameters: Dict, derivate_parameters: List[str],
        waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
        mode_array: ModeArray,
        convert_log: Optional[Iterable[str]] = None, convert_cos: Optional[Iterable[str]] = None,
        step=1e-7) -> np.ndarray:

        fixed_parameters = {
            key: value for key, value in parameters.items() if key not in derivate_parameters}

        parameters_vector = [parameters[para] for para in derivate_parameters]
        signal_func = self._signal_func_rotating(waveform_generator_from_mode, mode_array, derivate_parameters, fixed_parameters)
        signal_derivate = complex_gradient(signal_func, parameters_vector, step=step)

        return self._signal_derivate_to_fisher_matrix(parameters, derivate_parameters, signal_derivate, convert_log, convert_cos)
