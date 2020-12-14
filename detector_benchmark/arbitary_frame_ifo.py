import numpy as np

from bilby.gw.detector import Interferometer
from bilby.gw.detector.calibration import Recalibrate

from . import utils
from .coordinate import CoordinateFrame

from typing import Dict, Callable, Tuple
from .typing import WaveformGenerator, HarmonicMode, ModeArray


class ArbitaryFrameIfo(Interferometer):
    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency, length,
                 latitude, longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt=0, yarm_tilt=0, calibration_model=Recalibrate(),
                 frame_x=None, frame_y=None):
        """
        calculate response of the interferometer in arbitary reference frame specified by frame_x, frame_y

        Args:
            name, power_spectral_density, minimum_frequency, maximum_frequency, length,
            latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
            xarm_tilt=0, yarm_tilt=0, calibration_model : same as the Interferometer class
            frame_x (1d array, optional): x axis of reference frame. If not specified, set to be the x arm of the detector.
            frame_y (1d array, optional): y axis of reference frame. If not specified, set to be the y arm of the detector.
        """
        super().__init__(name, power_spectral_density, minimum_frequency, maximum_frequency, length, latitude, longitude,
                         elevation, xarm_azimuth, yarm_azimuth, xarm_tilt=xarm_tilt, yarm_tilt=yarm_tilt, calibration_model=calibration_model)
        self.frame_x = self.x if frame_x is None else frame_x
        self.frame_y = self.y if frame_y is None else frame_y
        self.frame = CoordinateFrame(x=self.frame_x, y=self.frame_y)

    @classmethod
    def from_ifo_instance(cls, ifo: Interferometer, frame_x=None, frame_y=None):
        return cls(ifo.name, ifo.power_spectral_density, ifo.minimum_frequency, ifo.maximum_frequency, ifo.length, ifo.latitude, ifo.longitude,
                   ifo.elevation, ifo.xarm_azimuth, ifo.yarm_azimuth, xarm_tilt=ifo.xarm_tilt, yarm_tilt=ifo.yarm_tilt, calibration_model=ifo.calibration_model,
                   frame_x=frame_x, frame_y=frame_y)

    def coordinate_shift(self, theta: float, phi: float, sideral_time_shift: float)->Tuple[float,float]:
        theta_cel, phi_cel = self.frame.inverse_transform_spherical(theta, phi)
        phi_cel -= sideral_time_shift
        return self.frame.coordinate_transform_spherical(theta_cel, phi_cel)

    @property
    def detector_tensor(self):
        x = self.frame.coordinate_transform(self.x)
        y = self.frame.coordinate_transform(self.y)
        return 0.5 * (np.einsum('i,j->ij', x, x) - np.einsum('i,j->ij', y, y))

    @utils.serialize(serializeable_paras=['sideral_time_shift'])
    def antenna_response(self, theta: float, phi: float, psi: float, mode:str, sideral_time_shift: float=0)->float:
        shifted_theta, shifted_phi = self.coordinate_shift(
            theta, phi, sideral_time_shift)
        polarization_tensor = utils.polarization_tensor_in_arbitary_frame(shifted_theta, shifted_phi, psi, mode)
        return np.einsum('ij,ij->', self.detector_tensor, polarization_tensor)

    def get_detector_response(self, waveform_polarizations, parameters, sideral_time_shift=0):
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.antenna_response(
                parameters['theta'],
                parameters['phi'],
                parameters['psi'],
                mode, sideral_time_shift=sideral_time_shift)

            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values())

        signal_ifo *= self.strain_data.frequency_mask

        time_delay = self.time_delay_from_geocenter(
            theta=parameters['theta'], phi=parameters['phi'], sideral_time_shift=sideral_time_shift)
        if isinstance(time_delay, np.ndarray):
            time_delay=time_delay[self.strain_data.frequency_mask]

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
        dt = dt_geocent + time_delay

        signal_ifo[self.strain_data.frequency_mask] = signal_ifo[self.strain_data.frequency_mask] * np.exp(
            -1j * 2 * np.pi * dt * self.strain_data.frequency_array[self.strain_data.frequency_mask])

        signal_ifo[self.strain_data.frequency_mask] *= self.calibration_model.get_calibration_factor(
            self.strain_data.frequency_array[self.strain_data.frequency_mask],
            prefix='recalib_{}_'.format(self.name), **parameters)

        return signal_ifo

    @utils.serialize(serializeable_paras=['sideral_time_shift'])
    def time_delay_from_geocenter(self, theta:float, phi:float, sideral_time_shift:float=0)->float:
        shifted_theta, shifted_phi = self.coordinate_shift(
            theta, phi, sideral_time_shift)
        transformed_vertex = self.frame.coordinate_transform(self.vertex)
        return utils.time_delay(transformed_vertex, shifted_theta, shifted_phi)

    def get_optimal_snr_stationary(self, parameters: Dict, waveform_generator: WaveformGenerator) -> float:
        """
        Calculate optimal SNR of GW signal, ignoring the Earth's rotation.

        Args:
            parameters (Dict): GW parameters
            waveform_generator (WaveformGenerator): bilby waveform generator

        Returns:
            float: optimal SNR
        """
        waveform_polarizations = waveform_generator.frequency_domain_strain(
            parameters)
        signal = self.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=waveform_generator.parameters)  # converted parameters
        return np.sqrt(self.optimal_snr_squared(signal)).real

    def get_optimal_snr_rotating(self, parameters: Dict,
                                 waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                                 mode_array: ModeArray) -> float:
        """
        Calculate optimal SNR of GW signal, considering the Earth's rotation.

        Args:
            parameters (Dict): GW parameters
            waveform_generator_from_mode (Callable[[HarmonicMode], WaveformGenerator]):
                function recieves a mode(tuple (l, m)), and returns a bilby waveform generator.
            mode_array (ModeArray): iterable containing all harmonic modes

        Returns:
            float: optimal SNR
        """
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
                waveform_polarizations=waveform, parameters=generator.parameters, sideral_time_shift=sideral_time_shift))
        signal = sum(single_mode_signals)
        return np.sqrt(self.optimal_snr_squared(signal)).real
