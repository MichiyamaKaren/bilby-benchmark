import pathos
import numpy as np

from .network import Network
from .result import SNRResult

from typing import Dict, List, Callable, Iterable, Optional
from .typing import HarmonicMode, ModeArray, WaveformGenerator


class SNREvaluater:
    def __init__(self, network: Network, default_parameters: Dict) -> None:
        self.network = network
        self.result = SNRResult(
            name=network.name, default_parameters=default_parameters)
        self.default_parameters = default_parameters  # default_parameters should be initialized after result. See default_parameter.setter.

    @property
    def default_parameters(self):
        return self._default_parameters

    @default_parameters.setter
    def default_parameters(self, value: Dict):
        self._default_parameters = value.copy()
        self.result.default_parameters = self.default_parameters.copy()

    def _SNR_wrapper(self, default_parameters: Dict, parameter_name: str, mode_array: ModeArray,
                     wf_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator]):
        def wrapped_SNR(para):
            parameters = default_parameters.copy()
            parameters[parameter_name] = para
            return self.network.get_optimal_snr_rotating(parameters, wf_generator_from_mode, mode_array)
        return wrapped_SNR

    def _SNR_stationary_wrapper(self, default_parameters: Dict, parameter_name: str, wf_generator: WaveformGenerator):
        def wrapped_SNR(para):
            parameters = default_parameters.copy()
            parameters[parameter_name] = para
            return self.network.get_optimal_snr_stationary(parameters, wf_generator)
        return wrapped_SNR

    def evaluate_SNR_on_varying_parameter(
            self, parameter_name: str, parameter_values: np.ndarray,
            stationary: bool = False,
            wf_generator: Optional[WaveformGenerator] = None,
            wf_generator_from_mode: Optional[Callable[[HarmonicMode], WaveformGenerator]] = None,
            mode_array: Optional[ModeArray] = None,
            nodes: int = 1) -> np.ndarray:
        """
        Evaluate SNR with one parameter varying, while the others fixed on their default value.

        Args:
            parameter_name (str): name of the varying parameter
            parameter_values (1d array): values of the varying parameter
            stationary (bool, optional): if True, don't consider rotation of the Earth. Defaults to False.
            wf_generator (WaveformGenerator, optional): waveform generator, only needed when stationary is True.
            wf_generator_from_mode (Callable, optional): function which recieves harmonic mode and returns waveform generator object, only needed when stationary is False.
            mode_array (ModeArray, optional): mode array, only needed when stationary is False.
            nodes (int, optional): number of processes (parallel computing SNR). Defaults to 1.

        Raises:
            ValueError: if parameter_name is not in self.default_parameters

        Returns:
            1d array: SNR values corresponding to parameter_values
        """
        if parameter_name not in self.default_parameters:
            raise ValueError(f'Wrong parameter name {parameter_name}')

        if stationary:
            assert wf_generator is not None
            wrapped_SNR = self._SNR_stationary_wrapper(
                self.default_parameters, parameter_name, wf_generator)
        else:
            assert wf_generator_from_mode is not None
            assert mode_array is not None
            wrapped_SNR = self._SNR_wrapper(
                self.default_parameters, parameter_name, mode_array, wf_generator_from_mode)

        if nodes == 1:
            SNR_values = np.array(list(map(wrapped_SNR, parameter_values)))
        else:
            pool = pathos.multiprocessing.ProcessPool(nodes=nodes)
            SNR_values = np.array(pool.map(wrapped_SNR, parameter_values))
        self.result.save_result(parameter_name=parameter_name,
                                parameter_values=parameter_values, SNR_values=SNR_values)
        return SNR_values

    def evaluate_SNR_on_sky(
            self, theta_values: np.ndarray, phi_values: np.ndarray,
            stationary: bool = False,
            wf_generator: Optional[WaveformGenerator] = None,
            wf_generator_from_mode: Optional[Callable[[HarmonicMode], WaveformGenerator]] = None,
            mode_array: Optional[ModeArray] = None,
            nodes: int = 1) -> np.ndarray:
        """
        Evaluate SNR with one parameter varying, while the others fixed on their default value.

        Args:
            theta_values (1d array): values of theta coordinate
            phi_values (1d array): values of phi coordinate
            stationary (bool, optional): if True, don't consider rotation of the Earth. Defaults to False.
            wf_generator (WaveformGenerator, optional): waveform generator, only needed when stationary is True.
            wf_generator_from_mode (Callable, optional): function which recieves harmonic mode and returns waveform generator object, only needed when stationary is False.
            mode_array (ModeArray, optional): mode array, only needed when stationary is False.
            nodes (int, optional): number of processes (parallel computing SNR). Defaults to 1.

        Returns:
            2d array: SNR values, [i, j] corresponding to SNR on theta[i] and phi[j]
        """
        default_parameters = self.default_parameters.copy()
        SNR_values = np.zeros((len(theta_values), len(phi_values)))
        for i, theta_i in enumerate(theta_values):
            default_parameters['theta'] = theta_i
            if stationary:
                assert wf_generator is not None
                wrapped_SNR = self._SNR_stationary_wrapper(
                    default_parameters, 'phi', wf_generator)
            else:
                assert wf_generator_from_mode is not None
                assert mode_array is not None
                wrapped_SNR = self._SNR_wrapper(
                    default_parameters, 'phi', mode_array, wf_generator_from_mode)

            if nodes == 1:
                SNR_on_theta_i = np.array(list(map(wrapped_SNR, phi_values)))
            else:
                pool = pathos.multiprocessing.ProcessPool(nodes=nodes)
                SNR_on_theta_i = np.array(pool.map(wrapped_SNR, phi_values))
            SNR_values[i, :] = SNR_on_theta_i
        self.result.save_sky_result(theta_values, phi_values, SNR_values)
        return SNR_values

    def dump_result(self, filename: str):
        self.result.dump_to_file(filename) 

    def load_result(self, filename: str):
        self.result = SNRResult.load_from_file(filename)
