# %%
import numpy as np

from bilby.gw import utils as gwutils
from .response import get_detector_response

from .typing import HarmonicMode, ModeArray
from .typing import Interferometer, InterferometerList, WaveformGenerator
from typing import Callable, List, Dict, Union, Optional

# %%
def derivative_ndim(func, ndim, x, steps):
    x = np.array(x)
    dx = np.einsum('ij,i->ij', np.eye(ndim), np.array(steps))
    return np.array([(func(*(x+dx[i, :]))-func(*x))/steps[i] for i in range(ndim)])


def single_ifo_fisher_matrix(ifo: Interferometer, mode_array: ModeArray,
                             generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                             parameters: Dict, parameter_names: List[str],
                             step_dict: Optional[Dict] = None) -> np.ndarray:
    default_parameters = {
        key: value for key, value in parameters.items() if key not in parameter_names}
    parameters_vector = [parameters[para] for para in parameter_names]
    ndim = len(parameter_names)

    def signal_func(*args):
        waveform_parameters = default_parameters.copy()
        derivate_parameters = {para: value for para,
                               value in zip(parameter_names, args)}
        waveform_parameters.update(derivate_parameters)
        return get_detector_response(ifo, mode_array, generator_from_mode, waveform_parameters)

    if step_dict is None:
        default_step_dict = dict(ra=1e-3, dec=1e-3, phase=1e-2, iota=1e-3,
                                 total_mass=1e-4, mass_ratio=1e-6,
                                 geocent_time=1e-5, psi=1e-5, lndL=1e-3)
        step_dict = {key: value for key,
                     value in default_step_dict.items() if key in parameter_names}
    try:
        steps = [step_dict[para] for para in parameter_names]
    except KeyError:
        raise ValueError('Wrong step_dict!')
    signal_derivate = derivative_ndim(
        signal_func, ndim, parameters_vector, steps)

    fisher_matrix = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i+1):
            fisher_matrix[i, j] = gwutils.noise_weighted_inner_product(
                signal_derivate[i][ifo.strain_data.frequency_mask],
                signal_derivate[j][ifo.strain_data.frequency_mask],
                power_spectral_density=ifo.power_spectral_density_array[
                    ifo.strain_data.frequency_mask],
                duration=ifo.strain_data.duration
            ).real
            fisher_matrix[j, i] = fisher_matrix[i, j]
    return fisher_matrix


def fisher_matrix(network: Union[Interferometer, InterferometerList, List[Interferometer]],
                  mode_array: ModeArray, generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                  parameters: Dict, parameter_names: List[str],
                  step_dict: Optional[Dict] = None, **kwargs) -> np.ndarray:
    parameters = parameters.copy()
    parameters.update(kwargs)
    if isinstance(network, Interferometer):
        return single_ifo_fisher_matrix(network, mode_array, generator_from_mode, parameters, parameter_names, step_dict)
    elif isinstance(network, InterferometerList) or isinstance(network, list):
        return sum([single_ifo_fisher_matrix(ifo, mode_array, generator_from_mode, parameters, parameter_names, step_dict) for ifo in network])


def single_ifo_fisher_matrix_stationary(
    ifo: Interferometer, waveform_generator: WaveformGenerator,
    parameters: Dict, parameter_names: List[str],
    step_dict: Optional[Dict] = None) -> np.ndarray:

    default_parameters = {
        key: value for key, value in parameters.items() if key not in parameter_names}
    parameters_vector = [parameters[para] for para in parameter_names]
    ndim = len(parameter_names)

    def signal_func(*args):
        waveform_parameters = default_parameters.copy()
        derivate_parameters = {para: value for para,
                               value in zip(parameter_names, args)}
        waveform_parameters.update(derivate_parameters)
        waveform = waveform_generator.frequency_domain_strain(
            waveform_parameters)
        return ifo.get_detector_response(waveform, waveform_parameters)

    if step_dict is None:
        default_step_dict = dict(ra=1e-3, dec=1e-3, phase=1e-2, iota=1e-3,
                                 total_mass=1e-4, mass_ratio=1e-6,
                                 geocent_time=1e-5, psi=1e-5, lndL=1e-3)
        step_dict = {key: value for key,
                     value in default_step_dict.items() if key in parameter_names}
    try:
        steps = [step_dict[para] for para in parameter_names]
    except KeyError:
        raise ValueError('Wrong step_dict!')
    signal_derivate = derivative_ndim(
        signal_func, ndim, parameters_vector, steps)

    fisher_matrix = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i+1):
            fisher_matrix[i, j] = gwutils.noise_weighted_inner_product(
                signal_derivate[i][ifo.strain_data.frequency_mask],
                signal_derivate[j][ifo.strain_data.frequency_mask],
                power_spectral_density=ifo.power_spectral_density_array[
                    ifo.strain_data.frequency_mask],
                duration=ifo.strain_data.duration
            ).real
            fisher_matrix[j, i] = fisher_matrix[i, j]
    return fisher_matrix


def fisher_matrix_stationary(
    network: Union[Interferometer, InterferometerList, List[Interferometer]],
    waveform_generator: WaveformGenerator,
    parameters: Dict, parameter_names: List[str],
    step_dict: Optional[Dict] = None, **kwargs) -> np.ndarray:

    parameters = parameters.copy()
    parameters.update(kwargs)
    if isinstance(network, Interferometer):
        return single_ifo_fisher_matrix_stationary(network, waveform_generator, parameters, parameter_names, step_dict)
    elif isinstance(network, InterferometerList) or isinstance(network, list):
        return sum([single_ifo_fisher_matrix_stationary(ifo, waveform_generator, parameters, parameter_names, step_dict) for ifo in network])


def localization_accuracy(fisher_matrix: np.ndarray, dec: float, ra_i: int, dec_i: int) -> float:
    fisher_inv = np.linalg.inv(fisher_matrix)
    delta_theta = fisher_inv[dec_i, dec_i]
    delta_phi = fisher_inv[ra_i, ra_i]
    delta_theta_phi = fisher_inv[ra_i, dec_i]
    return 2*np.pi*abs(np.sin(dec))*np.sqrt(delta_theta*delta_phi-delta_theta_phi**2)
