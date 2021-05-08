import numpy as np
from numdifftools import Gradient


def amplitude_and_phase(z):
    return np.abs(z), np.unwrap(np.angle(z))


def derivative_from_amplitude_and_phase(amplitude, phase, d_amplitude, d_phase):
    return d_amplitude * np.exp(1j*phase) + amplitude * np.exp(1j*phase) * 1j * d_phase


def complex_gradient(func, parameter_vector, step, **gradient_kwargs):
    """
    Calculate gradient for complex function by calculating derivative for amplitude and phase seperately.
    Using numerical derivative package numdifftools.

    Args:
        func : function for gradient calculation
        parameter_vector (List[float]): values of parameters(real)
        step (float) : step of numerical derivative
    """
    amplitude, phase = amplitude_and_phase(func(parameter_vector))

    def amplitude_func(parameter_vector):
        return amplitude_and_phase(func(parameter_vector))[0]
    d_amplitude = Gradient(amplitude_func, step=step, **gradient_kwargs)(parameter_vector).T

    def phase_func(parameter_vector):
        return amplitude_and_phase(func(parameter_vector))[1]
    d_phase = Gradient(phase_func, step=step, **gradient_kwargs)(parameter_vector).T

    return derivative_from_amplitude_and_phase(amplitude, phase, d_amplitude, d_phase)


def convert_derivate_log(value, derivate):
    return derivate*value


def convert_derivate_cos(value, derivate):
    return -1/np.sin(value)*derivate
