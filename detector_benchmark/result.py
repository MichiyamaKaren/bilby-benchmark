import numpy as np
import pickle

from typing import Dict, Optional, Tuple


def _load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def _dump_pkl(filename, obj):
    with open(filename, 'wb') as f:
        return pickle.dump(obj, f)


class SNRResult:
    def __init__(self, name: str, default_parameters: Dict,
                 SNR_values: Optional[Dict] = None, parameter_values: Optional[Dict] = None):
        self.name = name
        self.default_parameters = default_parameters.copy()

        self.SNR_values = SNR_values if SNR_values is not None else {}
        self.parameter_values = parameter_values.copy(
        ) if parameter_values is not None else {}

    def save_result(self, parameter_name: str, parameter_values: np.ndarray, SNR_values: np.ndarray) -> None:
        self.SNR_values[parameter_name] = SNR_values
        self.parameter_values[parameter_name] = parameter_values

    def save_sky_result(self, theta_values: np.ndarray, phi_values: np.ndarray, SNR_values: np.ndarray) -> None:
        self.parameter_values['_theta'] = theta_values
        self.parameter_values['_phi'] = phi_values
        self.SNR_values['_sky'] = SNR_values

    def get_result(self, parameter_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.parameter_values[parameter_name], self.SNR_values[parameter_name]

    def get_sky_result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.parameter_values['_theta'], self.parameter_values['_phi'], self.SNR_values['_sky']

    @classmethod
    def load_from_file(cls, filename: str):
        data_dict = _load_pkl(filename)
        return cls(**data_dict)

    def dump_to_file(self, filename: str):
        save_attrs = ['name', 'default_parameters',
                      'SNR_values', 'parameter_values']
        data_dict = {attr: self.__getattribute__(attr) for attr in save_attrs}
        _dump_pkl(filename, data_dict)
