# %%
import numpy as np

import bilby
from waveform import GR_waveform
from detector_benchmark import Network, SNREvaluater
from bilby.gw import WaveformGenerator

import os
import argparse
from datetime import datetime

from typing import List

# %%
default_parameters = dict(
    chirp_mass=50, mass_ratio=0.5,
    phase=0, iota=30*np.pi/180, theta=2/3*np.pi, phi=3/2*np.pi, psi=30*np.pi/180,
    redshift=1, geocent_time=1126259642.4,
    spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0.)

varying_parameters = dict(
    chirp_mass=np.linspace(10, 100, 100),
    mass_ratio=np.linspace(0, 1, 100)[1:],
    iota=np.linspace(0, np.pi, 100),
    psi=np.linspace(0, np.pi, 100),
    redshift=np.linspace(0.1, 2, 1000)
)

theta_values = np.linspace(0, np.pi, 100)
phi_values = np.linspace(0, 2*np.pi, 100)

# %%
mode_array = [[2, 2]]
sampling_frequency = 2048
duration = 4
waveform_arguments = dict(minimum_frequency=10,
                          waveform_approximant='IMRPhenomHM')


def GR_generator_from_mode(mode):
    wf_args = waveform_arguments.copy()
    wf_args['mode_array'] = [mode]
    return WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=GR_waveform,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=wf_args)


GR_generator_22 = GR_generator_from_mode([2, 2])

ifo_dir = 'bilby_detectors'
ifo_files = [os.path.join(ifo_dir, f) for f in os.listdir(ifo_dir)]
ifos = [bilby.gw.detector.networks.load_interferometer(f) for f in ifo_files]

networks: List[Network] = []
for ifo in ifos:
    network = Network(name=ifo.name, ifos=[])
    network.append_ifo(ifo)
    for network_ifo in network.ifos:
        network_ifo.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=default_parameters['geocent_time'])
    networks.append(network)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='process number')
args = parser.parse_args()

# %%
start_t = datetime.now()
for network in networks:
    evaluater = SNREvaluater(
        network=network, default_parameters=default_parameters)
    for para, value in varying_parameters.items():
        evaluater.evaluate_SNR_on_varying_parameter(
            para, value,
            stationary=True, wf_generator=GR_generator_22,
            nodes=args.n)
        evaluater.evaluate_SNR_on_sky(
            theta_values, phi_values,
            stationary=True, wf_generator=GR_generator_22,
            nodes=args.n)
    evaluater.dump_result(f'{network.name}_SNR.pkl')
end_t = datetime.now()
print('used time:', end_t-start_t)

# %%
