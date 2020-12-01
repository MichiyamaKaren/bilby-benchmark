# %%
import numpy as np
import matplotlib.pyplot as plt

import bilby
from detector_response import SNR
from bilby.gw import WaveformGenerator
from bilby.gw.detector import Interferometer, TriangularInterferometer
from pycbc.waveform import get_fd_waveform

import os
import pickle
import pathos
import argparse
from datetime import datetime

# %%
approx = 'IMRPhenomHM'


def GR_waveform(farray, mass_1, mass_2,
                phase, iota, ra, dec, psi, luminosity_distance, geocent_time,
                spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, mode_array, **kwargs):
    flow = 1e-4
    fhigh = farray[-1]
    deltaf = farray[1] - farray[0]

    hp, hc = get_fd_waveform(
        approximant=approx,
        mass1=mass_1, mass2=mass_2,
        distance=luminosity_distance,
        inclination=iota, coa_phase=phase,
        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
        delta_f=deltaf, f_lower=flow, f_final=fhigh,
        mode_array=mode_array)

    hp = np.array([hp.at_frequency(f) for f in farray])
    hc = np.array([hc.at_frequency(f) for f in farray])
    return dict(plus=hp, cross=hc)


def GR_waveform_from_mode_array(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, ra, dec, psi, luminosity_distance, geocent_time,
                 spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, **kwargs):
        return GR_waveform(farray, mass_1, mass_2, phase, iota, ra, dec, psi, luminosity_distance, geocent_time,
                           spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, mode_array, **kwargs)

    return waveform


def save_SNR_data(filename, all_SNR, default_parameters, varying_parameters):
    SNR_data = dict(all_SNR=all_SNR,
                    default_parameters=default_parameters,
                    varying_parameters=varying_parameters)
    with open(filename, 'wb') as f:
        pickle.dump(SNR_data, f)


def load_SNR_data(filename):
    with open(filename, 'rb') as f:
        SNR_data = pickle.load(f)
    return SNR_data['all_SNR'], SNR_data['varying_parameters'], SNR_data['default_parameters']


# %%
default_parameters = dict(
    total_mass=50, mass_ratio=0.5,
    phase=0, iota=30*np.pi/180, ra=3/2*np.pi, dec=2/3*np.pi, psi=30*np.pi/180,
    redshift=1, geocent_time=1126259642.4,
    spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0.)

varying_parameters = dict(
    total_mass=np.linspace(10, 100, 100),
    mass_ratio=np.linspace(0, 1, 100)[1:],
    iota=np.linspace(0, np.pi, 100),
    psi=np.linspace(0, np.pi, 100),
    redshift=np.linspace(0.1, 2, 1000)
)

mode_array = [[2, 2]]
# %%
sampling_frequency = 2048
duration = 4
waveform_arguments = dict(minimum_frequency=10)


def GR_generator_from_mode(mode):
    return WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=GR_waveform_from_mode_array([mode]),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments, init_log=False)


ifo_dir = 'bilby_detectors'
ifo_files = [os.path.join(ifo_dir, f) for f in os.listdir(ifo_dir)]
ifos = [bilby.gw.detector.networks.load_interferometer(f) for f in ifo_files]
for ifo in ifos:
    if isinstance(ifo, Interferometer):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=default_parameters['geocent_time'] - duration)
    elif isinstance(ifo, TriangularInterferometer):
        ifo.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=default_parameters['geocent_time'] - duration)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='process number')
parser.add_argument('--save', type=str, default='SNRdata.pkl',
                    help='save SNR result to file')
parser.add_argument('--load', type=str, help='load SNR result from file')
args = parser.parse_args()

# %%
def SNR_wrapper(ifo, para_name):
    def wrapped_SNR(para):
        return SNR(ifo, mode_array, GR_generator_from_mode, default_parameters, **{para_name: para})
    return wrapped_SNR


if args.load:
    all_SNR, varying_parameters, default_parameters = load_SNR_data(args.load)
else:
    pool = pathos.multiprocessing.ProcessPool(nodes=args.n)

    all_SNR = {para_name: {} for para_name in varying_parameters.keys()}
    for para_name, para in varying_parameters.items():
        start_t = datetime.now()
        for ifo in ifos:
            ifo_SNR = pool.map(SNR_wrapper(ifo, para_name), para)
            all_SNR[para_name][ifo.name] = np.array(ifo_SNR)
        end_t = datetime.now()
        print('used time:', end_t-start_t)
    save_SNR_data(args.save, all_SNR, default_parameters, varying_parameters)

# %%
for para_name, para in varying_parameters.items():
    for ifo_name, ifo_SNR in all_SNR[para_name].items():
        plt.plot(para, ifo_SNR, label=ifo_name)
    plt.title(f'{para_name} SNR')
    plt.legend()
    plt.savefig(f'{para_name}-SNR.png')
    plt.clf()

# %%
