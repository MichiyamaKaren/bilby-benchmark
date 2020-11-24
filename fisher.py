# %%
import numpy as np
from scipy import interpolate
from astrotools import skymap, healpytools as hpt

import bilby
from pycbc.waveform import get_fd_waveform
from detector_response import fisher_matrix, localization_accuracy, fisher_matrix_stationary
from bilby.gw import WaveformGenerator
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters as bilby_convert
from bilby.gw.detector import Interferometer, TriangularInterferometer, InterferometerList
from typing import Union

import os
import pickle
import pathos
import argparse
from functools import reduce
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


'''
def GR_waveform_from_mode_array(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, ra, dec, psi, luminosity_distance, geocent_time,
                 spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, **kwargs):
        return GR_waveform(farray, mass_1, mass_2, phase, iota, ra, dec, psi, luminosity_distance, geocent_time,
                           spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, mode_array, **kwargs)

    return waveform
'''


def GR_waveform_from_mode_array(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, ra, dec, psi,
                 luminosity_distance, geocent_time, **kwargs):
        return GR_waveform(farray, mass_1, mass_2, phase, iota, ra, dec, psi,
                           luminosity_distance, geocent_time, 0, 0, 0, 0, 0, 0, mode_array, **kwargs)

    return waveform


def save_fisher_data(filename, all_fisher, default_parameters, ra, dec):
    fisher_data = dict(all_fisher=all_fisher,
                       default_parameters=default_parameters,
                       ra=ra, dec=dec)
    with open(filename, 'wb') as f:
        pickle.dump(fisher_data, f)


def load_fisher_data(filename):
    with open(filename, 'rb') as f:
        fisher_data = pickle.load(f)
    return fisher_data['all_fisher'], fisher_data['default_parameters'], fisher_data['ra'], fisher_data['dec']


# %%

default_parameters = dict(
    total_mass=50, mass_ratio=0.5,
    phase=0, iota=30*np.pi/180, ra=3/2*np.pi, dec=2/3*np.pi, psi=30*np.pi/180,
    lndL=np.log(1e3), geocent_time=1126259642.4)
#    spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0.)

parameter_names = list(default_parameters.keys())
name_to_index_map = {name: i for i, name in enumerate(parameter_names)}

mode_array = [[2, 2]]
# %%
sampling_frequency = 4096
duration = 32
waveform_arguments = dict(minimum_frequency=10)


def parameter_converter(parameters):
    converted_parameters, add_keys = bilby_convert(parameters)
    if 'lndL' in converted_parameters:
        converted_parameters['luminosity_distance'] = np.exp(
            converted_parameters['lndL'])
        add_keys.append('luminosity_distance')
    return converted_parameters, add_keys


def GR_generator_from_mode(mode):
    return WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=GR_waveform_from_mode_array([mode]),
        parameter_conversion=parameter_converter,
        waveform_arguments=waveform_arguments)  # , init_log=False)


generator_22 = GR_generator_from_mode([2, 2])


# %%
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


def join_ifo_list(ifo_1: Union[Interferometer, TriangularInterferometer], ifo_2: Union[Interferometer, TriangularInterferometer]):
    if isinstance(ifo_1, Interferometer):
        ifo_1 = InterferometerList([ifo_1])
    if isinstance(ifo_2, Interferometer):
        ifo_2 = InterferometerList([ifo_2])
    return ifo_1+ifo_2


ifos_dict = {ifo.name: ifo for ifo in ifos}
name_of_networks = [['CE'], ['ET-B'], ['ET-D'], ['CE', 'ET-B'], ['CE', 'ET-D']]
ifo_networks = {
    '&'.join(names): reduce(join_ifo_list, [ifos_dict[name] for name in names])
    for names in name_of_networks}

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='process number')
parser.add_argument('--save', type=str, default='Fisherdata.pkl',
                    help='save fisher matrix data to file')
parser.add_argument('--load', type=str,
                    help='load fisher matrix data from file')
args = parser.parse_args()

# %%


def fisher_wrapper(network, ra):
    def wrapped_fisher(dec):
        return fisher_matrix(network, mode_array, GR_generator_from_mode,
                             default_parameters, parameter_names, ra=ra, dec=dec)
    return wrapped_fisher


def fisher_stationary_wrapper(network, ra):
    def wrapped_fisher_stationary(dec):
        return fisher_matrix_stationary(network, generator_22,
                                        default_parameters, parameter_names, ra=ra, dec=dec)
    return wrapped_fisher_stationary


if args.load:
    all_fisher, _, ra, dec = load_fisher_data(args.load)
else:
    pool = pathos.multiprocessing.ProcessPool(nodes=args.n)

    ra = np.linspace(0, 2*np.pi, 100)
    dec = np.linspace(0, np.pi, 100)
    all_fisher = {network_name: np.zeros(
        list(ra.shape)+list(dec.shape)+2*[len(parameter_names)]) for network_name in ifo_networks}

    start_t = datetime.now()

    for network_name, network in ifo_networks.items():
        for i, ra_i in enumerate(ra):
            ifo_fisher = pool.map(
                fisher_stationary_wrapper(network, ra_i), dec)
            all_fisher[network_name][i, :, :, :] = np.array(ifo_fisher)

    end_t = datetime.now()
    print('used time:', end_t-start_t)

    save_fisher_data(args.save, all_fisher,
                     default_parameters, ra, dec)

# %%
for ifo_name, ifo_fisher in all_fisher.items():
    accuracy = np.array([[
        localization_accuracy(
            ifo_fisher[i, j], dec[j],
            name_to_index_map['ra'], name_to_index_map['dec']
        ) for j in range(ifo_fisher.shape[1])
    ] for i in range(ifo_fisher.shape[0])])
    accuracy[accuracy > 4*np.pi] = 4*np.pi

    nside = 64
    accuracy_func = interpolate.interp2d(
        ra-np.pi, dec-np.pi/2, accuracy)
    accuracy_hp = np.hstack([accuracy_func(*hpt.pix2ang(nside, i))
                             for i in range(hpt.nside2npix(nside))])
    skymap.heatmap(accuracy_hp, label=ifo_name,
                   opath=f'{ifo_name}_localization.png')

# %%
