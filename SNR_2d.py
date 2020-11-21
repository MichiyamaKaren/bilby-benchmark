# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from astrotools import skymap, healpytools as hpt

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


def save_SNR_data(filename, position_SNR, default_parameters, ra_grid, dec_grid):
    SNR_data = dict(position_SNR=position_SNR,
                    default_parameters=default_parameters,
                    ra_grid=ra_grid, dec_grid=dec_grid)
    with open(filename, 'wb') as f:
        pickle.dump(SNR_data, f)


def load_SNR_data(filename):
    with open(filename, 'rb') as f:
        SNR_data = pickle.load(f)
    return SNR_data['position_SNR'], SNR_data['default_parameters'], SNR_data['ra_grid'], SNR_data['dec_grid']


# %%
default_parameters = dict(
    total_mass=50, mass_ratio=0.5,
    phase=0, iota=30*np.pi/180, ra=3/2*np.pi, dec=2/3*np.pi, psi=30*np.pi/180,
    luminosity_distance=1e3, geocent_time=1126259642.4,
    spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0.)

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


def SNR_wrapper(ifo, mode_array, generator_from_mode, default_parameters):
    def wrapped_SNR(ra, dec):
        return SNR(ifo, mode_array, generator_from_mode, default_parameters, ra=ra, dec=dec)
    return wrapped_SNR


if args.load:
    position_SNR, _, ra_grid, dec_grid = load_SNR_data(args.load)
else:
    pool = pathos.multiprocessing.ProcessPool(nodes=args.n)

    ra = np.linspace(0, 2*np.pi, 100)
    dec = np.linspace(0, np.pi, 100)
    ra_grid, dec_grid = np.meshgrid(ra, dec)
    position_SNR = {ifo.name: np.zeros(ra_grid.shape) for ifo in ifos}

    start_t = datetime.now()

    for ifo in ifos:
        for i in range(ra_grid.shape[0]):
            ifo_SNR = pool.map(SNR_wrapper(ifo, mode_array, GR_generator_from_mode,
                                           default_parameters), ra_grid[i, :], dec_grid[i, :])
            position_SNR[ifo.name][i, :] = np.array(ifo_SNR)

    end_t = datetime.now()
    print('used time:', end_t-start_t)

# %%
for ifo_name, ifo_SNR in position_SNR.items():
    nside = 64
    SNR_func = interpolate.interp2d(ra_grid-np.pi, dec_grid-np.pi/2, ifo_SNR)
    SNR_hp = np.hstack([SNR_func(*hpt.pix2ang(nside, i))
                        for i in range(hpt.nside2npix(nside))])
    skymap.heatmap(SNR_hp, label=ifo_name,
                   opath=f'{ifo_name}_position_SNR.png')
# %%
