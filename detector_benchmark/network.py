import numpy as np

from .arbitary_frame_ifo import ArbitaryFrameIfo

from typing import List, Dict, Union, Callable
from .typing import Interferometer, TriangularInterferometer, WaveformGenerator, HarmonicMode, ModeArray


class Network:
    def __init__(self, name: str, ifos: List[ArbitaryFrameIfo]) -> None:
        self.name = name
        self.ifos = ifos

    def append(self, ifo:ArbitaryFrameIfo) -> None:
        self.ifos.append(ifo)

    def append_ifo(self, ifo: Union[Interferometer, TriangularInterferometer]) -> None:
        if isinstance(ifo, Interferometer):
            self.ifos.append(ArbitaryFrameIfo.from_ifo_instance(ifo))
        elif isinstance(ifo, TriangularInterferometer):
            vertexes = [ifo_i.vertex for ifo_i in ifo]
            x = vertexes[1]-vertexes[0]
            y = vertexes[2]-(vertexes[0]+vertexes[1])/2
            self.ifos += [ArbitaryFrameIfo.from_ifo_instance(
                ifo_i, frame_x=x, frame_y=y) for ifo_i in ifo]

    def get_optimal_snr_stationary(self, parameters: Dict, waveform_generator: WaveformGenerator) -> float:
        snr_squares = [ifo.get_optimal_snr_stationary(
            parameters, waveform_generator)**2 for ifo in self.ifos]
        return np.sqrt(sum(snr_squares)).real
    
    def get_optimal_snr_rotating(self, parameters: Dict,
                                 waveform_generator_from_mode: Callable[[HarmonicMode], WaveformGenerator],
                                 mode_array: ModeArray) -> float:
        snr_squares = [ifo.get_optimal_snr_rotating(
            parameters, waveform_generator_from_mode, mode_array)**2 for ifo in self.ifos]
        return np.sqrt(sum(snr_squares)).real
