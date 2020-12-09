from bilby.gw import WaveformGenerator
from bilby.gw.detector import Interferometer, TriangularInterferometer, InterferometerList
from typing import Iterable, Union

HarmonicMode = Iterable[int]
ModeArray = Iterable[HarmonicMode]

Ifo = Union[Interferometer, TriangularInterferometer]