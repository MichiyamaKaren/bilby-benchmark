from bilby.gw import WaveformGenerator
from bilby.gw.detector import Interferometer, TriangularInterferometer
from typing import Iterable, Tuple, Union

HarmonicMode = Tuple[int, int]
ModeArray = Iterable[HarmonicMode]