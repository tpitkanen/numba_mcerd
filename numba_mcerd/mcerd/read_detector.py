import logging
from enum import Enum
from pathlib import Path
from typing import Sequence, Generator, Tuple

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import read_target


class ReadDetectorError(Exception):
    """Error while reading detector file"""


class DetectorSettingsLine(Enum):
    """Types of lines in detector settings file"""
    DETECTOR_TYPE = "Detector type:"
    DETECTOR_ANGLE = "Detector angle:"
    VIRTUAL_DETECTOR_SIZE = "Virtual detector size:"
    TIMING_DETECTOR_NUMBERS = "Timing detector numbers:"
    ENERGY_DETECTOR_LAYER = "Energy detector layer:"
    FOIL_DESCRIPTION_FILE = "Description file for the detector foils:"
    FOIL_TYPE = "Foil type:"
    FOIL_DIAMETER = "Foil diameter:"
    FOIL_SIZE = "Foil size:"
    FOIL_DISTANCE = "Foil distance:"


def _iter_lines(lines: Sequence[str]) -> Generator[Tuple[str, str], None, None]:
    for line in lines:
        # '----------' and '==========' are separators
        if not line or line.isspace() or line.startswith("=") or line.startswith("-"):
            yield line
            continue
        key, value = line.split(sep=":", maxsplit=1)
        key = key.strip() + ":"
        value = value.strip()
        yield key, value


def read_detector_file(filename: str, g: o.Global, detector: o.Detector, target: o.Target) -> None:
    """Read detector configuration from file

    Args:
        filename: file to read configuration from
        g: global object
        detector: detector to set
        target: target to set
    """
    g.virtualdet = False

    logging.info(f"Opening detector file {filename}")
    # fp = Path(filename).open("r")
    with Path(filename).open("r") as f:
        lines_gen = _iter_lines(f.readlines())

    # TODO: Line order wouldn't matter as much if this was implemented with switch-case

    key, dtype = next(lines_gen)
    if key != DetectorSettingsLine.DETECTOR_TYPE.value:
        raise ReadDetectorError
    try:
        detector.type = c.DetectorType[dtype]
    except KeyError as ex:
        raise ReadDetectorError(f"Detector of type '{ex}' not supported")

    key, angle = next(lines_gen)
    if key != DetectorSettingsLine.DETECTOR_ANGLE.value:
        raise ReadDetectorError
    detector.angle = float(angle) * c.C_DEG

    key, detector_size = next(lines_gen)
    detector_size = detector_size.split(maxsplit=1)
    if key != DetectorSettingsLine.VIRTUAL_DETECTOR_SIZE.value:
        raise ReadDetectorError
    detector.vsize[0], detector.vsize[1] = float(detector_size[0]), float(detector_size[1])

    key, timing_numbers = next(lines_gen)
    timing_numbers = timing_numbers.split(maxsplit=1)
    if key != DetectorSettingsLine.TIMING_DETECTOR_NUMBERS.value:
        raise ReadDetectorError
    detector.tdet[0], detector.tdet[1] = int(timing_numbers[0]), int(timing_numbers[1])

    if g.output_trackpoints:
        key, energy_detector_layer = next(lines_gen)
        if key != DetectorSettingsLine.ENERGY_DETECTOR_LAYER.value:
            raise ReadDetectorError("Give energy detector layer if you want to output trackpoints.")
        detector.edet[0] = int(energy_detector_layer)

    key, foil_file = next(lines_gen)
    if key != DetectorSettingsLine.FOIL_DESCRIPTION_FILE.value:
        raise ReadDetectorError

    # Read foils
    n = 0
    while next(lines_gen, False):  # Skip first line (should be empty)
        foil = detector.foil[n]

        key, foil_type = next(lines_gen)
        if key != DetectorSettingsLine.FOIL_TYPE.value:
            raise ReadDetectorError
        if foil_type == "circular":
            foil.type = c.FoilType.CIRC
            key, diameter = next(lines_gen)
            if key != DetectorSettingsLine.FOIL_DIAMETER.value:
                raise ReadDetectorError
            # TODO: Size array could be set to specific size, or Foil could be subclassed
            foil.size[0] = float(diameter) * 0.5 * c.C_MM
        elif foil_type == "rectangular":
            foil.type = c.FoilType.RECT
            key, sizes = next(lines_gen)
            sizes = sizes.split(maxsplit=1)
            if key != DetectorSettingsLine.FOIL_SIZE.value:
                raise ReadDetectorError
            foil.size[0] = float(sizes[0]) * 0.5 * c.C_MM
            foil.size[1] = float(sizes[1]) * 0.5 * c.C_MM
        else:
            ReadDetectorError(f"Detector foil type {foil_type} not supported")

        key, foil_distance = next(lines_gen)
        if key != DetectorSettingsLine.FOIL_DISTANCE.value:
            raise ReadDetectorError
        foil.dist = float(foil_distance) * c.C_MM
        n += 1

    detector.nfoils = n
    if detector.vsize[0] > 1.0 and detector.vsize[1] > 1.0:
        g.virtualdet = True

    read_target.read_target_file(foil_file, g, target)
