# FIXME borroed from spherex-utils

import logging

DEFAULT_FLAGS: tuple[str, ...] = ("ALL", "-FULLSAMPLE", "-SOURCE")
# Empty dict
# EMPTY_DICT: Mapping[Any, Any] = MappingProxyType({})

# Dict to map flag names to bit positions
FLAGS = {
    "TRANSIENT": 0,  # Transient detected during SUR
    "OVERFLOW": 1,  # Overflow detected during SUR
    "SUR_ERROR": 2,  # Error in onboard processing
    "PHANTOM": 4,  # Phantom pixel
    "REFERENCE": 5,  # Reference pixel
    "NONFUNC": 6,  # Permanently unusable
    "DICHROIC": 7,  # Low efficiency due to dichroic
    "MISSING_DATA": 9,  # Onboard data lost
    "HOT": 10,  # Hot pixel
    "COLD": 11,  # Anomalously low signal
    "FULLSAMPLE": 12,  # Pixel full sample history is available
    "PHANMISS": 14,  # Phantom correction was not applied
    "NONLINEAR": 15,  # Pixel for which a reliable nonlinearity correction
    "PERSIST": 17,  # Persistent charge above threshold
    "OUTLIER": 19,  # Pixel flagged by Detect Outliers
    "CROSSTALK": 20,  # Pixel affected by crosstalk
    "SOURCE": 21,  # Pixel mapped to a known source
    "GHOST": 22,  # Pixel affected by a bright source inside the exposure
    "GHOST_FPA": 23,  # Pixel affected by a bright source on another detector
    "GHOST_EXT": 24,  # Pixel affected by a bright source outside the field of view
    "BLOOM": 26,  # Pixel affected by a bloom artifact
    "SNOWBALL": 27,  # Pixel affected by a snowball artifact
    "HALO": 28,  # Pixel affected by a transient halo
}


def get_flagval(*args) -> int:  # noqa: ANN002
    """Return the flag value."""
    # Logger for the function
    logger = logging.getLogger("get_flagval")

    flags_to_include = set()
    flags_to_exclude = set()
    for flag in args:
        if len(flag) == 0:
            continue
        if flag[0] == "-" and (flag_upper := flag[1:].upper()) in FLAGS:
            flags_to_exclude.add(flag_upper)
        elif (
            (flag[0] == "+" and (flag_upper := flag[1:].upper()) in FLAGS)
            or (flag_upper := flag.upper()) in FLAGS
        ):
            flags_to_include.add(flag_upper)
        elif flag_upper == "ALL":
            flags_to_include |= FLAGS.keys()
        else:
            logger.warning("Ignore flag, '%s'.", flag)

    flags = flags_to_include - flags_to_exclude

    flagval = 0
    for flag in flags:
        flagval |= 1 << FLAGS[flag]

    return flagval

