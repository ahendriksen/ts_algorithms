# -*- coding: utf-8 -*-

"""Top-level package for Tomosipo algorithms."""

__author__ = """Allard Hendriksen"""
__email__ = 'allard.hendriksen@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Ensure torch support from tomosipo
import tomosipo.torch_support

from .sirt import sirt
from .fbp import fbp
from .tv_min import tv_min2d
from .operators import operator_norm, ATA_max_eigenvalue
from .fdk import fdk
from .nag_ls import nag_ls
from .callbacks import TrackMetricCb, TrackMseCb, TrackResidualMseCb, TimeoutCb
