""" Terrace analysis of stepped surfaces: terraces, their heights and the common step height. The code is NSFopen's
(NSFopen.process.terraces, NSFopen >= 2.5): this module is that module. """

import sys

from NSFopen.process import terraces as _shared

sys.modules[__name__] = _shared
