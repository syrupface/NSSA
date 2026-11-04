""" Spectroscopy files, force curve processing, results, statistics and maps. The code is NSFopen's (NSFopen.process.spectroscopy, NSFopen >= 2.4): this module is that
module, so the app and scripts give the same numbers. """

import sys

from NSFopen.process import spectroscopy as _shared

sys.modules[__name__] = _shared
