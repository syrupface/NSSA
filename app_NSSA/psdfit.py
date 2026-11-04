""" Fits of the radially averaged PSD. The code lives in NSFopen (NSFopen.process.psdfit, NSFopen >= 2.3.0); this module is that module, so
`import psdfit` in the app and NSFopen's own modules share the same functions. """

import sys

from NSFopen.process import psdfit as _shared

sys.modules[__name__] = _shared
