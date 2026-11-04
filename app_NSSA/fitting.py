""" Fits of functions to histograms (height distributions). The code lives in NSFopen (NSFopen.process.fitting, NSFopen >= 2.3.0); this module is that module, so
`import fitting` in the app and NSFopen's own modules share the same functions. """

import sys

from NSFopen.process import fitting as _shared

sys.modules[__name__] = _shared
