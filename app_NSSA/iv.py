""" I-V curves: dI/dV, normalized conductance, zero-bias conductance, gap, rectification. The code is NSFopen's (NSFopen.process.iv, NSFopen >= 2.4): this module is that
module, so the app and scripts give the same numbers. """

import sys

from NSFopen.process import iv as _shared

sys.modules[__name__] = _shared
