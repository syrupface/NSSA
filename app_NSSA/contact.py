""" Contact mechanics of force curves: Hertz, Sneddon, DMT, JKR, Oliver-Pharr for five tip shapes. The code is NSFopen's (NSFopen.process.contact, NSFopen >= 2.4): this module is that
module, so the app and scripts give the same numbers. """

import sys

from NSFopen.process import contact as _shared

sys.modules[__name__] = _shared
