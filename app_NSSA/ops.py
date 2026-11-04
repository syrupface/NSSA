""" Image processing operations, registered by name for the pipeline. The code lives in NSFopen (NSFopen.process.ops, NSFopen >= 2.3.0); this module is that module, so
`import ops` in the app and NSFopen's own modules share the same functions and, for ops, the same registry OPS. """

import sys

from NSFopen.process import ops as _shared

sys.modules[__name__] = _shared
