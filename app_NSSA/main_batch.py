""" Entry of the console program for batch processing (NSSA-batch.exe in the stand-alone build):
the same as 'main.py --batch ...'.

  NSSA-batch <recipe.json> <files or folders...> --out <folder> [--grains] [--png] [--gwy] ...
  NSSA-batch --spectroscopy <settings.json> <files or folders...> --out <folder>
"""

import sys

import main

if __name__ == "__main__":
    if not any(flag in sys.argv for flag in ("--batch", "--spectroscopy", "--selftest")):
        sys.argv.insert(1, "--batch")
    main.main()
