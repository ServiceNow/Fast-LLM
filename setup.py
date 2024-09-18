import sys

import setuptools

# Minimum setuptools version required to parse setup.cfg metadata.
_SETUPTOOLS_MIN_VERSION = "30.3"

if setuptools.__version__ < _SETUPTOOLS_MIN_VERSION:
    print(f"Error: setuptools version {_SETUPTOOLS_MIN_VERSION} " "or greater is required")
    sys.exit(1)

setuptools.setup()
