import sys

try:
    import numpy
except ImportError:
    pass

try:
    import torch
except ImportError:
    pass

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []

if sys.version_info < (3, 8):
    collect_ignore_glob.append("*py38*")

OPTIONAL_TEST_DEPENDENCIES = (
    "numpy",
    "torch",
)

for module in OPTIONAL_TEST_DEPENDENCIES:
    if module not in sys.modules:
        collect_ignore_glob.append(f"*{module}*")
