import sys

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []

if sys.version_info < (3, 8):
    collect_ignore_glob.append("hydra_utils/*py38*")
