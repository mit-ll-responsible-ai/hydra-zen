# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from setuptools import setup

import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    download_url="https://github.com/mit-ll-responsible-ai/hydra-zen/tarball/v"
    + versioneer.get_version(),
)
