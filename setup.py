# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

import versioneer

DISTNAME = "hydra_zen"
LICENSE = "MIT"
AUTHOR = "Justin Goodwin, Ryan Soklaski"
AUTHOR_EMAIL = "ryan.soklaski@ll.mit.edu"
URL = "https://github.com/mit-ll-responsible-ai/hydra_zen"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
]
KEYWORDS = "machine learning research configuration scalable reproducible"
INSTALL_REQUIRES = [
    "hydra-core >= 1.1.0dev7",
    "typing-extensions >= 3.7.4.1",
]
TESTS_REQUIRE = [
    "pytest >= 3.8",
    "hypothesis >= 5.32.0",
]

DESCRIPTION = "Utilities for making hydra scale to ML workflows"
LONG_DESCRIPTION = """
TBD
"""


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    download_url="https://github.com/mit-ll-responsible-ai/hydra-zen/tarball/"
    + versioneer.get_version(),
    python_requires=">=3.6",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
)
