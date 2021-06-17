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
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
]
KEYWORDS = "machine learning research configuration scalable reproducible"
INSTALL_REQUIRES = [
    "hydra-core >= 1.1.0",
    "typing-extensions >= 3.7.4.1",
]
TESTS_REQUIRE = [
    "pytest >= 3.8",
    "hypothesis >= 5.32.0",
]

DESCRIPTION = "Utilities for making hydra scale to ML workflows"
LONG_DESCRIPTION = """
hydra-zen helps you configure your project using the power of Hydra, while enjoying the Zen of Python!

hydra-zen eliminates the boilerplate code that you write to configure, orchestrate, and organize the results of large-scale projects, such as machine learning experiments. It does so by providing Hydra-compatible tools that dynamically generate "structured configurations" of your code, and enables Python-centric workflows for running configured instances of your code.

hydra-zen offers:

    - Functions for automatically and dynamically generating structured configs that can be used to fully or partially instantiate objects in your application.
    - The ability to launch Hydra jobs, complete with parameter sweeps and multi-run configurations, from within a notebook or any other Python environment.
    - Incisive type annotations that provide enriched context about your project's configurations to IDEs, type checkers, and other tooling.
    - Runtime validation of configurations to catch mistakes before your application launches.
    - Equal support for both object-oriented libraries (e.g., torch.nn) and functional ones (e.g., jax and numpy).

These functions and capabilities can be used to great effect alongside PyTorch Lightning to design boilerplate-free machine learning projects!
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
