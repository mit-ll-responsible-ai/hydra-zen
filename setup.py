# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

import versioneer

DISTNAME = "hydra_zen"
LICENSE = "MIT"
AUTHOR = "Justin Goodwin, Ryan Soklaski"
AUTHOR_EMAIL = "ryan.soklaski@ll.mit.edu"
URL = "https://github.com/mit-ll-responsible-ai/hydra-zen"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
]
KEYWORDS = (
    "machine learning research configuration scalable reproducible yaml Hydra dataclass"
)
INSTALL_REQUIRES = [
    "hydra-core >= 1.1.0",
    "typing-extensions >= 4.1.0",
]
TESTS_REQUIRE = [
    "pytest >= 3.8",
    "hypothesis >= 6.28.0",
]

DESCRIPTION = "Configurable, reproducible, and scalable workflows in Python, via Hydra"
LONG_DESCRIPTION = """
hydra-zen is a Python library that simplifies the process of writing code (research-grade or production-grade) that is:

- **Configurable**: you can configure all aspects of your code from a single interface (the command line or a single Python function).
- **Repeatable**: each run of your code will be self-documenting; the full configuration of your software is saved alongside your results.
- **Scalable**: launch multiple runs of your software, be it on your local machine or across multiple nodes on a cluster.

It builds off – and is fully compatible with – Hydra, a framework for elegantly
 configuring complex applications.

hydra-zen helps simplify the process of using Hydra by providing convenient functions
for creating and validating configs, as well as launching Hydra jobs. It also provides
novel functionality such as wrapped instantiation and meta fields in configs.
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
    download_url="https://github.com/mit-ll-responsible-ai/hydra-zen/tarball/v"
    + versioneer.get_version(),
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    package_data={"hydra_zen": ["py.typed"]},
    extras_require={
        "pydantic": [
            "pydantic>=1.8.2"
        ],  # don't reduce below 1.8.2 -- security vulnerability
        "beartype": ["beartype>=0.8.0"],
    },
)
