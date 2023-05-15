#!/usr/bin/env python

"""Setup script for the package."""

import logging
import os
import sys

import setuptools

PACKAGE_NAME = "vecs"
MINIMUM_PYTHON_VERSION = (3, 7, 0, "", 0)


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit(
            "At least Python {0}.{1}.{2} is required.".format(
                *MINIMUM_PYTHON_VERSION[:3]
            )
        )


def read_package_variable(key, filename="__init__.py"):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join("src", PACKAGE_NAME, filename)
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(" ", 2)
            if parts[:-1] == [key, "="]:
                return parts[-1].strip("'").strip('"')
    logging.warning("'%s' not found in '%s'", key, module_path)
    raise KeyError(key)


check_python_version()


DEV_REQUIRES = ["pytest", "parse", "numpy", "pytest-cov"]

REQUIRES = ["pgvector==0.1.*", "sqlalchemy==2.*"]


setuptools.setup(
    name=read_package_variable("__project__"),
    version=read_package_variable("__version__"),
    description="pgvector client",
    url="https://github.com/olirice/vecs",
    author="Oliver Rice",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["py.typed"]},
    tests_require=["pytest"],
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=REQUIRES,
    extras_require={"dev": DEV_REQUIRES},
)
