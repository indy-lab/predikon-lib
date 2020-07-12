from os import path
from setuptools import find_packages, setup

AUTHORS = "A. Immer, V. Kristof"
NAME = "predikon"
PACKAGES = find_packages()
DESCRIPTION = "Predikon: online prediction of regional and national election results"
LONG_DESCR = ""
VERSION = "0.1"
URL = "https://github.com/indy-lab/predikon"
LICENSE = "MIT"
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)

with open(REQUIREMENTS_FILE) as f:
    requirements = f.read().splitlines()

setup(
    author=AUTHORS,
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCR,
    install_requires=requirements,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.3",
)
