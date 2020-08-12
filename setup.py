from os import path
from setuptools import find_packages, setup

CURRENT_DIR = path.abspath(path.dirname(__file__))


def read_me(filename):
    with open(path.join(CURRENT_DIR, filename), encoding='utf-8') as f:
        return f.read()


def requirements(filename):
    with open(path.join(CURRENT_DIR, filename)) as f:
        return f.read().splitlines()


AUTHORS = "A. Immer, V. Kristof"
NAME = "predikon"
PACKAGES = find_packages()
DESCR = "Predikon: Sub-Matrix Factorization for Real-Time Vote Prediction"
LONG_DESCR = read_me('README.md')
LONG_DESCR_TYPE = 'text/markdown'
REQUIREMENTS = requirements('requirements.txt')
VERSION = "0.2"
URL = "https://github.com/indy-lab/predikon"
DOWNLOAD_URL = 'https://github.com/indy-lab/predikon/archive/v0.2.tar.gz'
LICENSE = "MIT"


setup(
    author=AUTHORS,
    name=NAME,
    version=VERSION,
    description=DESCR,
    long_description=LONG_DESCR,
    long_description_content_type=LONG_DESCR_TYPE,
    install_requires=REQUIREMENTS,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.5",
)
