from distutils.core import setup

from setuptools import find_packages

setup(
    name='go1_gym',
    version='1.0.0',
    author='Guoping Pan',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='panguoping02@gmail.com',
    description='Toolkit for deployment of sim-to-real RL on the Unitree Go1/2+ARX.',
    install_requires=[
                      'jaynes>=0.9.2',
                      'params-proto==2.10.5',
                      'gym>=0.14.0',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.23.5'
                      ]
)
