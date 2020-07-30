from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

requirements = ['numpy>=1.16']

setup(packages=find_packages(exclude=['docs']),
    name="ConvCNP-Climate", # Replace with your own username
    version="0.0.1",
    author="Anna Vaughan",
    author_email="av555@cam.ac.uk",
    python_requires='>=3.6',
    install_requires=requirements,
    include_package_data=True)