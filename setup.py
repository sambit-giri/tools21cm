'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

import setuptools
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy as np
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Check if the Cython source file exists
cython_file = "src/tools21cm/ViteBetti_cython.pyx"

if not os.path.isfile(cython_file):
    raise FileNotFoundError(f"Required file not found: {cython_file}")

# Define the extension module
extensions = [
    Extension(
        'tools21cm.ViteBetti_cython',
        [cython_file],
        language="c++",
        include_dirs=[np.get_include()]
    )
]

setup(
    name='tools21cm',
    version='2.1.10',
    author='Sambit Giri',
    author_email='sambit.giri@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={'tools21cm': ['input_data/*']},
    install_requires=requirements,
    include_package_data=True,
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
