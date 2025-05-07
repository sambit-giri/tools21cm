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

# Check if the Cython source files exist
cython_files = [
    "src/tools21cm/ViteBetti_cython.pyx",
]

# Check if each Cython file exists
for cython_file in cython_files:
    if not os.path.isfile(cython_file):
        raise FileNotFoundError(f"Required file not found: {cython_file}")

# Define the extension modules
extensions = [
    Extension(
        'tools21cm.ViteBetti_cython',
        [cython_files[0]],
        language="c++",
        include_dirs=[np.get_include()]
    ),
]

setup(
    name='tools21cm',
    version='2.3.1',
    author='Sambit Giri',
    author_email='sambit.giri@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={'tools21cm': ['input_data/*']},
    install_requires=requirements,
    setup_requires=[
        'setuptools>=18.0',
        'cython',
        'numpy',
    ],
    include_package_data=True,
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    options={
        "bdist_wheel": {"universal": True}  # Only if truly OS-independent!
    },
)
