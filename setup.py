'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

from Cython.Build import cythonize
import numpy as np

setup(name='tools21cm',
      version='2.1.8',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={'tools21cm': ['input_data/*']},
      install_requires=['numpy', 'scipy', 'matplotlib', 'numba',
                        'scikit-learn', 'scikit-image', 'astropy',
                        'tqdm', 'joblib', #'pyfftw', #'pyfits',
                        'pytest', 'cython', 'pandas'],
      include_package_data=True,
      ext_modules = cythonize("src/tools21cm/ViteBetti_cython.pyx", language="c++"),
      include_dirs=[np.get_include()],
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
)
