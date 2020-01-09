'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

#from setuptools import setup, find_packages
from distutils.core import setup


setup(name='tools21cm',
      version='2.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      package_dir = {'tools21cm' : 'tools21cm'},
      packages=['tools21cm'],
      package_data={'share':['*'],},
      install_requires=['numpy','scipy','scikit-learn','scikit-image','numba'],
      #include_package_data=True,
)
