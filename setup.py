'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
setup(name='py21cmtools',
      version='0.1',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      package_dir = {'py21cmtools' : 'src'},
      packages=['py21cmtools'],
      include_package_data=True,
)
