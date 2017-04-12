'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from distutils.core import setup
setup(name='21cmtools',
      version='0.1',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      package_dir = {'21cmtools' : 'source'},
      packages=['21cmtools'],
      )
