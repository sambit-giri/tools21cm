'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup


setup(name='tools21cm',
      version='2.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      packages=find_packages("src"),
      package_dir={"": "src"},
      # package_dir = {'tools21cm' : 'src'},
      # packages=['tools21cm'],
      # package_data={'share':['*'],},
      package_data={'tools21cm': ['input_data/*']},
      install_requires=['numpy', 'scipy', 'matplotlib', 'numba',
                        'scikit-learn', 'scikit-image', 'astropy',
                        'tqdm', 'joblib', 'pyfftw', #'pyfits',
                        'cosmospectra', #'cosmospectra@git+https://github.com/sambit-giri/cosmospectra.git',
                        'pytest'],
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
)
