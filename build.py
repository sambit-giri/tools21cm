from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

# Define Cython extension modules
extensions = [
    Extension(
        'tools21cm.ViteBetti_cython',
        ['src/tools21cm/ViteBetti_cython.pyx'],
        language="c++",
        include_dirs=[np.get_include()]
    ),
]

def build(setup_kwargs):
    setup_kwargs.update({
        'ext_modules': cythonize(extensions, language_level=3),
        'include_dirs': [np.get_include()],
    })
