from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'tools21cm.ViteBetti_cython',
        ['src/tools21cm/ViteBetti_cython.pyx'],
        language="c++",
        include_dirs=[np.get_include()]
    ),
]

setup(ext_modules=cythonize(extensions, language_level=3))
