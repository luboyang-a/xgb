from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "hist",
        sources=["hist.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "tree",
        sources=["tree.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++11"],
    ),
]

setup(
    name="XGB_Lightning",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
)