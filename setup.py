from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "hist",
    sources=["hist.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3']
)

setup(
    ext_modules=cythonize(
        [ext],
        compiler_directives={'language_level': "3"}
    )
)