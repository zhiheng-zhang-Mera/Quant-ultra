from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ROOT = Path(__file__).parent
extensions = [Extension("quant_ultra_fast", ["cython/quant_ultra_fast.pyx"], include_dirs=[numpy.get_include()])]
setup(name="quant-ultra-fast", version="0.1.0", ext_modules=cythonize(extensions, compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False, "cdivision": True}))

