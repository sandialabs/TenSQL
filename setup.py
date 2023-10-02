#!/usr/bin/env python
"""
Setup script for TenSQL
"""

from setuptools import setup, find_packages, Extension

extra_params = {}
setup_requires = ['cython', 'numpy<1.21', 'wheel']
try:
    import pip
    pip.main(['install', '-v'] + setup_requires)
    setup_requires = []
except Exception:
    # Going to use easy_install for
    traceback.print_exc()

import versioneer
import numpy as np

_DEBUG = False

extra_compile_args = [
  '-fopenmp', '-Wall'
]
if _DEBUG:
  extra_compile_args += [
    '-g'
  ]
else:
  extra_compile_args += [
    '-O3', '-DNDEBUG'
  ]

ext_modules = []
ext_modules.append(
  Extension(
    "tensql._PyHashOCNT",
    ["src/PyHashOCNT.pyx"],
    extra_compile_args=["-std=c++17", "-O3"],
    include_dirs=[np.get_include()],
    language="c++"
  )
)


cffi_modules = []

cffi_modules.append("build_PyObjectHelper.py:ffibuilder")

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['tensql', 'tensql.QIR', 'tensql.LAIR', 'tensql.LAIR.visitors'],
    #packages=find_packages(include=['tensql'], exclude=['tensql.tests', 'tensql.tests.*']),
    ext_modules = ext_modules,
    cffi_modules = cffi_modules
)
