#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

from distutils.core import setup
from Cython.Build import cythonize
import numpy

from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler import Options


print("numpy.get_include() = %s"%numpy.get_include())
setup(
    name = 'calcgau',
    ext_modules = cythonize('calcgau.pyx'),
    #include_path = [numpy.get_include()]
    include_dirs = [numpy.get_include()]
    )

#setup(
    #name = 'calcVES',
    #ext_modules = cythonize('calcVES.pyx'),
    #include_dirs = [numpy.get_include()]
    #)

ext_modules = [Extension("calcRCMC", ["calcRCMC.pyx"], language="c++")]#,
               #Extension("module2", ["module2.pyx"], language="c++")]

setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules, include_dirs = [numpy.get_include()])

#setup(
    #name = 'calcRCMC',
    #ext_modules = cythonize('calcRCMC.pyx'),
    ##include_path = [numpy.get_include()]
    #include_dirs = [numpy.get_include()]
    ##)
