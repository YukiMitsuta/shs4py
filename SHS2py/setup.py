#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

from distutils.core import setup
from Cython.Build import cythonize
import numpy

print("numpy.get_include() = %s"%numpy.get_include())
setup(
    name = 'calcgau',
    ext_modules = cythonize('calcgau.pyx'),
    #include_path = [numpy.get_include()],
    include_dirs = [numpy.get_include()]
    )
