#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
the calculation of minimum path that is calculated to gradient down form Transiton state

Available functions:
    main: main part of the calculation of minimum path

"""

import os, glob, shutil, re, gc
import time, math, copy, inspect, datetime
import numpy as np
from   scipy.optimize import minimize

#from . import const, functions, IOpack
from . import functions, IOpack

def main(tspoint, f, grad, hessian, dirname, SHSrank, SHSroot, SHScomm, const):
    """
    main: main part of the calculation of minimum path
    Args:
        tspoint : Coordinate of ts point
        f       : function to calculate potential as f(x)
        grad    : function to calculate gradient as grad(x)
        hessian : function to calculate hessian as hessian(x)
        dirname : name of directory to calculate minimum path
        SHSrank : rank of MPI
        SHSroot : root rank of MPI
        SHScomm : communicate class of mpi4py
        const   : class of constants

    """
    if SHSrank == SHSroot:
        print("start Minimum Path calculation of %s"%dirname)
    os.chdir(dirname)
    dim    = len(tspoint)
    TShess = hessian(tspoint)
    eigNlist, _eigV = np.linalg.eigh(TShess)
    eigVlist = []
    for i in range(dim):
        eigVlist.append(_eigV[:,i])
    pathlist  = []
    if SHSrank == SHSroot:
        for i, pm in enumerate([1, -1]):
            writeline = ""
            for p in tspoint:
                writeline += "% 3.10f, "%p
            writeline += ": TS point\n"
            with open("./pathlist%s.csv"%i, "a") as wf:
                wf.write(writeline)
            #pathlist.append(tspoint + pm * const.deltas0 * eigVlist[0])
            x = tspoint + pm * const.deltas0 * eigVlist[0]
            pathlist.append(x)
            writeline = ""
            for p in x:
                writeline += "% 3.10f, "%p
            writeline += ":deltas0 \n"
            with open("./pathlist%s.csv"%i, "a") as wf:
                wf.write(writeline)
    else:
        pathlist = None
    if const.calc_mpiQ:
        pathlist = SHScomm.bcast(pathlist, root = 0)


    downQlist = [True, True]
    beforeEgradlist = [None, None]
    whileN = 0
    while any(downQlist):
        whileN += 1
        if 1000 < whileN:
            if SHSrank == SHSroot:
                print("in MinimumPath: whileN over 1000")
            return []
        for i, pm in enumerate([-1, 1]):
            if downQlist[i] is False:
                continue
            x = pathlist[i]
            grad_x = grad(x)
            if np.linalg.norm(grad_x) < const.threshold * 5.0:
                grad_x = pm * eigVlist[0]
            else:
                Egrad = grad_x / np.linalg.norm(grad_x)
                beforeEgradlist[i] = Egrad
                downQlist[i] = False
                continue

            if np.linalg.norm(grad_x) == 0.0:
                if SHSrank == SHSroot:
                    print("ERROR: gradient become 0.0 in %s"%x)
                return []
            Egrad = grad_x / np.linalg.norm(grad_x)
            x -= const.deltas0 * Egrad
            beforeEgradlist[i] = Egrad
            pathlist[i]        = copy.copy(x)
            if SHSrank == SHSroot:
                writeline = ""
                for p in x:
                    writeline += "% 3.10f, "%p
                writeline += ":% 3.10f\n"%np.linalg.norm(grad_x)
                with open("./pathlist%s.csv"%i, "a") as wf:
                    wf.write(writeline)
    downQlist = [True, True]

    while any(downQlist):
        whileN += 1
        if 1000 < whileN:
            if SHSrank == SHSroot:
                print("in MinimumPath: whileN over 1000")
            return []
            #break
        for i in [0, 1]:
            if not downQlist[i]:
                continue
            x = pathlist[i]
            grad_x = grad(x)
            if np.linalg.norm(grad_x) == 0.0:
                if SHSrank == SHSroot:
                    print("ERROR: gradient become 0.0 in %s"%x)
                return []
            Egrad = grad_x / np.linalg.norm(grad_x)
            if np.dot(Egrad, beforeEgradlist[i]) < 0.0:
                #print("find near point of new EQ")
                downQlist[i] = False
            x -= const.deltas * Egrad
            if SHSrank == SHSroot:
                writeline = ""
                for p in x:
                    writeline += "% 3.10f, "%p
                #writeline += ":% 3.10f\n"%np.dot(Egrad, beforeEgradlist[i])
                writeline += ":% 3.10f\n"%np.linalg.norm(grad_x)
                with open("./pathlist%s.csv"%i, "a") as wf:
                    wf.write(writeline)
            beforeEgradlist[i] = copy.copy(Egrad)
            pathlist[i]        = copy.copy(x)
    os.chdir("../")
    return pathlist

