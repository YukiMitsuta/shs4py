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
from . import OPT, functions

def main(tspoint, f, grad, hessian, dirname, SHSrank, SHSroot, SHScomm, optdigTH, const):
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
    dflist = []
    dxlist = []
    if SHSrank == SHSroot:
        for i, pm in enumerate([1, -1]):
            writeline = ""
            for p in tspoint:
                writeline += "% 3.10f, "%p
            writeline += "% 3.10f, TS point\n"%(0.0)
            with open("./pathlist%s.csv"%i, "a") as wf:
                wf.write(writeline)
            #pathlist.append(tspoint + pm * const.deltas0 * eigVlist[0])
            x = tspoint + pm * const.deltas0 * eigVlist[0]
            grad_x = grad(x)
            df = - const.deltas0*np.linalg.norm(grad_x)
            pathlist.append(x)
            writeline = ""
            for p in x:
                writeline += "% 3.10f, "%p
            writeline += "% 3.10f, deltas0\n"%np.linalg.norm(grad_x)
            with open("./pathlist%s.csv"%i, "a") as wf:
                wf.write(writeline)
            dflist.append(df)
            dxlist.append(const.deltas0)
            with open("./potential%s.csv"%i, "a") as wf:
                wf.write("% 3.10f, % 3.10f \n"%(0.0,0.0))
                wf.write("% 3.10f, % 3.10f \n"%(const.deltas0,df))
    else:
        pathlist = None
    if const.calc_mpiQ:
        pathlist = SHScomm.bcast(pathlist, root = 0)


    downQlist = [True, True]
    #beforeEgradlist = [None, None]
    whileN = 0
    while any(downQlist):
        whileN += 1
        if 1000 < whileN:
            if SHSrank == SHSroot:
                print("in MinimumPath: whileN over 1000")
            return []
        for i, pm in enumerate([1, -1]):
            if downQlist[i] is False:
                continue
            x = pathlist[i]
            grad_x = grad(x)
            #if np.linalg.norm(grad_x) < const.OPTthreshold * 10.0:
                #grad_x = pm * eigVlist[0]
            #else:
            if const.OPTthreshold*10.0 < np.linalg.norm(grad_x):
                #Egrad = grad_x / np.linalg.norm(grad_x)
                #beforeEgradlist[i] = Egrad
                downQlist[i] = False
                continue
            if np.linalg.norm(grad_x) == 0.0:
                if SHSrank == SHSroot:
                    print("ERROR: gradient become 0.0 in %s"%x)
                return []
            #Egrad = grad_x / np.linalg.norm(grad_x)
            #x -= const.deltas0 * Egrad
            x += pm* const.deltas0 *eigVlist[0]
            #beforeEgradlist[i] = Egrad
            pathlist[i]        = copy.copy(x)
            if SHSrank == SHSroot:
                writeline = ""
                for p in x:
                    writeline += "% 3.10f, "%p
                writeline += "% 3.10f, deltas0\n"%np.linalg.norm(grad_x)
                with open("./pathlist%s.csv"%i, "a") as wf:
                    wf.write(writeline)
                #df = - const.deltas0*np.linalg.norm(grad_x)
                df = - const.deltas0*np.abs(np.dot(grad_x, eigVlist[0]))
                dflist[i] += df
                dxlist[i] += const.deltas0
                with open("./potential%s.csv"%i, "a") as wf:
                    wf.write("% 3.10f, % 3.10f \n"%(dxlist[i],dflist[i]))
            
    downQlist = [True, True]
    before_pathlist = copy.copy(pathlist)

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
            if grad_x is False:
                if SHSrank == SHSroot:
                    print("ERROR: gradient cannot calculate")
                downQlist[i] = False
                continue
                
                
            if np.linalg.norm(grad_x) == 0.0:
                if SHSrank == SHSroot:
                    print("ERROR: gradient become 0.0 in %s"%x)
                return []
            Egrad = grad_x / np.linalg.norm(grad_x)
            if np.linalg.norm(grad_x) < const.OPTthreshold*10:
                EQhess = hessian(x)
                eigNlist, _eigV = np.linalg.eigh(EQhess)
                if min(eigNlist) < 0.0:
                    print("MFEP was stoped at TS point")
                    eigV = _eigV[:,0]
                    beforedamp = functions.periodicpoint(before_pathlist[i], const, pathlist[i])
                    pathvector = pathlist[i] - beforedamp
                    if np.dot(pathvector, eigV) < 0.0:
                        eigV *= -1.0
                    x += eigV*const.deltas0
                    if SHSrank == SHSroot:
                        #dflist[i] -= const.deltas0*np.linalg.norm(grad_x)
                        dflist[i] -= const.deltas0*np.abs(np.dot(grad_x, eigV))
                        dxlist[i] += const.deltas0
                else:
                    print("norm(grad_x) = %s < %s; stop MinimumPath"%(np.linalg.norm(grad_x),const.OPTthreshold*10))
                    eqpoint, eqhess_inv = OPT.PoppinsMinimize(x, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
                    xdamp = functions.periodicpoint(x, const, eqpoint)
                    deltavec = eqpoint - xdamp
                    delta = np.linalg.norm(deltavec)
                    if type(const.sameEQthreshold) is float:
                        disMax = max([abs(x) for x in delta])
                        if const.sameEQthreshold < disMax:
                            samepointQ = False
                        else:
                            samepointQ = True
                    elif type(const.sameEQthreshold) is list:
                        samepointQ = all([ abs(x) < const.sameEQthreshold[i] for i,x in enumerate(deltavec)])
                    if samepointQ:
                        grad_x = grad(x)
                        if SHSrank == SHSroot:
                            dflist[i] -= delta*np.linalg.norm(grad_x)
                            dxlist[i] += delta
                        #x += eigV*const.deltas0
                        grad_x = grad(eqpoint)
                        pathlist[i] = eqpoint
                        downQlist[i] = False
                    else:
                        grad_x = grad(x)
                        x += deltavec/delta*const.deltas0
                        if SHSrank == SHSroot:
                            dflist[i] -= delta*np.abs(np.dot(deltavec/delta,grad_x))
                            dxlist[i] += const.deltas0
            #if np.dot(Egrad, beforeEgradlist[i]) < 0.0:
                #print("find near point of new EQ")
                #downQlist[i] = False
            else: 
                x -= const.deltas * Egrad
                if SHSrank == SHSroot:
                    dflist[i] -= const.deltas*np.linalg.norm(grad_x)
                    dxlist[i] += const.deltas
            pathlist[i] = functions.periodicpoint(pathlist[i], const, np.zeros(const.dim))
            
            os.chdir(dirname)
            if SHSrank == SHSroot:
                writeline = ""
                for p in pathlist[i]:
                    writeline += "% 3.10f, "%p
                #writeline += ":% 3.10f\n"%np.dot(Egrad, beforeEgradlist[i])
                writeline += "% 3.10f, deltas\n"%np.linalg.norm(grad_x)
                with open("./pathlist%s.csv"%i, "a") as wf:
                    wf.write(writeline)
                with open("./potential%s.csv"%i, "a") as wf:
                    wf.write("% 3.10f, % 3.10f\n"%(dxlist[i],dflist[i]))
            if downQlist[i]:
                #beforeEgradlist[i] = copy.copy(Egrad)
                before_pathlist[i] = copy.copy(pathlist[i])
                pathlist[i]        = copy.copy(x)
    os.chdir("../")
    pathlist = [[pathlist[i], dxlist[i], -dflist[i]] for i in range(2)]
    return pathlist

