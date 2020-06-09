#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
the calculation of Optimization of EQ and TS points.

Available functions:
    PoppinsMinimize:      optimization of minimize EQ points by using L-BFGS-B
    PoppinsNewtonRaphson: optimization of minimize EQ points by using NewtonRaphson
    PoppinsDimer:         optimization to find TS points by using Dimer method
    cons:                 construction to restrect L-BFGS-B calculation to f(x) < optdigTH
    calcboundmax:         calculation of the max of x to f(x) < optdigTH  in the dimer calculation

"""

import os, glob, shutil, sys, re 
import time, datetime, copy, inspect, fcntl
import numpy      as np
from   scipy.optimize import minimize


#from . import const
from . import functions

import fasteners

def PoppinsMinimize(initialpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const):
    """
    PoppinsMinimize:      optimization of minimize EQ points by using L-BFGS-B
    """

    boundlist = []
    if const.periodicQ:
        if type(const.periodicmin) is float:
            for i in range(len(initialpoint)):
                boundlist.append((initialpoint[i] + const.periodicmin,
                                  initialpoint[i] + const.periodicmax))
        else:
            for i in range(len(initialpoint)):
                boundlist.append((initialpoint[i] + const.periodicmin[i],
                                  initialpoint[i] + const.periodicmax[i]))
    else:
        for i in range(len(initialpoint)):
            boundlist.append((const.wallmin[i], const.wallmax[i]))

    if True:
        #result = minimize(f, initialpoint, jac=grad, hess=hessian)
        result = minimize(f, initialpoint, jac=grad, bounds = boundlist, method="L-BFGS-B")
        #result = minimize(f, initialpoint, jac=grad, method="BFGS")
    else:
        f_ini = f(initialpoint)
        if optdigTH < f_ini:
            return initialpoint, 0.0
        consf = (
                {'type': 'ineq', 'fun': lambda x: cons(x, f, optdigTH)}
                )
        result = minimize(f, initialpoint, jac=grad, constraints=consf, method="SLSQP")
    #if SHSrank == SHSroot:
        #print("%s: %s"%(SHSrank, np.linalg.norm(result.x - initialpoint)))
        #print("%s: %s"%(SHSrank, result.message))
        
    x_0 = functions.periodicpoint(result.x, const)
    #exit()
    #return x_0, result.hess_inv
    return x_0, 0.0
def PoppinsNewtonRaphson(initialpoint, f, grad, hessian, const):
    """
    PoppinsNewtonRaphson: optimization of minimize EQ points by using NewtonRaphson
    """

    result = copy.copy(initialpoint)
    dim    = len(result)
    whileN = 0
    x0list = []
    while whileN < 1000:
        whileN += 1
        hessinv      = hessian(result)
        hessinv      = np.linalg.inv(hessinv)
        graddamp     = grad(result)
        if np.linalg.norm(graddamp) < const.threshold:
            break
        beforeresult = copy.copy(result)
        for x in range(dim):
            for y in range(dim):
                result[x] -= hessinv[x, y] * graddamp[y]
    else:
        print("in PoppinsNewtonRaphson, whileN over 1000")
        return False
    return result
def PoppinsDimer(initialpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const):
    """
    PoppinsDimer:         optimization to find TS points by using Dimer method
    """
    x_0 = copy.copy(initialpoint)
    x_0 = functions.periodicpoint(x_0, const)
    #print("%s: 61"%(SHSrank), flush = True)
    g0 = grad(x_0)
    #print("%s: 63"%(SHSrank), flush = True)
    if np.linalg.norm(g0) == 0.0:
        return x_0
    #print("%s: 66"%(SHSrank), flush = True)
    Egrad = g0 / np.linalg.norm(g0)
    x1   = x_0 + const.Ddimer * Egrad
    x2   = x_0 - const.Ddimer * Egrad
    tau  = (x1 - x2) * 0.5
    whileN = 0
    #print("%s: 69"%(SHSrank), flush = True)
    Ddimerdamp = copy.copy(const.Ddimer)
    NRerrorN = 0
    while whileN < 1000:
        whileN += 1
        phiturnN = 0
        phiendQ = True
        while phiturnN < 100:
            #print("%s: 74"%(SHSrank), flush = True)
            phiturnN += 1
            Etau = tau / np.linalg.norm(tau)
            g0   = grad(x_0)
            g1   = grad(x1)
            F_R  = -2.0 * (g1 - g0) + 2.0 * np.dot(g1 - g0, Etau) * Etau
            if np.linalg.norm(F_R) == 0.0:
                break
            TH   = F_R / np.linalg.norm(F_R)
            Ctau    =       np.dot(g1 - g0, Etau) / Ddimerdamp
            DelCtau = 2.0 * np.dot(g1 - g0, TH  ) / Ddimerdamp
            phi1    = -0.5 * np.arctan(DelCtau * 0.5 / abs(Ctau))
            if abs(phi1) < const.phitol:
                break
            x1prime   = x_0 + Etau * Ddimerdamp * np.cos(phi1) + TH * Ddimerdamp * np.sin(phi1)
            g1prime   = grad(x1prime)
            tauprime  = x1prime - x_0
            Etauprime = tauprime / np.linalg.norm(tauprime)
            Ctauprime = np.dot(g1prime - g0, Etauprime) / Ddimerdamp

            b1 = DelCtau * 0.5
            a1 = (Ctau - Ctauprime + b1 * np.sin(2.0 * phi1)) / (1.0 - np.cos(2.0 * phi1))
            a0 = 2.0 * (Ctau - a1)

            phiMIN  = 0.5 * np.arctan(b1 / a1)
            x1MIN   = x_0 + Etau * Ddimerdamp * np.cos(phiMIN) + TH * Ddimerdamp * np.sin(phiMIN)
            tauMIN  = x1MIN - x_0
            EtauMIN = tauMIN / np.linalg.norm(tauMIN)
            gMIN    = grad(x1MIN)
            CtauMIN = np.dot(g1prime - g0, Etauprime) / Ddimerdamp
            if Ctau < CtauMIN:
                phiMIN += np.pi * 0.5
            x1MIN   = x_0 + Etau * Ddimerdamp * np.cos(phiMIN) + TH * Ddimerdamp * np.sin(phiMIN)
            tauMIN  = x1MIN - x_0
            EtauMIN = tauMIN / np.linalg.norm(tauMIN)
            gMIN    = grad(x1MIN)
            CtauMIN = np.dot(g1prime - g0, Etauprime) / Ddimerdamp

            x1  = copy.copy(x1MIN)
            tau = copy.copy(tauMIN)
            if abs(phiMIN) < const.phitol:
                break
            #print("%s: 116"%(SHSrank), flush = True)
            #if SHSrank == SHSroot:
                #print("phiMIN = %s"%phiMIN, flush = True)
        else:
            Ddimerdamp += const.Ddimer
            x1   = x_0 + Ddimerdamp * Egrad
            x2   = x_0 - Ddimerdamp * Egrad
            tau  = (x1 - x2) * 0.5
            #if SHSrank == SHSroot:
                #print("in PoppinsDimer, phiturnN over 100", flush = True)
                #print("Ddimer is changed to %s"%Ddimerdamp, flush = True)
            if Ddimerdamp <= const.Ddimer_max:
                continue
            #return False
            #c = inspect.currentframe()
            #with open("./optlist.dat","a") as wf:
                #wf.write("#ERROR(%s): cannot calculate phi \n"%c.f_lineno)
            phiendQ = False
        if phiendQ:
            F_T = -g0 + 2.0 * np.dot(g0, Etau) * Etau
            if optdigTH is False:
                result = minimize(lambda x: np.linalg.norm(grad(x_0 + F_T * x)),
                            x0 = 0.0, method = "L-BFGS-B")
                resultx = result.x
                x_0 = x_0 + F_T * result.x
            else:
                f_0 = f(x_0)
                if optdigTH < f_0:
                    if SHSrank == SHSroot:
                        c = inspect.currentframe()
                        print("ERROR(%s): %s: f(x_0) = %s over optdigT, flush = TrueH(% 3.2f)"%(c.f_lineno, whileN, f_0, optdigTH), flush = True)
                    return False
                #consf = (
                    #{'type': 'ineq', 'fun': lambda x: cons(x_0 + F_T * x, f, optdigTH)}
                    #)
                #result = minimize(lambda x: np.linalg.norm(grad(x_0 + F_T * x)),
                            #x0 = 0.0, constraints=consf, method = "cobyla")
                            #x0 = 0.0, constraints=consf, method = "SLSQP")
                xmax = calcboundmax(x_0, F_T, f, optdigTH, const, Ddimerdamp)
                #if SHSrank == SHSroot:
                    #print("xmax = %s"%xmax, flush = True)
                #if SHSrank == SHSroot:
                    #print("%s:start minimize"%datetime.datetime.now(), flush = True)
                result = minimize(lambda x: np.linalg.norm(grad(x_0 + F_T * x, debagQ = True)),
                            x0 = 0.0, bounds = [(0.0, xmax)], method = "L-BFGS-B")
                #print("%s: %s"%(SHSrank,xmax), flush = True)
                #if SHSrank == SHSroot:
                    #print("%s:end minimize"%datetime.datetime.now(), flush = True)
                resultx = result.x[0]
                x_0 = x_0 + F_T * result.x
        else:
            resultx = 0.0
        if resultx < 1.0e-5:
            Ddimerdamp += const.Ddimer
            x1   = x_0 + Ddimerdamp * Egrad
            x2   = x_0 - Ddimerdamp * Egrad
            tau  = (x1 - x2) * 0.5
            #if SHSrank == SHSroot:
                #print("resultx = 0", flush = True)
                #print("Ddimer is changed to %s"%Ddimerdamp, flush = True)
            if Ddimerdamp <= const.Ddimer_max:
                continue
        x_0 = functions.periodicpoint(x_0, const)
        f_0 = f(x_0)
        #if SHSrank == SHSroot:
            #print("resultx, f_0 = %s, %s"%(resultx, f_0))
        graddamp     = grad(x_0)
        if SHSrank == SHSroot:
            print("%s: norm(grad) = %s"%(whileN, np.linalg.norm(graddamp)), flush = True)
        if np.linalg.norm(graddamp) < const.threshold:
            return x_0
        elif optdigTH < f_0:
            if SHSrank == SHSroot:
                print("ERROR: %s: f(x_0) = %s over optdigTH(% 3.2f)"%(whileN, f_0, optdigTH))
            return False
        #if result.x <= const.threshold:
        if resultx < 1.0e-5:
            hessinv      = hessian(x_0)
            #if SHSrank == SHSroot:
                #print("Try Newton-Raphthon", flush = True)
                #if len(hessinv) == 1:
                    #print("ERROR: hessinv.ndim = 1", flush = True)
                    #print("hessinv = %s"%hessinv,    flush= True)
            hessinv      = np.linalg.inv(hessinv)
            graddamp     = grad(x_0)
            dim          = len(x_0)
            #beforeresult = copy.copy(x_0)
            x_delta      = np.zeros(dim)
            for i in range(dim):
                for j in range(dim):
                    #x_0[i] -= hessinv[i, j] * graddamp[j]
                    x_delta[i] -= hessinv[i, j] * graddamp[j]
            f_next = f(x_0 + x_delta)
            #if SHSrank == SHSroot:
                #print("norm(x_delta) = %s"%np.linalg.norm(x_delta))
                #print("f_next        = %s"%f_next)
            if not optdigTH is False:
                if optdigTH < f_next:
                    if SHSrank == SHSroot:
                        c = inspect.currentframe()
                        print("ERROR(%s) %s: f(x_next) = %s over optdigTH(% 3.2f)"%(c.f_lineno, whileN, f_next, optdigTH), flush = True)
                    while True:
                        x_delta *= 0.5
                        if SHSrank == SHSroot:
                            print("np.linalg.norm(x_delta) = %s"%np.linalg.norm(x_delta))
                        if np.linalg.norm(x_delta) < 1.0e-5:
                            if SHSrank == SHSroot:
                                print("ERROR: cannot move: return False")
                            return False
                        f_next = f(x_0 + x_delta)
                        if f_next < optdigTH:
                            #x_0 = x_0 + x_delta
                            break

                    #x_0 += xmax * x_delta / np.linalg.norm(x_delta)
                #else:
                    #x_0 += x_delta 
            #else:
                #x_0 += x_delta 
            g_next = grad(x_0 + x_delta)
            if np.linalg.norm(graddamp) < np.linalg.norm(g_next):
            #if False:
                if SHSrank == SHSroot:
                    c = inspect.currentframe()
                    print("ERROR(%s): %s: Norm(graddamp) < Norm(g_next)"%(c.f_lineno, whileN), flush = True)
                if 20 < NRerrorN:
                    return False
                NRerrorN += 1
                while True:
                    x_delta *= 0.5
                    #if SHSrank == SHSroot:
                        #print("np.linalg.norm(x_delta) = %s"%np.linalg.norm(x_delta), flush = True)
                    if np.linalg.norm(x_delta) < 10e-5:
                        if SHSrank == SHSroot:
                            print("ERROR: cannot move: return False")
                        return False
                    g_next = grad(x_0 + x_delta)
                    if np.linalg.norm(g_next) < np.linalg.norm(graddamp):
                        x_0 = x_0 + x_delta
                        break
            else:
                x_0 += x_delta 
            Ddimerdamp = copy.copy(const.Ddimer)
            x1   = x_0 + Ddimerdamp * Egrad
            x2   = x_0 - Ddimerdamp * Egrad
            tau  = (x1 - x2) * 0.5
            #if SHSrank == SHSroot:
                #print("Ddimer is changed to %s"%Ddimerdamp)
                #print("x_delta_ini = %s"%np.linalg.norm(x_0 - initialpoint), flush = True)

    else:
        if SHSrank == SHSroot:
            print("in PoppinsDimer, whileN over 1000")
            print("x_0 = %s"%x_0)
        return False
    return False
def cons(x, f, optdigTH):
    """
    constructionn to restrect L-BFGS-B calculation to f(x) < optdigTH
    """
    return optdigTH - f(x)
def calcboundmax(x_0, F_T, f, optdigTH, const, Ddimerdamp):
    """
    calcboundmax:         calculation of the max of x to f(x) < optdigTH  in the dimer calculation
    """
    #xdelta = copy.copy(const.Ddimer)
    xdelta = copy.copy(Ddimerdamp)
    xmax   = 0.0
    whileN = 0
    while True:
        whileN += 1
        xmax  += xdelta
        xpoint = x_0 + F_T * xmax
        f_x = f(xpoint)
        if optdigTH < f_x:
            xmax -= xdelta
            break
        if np.pi < xmax:
            return xmax
    while True:
        whileN += 1
        if 1000 < whileN:
            print(" in OPT.calcboundmax, whileN over 1000")
            return 0.0
        xdelta *= 0.5
        if xdelta < 0.001:
            break
        xmax  += xdelta
        xpoint = x_0 + F_T * xmax
        f_x = f(xpoint)
        if optdigTH < f_x:
            xmax -= xdelta
    return xmax

