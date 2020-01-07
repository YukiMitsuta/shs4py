#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
the calculation of anharmonic downword distorsion (ADD).

Available class:
    ADDthetaClass: the deta class of one ADD
Available functions:
    main: main part of ADD calculation.
    Opt_hyper_sphere:  Optimization on a hypersphere

"""

import os, glob, shutil, re, gc
import time, math, copy, inspect, datetime
import numpy as np
from   scipy.optimize import minimize

from . import functions, IOpack

class ADDthetaClass(object):
    """
    ADDthetaClass: the deta class of one ADD

    Pblic attributes:
        IDnum      : The ID of this Class in ADDths
        dim        : Dimension of CVs
        SQ         : Matrix of axes of Supersphere 
        SQ_inv     : Inerse of SQ
        nADD       : Vector for ADD from eq point
        thetalist  : Angle list of polar coordinate of the nADD
        x          : Point of ADD on spheresphere in cartitian coordinate
        A          : Potential on x
        ADD        : Potential of ADD
        ADD_IOE    : Potential of ADD after IOE is applied
        findTSQ    : Is TS point is found along this ADD? (if it is found, findTSQ = True)
        ADDoptQ    : IS this ADD is optimised on the SH?
        ADDremoveQ : If ADDremoveQ = True, the optimization of this ADD is skilped 
        grad       : The gradient on x
        grad_vec   : The dot of grad and nADD (if grad_vec < 0, the ADD is over TS)
        
    Available functions:
        calc_onHS : calculation of f on hypersphere
        f_IOE     : IOE calculation of f
        IOE_total : sum up of IOE ilumination
        IOE       : IOE calculation between self and neiborADDth
        grad_theta: the gradient along theta (this function is not work now).
    """
    def SuperSphere_cartesian(self, A, thetalist, const):
        """
        vector of super sphere by cartetian 
        the basis transformathon from polar coordinates to cartesian coordinates
        {sqrt(2*A), theta_1,..., theta_n-1} ->  {q_1,..., q_n} -> {x_1, x_2,..., x_n}
        cf: thetalist = {theta_n-1,..., theta_1}

        Args:
            A          : Potential on the end of the vector
            thetalist  : Angle list of polar coordinate of the vector
            const      : Class of constants
        """
        if const.cythonQ:
            return const.calcgau.SuperSphere_cartesian(A, thetalist, self.SQ, self.dim)

        qlist = [np.sqrt(2.0 * A) for i in range(self.dim)]
        a_k = 1.0
        for i, theta in enumerate(thetalist):
            qlist[i] *= a_k * np.cos(theta)
            a_k *= np.sin(theta)
        qlist[-1] *= a_k

        SSvec =  np.dot(self.SQ, qlist)
        return SSvec
    def calc_onHS(self, deltaTH, func, eqpoint, IOEsphereA, A_eq, ADDths, const):
        """
        calc_onHS: calculation of f on hypersphere

        Args:
            deltaTH    : difference from self.thetalist
            func       : function of f
            eqpoint    : CVs of eqpoint
            IOEsphereA : harmonic potential of the supersphere that is calculated now
            A_eq       : potential of EQ point
            ADDths     : list of ADDthetaClass
            const      : Class of constants
        """
        thetalist = self.thetalist + deltaTH 
        nADD = self.SuperSphere_cartesian(IOEsphereA, thetalist, const)
        x = eqpoint + nADD
        x = functions.periodicpoint(x, const)
        return  self.f_IOE(nADD, x, func, IOEsphereA, A_eq, ADDths, const)
    def f_IOE(self, nADD, x, f, IOEsphereA, A_eq, ADDths, const):
        """
        f_IOE: IOE calculation of f
        """
        result = f(x) - IOEsphereA - A_eq
        ADDhere = result
        #result += self.IOE_total(nADD, ADDths, const)
        #result += self.IOE_total(nADD, ADDths, const)
        resultIOE =  self.IOE_total(nADD, ADDths, ADDhere, const)
        return result + resultIOE
    def IOE_total(self, nADD, ADDths, ADDhere, const):
        """
        IOE_total: sum up of IOE ilumination
        """
        result = 0.0
        if const.calc_mpiQ and const.SHSsize < len([ADDth for ADDth in ADDths if not ADDth.ADDoptQ]):
        #if False:
        #if const.calc_IOEmpiQ:
            IOEcounter  = 0
            for neiborADDth in ADDths:
                if self.IDnum == neiborADDth.IDnum:
                    continue
                if neiborADDth.ADDoptQ:
                    continue
                if neiborADDth.ADD <= self.ADD:
                #if neiborADDth.ADD <= ADDhere:
                    if IOEcounter % const.SHSsize == const.SHSrank:
                        result -= self.IOE(nADD, neiborADDth, const)
                    IOEcounter += 1
            resultg = const.SHScomm.gather(result, root=0)
            if const.SHSrank == const.SHSroot:
                allresult = 0.0
                for res in resultg:
                    allresult += res
            else:
                allresult = None
            allresult = const.SHScomm.bcast(allresult, root=0)
            return allresult
        else:
            for neiborADDth in ADDths:
                if self.IDnum == neiborADDth.IDnum:
                    continue
                if neiborADDth.ADDoptQ:
                    continue
                #if neiborADDth.ADD <= ADDhere:
                if neiborADDth.ADD <= self.ADD:
                    result -= self.IOE(nADD, neiborADDth, const)
            return result
    def IOE(self, nADD, neiborADDth, const):
        """
        IOE      : IOE calculation between self and neiborADDth
        """
        if const.cythonQ:
            return  const.calcgau.IOE(nADD, neiborADDth.nADD, self.SQ_inv, neiborADDth.ADD_IOE)
        deltaTH = functions.angle_SHS(nADD, neiborADDth.nADD, self.SQ_inv, const)
        if deltaTH <= np.pi * 0.5:
            cosdamp = np.cos(deltaTH)
            return neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
        else:
            return 0.0
    def IOE_grad(self, nADD, r, neiborADDth, const):
        if const.cythonQ:
            return  const.calcgau.IOE_grad(nADD, neiborADDth.nADD, self.SQ_inv, neiborADDth.ADD_IOE, r)
        #deltaTH = functions.angle_SHS(nADD, neiborADDth.nADD, self.SQ_inv, const)
        q_x     = np.dot(self.SQ_inv, nADD)
        q_y     = np.dot(self.SQ_inv, neiborADDth.nADD)
        deltaTH = functions.angle(q_x, q_y)
        if np.pi * 0.5 < deltaTH:
            return 0.0
        cosdamp = np.cos(deltaTH)
        xydot   = np.dot(q_x, q_y)
        returngradlist = []
        for i in range(len(nADD)):
            returngrad   = q_y[i] * r * r
            returngrad  -= 2.0 * q_x[i] * xydot
            returngradlist.append(returngrad)
        returngradlist  = np.array(returngradlist)
        returngradlist *= + 3.0 * neiborADDth.ADD_IOE * cosdamp * cosdamp / r / r / r / r
        return returngradlist
    def grad_theta(self, f, grad, eqpoint, IOEsphereA, deltaTH, A_eq, ADDths, const):
        """
        the gradient along theta (this function is not work now.)
        cf: thetalist = {theta_n-1,..., theta_1}
        """
        thetalist   = self.thetalist + deltaTH 
        nADD        = self.SuperSphere_cartesian(IOEsphereA, thetalist, const)
        r           = np.sqrt(2.0 * IOEsphereA)
        tergetpoint = eqpoint + nADD
        if const.periodicQ:
            tergetpoint = functions.periodicpoint(tergetpoint, const)
        grad_x = grad(tergetpoint)
        #f_x    = f(tergetpoint)
        #ADD_x  = f_x - IOEsphereA - A_eq
        ADD_x  = 0.0
        #grad_q = np.dot(self.SQ_inv, grad_x)
        grad_q = np.dot(grad_x, self.SQ)
        thetalistdamp = list(thetalist)
        #thetalistdamp.reverse()
        #thetalistdamp = np.array(thetalistdamp)
        returngrad = [0.0 for _ in range(len(thetalist))]
        delfdamp = self.delf_delx(grad_q, ADD_x, ADDths, nADD, r, const)
        for x_i in range(len(nADD)):
            for theta_i in range(len(returngrad)):
                if x_i < theta_i:
                    continue
                returngrad[theta_i] += self.delx_deltheta(r, thetalistdamp, theta_i, x_i) * \
                                        delfdamp[x_i]
        #returngrad.reverse()
        returngrad = np.array(returngrad)
        return returngrad
    def delx_deltheta(self, r, thetalistdamp, theta_i, x_i):
        returngrad = copy.copy(r)
        #if x_i == 0:
            #if theta_i == 0:
                #returngrad *= - np.sin(thetalistdamp[0])
            #else:
                #returngrad *= 0.0
            #return returngrad
        for i in range(x_i - 1):
            if i == theta_i:
                returngrad *= np.cos(thetalistdamp[i])
            else:
                returngrad *= np.sin(thetalistdamp[i])
        #if x_i == len(thetalistdamp):
            #if theta_i == x_i:
                #returngrad *= np.cos(thetalistdamp[-1])
            #else:
                #returngrad *= np.sin(thetalistdamp[-1])
        #else:
        if x_i != len(thetalistdamp):
            if x_i == theta_i:
                returngrad *= - np.sin(thetalistdamp[x_i])
            else:
                returngrad *=   np.cos(thetalistdamp[x_i])
        return returngrad
    def delf_delx(self, grad_q, ADD_x, ADDths, nADD, r, const):
        returngrad = copy.copy(grad_q)
        return returngrad
        for neiborADDth in ADDths:
            if self.IDnum == neiborADDth.IDnum:
                continue
            if neiborADDth.ADDoptQ:
                continue
            #if neiborADDth.ADD <= ADD_x:
            if neiborADDth.ADD <= self.ADD:
                returngrad += self.IOE_grad(nADD, r, neiborADDth, const)
        return returngrad
def main(eqpoint, f, grad, hessian, dirname, optdigTH, SHSrank, SHSroot, SHScomm, SHSsize, const):
    """
    main part of ADD calculation.
    """
    const.SHSrank = SHSrank
    const.SHSroot = SHSroot
    const.SHScomm = SHScomm
    const.SHSsize = SHSsize
    if SHSrank == SHSroot:
        print("start ADD calculation of %s"%dirname, flush = True)
    os.chdir(dirname)
    dim    = len(eqpoint)
    EQhess = hessian(eqpoint)
    eigNlist, _eigV = np.linalg.eigh(EQhess)
    eigVlist = []
    for i in range(dim):
        eigVlist.append(_eigV[:,i])
    eigVlist = np.array(eigVlist)
    if min(eigNlist) < 0.0:
        print("ERROR: %s is not EQ point!"%eqpoint, flush = True)
        TSinitialpoints = []
        return TSinitialpoints

    thetalist = functions.calctheta(eigVlist[-1], eigVlist, eigNlist)
    A_eq         = f(eqpoint)
    IOEsphereA_initial = functions.calcsphereA(const.IOEsphereA_initial * eigVlist[-1], eigNlist, eigVlist)
    IOEsphereA   = copy.copy(IOEsphereA_initial)
    IOEsphereA_r = functions.calcsphereA(const.IOEsphereA_dist * eigVlist[-1], eigNlist, eigVlist)
    if SHSrank == SHSroot:
        print("IOEsphereA_initial, IOEsphereA_r = %s, %s"%(IOEsphereA_initial, IOEsphereA_r), flush = True)
    if SHSrank == SHSroot:
        with open("./sphere.txt", "a")  as wf:
            wf.write("-*-"*10 + "\n")
            wf.write("start initial optimization\n")
            wf.write("-*-"*10 + "\n")
    IOEsphereA_before = 0.0

    ADDths = []
    lenADDthsbefore = 0
    IDnum   = 0
    sphereN = 0
    if const.lADDnQ:
        eigVnum = const.IOEl_forcollect
    else:
        eigVnum = len(eigVlist)
    while True:
        for eigV in eigVlist[:eigVnum]:
            for pm in [-1, 1]:
                # please look at the documantation of class ADDthetaClass about these public attributes
                ADDth              = ADDthetaClass()
                ADDth.IDnum        = copy.copy(IDnum)
                ADDth.dim          = len(eqpoint)
                ADDth.SQ           =  functions.SQaxes(eigNlist, eigVlist, ADDth.dim)
                ADDth.SQ_inv       =  functions.SQaxes_inv(eigNlist, eigVlist, ADDth.dim)
                ADDth.thetalist    = functions.calctheta(pm * eigV, eigVlist, eigNlist)
                ADDth.nADD         = ADDth.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, const)
                ADDth.x            = eqpoint + ADDth.nADD
                ADDth.A            = f(ADDth.x)
                ADDth.ADD          = ADDth.A - IOEsphereA - A_eq
                ADDth.ADD_IOE      = ADDth.A - IOEsphereA - A_eq
                ADDths.append(ADDth)
                IDnum += 1
        ADDths = sorted(ADDths, key = lambda ADDC: ADDC.ADD)
        ADDths, IOEsphereA = Opt_hyper_sphere(ADDths, f, grad, eqpoint, eigNlist, eigVlist, IOEsphereA, 
                IOEsphereA_r, A_eq, sphereN, dim, SHSrank, SHSroot, const)
        if len(ADDths) <= lenADDthsbefore:
            break
        if const.lADDnQ:
            if const.IOEl_forcollect < len(ADDths):
                break
        lenADDthsbefore = len(ADDths)

    if const.lADDnQ:
        ADDths = ADDths[:const.IOEl_forADDstart]
    TSinitialpoints = []
    for ADDth in ADDths:
        ADDth.findTSQ = False

    while sphereN < 1000:
        sphereN += 1
        if SHSrank == SHSroot:
            with open("./sphere.txt", "a")  as wf:
                wf.write("*"*50 + "\n")
                wf.write("sphereN = %s\n"%sphereN)
                wf.write("*"*50 + "\n")
        ADDths, IOEsphereA = Opt_hyper_sphere(ADDths, f, grad, eqpoint, eigNlist, eigVlist, IOEsphereA,
                IOEsphereA_r, A_eq, sphereN, dim, SHSrank, SHSroot, const)
        writeline = "%05d: "%sphereN
        for ADDth in ADDths:
            ADDth.grad     = grad(ADDth.x)
            ADDth.grad_vec = np.dot(ADDth.grad, ADDth.nADD/np.linalg.norm(ADDth.nADD))
            if ADDth.findTSQ:
                writeline += " % 5.2f,"%0.0
            else:
                writeline += " % 5.2f,"%ADDth.grad_vec
                if ADDth.grad_vec < 0.0:
                    print("New TS point is found.")
                    ADDth.findTSQ = True
                    nADDbefore = ADDth.SuperSphere_cartesian(IOEsphereA_before, ADDth.thetalist, const)
                    for i in range(11):
                        x_ts = eqpoint + nADDbefore + (ADDth.nADD - nADDbefore) * 0.1 * i 
                        grad_ts = grad(x_ts)
                        grad_ts = np.dot(grad_ts, ADDth.nADD / np.linalg.norm(ADDth.nADD))
                        if grad_ts < 0.0:
                            TSinitialpoints.append(x_ts)
                            break
                elif functions.wallQ(ADDth.x, const):
                    if SHSrank == SHSroot:
                        print("out of range!", flush = True)
                        print(ADDth.x, flush = True)
                    ADDth.findTSQ = True
                elif not optdigTH is False:
                    if ADDth.A > optdigTH:
                        if SHSrank == SHSroot:
                            print("ADDth.A = %s > optdigTH(%s): removed for metadynamics"%(ADDth.A, optdigTH), flush = True)
                        ADDth.findTSQ = True

        writeline = writeline.rstrip(",")
        writeline += "\n"
        if SHSrank == SHSroot:
            with open("./gradlist.txt", "a")  as wf:
                wf.write(writeline)

        if all([ADDth.findTSQ for ADDth in ADDths]):
            break
        if const.lADDnQ:
            if const.IOEl < len(TSinitialpoints):
                break

        IOEsphereA_before = copy.copy(IOEsphereA)
        IOEsphereA = np.sqrt(IOEsphereA) + IOEsphereA_r
        IOEsphereA = IOEsphereA * IOEsphereA
        if SHSrank == SHSroot:
            with open("./IOEsphereA.txt", "a")  as wf:
                wf.write("% 5.2f\n"%IOEsphereA)
    if SHSrank == SHSroot:
        writeline = ""
        for TSinitialpoint in TSinitialpoints:
            for p in TSinitialpoint:
                writeline += " % 5.2f,"%p
            writeline += "\n"
        with open("./TStergetpoint.txt", "a")  as wf:
            wf.write(writeline)
    os.chdir("../")
    return TSinitialpoints
def Opt_hyper_sphere(ADDths, f, grad, eqpoint, eigNlist, eigVlist, IOEsphereA, IOEsphereA_r,
                        A_eq, sphereN, dim, SHSrank, SHSroot, const):
    """
    Optimization on a hypersphere
    """
    thetaN = len(ADDths)
    for ADDth in ADDths:
        ADDth.thetalist  = functions.calctheta(ADDth.nADD, eigVlist, eigNlist)
        ADDth.ADDoptQ    = True
        ADDth.ADDremoveQ = False
    opt_calcQlist = [True for _ in range(len(ADDths))]
    optturnN = 0
    turnNrecall = 0
    newADDths = []
    nonoptlist = []
    while any(opt_calcQlist):
        optturnN += 1
        if optturnN == 1000:
            if SHSrank == SHSroot:
                print("optturnN = %s: break"%optturnN, flush = True)
            if sphereN == 0:
                IOEsphereA *= 2.0
                if SHSrank == SHSroot:
                    print("IOEsphereA is changed as %s"%IOEsphereA, flush = True)
                for ADDth in ADDths:
                    ADDth.thetalist  = functions.calctheta(ADDth.nADD, eigVlist, eigNlist)
                    ADDth.ADDoptQ    = True
                    ADDth.ADDremoveQ = False
                turnNrecall += 1
                optturnN = 0
                if 5 < turnNrecall:
                    if SHSrank == SHSroot:
                        print("turnNrecall over 5: break", flush = True)
                    break
            else:
                break
            continue
        if SHSrank == SHSroot:
            with open("./sphere.txt", "a")  as wf:
                wf.write("=" * 50 + "\n")
                wf.write("optturnN = %s\n"%optturnN)
                wf.write("%s\n"%datetime.datetime.now())
                wf.write("=" * 50 + "\n")
        ADDths = sorted(ADDths, key = lambda ADDth: ADDth.ADD)
        newADDths = []
        for ADDth in ADDths:
            if ADDth.ADDremoveQ:
                if SHSrank == SHSroot:
                    with open("./sphere.txt", "a")  as wf:
                        wf.write("%s is skilped\n"%ADDth.IDnum)
                ADDth.ADDoptQ = False
                continue
            if ADDth.ADDoptQ:
                boundlist = [[- np.pi, np.pi] for _ in range(len(ADDth.thetalist))]
                time2 = time.time()
                if const.use_jacQ:
                #if False:
                #result_jac = minimize(
                    result = minimize(
                        lambda deltaTH: ADDth.calc_onHS(deltaTH, f,
                                eqpoint, IOEsphereA, A_eq, ADDths, const
                                ),
                        x0 = np.zeros(len(ADDth.thetalist)),
                        jac = lambda deltaTH: ADDth.grad_theta(f, grad, eqpoint,
                            IOEsphereA, deltaTH, A_eq, ADDths, const),
                        #bounds = boundlist,
                        #tol = const.minimize_threshold,
                        #options = {"gtol":const.minimize_threshold},
                        method="L-BFGS-B")

                else:
                    result= minimize(
                #result= minimize(
                        lambda deltaTH: ADDth.calc_onHS(deltaTH, f, 
                                eqpoint, IOEsphereA, A_eq, ADDths, const
                                ),
                        x0 = np.zeros(len(ADDth.thetalist)),
                        #tol = const.minimize_threshold,
                        #options = {"gtol":const.minimize_threshold},
                        method="L-BFGS-B")
                #if SHSrank == SHSroot:
                    #print("result, result_jac = (% 5.4f, % 5.4f)"%(result.x[0], result_jac.x[0]), flush = True)
                time3 = time.time()
                ADDth.thetalist = ADDth.thetalist + result.x
                #ADDth.thetalist = ADDth.thetalist + result_jac.x
                ADDth.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, const)
                ADDth.x         = eqpoint + ADDth.nADD
                ADDth.x         = functions.periodicpoint(ADDth.x, const)
                ADDth.A         = f(ADDth.x)
                ADDth.ADD       = ADDth.A - IOEsphereA - A_eq
                if ADDth.ADD != ADDth.ADD:
                    if SHSrank == SHSroot:
                        print("ERROR: ADD is nan")
                        print("ADDth.A    = %s"%ADDth.A)
                        print("IOEsphereA = %s"%IOEsphereA)
                    functions.TARGZandexit()
                #ADDth.ADD_IOE   = f_IOE(ADDth.nADD, ADDth.x, f, IOEsphereA, A_eq, ADDth, ADDths, eigNlist, eigVlist, const)
                ADDth.ADD_IOE   = ADDth.f_IOE(ADDth.nADD, ADDth.x, f, IOEsphereA, A_eq, ADDths, const)
#                if ADDth.ADD != ADDth.ADD_IOE:
#                    result = minimize(
#                        lambda deltaTH: calc_onHS(ADDth.thetalist + deltaTH, f, f_IOE, eqpoint, eigNlist, eigVlist, IOEsphereA, A_eq, ADDth, ADDths), 
#                        x0 = np.zeros(len(ADDth.thetalist)),
#                        method="L-BFGS-B")
#                    ADDth.thetalist = ADDth.thetalist + result.x
#                    ADDth.x         = eqpoint + ADDth.nADD
#                    ADDth.x         = functions.periodicpoint(ADDth.x)
#                    ADDth.A         = f(ADDth.x)
#                    ADDth.ADD       = ADDth.A - IOEsphereA - A_eq
#                    ADDth.ADD_IOE   = f_IOE(ADDth.nADD, ADDth.x, f, IOEsphereA, A_eq, ADDth, ADDths, eigNlist, eigVlist)
#                else:
#                    if SHSrank == SHSroot:
#                        with open("./sphere%05d.txt"%sphereN, "a")  as wf:
#                            wf.write("jac\n")

                ADDth.ADDoptQ   = False
                if SHSrank == SHSroot:
                    with open("./sphere.txt", "a")  as wf:
                        wf.write("(IDnum, ADD, ADD_IOE), time = (%s, % 5.3f, % 5.3f), % 5.3f\n"%(
                                ADDth.IDnum, ADDth.ADD, ADDth.ADD_IOE, time3 - time2))
                returnoptQ = False
                for beforeADDth in ADDths:
                    if ADDth.IDnum == beforeADDth.IDnum:
                        continue
                    if not beforeADDth.ADDoptQ:
                        if ADDth.ADD  + abs(ADDth.ADD) * const.neiborfluct_threshold < beforeADDth.ADD:
                            if functions.angle_SHS(ADDth.nADD, beforeADDth.nADD, ADDth.SQ_inv, const) <= np.pi * 0.5:
                                returnoptQ    = True
                                beforeADDth.ADDoptQ = True
            else:
                returnoptQ = False
            if returnoptQ:
                ADDth.ADDoptQ = True
                break
            else:
                if 0.0 < ADDth.ADD_IOE:
                    continue
                for newADDth in newADDths:
                    if functions.angle_SHS(ADDth.nADD, newADDth.nADD, ADDth.SQ_inv, const) < 0.01:
                        if SHSrank == SHSroot:
                            with open("./sphere.txt", "a")  as wf:
                                wf.write("angle of %s and %s is smaller than 0.01:break\n"%(ADDth.IDnum, newADDth.IDnum))
                        ADDth.ADDremoveQ = True
                        ADDth.ADDoptQ    = False
                        break
                else:
                    newADDths.append(ADDth)
        opt_calcQlist = [ADDth.ADDoptQ for ADDth in ADDths]
        #if sphereN == 0:
        if True:
            nonoptlist.append([ADDth.IDnum for ADDth in ADDths if ADDth.ADDoptQ])
            for nonopt in nonoptlist:
                if nonoptlist.count(nonopt) != 1:
                    #IOEsphereA *= 2.0
                    IOEsphereA = np.sqrt(IOEsphereA) + IOEsphereA_r
                    IOEsphereA = IOEsphereA * IOEsphereA
                    optturnN   = 0
                    if SHSrank == SHSroot:
                        print("There is repeat of calculation",         flush = True)
                        print("IOEsphereA is changed as %s"%IOEsphereA, flush = True)
                        with open("./sphere.txt", "a")  as wf:
                            wf.write("IOEsphereA is changed as %s\n"%IOEsphereA)
                    nonoptlist = []
                    for ADDth in ADDths:
                        ADDth.thetalist  = functions.calctheta(ADDth.nADD, eigVlist, eigNlist)
                        ADDth.ADDoptQ    = True
                        ADDth.ADDremoveQ = False
                    break
    ADDths = copy.copy(newADDths)
    if const.exportADDpointsQ:
        if SHSrank == SHSroot:
            writeline = ""
            for ADDth in ADDths:
                writeline += ",".join(["% 5.3f"%p for p in eqpoint + ADDth.nADD])
                writeline += "\n"
            with open("./ADDpoints.csv", "a")  as wf:
                    wf.write(writeline)
    return ADDths, IOEsphereA
