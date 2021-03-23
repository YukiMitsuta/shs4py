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
import pickle
import numpy as np
from   scipy.optimize import minimize
from   scipy.optimize import approx_fprime

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
        ADDremoveQ : If ADDremoveQ = True, the optimization of this ADD is skipped 
        grad       : The gradient on x
        grad_vec   : The dot of grad and nADD (if grad_vec < 0, the ADD is over TS)

    Available functions:
        calc_onHS : calculation of f on hypersphere
        f_IOE     : IOE calculation of f
        IOE_total : sum up of IOE ilumination
        IOE       : IOE calculation between self and neiborADDth
        (next functions are testing now)
        grad_theta    : the gradient of polar coordinates without r.
        delx_deltheta : d(x)/d(theta)
        delf_delx     : d(f)/d(x)
        hessian_theta : the gradient of polar coordinates without r. 
        delx_delthetadeltheta : d^2(x)/d(theta)d(theta)
        
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
        #print("x=%s"%x)
        return  self.f_IOE(nADD, x, func, IOEsphereA, A_eq, ADDths, const)
    def f_IOE(self, nADD, x, f, IOEsphereA, A_eq, ADDths, const):
        """
        f_IOE: IOE calculation of f
        """
        #result = f(x, ADDQ = True) - IOEsphereA - A_eq
        #result = f(x, ADDQ = False) - IOEsphereA - A_eq
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
                if neiborADDth.ADD_IOE <= -1000000:
                    continue
                if 10000000 < neiborADDth.ADD_IOE:
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
                if neiborADDth.ADD_IOE <= -1000000:
                    continue
                if 10000000 < neiborADDth.ADD_IOE:
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
        returngrad = np.zeros(len(nADD))
        eps = 1.0e-5
        if deltaTH <= np.pi * 0.5:
            cosdamp    = np.cos(deltaTH)
            xydot      = np.dot(q_x, q_y)
            IOE_center = neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
            for i in range(len(nADD)):
                qx_i     = copy.copy(q_x)
                qx_i[i] += eps
                deltaTH  = functions.angle(qx_i, q_y)
                cosdamp  = np.cos(deltaTH)
                IOE_eps  = neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
                returngrad[i] = (IOE_eps - IOE_center) / eps
#        if np.pi * 0.5 < deltaTH:
#            return 0.0
#        cosdamp = np.cos(deltaTH)
#        xydot   = np.dot(q_x, q_y)
#        returngradlist = []
#        for i in range(len(nADD)):
#            returngrad  = q_y[i] * r * r
#            #returngrad -= 2.0 * q_x[i] * xydot
#            returngrad -= q_x[i] * xydot
#            returngradlist.append(returngrad)
#        returngradlist  = np.array(returngradlist)
#        #returngradlist *= 3.0 * neiborADDth.ADD_IOE * cosdamp * cosdamp / r / r / r / r
#        returngradlist *= neiborADDth.ADD_IOE * cosdamp * cosdamp
#        #print(returngradlist)
        return returngrad
    def IOE_gradgrad(self, nADD, IOEsphereA, r, neiborADDth, thetalistdamp, const):
        returngrad = np.zeros((len(thetalistdamp), len(nADD)))
        return returngrad
        q_x     = np.dot(self.SQ_inv, nADD)
        q_y     = np.dot(self.SQ_inv, neiborADDth.nADD)
        deltaTH = functions.angle(q_x, q_y)
        eps = 1.0e-5
        if deltaTH <= np.pi * 0.5:
            #cosdamp  = cosADD(q_x, q_y)
            DthetaDq = np.zeros(len(nADD))
            for i in range(len(nADD)):
                qx_i          = copy.copy(q_x)
                qx_i[i]      += eps
                deltaTH_eps   = functions.angle(qx_i, q_y)
                DthetaDq[i]   = (deltaTH_eps - deltaTH) / eps
            DthetaDpsi   = np.zeros(len(thetalistdamp))
            for k in range(len(thetalistdamp)):
                thetalist_eps = np.array(thetalistdamp)
                thetalist_eps[k] += eps
                nADD_eps      = self.SuperSphere_cartesian(IOEsphereA, thetalist_eps, const)
                q_x_eps       = np.dot(self.SQ_inv, nADD_eps)
                deltaTH_eps   = functions.angle(q_x_eps, q_y)
                DthetaDpsi[k] = (deltaTH_eps - deltaTH) / eps
            for k in range(len(thetalistdamp)):
                DthetaDpsiDq = np.zeros(len(nADD))
                for i in range(len(nADD)):
                    qx_i            = copy.copy(q_x_eps)
                    qx_i[i]        += eps
                    deltaTH_eps     = functions.angle(qx_i, q_y)
                    DthetaDpsiDq[i] = ((deltaTH_eps - deltaTH) / eps - DthetaDpsi[k]) / eps
                cosdamp = np.cos(deltaTH)
                sindamp = np.sin(deltaTH)
                for i in range(len(nADD)):
                    returngrad[k][i]  = DthetaDpsiDq[i] * cosdamp * cosdamp * sindamp
                    returngrad[k][i] += DthetaDpsi[k] * DthetaDq[i] * \
                        (2.0 * cosdamp * sindamp * sindamp - cosdamp * cosdamp * cosdamp)
        returngrad *= 3.0 * neiborADDth.ADD_IOE
        return returngrad

    def grad_theta(self, f, grad, eqpoint, IOEsphereA, deltaTH, A_eq, ADDths, const, eigVlist, eigNlist):
        """
        the gradient along theta (this function is not work now.)
        cf: thetalist = {theta_n0,..., theta_1}
        """
        thetalist   = self.thetalist + deltaTH 
        nADD        = self.SuperSphere_cartesian(IOEsphereA, thetalist, const)
        #print(nADD)
        thetalist   = functions.calctheta(nADD, eigVlist, eigNlist)
        r           = np.sqrt(2.0 * IOEsphereA)
        tergetpoint = eqpoint + nADD
        if const.periodicQ:
            tergetpoint = functions.periodicpoint(tergetpoint, const)
        grad_x = grad(tergetpoint)
        grad_q = np.dot(grad_x, self.SQ)
        thetalistdamp = list(thetalist)
        returngrad = [0.0 for _ in range(len(thetalist))]
        delfdamp = self.delf_delx(grad_q, ADDths, nADD, r, const)
        for theta_i in range(len(returngrad)): # the number of derivative
            Dqlist = self.delx_deltheta(r, thetalistdamp, theta_i, const)
            for x_i in range(len(nADD)):
                returngrad[theta_i] += Dqlist[x_i] * delfdamp[x_i]
        returngrad = np.array(returngrad)
        return returngrad
    def delx_deltheta(self, r, thetalist, theta_i, const):
        if const.cythonQ:
            return  const.calcgau.delx_deltheta(r, thetalist, theta_i)
        Dqlist = [r for _ in range(len(thetalist) + 1)]
        a_k = 1.0
        for i, theta in enumerate(thetalist):
            if i == theta_i:
                Dqlist[i] *= - a_k * np.sin(theta)
                a_k *= np.cos(theta)
            else:
                Dqlist[i] *= a_k * np.cos(theta)
                a_k *= np.sin(theta)
            if i < theta_i:
                Dqlist[i] *= 0.0
        Dqlist[-1] *= a_k
        return Dqlist
    def delf_delx(self, grad_q, ADDths, nADD, r, const):
        returngrad = np.zeros(len(nADD))
        for neiborADDth in ADDths:
            if self.IDnum == neiborADDth.IDnum:
                continue
            if neiborADDth.ADDoptQ:
                continue
            if neiborADDth.ADD <= self.ADD:
                returngrad -= self.IOE_grad(nADD, r, neiborADDth, const)
        returngrad += grad_q
        return returngrad
    def hessian_theta(self, f, grad, eqpoint, IOEsphereA, A_eq, ADDths, const, eigVlist, eigNlist):

        returnhess = np.zeros((len(self.thetalist), len(self.thetalist)))
        deltaTH    = np.zeros(len(self.thetalist))
        gradtheta  =  self.grad_theta(f, grad, eqpoint,
                                IOEsphereA, deltaTH, A_eq, ADDths, const,
                                eigVlist ,eigNlist)
        eps = 1.0e-5
        for theta_i in range(len(self.thetalist)): # the number of derivative
            deltaTH = np.zeros(len(self.thetalist))
            deltaTH[theta_i] += eps
            gradtheta_eps = self.grad_theta(f, grad, eqpoint,
                                IOEsphereA, deltaTH, A_eq, ADDths, const,
                                eigVlist ,eigNlist)
            gradgrad = (gradtheta_eps - gradtheta) / eps
            #print("gradgrad = %s"%gradgrad)
            for theta_j in range(len(self.thetalist)): # the number of derivative
                returnhess[theta_i, theta_j] += gradgrad[theta_j]
                #returnhess[theta_i, theta_j] = returnhess[theta_j, theta_i] 
        return returnhess



#        thetalist   = copy.copy(self.thetalist)
#        nADD        = self.SuperSphere_cartesian(IOEsphereA, thetalist, const)
#        thetalist   = functions.calctheta(nADD, eigVlist, eigNlist)
#        r           = np.sqrt(2.0 * IOEsphereA)
#        tergetpoint = eqpoint + nADD
#        if const.periodicQ:
#            tergetpoint = functions.periodicpoint(tergetpoint, const)
#        grad_x           = grad(tergetpoint)
#        grad_q           = np.dot(grad_x, self.SQ)
#        thetalistdamp    = list(thetalist)
#        delfdamp         = self.delf_delx(grad_q, ADDths, nADD, r, const)
#        delxdelthetalist = self.delf_delxdeltheta(grad, grad_x, thetalistdamp, ADDths, nADD, IOEsphereA, r, eqpoint, const)
#        print("delf_delxdeltheta = %s"%delxdelthetalist)
#
#        returnhess = np.zeros((len(thetalist), len(thetalist)))
#        for theta_i in range(len(thetalist)): # the number of derivative
#            for theta_j in range(theta_i, len(thetalist)): # the number of derivative
#                Dqlist = self.delx_delthetadeltheta(r, thetalistdamp, theta_i, theta_j, const)
#                #print(Dqlist)
#                for x_i in range(len(nADD)):
#                    returnhess[theta_i, theta_j] += Dqlist[x_i] * delfdamp[x_i]
#                Dqlist = self.delx_deltheta(r, thetalistdamp, theta_i, const)
#                #print(Dqlist)
#                for x_i in range(len(nADD)):
#                    returnhess[theta_i, theta_j] += Dqlist[x_i] * delxdelthetalist[theta_j][x_i]
#                returnhess[theta_i, theta_j] = returnhess[theta_j, theta_i] 
        return returnhess
    def delf_delxdeltheta(self, grad, grad_x, thetalistdamp, ADDths, nADD, IOEsphereA , r, eqpoint, const):
        returngrad = np.zeros((len(thetalistdamp), len(nADD)))
        for neiborADDth in ADDths:
            if self.IDnum == neiborADDth.IDnum:
                continue
            if neiborADDth.ADDoptQ:
                    continue
            if neiborADDth.ADD <= self.ADD:
                #print(returngrad.ndim)
                returngrad -= self.IOE_gradgrad(nADD, IOEsphereA, r, neiborADDth, thetalistdamp, const)
        Dgrad = self.delgrad_delxdeltheta(grad, grad_x, ADDths, thetalistdamp, nADD, r, IOEsphereA, eqpoint, const)
        for k in range(len(thetalistdamp)):
            returngrad[k] += Dgrad[k]
        return returngrad
    def delgrad_delxdeltheta(self, grad, grad_x, ADDths, thetalistdamp, nADD, r, IOEsphereA, eqpoint, const):
        #returngrad = np.zeros((len(thetalistdamp), len(nADD)))
        returngrad = []
        eps = 1.0e-5
        for k in range(len(thetalistdamp)):
            thetalist_eps     = np.array(thetalistdamp)
            thetalist_eps[k] += eps
            #print(thetalist_eps)
            nADD_eps          = self.SuperSphere_cartesian(IOEsphereA, thetalist_eps, const)
            tergetpoint_eps   = eqpoint + nADD_eps
            if const.periodicQ:
                tergetpoint_eps = functions.periodicpoint(tergetpoint_eps, const)
            grad_x_eps      = grad(tergetpoint_eps)
            gradgradxdtheta = (grad_x_eps - grad_x) / eps
            gradgradxdtheta = np.dot(gradgradxdtheta, self.SQ)
            returngrad.append(gradgradxdtheta)
        return returngrad
    def delx_delthetadeltheta(self, r, thetalist, theta_i, theta_j, const):
        #if const.cythonQ:
            #return  const.calcgau.delx_deltheta(r, thetalist, theta_i)
        Dqlist = [r for _ in range(len(thetalist) + 1)]
        a_k = 1.0
        for i, theta in enumerate(thetalist):
            if i == theta_i and i == theta_j:
                Dqlist[i] *= - a_k * np.cos(theta)
                a_k *= - np.sin(theta)
            elif i == theta_i or i == theta_j:
                Dqlist[i] *= - a_k * np.sin(theta)
                a_k *= np.cos(theta)
            else:
                Dqlist[i] *= a_k * np.cos(theta)
                a_k *= np.sin(theta)
            if i < theta_i or i < theta_j:
                Dqlist[i] *= 0.0
        Dqlist[-1] *= a_k
        print("delx_delthetadeltheta = %s"%Dqlist)
        return Dqlist
def main(eqpoint, f, grad, hessian, dirname, optdigTH, SHSrank, SHSroot, SHScomm, SHSsize, const, metaDclass):
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
    picklelist   = glob.glob("./ADDths*.pickle")
  
    if SHSrank == SHSroot:
        print("IOEsphereA_initial, IOEsphereA_r = %s, %s"%(IOEsphereA_initial, IOEsphereA_r), flush = True)
    if SHSrank == SHSroot:
        with open("./sphere.txt", "a")  as wf:
            wf.write("-*-"*10 + "\nstart initial optimization\n" + "-*-"*10 + "\n")
    IOEsphereA_before = 0.0

    lenADDthsbefore = 0
    IDnum   = 0
    sphereN = 0
    if len(picklelist) == 0:
        start0sphereQ = True
    else:
        start0sphereQ = False
        if SHSrank == SHSroot:
            try:
                while sphereN < 10000:
                    ADDthpiclename = "./ADDths%s.pickle"%sphereN
                    if os.path.exists(ADDthpiclename):
                        with open(ADDthpiclename, mode="rb") as pic:
                            ADDths = pickle.load(pic)
                        break
                    sphereN += 1
                with open("./IOEsphereA.pickle", mode="rb") as pic:
                    IOEsphereA = pickle.load(pic)
                if os.path.exists("./TSinitialpoints.pickle"):
                    with open("./TSinitialpoints.pickle", mode="rb") as pic:
                        TSinitialpoints = pickle.load(pic)
                else:
                    TSinitialpoints = []
            except EOFError:
                #ADDths = []
                #IOEsphereA = IOEsphereA_initial
                #TSinitialpoints = []
                start0sphereQ = True
        else:
            ADDths = None
            IOEsphereA = None
            TSinitialpoints = None
        if const.calc_mpiQ:
            ADDths = const.SHScomm.bcast(ADDths, root=0)
            IOEsphereA = const.SHScomm.bcast(IOEsphereA, root=0)
            TSinitialpoints = const.SHScomm.bcast(TSinitialpoints, root=0)
            start0sphereQ = const.SHScomm.bcast(start0sphereQ, root=0)
    #if len(picklelist) == 0:
    if start0sphereQ:
        ADDths = []
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
                    #ADDth.ADD_IOE      = ADDth.A - IOEsphereA - A_eq
                    ADDth.ADD_IOE      = -1.0e30
                    ADDth.grad     = grad(ADDth.x)
                    ADDth.grad_vec = np.dot(ADDth.grad, ADDth.nADD/np.linalg.norm(ADDth.nADD))
                    ADDth.findTSQ = False
                    ADDths.append(ADDth)
                    IDnum += 1
            if const.calc_cupyQ:
                updateCupyData(metaDclass, ADDth, IOEsphereA, eqpoint, eigNlist, eigVlist, const)
            ADDths = sorted(ADDths, key = lambda ADDC: ADDC.ADD)
            ADDths, IOEsphereA = Opt_hyper_sphere(
                    ADDths, f, grad, eqpoint, eigNlist, eigVlist, IOEsphereA, 
                    IOEsphereA_r, A_eq, sphereN, dim, SHSrank, SHSroot, const)
            newADDths = []
            for ADDth in ADDths:
                if not ADDth.ADDremoveQ:
                    newADDths.append(ADDth)
            ADDths = copy.copy(newADDths)
            if SHSrank == SHSroot:
                with open("./sphere.txt", "a")  as wf:
                    wf.write(" ====> len(ADDths) = %s\n"%len(ADDths))
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

    while sphereN < 10000:
        if SHSrank == SHSroot:
            beforepiclename = "./ADDths%s.pickle"%(sphereN-1)
            if os.path.exists(beforepiclename):
                os.remove(beforepiclename)
            ADDthpiclename = "./ADDths%s.pickle"%sphereN
            with open(ADDthpiclename, mode="wb") as pic:
                pickle.dump(ADDths, pic)
            with open("./IOEsphereA.pickle", mode="wb") as pic:
                pickle.dump(IOEsphereA, pic)
        if os.path.exists("./exit.txt"):
            if SHSrank == SHSroot:
                print("exit.txt is found; Forced termination")
                time.sleep(3)
                os.remove("./exit.txt")
                if os.path.exists("./running.txt"):
                    os.remove("./running.txt")
            exit()
        sphereN += 1
        if SHSrank == SHSroot:
            #with open("./sphere.txt", "a")  as wf:
                #wf.write("*"*50 + "\n")
                #wf.write("sphereN = %s\n"%sphereN)
                #wf.write("*"*50 + "\n")
            spherestr  = ""
            spherestr += "*"*50 + "\n"
            spherestr += "sphereN = %s\n"%sphereN
            spherestr += "*"*50 + "\n"
            with open("./sphere.txt", "a")  as wf:
                wf.write(spherestr)
            spherestr = False
        ADDths, IOEsphereA = Opt_hyper_sphere(
                ADDths, f, grad, eqpoint, eigNlist, eigVlist, IOEsphereA,
                IOEsphereA_r, A_eq, sphereN, dim, SHSrank, SHSroot, const)
        writeline = "%05d: "%sphereN
        for ADDth in ADDths:
            ADDth.grad     = grad(ADDth.x)
            ADDth.grad_vec = np.dot(ADDth.grad, ADDth.nADD/np.linalg.norm(ADDth.nADD))
            ADDth.grad_norm = np.linalg.norm(ADDth.grad)
            if ADDth.findTSQ:
                writeline += " % 5.2f,"%0.0
            else:
                writeline += " % 5.2f,"%ADDth.grad_norm
                #if sphereN > 5 and ADDth.grad_vec < 0.0:
                if ADDth.grad_norm < const.threshold:
                    if SHSrank == SHSroot:
                        print("New TS point is found.", flush = True)
                    ADDth.findTSQ = True
#                    nADDbefore = ADDth.SuperSphere_cartesian(IOEsphereA_before, ADDth.thetalist, const)
#                    for i in range(11):
#                        x_ts    = eqpoint + nADDbefore + (ADDth.nADD - nADDbefore) * 0.1 * i 
#                        grad_ts = grad(x_ts)
#                        grad_ts = np.dot(grad_ts, ADDth.nADD / np.linalg.norm(ADDth.nADD))
#                        if grad_ts < 0.0:
#                            TSinitialpoints.append(x_ts)
#                            if SHSrank == SHSroot:
#                                with open("./TSinitialpoints.pickle", mode="wb") as pic:
#                                    pickle.dump(TSinitialpoints, pic)
#                            break
                    TSinitialpoints.append(ADDth.x)
                    if SHSrank == SHSroot:
                        with open("./TSinitialpoints.pickle", mode="wb") as pic:
                            pickle.dump(TSinitialpoints, pic)
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
        if const.calc_cupyQ:
            updateCupyData(metaDclass, ADDth, IOEsphereA, eqpoint, eigNlist, eigVlist, const)
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
        #print(ADDth.thetalist)
        ADDth.ADDoptQ    = True
        ADDth.ADDremoveQ = False
    opt_calcQlist = [True for _ in range(len(ADDths))]
    optturnN      = 0
    turnNrecall   = 0
    newADDths     = []
    nonoptlist    = []
    spherestr = False
    while any(opt_calcQlist):
        optturnN += 1
        if optturnN == 10000:
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
            #with open("./sphere.txt", "a")  as wf:
                #wf.write("=" * 50 + "\n")
                #wf.write("optturnN = %s\n"%optturnN)
                #wf.write("%s\n"%datetime.datetime.now())
                #wf.write("=" * 50 + "\n")
            if spherestr is not False:
                if optturnN % 10 == 0:
                    with open("./sphere.txt", "a")  as wf:
                        wf.write(spherestr)
                    spherestr  = "=" * 50 + "\n"
                else:
                    spherestr += "=" * 50 + "\n"
            else:
                spherestr  = "=" * 50 + "\n"

            #spherestr  = "=" * 50 + "\n"
            spherestr += "optturnN = %s\n"%optturnN
            spherestr += "%s\n"%datetime.datetime.now()
            spherestr += "=" * 50 + "\n"
        newADDths = []
        for ADDth in ADDths:
            #if not ADDth.ADDoptQ:
            if True:
                if 0.0 < ADDth.ADD_IOE:
                    continue
            newADDths.append(ADDth)
        ADDths = copy.copy(newADDths)
        ADDths = sorted(ADDths, key = lambda ADDth: ADDth.ADD)
        newADDths = []
        #if SHSrank == SHSroot:
            #print([x.IDnum for x in ADDths], flush = True)
        bifQ = False
        for ADDth in ADDths:
            if ADDth.ADDremoveQ:
                #if SHSrank == SHSroot:
                    #with open("./sphere.txt", "a")  as wf:
                        #wf.write("%s is skipped\n"%ADDth.IDnum)
                ADDth.ADDoptQ = False
                continue
            if const.x0randomQ:
                if SHSrank == SHSroot:
                    x_initial = np.array([np.random.rand() * 0.01 - 0.005 for _ in range(len(ADDth.thetalist))])
                else:
                    x_initial = None
                if const.calc_mpiQ:
                    x_initial = const.SHScomm.bcast(x_initial, root=0)

            else:
                x_initial = np.zeros(len(ADDth.thetalist)),
            if ADDth.ADDoptQ:
                time2 = time.time()
                if const.use_jacQ:
                    #boundlist = []
                    #for _ in range(len(ADDth.thetalist) - 1):
                        #boundlist.append([-np.pi * 0.5, np.pi * 0.5])
                    #boundlist.append([-np.pi, np.pi])
                    result = minimize(
                            lambda deltaTH: ADDth.calc_onHS(deltaTH, f,
                                    eqpoint, IOEsphereA, A_eq, ADDths, const,
                                    ),
                            #x0 = np.zeros(len(ADDth.thetalist)),
                            x0 = x_initial,
                            jac = lambda deltaTH: ADDth.grad_theta(f, grad, eqpoint,
                                IOEsphereA, deltaTH, A_eq, ADDths, const,
                                eigVlist ,eigNlist),
                            #bounds = boundlist,
                            #tol = const.minimize_threshold,
                            #options = {"gtol":const.minimize_threshold},
                            method="L-BFGS-B")
                            #method="tnc")
                    resultx = result.x

                    #eps = np.sqrt(np.finfo(float).eps)
                    #grad_app = approx_fprime(np.zeros(len(ADDth.thetalist)), 
                    #grad_app = approx_fprime(result.x, 
                        #lambda deltaTH: ADDth.calc_onHS(deltaTH, f,
                                #eqpoint, IOEsphereA, A_eq, ADDths, const
                                #), [eps for _ in range(len(ADDth.thetalist))])
                    #grad_jac = ADDth.grad_theta(f, grad, eqpoint,
                            ##IOEsphereA, result.x, A_eq, ADDths, const)
                            #IOEsphereA, np.zeros(len(ADDth.thetalist)), A_eq, ADDths, 
                            #const, eigVlist, eigNlist)
#                    if SHSrank == SHSroot:
#                        #print("ADDth.x = %s"%ADDth.x)
#                        for i in range(len(result.x)):
#                            print("thetalist[%s] = % 5.4f"%(i, ADDth.thetalist[i]))
#                        #for i in range(len(result.x)):
#                            #print("result.x[%s] = % 5.4f"%(i, result.x[i]))
#                        #print("grad_app = %s"%grad_app)
#                        #print("grad_jac = %s"%grad_jac)
#                        for i in range(len(grad_app)):
#                            print("% 5.4f, % 5.4f"%(grad_app[i], grad_jac[i]))
#                        print("grad_app, grad_jac = (% 5.4f, % 5.4f)"%(np.linalg.norm(grad_app), np.linalg.norm(grad_jac)), flush = True)
##                    exit()

                else:
                    result= minimize(
                        lambda deltaTH: ADDth.calc_onHS(deltaTH, f, 
                                eqpoint, IOEsphereA, A_eq, ADDths, const
                                ),
                        #x0 = np.zeros(len(ADDth.thetalist)),
                        x0 = x_initial,
                        #tol = const.minimize_threshold,
                        #options = {"gtol":const.minimize_threshold},
                        method="L-BFGS-B")
                    resultx = result.x

                time3 = time.time()
                ADDth.thetalist = ADDth.thetalist + resultx
                #ADDth.thetalist = ADDth.thetalist + result.x
                #ADDth.thetalist = ADDth.thetalist + result_jac.x
                ADDth.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, const)
                ADDth.thetalist = functions.calctheta(ADDth.nADD, eigVlist, eigNlist)
                ADDth.x         = eqpoint + ADDth.nADD
                ADDth.x         = functions.periodicpoint(ADDth.x, const)
                ADDth.A         = f(ADDth.x)
                ADDth.ADD       = ADDth.A - IOEsphereA - A_eq
                #print(ADDth.ADD)
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

#                eps = np.sqrt(np.finfo(float).eps)
#                grad_app = approx_fprime(np.zeros(len(ADDth.thetalist)), 
#                        lambda deltaTH: ADDth.calc_onHS(deltaTH, f,
#                                eqpoint, IOEsphereA, A_eq, ADDths, const
#                                ), [eps for _ in range(len(ADDth.thetalist))])
#                grad_jac = ADDth.grad_theta(f, grad, eqpoint,
#                            IOEsphereA, np.zeros(len(ADDth.thetalist)), A_eq, ADDths, const)
#                if SHSrank == SHSroot:
#                    print("ADDth.x = %s"%ADDth.x)
#                    print("grad_app, grad_jac = (% 5.4f, % 5.4f)"%(grad_app, grad_jac), flush = True)
#                    exit()
                ADDth.ADDoptQ = False
                if SHSrank == SHSroot:
                    #with open("./sphere.txt", "a")  as wf:
                        #wf.write("(IDnum, ADD, ADD_IOE), time = (%s, % 5.3f, % 5.3f), % 5.3f\n"%(
                                #ADDth.IDnum, ADDth.ADD, ADDth.ADD_IOE, time3 - time2))
                    #print("IOE = % 5.4f"%(ADDth.ADD_IOE - ADDth.ADD), flush = True)
                    spherestr += "(IDnum, ADD, ADD_IOE), time = (%s, % 5.3f, % 5.3f), % 5.3f\n"%(
                                ADDth.IDnum, ADDth.ADD, ADDth.ADD_IOE, time3 - time2)
                returnoptQ = False
                bifQ = False
                bifADDths = []

                #print("ADD_IOE = %s"%ADDth.ADD_IOE)
                #if ADDth.ADD_IOE < 0.0 and const.chkBifurcationQ and not ADDth.findTSQ:
                calcbifQ = False
                if ADDth.ADD_IOE < 0.0 and not ADDth.findTSQ:
                    if const.chkBifurcationQ:
                        calcbifQ = True
                    elif const.chkinitialTSQ  and sphereN == 0:
                        calcbifQ = True
                if calcbifQ:
                    bifQ, returnoptQ, _ADDoptQ, _ADDremoveQ, bifADDths = chkBifurcation(
                            f, grad, eqpoint, IOEsphereA, A_eq, ADDth, ADDths, const, 
                            sphereN, eigVlist, eigNlist, dim, SHSrank, SHSroot)
                    ADDths.extend(bifADDths)
                    ADDth.ADDoptQ    = _ADDoptQ 
                    ADDth.ADDremoveQ = _ADDremoveQ 
                    if ADDth.ADDremoveQ:
                        if SHSrank == SHSroot:
                            #with open("./sphere.txt", "a")  as wf:
                                #wf.write("%s is removed for bifurcation\n"%ADDth.IDnum)
                            spherestr += "%s is removed for bifurcation\n"%ADDth.IDnum
                        #exit()

                    for beforeADDth in ADDths:
                        if ADDth.IDnum == beforeADDth.IDnum:
                            continue
                        if not beforeADDth.ADDoptQ:
                            for bifADDth in bifADDths:
                                if bifADDth.ADD  + abs(bifADDth.ADD) * const.neiborfluct_threshold < beforeADDth.ADD:
                                    if functions.angle_SHS(bifADDth.nADD, beforeADDth.nADD, ADDth.SQ_inv, const) <= np.pi * 0.5:
                                        returnoptQ    = True
                                        beforeADDth.ADDoptQ = True
                if not ADDth.ADDremoveQ:
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
            if bifQ:
                break
            if returnoptQ:
                ADDth.ADDoptQ = True
                break
            else:
                if 0.0 < ADDth.ADD_IOE:
                    continue
                elif ADDth.ADD_IOE < -100000:
                    continue
                for newADDth in newADDths:
                    if functions.angle_SHS(ADDth.nADD, newADDth.nADD, ADDth.SQ_inv, const) < 0.01:
                        if SHSrank == SHSroot:
                            #with open("./sphere.txt", "a")  as wf:
                                #wf.write("angle of %s and %s is smaller than 0.01:break\n"%(ADDth.IDnum, newADDth.IDnum))
                            spherestr += "angle of %s and %s is smaller than 0.01:break\n"%(ADDth.IDnum, newADDth.IDnum)
                        ADDth.ADDremoveQ = True
                        ADDth.ADDoptQ    = False
                        break
                else:
                    newADDths.append(ADDth)
        opt_calcQlist = [ADDth.ADDoptQ for ADDth in ADDths]
        #if sphereN == 0:
        #if True:
        if bifQ is False:
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
                        #with open("./sphere.txt", "a")  as wf:
                            #wf.write("IOEsphereA is changed as %s\n"%IOEsphereA)
                        spherestr += "IOEsphereA is changed as %s\n"%IOEsphereA
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
    if SHSrank == SHSroot:
        if spherestr is not False:
            with open("./sphere.txt", "a")  as wf:
                wf.write(spherestr)
    return ADDths, IOEsphereA
def updateCupyData(metaDclass, ADDth, IOEsphereA, eqpoint, eigNlist, eigVlist, const):
    return
#    if const.parallelMetaDQ:
#        updateCupyQ = True
#    else:
#        if const.SHSrank == const.SHSroot:
#            updateCupyQ = True
#        else:
#            updateCupyQ = False
#    if updateCupyQ:
    h_part =[]
    slist_part = []
    sigmainvlist_part =[]
    DP2CUTOFF = 6.25 * 4.0 # this cutoff is same with PLUMED (look at MetaD.cpp)
    #DP2CUTOFF = 6.25 * 2.0 # this cutoff is same with PLUMED (look at MetaD.cpp)
    #DP2CUTOFF = 6.25 
    for hillN, hillC in enumerate(metaDclass.hillCs):
        if const.calc_cupyQ:
            if not const.parallelMetaDQ:
                if const.SHSrank == const.SHSroot:
                    if hillN % const.SHSsize != const.SHSrank:
                        continue
                    #print(const.SHSsize)
                    #print(hillN%const.SHSsize)
        v         = hillC.s - eqpoint
        thetalist = functions.calctheta(v, eigVlist, eigNlist)
        nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, thetalist, const)
        p         = eqpoint + nADD - hillC.s
        dp2 = 0.0
        for i in range(len(eqpoint)):
            dp2 += p[i] * p[i] * hillC.sigmainv[i] * hillC.sigmainv[i]
        if dp2 < DP2CUTOFF:
            h_part.append(hillC.h)
            slist_part.append(hillC.s)
            sigmainvlist_part.append(hillC.sigmainv)
    #print("len(hillCs) = %s"%(hillN+1), flush = True)
    if const.calc_mpiQ:
        h_partG            = const.SHScomm.gather(h_part, root=0)
        slist_partG        = const.SHScomm.gather(slist_part, root=0)
        sigmainvlist_partG = const.SHScomm.gather(sigmainvlist_part, root=0)
        h_partR = []
        slist_partR = []
        sigmainvlist_partR = []
        if const.SHSrank == const.SHSroot:
            for h_part in h_partG:
                h_partR.extend(h_part)
            for slist_part in slist_partG:
                slist_partR.extend(slist_part)
            for sigmainvlist_part in sigmainvlist_partG:
                sigmainvlist_partR.extend(sigmainvlist_part)
    else:
        h_partR = h_part
        slist_partR = slist_part
        sigmainvlist_partR = sigmainvlist_part
    if const.SHSrank == const.SHSroot:
        #print("len(h_part) = %s"%len(h_partR), flush = True)
        metaDclass.h_part            = const.cp.array(h_partR)
        metaDclass.slist_part        = const.cp.array(np.array(slist_partR).transpose())
        metaDclass.sigmainvlist_part = const.cp.array(np.array(sigmainvlist_partR).transpose())
        metaDclass.hillClength = len(h_partR)
def chkBifurcation(f, grad, eqpoint, IOEsphereA, A_eq, ADDth, ADDths, const, 
         sphereN, eigVlist, eigNlist, dim, SHSrank, SHSroot):
    returnoptQ = False
    bifQ = False
    bifADDths = []
    #returnADDth = copy.copy(ADDth)
    _ADDoptQ    = copy.copy(ADDth.ADDoptQ)
    _ADDremoveQ = copy.copy(ADDth.ADDremoveQ)
    hessTheta = ADDth.hessian_theta(f, grad, eqpoint, IOEsphereA, A_eq, ADDths, const, eigVlist, eigNlist)
    try:
        eigNlistTH, _eigVTH = np.linalg.eigh(hessTheta)
    except:
        print("ERROR: eigenvector cannot calculate; skip")
        return bifQ, returnoptQ, _ADDoptQ, _ADDremoveQ, bifADDths
    #if SHSrank == SHSroot:
        #print("% 6.5f"%min(eigNlistTH), flush = True)
    if const.bifucation_eigTH < min(eigNlistTH):
        return bifQ, returnoptQ, _ADDoptQ, _ADDremoveQ, bifADDths
    
    bifQ = True
    if SHSrank == SHSroot:
        writeline = "eigNth = "
        for eigNTH in eigNlistTH:
            if eigNTH < 0.0: 
                writeline += "% 7.6f, "%eigNTH
        #for i in range(len(result.x)):
            #print("thetalist[%s] = % 5.4f"%(i, ADDth.thetalist[i]), flush = True)
            #print("x[%s] = % 5.4f"%(i, ADDth.x[i]), flush = True)
        #xstr  = ", ".join(["%5.4f"%x for x in ADDth.x])
        xstr     = ", ".join(["%5.4f"%x for x in ADDth.x])
        nADDstr  = ", ".join(["%5.4f"%x for x in ADDth.nADD])
        #print("x = %s"%xstr)
        #print("(IDnum, ADD, ADD_IOE), time = (%s, % 5.3f, % 5.3f), % 5.3f"%(
                #ADDth.IDnum, ADDth.ADD, ADDth.ADD_IOE, time3 - time2), flush = True)
        #print(writeline, flush = True)
        print(" this point is not Equiblium on SH; chk bifurcation", flush = True)
    if sphereN == 0:
        csvname = "./initialTS.csv"
    else:
        csvname = "./Bifurcation.csv"
    BifnADDcsv = "./BifnADD%s.csv"%sphereN
    #if os.path.exists(csvname):
    if os.path.exists(BifnADDcsv):
        Bifdismin = 1.0e30
        for line in open(BifnADDcsv):
            line = line.split(",")
            #BifPoint = np.array(line, dtype = float)
            #dis = np.linalg.norm(BifPoint - ADDth.x)
            #dis = BifPoint - ADDth.x
            #dis = max(np.abs(x) for x in dis)
            #if dis < Bifdismin:
                #Bifdismin = copy.copy(dis)
            BifnADD = np.array(line, dtype = float)
            disangle = functions.angle(BifnADD, ADDth.nADD)
            if disangle < Bifdismin:
                Bifdismin = copy.copy(disangle)
        #if dis < const.sameEQthreshold:
        if Bifdismin < const.bifucationTH:
            if SHSrank == SHSroot:
                print("This bifurcation is calculated; remove", flush = True)
            _ADDremoveQ = True
    if _ADDremoveQ:
        return bifQ, returnoptQ, _ADDoptQ, _ADDremoveQ, bifADDths

    eigVlist_TH = []
    for i in range(dim-1):
        eigVlist_TH.append(_eigVTH[:,i])
    eigVlist_TH = np.array(eigVlist_TH)
    #print(eigVlist_TH[0])
    _ADDoptQ = True
    _ADDremoveQ = True
    for ADDthdamp in ADDths:
        if ADDthdamp.IDnum == ADDth.IDnum:
            ADDthdamp.ADDoptQ = True
            ADDthdamp.ADDremoveQ = True

    bifADDth = copy.copy(ADDth)
    bifADDth.ADDremoveQ = False
    bifADDth.IDnum     = max(x.IDnum for x in ADDths) + 1
    bifADDth.thetalist = bifADDth.thetalist + eigVlist_TH[0] * const.s_bif0
    bifADDth.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, bifADDth.thetalist, const)
    bifADDth.thetalist = functions.calctheta(bifADDth.nADD, eigVlist, eigNlist)

    deltaTH = np.zeros(len(ADDth.thetalist))
    jac = bifADDth.grad_theta(f, grad, eqpoint,
            IOEsphereA, deltaTH, A_eq, ADDths, const,
            eigVlist ,eigNlist)
    Ejac = jac / np.linalg.norm(jac)
    whileN = 0
    while whileN < 100:
        whileN += 1
        bifADDth.thetalist = bifADDth.thetalist - Ejac * const.s_bif
        bifADDth.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, bifADDth.thetalist, const)
        bifADDth.thetalist = functions.calctheta(bifADDth.nADD, eigVlist, eigNlist)
        newjac = bifADDth.grad_theta(f, grad, eqpoint,
            IOEsphereA, deltaTH, A_eq, ADDths, const,
            eigVlist ,eigNlist)
        Enewjac = newjac / np.linalg.norm(newjac)
        #if SHSrank == SHSroot:
            #rint("%s, % 3.2f"%(whileN, np.linalg.norm(newjac)), flush = True)
        if np.dot(Ejac, Enewjac) < 0.0:
            break
        Ejac = copy.copy(Enewjac)

    x_initial = np.zeros(len(ADDth.thetalist)),
    result = minimize(
        lambda deltaTH: bifADDth.calc_onHS(deltaTH, f,
            eqpoint, IOEsphereA, A_eq, ADDths, const,
            ),
        jac = lambda deltaTH: bifADDth.grad_theta(f, grad, eqpoint,
            IOEsphereA, deltaTH, A_eq, ADDths, const,
            eigVlist ,eigNlist),
            x0 = x_initial,
        #options = {"gtol":const.minimize_threshold},
        method="L-BFGS-B")
    bifADDth.thetalist += result.x
    bifADDth.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, bifADDth.thetalist, const)
    bifADDth.thetalist = functions.calctheta(bifADDth.nADD, eigVlist, eigNlist)

    bifADDth.x         = eqpoint + bifADDth.nADD
    bifADDth.x         = functions.periodicpoint(bifADDth.x, const)

    bifADDth2 = copy.copy(ADDth)
    bifADDth2.ADDremoveQ = False
    bifADDth2.IDnum     = max(x.IDnum for x in ADDths) + 2
    bifADDth2.thetalist = bifADDth2.thetalist - eigVlist_TH[0] * const.s_bif0
    bifADDth2.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, bifADDth2.thetalist, const)
    bifADDth2.thetalist = functions.calctheta(bifADDth2.nADD, eigVlist, eigNlist)

    deltaTH = np.zeros(len(ADDth.thetalist))
    jac = bifADDth2.grad_theta(f, grad, eqpoint,
            IOEsphereA, deltaTH, A_eq, ADDths, const,
            eigVlist ,eigNlist)
    Ejac = jac / np.linalg.norm(jac)
    whileN = 0
    while whileN < 100:
        whileN += 1
        bifADDth2.thetalist = bifADDth2.thetalist - Ejac * const.s_bif
        bifADDth2.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, bifADDth2.thetalist, const)
        bifADDth2.thetalist = functions.calctheta(bifADDth2.nADD, eigVlist, eigNlist)
        newjac = bifADDth2.grad_theta(f, grad, eqpoint,
            IOEsphereA, deltaTH, A_eq, ADDths, const,
            eigVlist ,eigNlist)
        Enewjac = newjac / np.linalg.norm(newjac)
        #if SHSrank == SHSroot:
            #print("%s, % 3.2f"%(whileN, np.linalg.norm(newjac)), flush = True)
        if np.dot(Ejac, Enewjac) < 0.0:
            break
        Ejac = copy.copy(Enewjac)
    x_initial = np.zeros(len(ADDth.thetalist)),
    result = minimize(
        lambda deltaTH: bifADDth2.calc_onHS(deltaTH, f,
            eqpoint, IOEsphereA, A_eq, ADDths, const,
            ),
        jac = lambda deltaTH: bifADDth2.grad_theta(f, grad, eqpoint,
            IOEsphereA, deltaTH, A_eq, ADDths, const,
            eigVlist ,eigNlist),
            x0 = x_initial,
        #options = {"gtol":const.minimize_threshold},
        method="L-BFGS-B")
    bifADDth2.thetalist += result.x
    bifADDth2.nADD      = ADDth.SuperSphere_cartesian(IOEsphereA, bifADDth2.thetalist, const)
    bifADDth2.thetalist = functions.calctheta(bifADDth2.nADD, eigVlist, eigNlist)
    bifADDth2.x         = eqpoint + bifADDth2.nADD
    bifADDth2.x         = functions.periodicpoint(bifADDth2.x, const)
    #bifdis = np.linalg.norm(bifADDth.x - bifADDth2.x)
    #bifdis = np.linalg.norm(bifADDth.x - bifADDth2.x)
    #bifdis = functions.periodicpoint(bifADDth.x, const, bifADDth2.x)
    #bifdis = np.linalg.norm(bifdis - bifADDth2.x)
    bifangle = functions.angle(bifADDth2.nADD, bifADDth.nADD)
    if SHSrank == SHSroot:
        #print("np.linalg.norm(bifADDth.x - bifADDth2.x) = %s"%bifdis, flush = True)
        print("angle(bifADDth.nADD, bifADDth2.nADD) = % 4.3f"%bifangle, flush = True)

    #if const.bifucationTH < bifdis:
    if const.bifucationTH < bifangle:
        if SHSrank == SHSroot:
            print("the bifurcation ADD is added", flush = True)
            with open(csvname, "a")  as wf:
                wf.write("%s\n"%xstr)
            with open(BifnADDcsv, "a")  as wf:
                wf.write("%s\n"%nADDstr)
        bifADDth.ADDoptQ = True
        bifADDth2.ADDoptQ = True
        #ADDths.append(bifADDth)
        #ADDths.append(bifADDth2)
        bifADDths = [bifADDth, bifADDth2]

        _ADDoptQ    = True
        _ADDremoveQ = True

        returnoptQ = True
    else:
        if sphereN == 0:
            if SHSrank == SHSroot:
                print("this ADD is TS and removed", flush = True)
            _ADDremoveQ = True
        else:
            _ADDoptQ = False
            _ADDremoveQ = False
    return bifQ, returnoptQ, _ADDoptQ, _ADDremoveQ, bifADDths

