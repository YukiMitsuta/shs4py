#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021/06/29 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, copy,shutil,sys
import numpy as np
import subprocess as sp

from . import functions
from . import IOpack
from . import umbrellaINT as UIint
from . import callwindows

#class const():
    #pass
class CallUmbIntClass():
    def __init__(self, constC,windowlist=None, rank=0, size=1, comm=None):
        if not os.path.exists("jobfiles"):
            os.mkdir("jobfiles")
        _keylist = constC.__dict__.keys()
        const = constC
        #print("constC = ",constC)
        #const.dim         = constC.dim             # dimension of CVs
        if not "betainv" in _keylist:
            const.Temp        = 298.0            # tempeture (K)
            const.k_B         = 1.38065e-26      # Boltzmann constant (kJ / K)
            const.N_A         = 6.002214e23      # Avogadro constant (mol^-1)
            const.betainv     = const.Temp * const.k_B * const.N_A # 1/beta (kJ / mol)
        const.pi          = np.pi            #     pi
        const.pi2         = np.pi * 2.0         # 2 * pi
        if not "grofilename" in _keylist:
            #const.grofilename = "npt.gro"
            const.grofilename = "run.gro"
        const.argvs       = sys.argv
        if len(const.argvs) < 2:
            const.systemname = "main"
            const.pwdpath    = os.getcwd()
        else:
                 systemname = argvs[1]
                 if os.path.exists("./pwdpath.dat"):
                     const.pwdpath = open("./pwdpath.dat", "r").read()
                 elif os.path.exists("../pwdpath.dat"):
                     const.pwdpath = open("../pwdpath.dat", "r").read()
                 elif os.path.exists("../../pwdpath.dat"):
                     const.pwdpath = open("../../pwdpath.dat", "r").read()
                 elif os.path.exists("../../../pwdpath.dat"):
                     const.pwdpath = open("../../../pwdpath.dat", "r").read()
                 else:
                     print("ERROR: there is not pwdpath.dat")
                     return False

        const.moveQ           = False  # move temp dir or not (for mir2 system)
        const.shellname       = "bash"
        const.needgpuid       = False
        const.allTScalcQ      = False # calculate of all TS point or not calculate
        const.chkDihedralQ    = False  # check dihedral of amide plane
        const.Ndihedral       = 2     # the number of dihedrals 
        const.KoffdiagonalQ   = True
        const.nearestWindowQ  = True
        const.parallelPy      =    1         # minimum of parallel number of python
        const.parallelMDmax   = 1
        const.parallelMDs     = []
        for i in range(1,const.parallelMDmax + 1):
            if const.parallelMDmax % i == 0:
                const.parallelMDs.append(i)
        const.mpirunlist      = ["mpirun", "-np", "%s"%const.parallelPy]
        const.maxtimeEQUI        = "50"           # chk the end of calculation (ps)
        const.maxtime            = "100"           # chk the end of calculation (ps)
        const.maxtimeOPT         = "1000"          # chk the end of calculation (ps)
        const.maxtimeMFEP        = "1000"          # chk the end of calculation (ps)
        const.maxtimeEQ          = "1000"          # chk the end of calculation (ps)
        const.ADDturnNmax     =   999
        
        if not "allperiodicQ" in _keylist:
            const.allperiodicQ    = False
            const.periodicmax     = np.array([ 1.0e30     for x in range(const.dim)])
            const.periodicmin     = np.array([-1.0e30     for x in range(const.dim)])
        if not "wallmax" in _keylist:
            const.wallmax         = np.array([ 1.0e30 for x in range(const.dim)])
        if not "wallmin" in _keylist:
            const.wallmin         = np.array([-1.0e30 for x in range(const.dim)])
        const.lockfilepath = const.pwdpath + '/CONTROL/LockFile'
        const.lockfilepath_UIlist = const.pwdpath + '/CONTROL/LockFile_UIlist'
        if not os.path.exists(const.lockfilepath):
            with open(lockfilepath, "w") as wf:
                wf.write("")
        if not os.path.exists(const.lockfilepath_UIlist):
            with open(lockfilepath_UIlist, "w") as wf:
                wf.write("")
        const.OPTthreshold    = 1.0
        const.nextsteproundN  = 10
        const.phitol          = 0.08
        # the start threshold to choose the constrain of windows (distance between average to target point)
        const.threshold       = 0.05
        #const.threshold       = 1.0 
        const.UIminimizethreshold = 0.01
        #UIminimizethreshold =  0.001
        #UIminimizethreshold = 0.00001
        # the start threshold to choose the constrain of windows (distance between target point to reference point)
        const.refthreshold    = 10.00
        # the calculated point is move to end point if the ratio of histgram is larger than this parameter
        const.didpointsigmaTH = 1.20
        const.didpointTH      = const.didpointsigmaTH
        #didpointTH      = np.exp(-didpointsigmaTH * didpointsigmaTH * 0.5)
        #print("didpoinitTH = %s"%didpointTH)
        const.connectionsigmaTH = 0.5
        const.recallsigmaTH   = 3.0
        const.recallTH        = const.recallsigmaTH
        #recallTH        = np.exp(-recallsigmaTH * recallsigmaTH * 0.5)
        const.recallturnmax   = 10
        # the threshold of nextstep point (if the nextpoint is calculated, more larger parameter was used)
        #ADDstepsigmaTH  = 2.5
        const.ADDstepsigmaTH  = 1.5
        #ADDstepsigmaTH  = 1.0
        const.ADDstepTH       = const.ADDstepsigmaTH
        #ADDstepTH       = np.exp(-ADDstepsigmaTH * ADDstepsigmaTH * 0.5)
        const.firstspheresigmaTH = 3.0
        const.neighborwindowTH  = 3.0
        #nextstepsigmamaxTH = 3.0
        #nextstepsigmaminTH = 3.0
        const.nextstepsigmamaxTH = 3.0
        const.nextstepsigmaminTH = 2.5
        #nextstepTH      = np.exp(-nextstepTH * nextstepTH * 0.5)
        #print("nextstepTH = %s"%nextstepTH)
        #neighborsigmaTH = 1.0e30
        #neighborsigmaTH = 10.0
        const.neighborsigmaTH = 2.5
        #neighborTH      =  5.0
        const.MFEPsigmaTH      =  1.5
        const.HESSIANsigmaTH      =  3.0
        const.integratesigmaTH =  5.0
        #neighborTH      = np.exp(-neighborsigmaTH * neighborsigmaTH * 0.5)
        const.edgelistsigmaTH =  3.0
        const.errorwindowsigmaTH = 1.2
        const.lADDnQ           = True
        #const.IOEl             = 4
        #const.IOEl_forADDstart = IOEl * 2
        #const.IOEl_forcollect  = IOEl * 3
        # the threshold of gradient to gain saddle point in ADD step
        const.gradTH          = 10.0
        const.gradmaxTH       = 10.0
        #ADDdeltaA       =  0.5
        #ADDdeltaAtimes  =  0.01
        const.ADDdeltaAtimes  =  0.02
        #ADDdeltaAtimes  =  0.05
        #ADDdeltaAtimes  =  0.5
        const.ADDfesphereAmax =  0.5
        const.minimizeBoundadd =  np.pi * 0.05 # Bound of minimize in L-BFES-B
        const.minimizeBound    =  np.pi * 0.05 # Bound of minimize in L-BFES-B
        const.minimizeBoundmax =  np.pi * 0.05 # Bound of minimize in L-BFES-B in firstADD
        const.deltaBoundsigmaMIN = 0.2 # see ADD.mkminimizeBound
        const.deltaBoundsigmaMAX = 1.0 # see ADD.mkminimizeBound
        const.deltaBoundTIMES    = 5   # see ADD.mkminimizeBound
        const.deltaBoundrecallTH = 0.5
        const.minimizeTH       = 0.01       # threshold of minimize
        const.minimizeBoundL   = 0.05       # Bound of minimize with length
        const.minimizeboundsigmaTH = 2.0
        const.minimizeboundsigma_allTH = 1.0 
        const.IOEsphereAnum   = 1
        #gradTHforM = 10.0
        const.feTH            = -0.5
        const.IOETH           = -0.1
        const.E               = np.identity(const.dim) # Identity matrix
        const.Kmin            =     100.0        # minimum of constrain
        const.Kminchunemax    =     200.0
        const.optK            =     100.0        # minimum of constrain
        const.eqK             =     100.0        # minimum of constrain
        const.mfepK           = const.E * 100.0        # minimum of constrain
        const.whamK           = const.E * 100.0        # minimum of constrain
        const.Kmax            =    5000.0        # minimum of constrain
        #deltas0         = 0.15
        const.deltas0_sigma   = 1.00
        #deltas0_sigma   = 1.00
        const.deltas0_sigmaIF = 2.00
        #deltas          = 0.15
        const.deltas_sigma    = 0.50
        #deltas_sigma    = 0.50
        const.epsilon         = 0.50
        const.minimumAmax     = 1000.0
        const.optgradmax      = 1000.0
        const.Pbiasconst      = 1.0 / np.power(2.0 * np.pi, const.dim * 0.5)
        #ADDlengthTH     = 0.5
        const.sameANthreshold =  20
        const.optturnNmax     = 20
        #optturnNmax     = 10
        const.TSoptdimerturnNmax = 1000
        #TSoptdimerturnNmax = 1
        const.parallelADDQ = False
        const.edgeNmax = 1.0e30
        const.partADDQ = False
        const.partOPTQ = False
        const.partdim  = 8
        const.Krandommax = 1000
        const.covbinwidthN    =  5000
        const.nextstepconstQ  = False
        const.nextstepDIOE    = const.pi2 / 15.0
        const.nextstepDADD    = const.pi2 / 15.0
        const.nextKconstQ     = False
        const.nextKconst      = const.E * 300.0       # nextK with constant
        const.calltime        = 60 * 60 * 60 # max time of np.call job (second)
        const.usefcntlQ = False
        os.environ["OMP_NUM_THREADS"] = "1"
        const.WindowDataType = "hdf5"
        const.UIlistchunksize = 1000
        
        const.UIoptMethod = "ADAM"

        self.const = const
        self.rank = rank
        self.size = size
        self.comm = comm
        self.root = 0
    def readUIeq(self, UIeqCOLpath, UIeqplumedpath):
        UIeq = UIint.UIstep(self.const)
        pldiclist = functions.importplumed(UIeqplumedpath)
        UIeq.calculationpath =  UIeqplumedpath.replace("plumed.dat","")
        UIeq.ref = np.zeros(self.const.dim)
        UIeq.K =  np.zeros((self.const.dim, self.const.dim))
        arglist = []
        for pldic in pldiclist:
            if any("CV" in x for x in pldic["comments"]):
                arglist.append(pldic["LABEL"])
        for dic in pldiclist:
            if "RESTRAINT" in dic["options"]:
                for i_ref, arg in enumerate(arglist):
                    if dic["ARG"] == arg:
                        UIeq.ref[i_ref] = float(dic["AT"])
                        UIeq.K[i_ref,i_ref] = float(dic["KAPPA"])
            elif "RESTRAINTMATRIX" in dic["options"]:
                UIeq.ref[i_ref] = float(dic["AT"])
                for i in range(self.const.dim):
                    k = ",".split(dic["KAPPA%s"%i])
                    for j in range(self.const.dim):
                        UIeq.k[i,j]=float(k[j])
                
        UIeq.readdat(UIeqCOLpath)
        #print("UIeq.const.dim ",UIeq.const.dim)
        #print("UIeq.covinv ",UIeq.covinv)
        #exit()
        UIeq.stepN = 0
        UIeq.A = 0.0
        UIeq.aveT = UIeq.ref
        UIeq.aveTinitial = UIeq.ref

        self.WGraph = False
        self.UIlistall = []
        self.WGraph, self.UIlistall = UIint.importUIlistall_exclusion(self.const,self.WGraph, self.UIlistall)
        UIlistpathset = set(UI.calculationpath for UI in self.UIlistall)
        if not UIeq.calculationpath in UIlistpathset:
            #print("add UIeq")
            #print(UIlistpathset)
            #for x in UIlistpathset:
                #print(x)
                #print(UIeq.calculationpath)
            #exit()
            UIeq, self.WGraph = UIint.exportUI_exclusion(self.const,UIeq, self.WGraph, self.UIlistall)
        self.UIeq = UIeq
        #print(UIeq.path)
        #print(self.UIlistall[0].path)
        #print(self.UIlistall[1].path)
        #print([UI.path for UI in self.UIlistall])
        #exit()
        #self.UIlist = self.UIlistall
        self.UIlist = []

        self.parallelC = functions.parallelClass()
        self.parallelC.UIlist = []
        self.WGraph, self.UIlistall = UIint.importUIlistall_exclusion(self.const,self.WGraph, self.UIlistall)
        self.updateUIlist([UIeq.ave])
        self.parallelC.UIeq = self.UIeq
        self.parallelC.minimumA = 0.0
        #self.parallelC.UIlist = self.UIlist
        self.parallelC.UIlistall = self.UIlistall
        #self.parallelC.UIlistdamp = self.UIlist
        #self.parallelC.UIlist_initial = self.UIlist
        #self.parallelC.UIlist_global = self.UIlist
        self.parallelC.WGraph = self.WGraph
        self.parallelC.const = self.const

        self.parallelC.edgelist = self.UIlist
        self.parallelC.edgeN = 0
        self.parallelC.endlist = []

        self.parallelC.nextsteps_global    = []
        self.parallelC.endsteps_global     = []
        self.parallelC.errorsteps_global   = []
        self.parallelC.processturnN_global = []
        self.parallelC.runN_global = []
        self.parallelC.mpID = 0

        self.parallelC.forcecallwindowQ = False
        self.parallelC.initialpoint = self.UIeq.ave

        self.parallelC.callN = 0
    def f(self,x):
        if not isinstance(x,list):
            x = [x]
        self.parallelC.callN += 1
        if self.parallelC.callN > 100:
            if len(self.parallelC.UIlist) > 300:
                self.parallelC.callN = 0
                self.updateUIlist(x)
        returnf = []
        cwdhere = os.getcwd()
        for finishpoint in x:
            self.parallelC.finishpoint = finishpoint
            UIlistdamp = callwindows.chkneedwindow(self.parallelC)
            if UIlistdamp is False:
                print("f@UmbInterface: error in chkneedwindow")
                #exit()
                os.chdir(cwdhere)
                return False
            self.parallelC.UIlistdamp = UIlistdamp
            self.parallelC.UIlist = UIlistdamp
            A, varA = UIint.calcUIall_nearW_ave(self.const,self.parallelC.UIlist, finishpoint)
            if A is False:
                self.parallelC.finishpoint = finishpoint
                UIlistdamp = callwindows.chkneedwindow(self.parallelC)
                self.parallelC.UIlistdamp = UIlistdamp
                self.parallelC.UIlist = UIlistdamp
                A, varA = UIint.calcUIall_nearW_ave(
                    self.const,self.parallelC.UIlist, finishpoint)
                if A is False:
                    #print(self.parallelC.UIlist)
                    #print("sigma ",self.parallelC.UIlist[0].sigma)
                    print("ref ",self.parallelC.UIlist[0].ref)
                    print("ave", self.parallelC.UIlist[0].ave)
                    raise ValueError("free energy cannot calculated") 
        

            returnf.append(A)
        os.chdir(cwdhere)
        if len(x) == 1:
            return returnf[0]
        else:
            return returnf
    def g(self,x, debagQ=False):
        returnfs = self.f(x)
        if returnfs is False:
            print("g@UmbInterface: error in chkneedwindow")
            #os.chdir(cwdhere)
            return False, False
        returngrads = []
        #print("287: pwd = ",os.getcwd())
        cwdhere = os.getcwd()
        if not isinstance(x,list):
            x = [x]
        for finishpoint in x:
            grad = UIint.gradUIall(self.const,self.parallelC.UIlist,finishpoint)
            returngrads.append(grad)
        #os.chdir(self.const.pwdpath)
        #os.chdir(self.const.pwdpath+"/jobfiles")
        os.chdir(cwdhere)
        #print("295: pwd = ",os.getcwd())
        if len(x) == 1:
            return returnfs, returngrads[0]
        else:
            return returnfs, returngrads
    def grad(self,x, debagQ=False):
        _, returngrad = self.g(x)
        return returngrad
    def hessian(self,x, debagQ=False):
        #print("len(UIlist) = ",len(self.UIlist))
        self.WGraph, self.UIlistall = UIint.importUIlistall_exclusion(self.const,self.WGraph, self.UIlistall)
        self.parallelC.UIlistall = self.UIlistall
        print("len(UIlistall) = ",len(self.UIlistall))
        print("x =",x)
        cwdhere = os.getcwd()
        self.parallelC.finishpoint = x
        getwindowQ = callwindows.chkneedwindow(self.parallelC)
        if getwindowQ is False:
            print("hessian@UmbInterface: error in chkneedwindow")
            os.chdir(cwdhere)
            return False
        dmin = UIint.Dmin(self.parallelC.UIlist,x)
        #if False:
        if self.const.didpointsigmaTH < dmin: 
            print("hessian@UmbInterface: dmin = %s; add window"%dmin)
            for _ in range(10):
                parallelCdamp = copy.copy(self.parallelC)
                #parallelCdamp = self.parallelC
                parallelCdamp.forcecallwindowQ = True
                parallelCdamp.finishpoint = x
                getwindowQ = callwindows.chkneedwindow(parallelCdamp)
                self.parallelC.forcecallwindowQ = False
                self.parallelC.runN_global = []
                dmin = UIint.Dmin(self.parallelC.UIlist,x)
                if dmin < self.const.didpointsigmaTH: 
                    break
            if getwindowQ is False:
                print("hessian@UmbInterface: error in chkneedwindow")
                os.chdir(cwdhere)
                exit()
                return False
            dmin = UIint.Dmin(self.parallelC.UIlist,x)
            if self.const.didpointsigmaTH < dmin: 
                print("hessian@UmbInterface: error ; dmin = %s"%dmin)
                os.chdir(cwdhere)
                return False
        UIdamp = []
        for UI in self.parallelC.UIlist:
            if UI.Dbias(x) < self.const.HESSIANsigmaTH:
                UIdamp.append(UI)
        hess= UIint.HessianUIall(self.const,UIdamp, x)
#        dim  = len(x)
#        eps  = 1.0e-3
#        hess = np.empty((dim,dim))
#        for i in range(dim):
#            xdeltap = copy.copy(x)
#            xdeltap[i] += eps
#            gradp = UIint.gradUIall(self.const,self.UIlist,xdeltap)
#            xdeltam = copy.copy(x)
#            xdeltam[i] -= eps
#            gradm = UIint.gradUIall(self.const,self.UIlist,xdeltam)
#            gradgrad = (gradp - gradm)/eps*0.5
#            for j in range(dim):
#                hess[i,j] = gradgrad[j]
        #print("hess = ",hess)
        return hess
    def SLsampling(self):
        whileN = 0
        endsteplist = []
        while whileN < 1000:
            whileN += 1
            UIlistdamp = self.UIlist
            UIlistdamp.sort(key = lambda x: x.A)
            for UI in UIlistdamp:
                #print("UI.stepN = ", UI.stepN)
                if not UI.stepN in endsteplist:
                    break
            else:
                print("All points are calculated")
                exit()
            dminMax = 0.0
            UIlist_outer = [UIround for UIround in UIlistdamp if UIround.stepN != UI.stepN]
            for i in range(20):
                theta= 2.0*np.pi*0.05*i
                vec = np.array([np.cos(theta),np.sin(theta)])
                #print("%s,%s"%(vec[0],vec[1]))
                dist = self.finddist(UI, vec)
                point = UI.ave + vec*dist
                dmin = UIint.Dmin(UIlist_outer,point)
                if dminMax <dmin:
                    dminMax = dmin
                    minimumpoint = point
            print("dminMax = ",dminMax)
            print("minimumpoint =", minimumpoint)
            if dminMax < 1.5:
                dminMax = 0.0
                for i in range(20):
                    theta= 2.0*np.pi*0.05*i
                    vec = np.array([np.cos(theta),np.sin(theta)])
                    dist = self.finddist(UI, vec)
                    for j in range(1,21):
                        point = UI.ave + 0.05*j*vec*dist
                        dmin = UIint.Dmin(UIlistdamp,point)
                        if dminMax <dmin:
                            dminMax = dmin
                            minimumpoint = point
                print("dminMax = ",dminMax)
                print("minimumpoint =", minimumpoint)
                if dminMax < 1.5:
                    endsteplist.append(UI.stepN)
                    continue
            beforeUIlen = len(self.UIlist)
            fsampl = self.f(minimumpoint)
            afterUIlen = len(self.UIlist)
            if beforeUIlen == afterUIlen:
                endsteplist.append(UI.stepN)
                print("len(endsteplist) =", len(endsteplist))

            #exit()
    def finddist(self,UI, vec):
        dist  = 0.1
        delta = 0.1
        for _ in range(50):
            point = UI.ave+dist*vec
            dbias = UI.Dbias(point)
            if dbias < 3.0:
                dist += delta
            else:
                break
        delta = dist*0.5
        dist -= delta
        for _ in range(50):
            point = UI.ave+dist*vec
            dbias = UI.Dbias(point)
            #print(dbias)
            if dbias < 3.0:
                dist += delta
            else:
                delta *= 0.5
                if delta < 0.001:
                    break
                dist -= delta
        return dist
    def updateUIlist(self,pointlist):
        print("before: len(UIlist) = ",len(self.parallelC.UIlist))
        UIlistpathset = set(UI.path for UI in self.parallelC.UIlist)
        UIlist = []
        for newUI in self.UIlistall:
            #print(newUI.calculationpath)
            #print(newUI.Dbias(point))
            for point in pointlist:
                if newUI.Dbias(point) < self.const.integratesigmaTH:
                    newUI.importdata(newUI.path)
                    UIlist.append(newUI)
                    break
        self.UIlist = UIlist
        self.parallelC.UIlist = UIlist
        self.parallelC.UIlistdamp = UIlist
        self.parallelC.UIlist_initial = UIlist
        self.parallelC.UIlist_global = UIlist
        print("after:  len(UIlist) = ",len(self.parallelC.UIlist))
