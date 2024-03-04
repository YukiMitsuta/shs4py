#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#! /work/mitsutay13/tools/python-3.4.3/bin/python3.4
#
# Distributed under terms of the MIT license.
import os, glob, shutil, sys, re, gc, datetime
import copy, inspect, itertools, fcntl, time
import subprocess as sp
import numpy      as np
from   scipy import stats
from   scipy.optimize import minimize
import multiprocessing as mp
import networkx   as nx
import h5py


#from . import *
from . import functions 
from . import IOpack    

import pyximport ## for cython
pyximport.install()
try:
    from . import UIstepCython
except ImportError:
    import UIstepCython
include_dirs = [np.get_include()]

#if not usefcntlQ:
    #import fasteners
import fasteners

rank = 0
size = 1
root = 0

"""
Updated in 
2022-06-16
"""

class UIpotential():
    def __init__(self,const, rank, root, size, comm, UIinitial = False):
        self.const = const
        self.rank  = rank
        self.root  = root
        self.size  = size
        self.comm  = comm
        self.plumeddic = functions.importplumed(self.const.plumedpath)

        windowspath = "%s/%s/windows"%(self.const.pwdpath,self.const.jobfilepath)
        if os.path.exists(windowspath) is False:
            os.mkdir(windowspath)
            with open("%s/pathlist.dat"%windowspath, "w") as wf:
                wf.write("#  calculation path  -> window data path\n")

        self.UIlistall = False
        self.WGraph = False
        if not UIinitial is False:
            UIinitial, self.WGraph = exportUI_exclusion(UIinitial, self.const, self.WGraph, self.UIlistall)
            self.UIlistall = [UIinitial]
        self.WGraph    = False
        self.WGraph, self.UIlistall = importUIlistall_exclusion(self.WGraph, self.UIlistall, self.const)
        #print(len(self.UIlistall))
    def f(self, x):
        if self.chkneedwindow(x):
            _Dmin, UIbefore = self.Dmin(x)
            deltaA_UIall, varA = calcUIall(self.UIlistall, UIbefore.ave, x, self.const)
            returnf = UIbefore.A + deltaA_UIall
            return returnf
        else:
            return 1.0e30
        
    def grad(self, x):
        if self.chkneedwindow(x):
            eps = np.sqrt(np.finfo(float).eps)
            if self.const.periodicQ:
                return UIstepCython.cgradUIall_periodic(self.UIlistall, x,
                    self.const.periodicmax, self.const.periodicmin, self.const.dim, self.const.betainv, eps)
            else:
                return UIstepCython.cgradUIall(self.UIlistall, x, self.const.dim, self.const.betainv, eps)
        else:
            return [1.0e30 for _ in range(len(x))]
    def hessian(self, x):
        if self.chkneedwindow(x):
            hessdamp    = HessianUIall(self.UIlistall, x, self.const)
            return hessdamp
        else:
            return 1.0e30
    def chkneedwindow(self, x):
        whileN = 0
        while whileN < 1000:
            whileN += 1
            _Dmin, UIbefore = self.Dmin(x)
            #if _Dmin < self.const.neighborsigmaTH:
            if _Dmin < self.const.ADDstepsigmaTH:
                break
            print("x    = ",x)
            print("Dmin = ",_Dmin)
            print("new window is required.")
            if self.calcEdgeWindow(x,UIbefore) is False:
                return False
        else:
            print("ERROR in chkneedwindow: whileN over 1000")
            return False
        return True
    def Dmin(self, xi):
        minD = 1.0e30
        UIbefore = False
        for UI in self.UIlistall:
            if UI.stepN < 0:
                print(UI.stepN)
                continue
            UID = UI.Dbias(xi)
            if UID < minD:
                minD = UID
                UIbefore = UI
        return minD, UIbefore
    def getNeedWindowPoint(self,x,sigmaminTH, UIbefore):
        initialpoint = copy.copy(UIbefore.ave)
        initialpoint = functions.periodicpoint(initialpoint, self.const)
        if self.const.periodicQ:
            xdamp = UIstepCython.periodic(x,
                    self.const.periodicmax, self.const.periodicmin, initialpoint, self.const.dim)
        else:
            xdamp = x
        nADD = xdamp - initialpoint
        #acceptQ, needwindowpoint = UIstepCython.getneedwindowC(self.UIlistall, self.UIlistall,
            #initialpoint, xdamp, nADD,
            #self.const.ADDstepsigmaTH, self.const.minimizeTH, sigmaminTH,
            #self.const.periodicQ, self.const.periodicmin, self.const.periodicmax, self.const.dim)
        i = 0
        while True:
            i += 1
            if 1000 < i:
                print("i over 1000: Dmin_max = %s"%Dmin_max)
            needwindowpoint = initialpoint + (i * 0.01) * nADD
            _dist = UIbefore.Dbias(needwindowpoint)
            if sigmaminTH < _dist:
                break
        return needwindowpoint
    def calcEdgeWindow(self,x,UIbefore):
        Kminchune = self.const.Kmin
        refthresholdchune = self.const.refthreshold
        sigmaminTHchune = self.const.nextstepsigmaminTH
        UIdamp = copy.copy(UIbefore)
        whileN = 0
        #while whileN < 1000:
        while whileN < 5:
            whileN += 1
            targetpoint = self.getNeedWindowPoint(x, sigmaminTHchune, UIbefore)
            print("targetpoint = ",targetpoint)
            print("D_before(targetpoint) = ",UIbefore.Dbias(targetpoint))

            K, newref, gradvector = UIdamp.nextK(targetpoint, refthresholdchune, Kminchune)
            UI = self.callwindow(K,newref,gradvector, targetpoint, UIbefore)
            D1= UIbefore.Dbias(UI.ave)
            D2= UI.Dbias(UIbefore.ave)

            print("UIdist (->) = ", D1)
            print("UIdist (<-) = ", D2)
            print("D(targetpoint) = ",UI.Dbias(targetpoint))
            c = inspect.currentframe()
            if self.const.nextstepsigmamaxTH < D1 or self.const.nextstepsigmamaxTH < D2:
            #if D1 < self.const.ADDstepsigmaTH or D2 < self.const.ADDstepsigmaTH:
                print("D1 and/or D2 are larger than %s"%self.const.nextstepsigmamaxTH)
                sigmaminTHchune -= 0.2
                #Kminchune += self.const.Kmin
                #UIdamp = copy.copy(UIbefore)
                if self.const.Kmax < Kminchune:
                    print("ERROR(%s): cannot calulate neighbor window"%c.f_lineno, flush = True)
                    UI = UIdamp
                    #return False
                    #exit()
                    break
            #elif self.const.nextstepsigmamaxTH < D1 or self.const.nextstepsigmamaxTH < D2:
            elif self.const.ADDstepsigmaTH < UI.Dbias(targetpoint):
                print("D(target) is larger than %s"%self.const.ADDstepsigmaTH)
                #Kminchune += self.const.Kmin
                #refthresholdchune *= 0.5
                #sigmaminTHchune -= 0.2
                Kminchune += self.const.Kmin
                print("sigmaminTHchune = ",sigmaminTHchune)
                UIdamp = copy.copy(UIbefore)
                #if refthresholdchune < 0.001:
                #if self.const.Kmax < Kminchune:
                #if sigmaminTHchune < self.const.ADDstepsigmaTH:
                if self.const.Kmax < Kminchune:
                #if sigmaminTHchune < 0.0:
                    print("ERROR(%s): cannot calulate neighbor window"%c.f_lineno, flush = True)
                    UI = UIdamp
                    #return False
                    #exit()
            
                    break
            else:
                break
        else:
            return False
        if UIbefore.path == UI.path:
            print("ERROR(%s): cannot calulate neighbor window"%c.f_lineno, flush = True)
            print("UIbefore.path = ",UIbefore.path)
            print("UI.path = ",UI.path)
            return False
        deltaA_UIall, varA = calcUIall(self.UIlistall+[UI], UIbefore.ave, UI.ave, self.const)
        UI.A  = UIbefore.A + deltaA_UIall
        UI.exportdata(UI.path)
        self.UIlistall.append(UI)
        self.UIinitial, self.WGraph = exportUI_exclusion(UI, self.const, self.WGraph, self.UIlistall)
        return True
    def callwindow(self,K,newref,gradvector,targetpoint,UIbefore):
        i = 0
        cwddamp = os.getcwd()
        while True:
            i += 1
            #stepdirname = "%s/step%s"%(os.getcwd(), i)
            stepdirname = "%s/%s/windows/step%s"%(self.const.pwdpath,self.const.jobfilepath, i)
            if not os.path.exists(stepdirname):
                os.mkdir(stepdirname)
                stepdirname = copy.copy(stepdirname)
                stepN       = copy.copy(i)
                break
        os.chdir(stepdirname)
        currentstep  = UIbefore.stepN
        callstepNdamp = 0
        dirnames = ""
        if UIbefore.mkplumed(newref, K, currentstep):
            dirnames += " step%s"%stepN
            callstepNdamp += 1
        os.chdir("../")
        #try:
        if True:
            if self.const.shellname == "csh":
                sp.call(["%s/calljob.csh"%tmppath, str(callstepNdamp), dirnames, "equiblium", str(self.const.parallelPy)],
                    timeout=self.const.calltime)
            elif self.const.shellname == "bash":
                sp.call(["%s/calljob.sh"%self.const.pwdpath, str(callstepNdamp), dirnames, "equiblium", str(self.const.parallelPy)],
                    timeout=self.const.calltime)
            else:
                c = inspect.currentframe()
                print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, self.const.shellname), flush = True)
    
        #except:
            #c = inspect.currentframe()
            #print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno, self.const.calltime))
    #####
    ##### now the recallturn is ignored in DC calculation: future task
    #####
    
#        #try:
#        if True:
#            if self.const.shellname == "csh":
#                sp.call(["%s/calljob.csh"%self.const.tmppath, str(callstepNdamp), dirnames, "MFEP", str(self.coonst.parallelPy)],
#                    timeout=self.const.calltime)
#            elif self.const.shellname == "bash":
#                sp.call(["%s/calljob.sh"%self.const.pwdpath, str(callstepNdamp), dirnames, "MFEP", str(self.const.parallelPy)],
#                    timeout=self.const.calltime)
#            else:
#                c = inspect.currentframe()
#                print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
#        #except:
#        else:
#            c = inspect.currentframe()
#            print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno, self.const.calltime))
#            exit()
    
        dirname        = stepdirname
        UI             = UIstep(self.const)
        UI.K           = K
        #periodicQ      = periodicQdic
        refpoint       = newref
        UI.ref         = np.array(refpoint, dtype=float)
        UI.aveT        = targetpoint 
        #UI.aveTinitial = aveTinitial 
        UI.aveTinitial = targetpoint
        UI.stepN       = stepN
        #UIbefore_initial = UIbefore_initial

        COLlist = glob.glob("%s/COLVAR"%dirname) 
        if not COLlist:
            COLlist += glob.glob("%s/COLVAR.*"%dirname)
        errorstepQ = False
        if len(COLlist) == 0:
            with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                c = inspect.currentframe()
                wf.write("Debag(%s): there is not COLVAR in %s\n"%(c.f_lineno, dirname))
        elif UI.readdat(COLlist[0]) is False:
            with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                c = inspect.currentframe()
                wf.write("Debag(%s): In %s, this calculation is not finished.\n"%(c.f_lineno, dirname))
        UI.calculationpath = UI.path
        os.chdir(cwddamp)
        return UI

class UIstep():
    """
    This is the Class of UIstep
    to analyze .dat file and obtain gradient and hessian
    with Umbrella Integration.
    """
    def __init__(self,const):
        self.const = const
        
    def readdat(self, filename, filename_fix = None, maxtimehere = False, 
            whamQ = False, calccovQ = False, rmdataQ = True):
        """
        read $filename and analyse data
        """
        if maxtimehere is False:
            maxtimehere = self.const.maxtime
        self.path   = os.path.dirname(filename)
        self.connections = [0]
        if os.path.exists(filename) is False:
            c       = inspect.currentframe()
            print(
"""ERROR(%s): There is not %s
: The calculation may not run.
***EXIT***"""%(c.f_lineno, filename))
            return False
        if filename_fix is None:
            fieldnames, ldamp, t = self.readCOORD(filename)
            if fieldnames is False:
                return False
        else:
            fieldnames,     ldamp_calc, t = self.readCOORD(filename)
            if fieldnames is False:
                return False
            fieldnames_fix, ldamp_fix,  t = self.readCOORD(filename_fix)
            if fieldnames_fix is False:
                return False
            indexdic = {}
            colname = filename.split("/")[-1]
            for line in open(filename.replace(colname, "plumed.dat")):
                line = line.split()
                if len(line) == 0:
                    continue
                line[0]   = line[ 0].replace( ":", "")
                fullindex = line[-1].replace("\n", "")
                if line[0] in fieldnames:
                    indexdic[int(fullindex)] = (ldamp_calc, fieldnames.index(line[0]))
                elif line[0] in fieldnames_fix:
                    indexdic[int(fullindex)] = (ldamp_fix, fieldnames_fix.index(line[0]))
            ldamp = []
            for i in range(len(ldamp_calc)):
                l = []
                for j in range(self.const.dim):
                    ldamp_x, fieldindex = indexdic[j]
                    l.append(ldamp_x[i][fieldindex])
                #if i == 1:
                    #print(l)

                ldamp.append(l)

        if t != maxtimehere:
            c = inspect.currentframe()
            print(
"""ERROR(%s): The calculation of %s is not finished(time is %s),
or maxtime(%s) is not exact.
***EXIT***"""%(c.f_lineno, filename, t, maxtimehere)
                 )
            return False
        self.N               = len(ldamp) * 0.001
        #self.N               = len(ldamp)
        self.data            = np.array(ldamp, dtype=float)
        self.cov             = np.cov(self.data.T)
        #print(self.cov)
        self.covinv          = np.linalg.inv(self.cov)
        self.ave             = self.data.mean(0)
        self.hes             = self.const.betainv * self.covinv - self.K
        self.hesinv          = np.linalg.inv(self.hes)
        self.eigN, _eigV     = np.linalg.eigh(self.hes)
        self.eigV = []
        for i in range(self.const.dim):
            self.eigV.append(_eigV[:,i])
        self.Pbiasave        = self.const.Pbiasconst / np.sqrt(np.linalg.det(self.cov))
        self.cov_eigN, _cov_eigV = np.linalg.eigh(self.cov)
        self.cov_eigV = []
        for i in range(self.const.dim):
            self.cov_eigV.append(_cov_eigV[:,i])
        self.cov_eigV = np.array(self.cov_eigV)

        minimumangleMax = 1.0e30
        segN = 0
        while 0.05 < minimumangleMax:
            segN += self.const.covbinwidthN
    #        while self.N / n > 24:
    #            shapiroQ = []
    #            i = 0
    #            while i * n + n < self.N:
    #                if len(self.data[i * n: i * n + n]) < n:
    #                    #print(self.data[i * n: i * n + n].T[j])
    #                    continue
    #                W, p = self.Shapiro_Maharanobis(self.data[i * n: i * n + n])
    #                if p < 0.05:
    #                    print("   (n, p) = (%s, %s)"%(n, p))
    #                    shapiroQ.append(False)
    #                else:
    #                    print("==>(n, p) = (%s, %s)"%(n, p))
    #                    shapiroQ.append(True)
    #                i += 1
    #            if all(shapiroQ):
    #                break
    #            else:
    #                n += covbinwidthN 
    #        else:
    #            exit()
            self.covinv_eigN, covinv_eigVdamp = np.linalg.eigh(self.covinv)
            self.covinv_eigV = []
            for i in range(self.const.dim):
                self.covinv_eigV.append(covinv_eigVdamp[:,i])
            #if not calccovQ:
            if True:
                break
            if calccovQ:
                self.ave_traj     = [[] for _ in range(self.const.dim)]
                self.sigmasq_traj = [[] for _ in range(self.const.dim)]
                #P      = self.cov_eigV
                P      = np.array(self.covinv_eigV)
                Pinv = np.linalg.inv(P)
                i = 0
                minimumanglelist = []
                #while i * covvindeltaN + segN <= self.N / 0.001:
                    #datadamp = self.data[i * covvindeltaN: i * covvindeltaN + segN]
                while i * segN + segN <= self.N / 0.001:
                    datadamp = self.data[i * segN: i * segN + segN]
                    avedamp = datadamp.mean(0)
                    #print(len(datadamp))
                    covdamp = np.cov(datadamp.T)
                    cov_eigNdamp, _cov_eigVdamp = np.linalg.eigh(covdamp)
                    cov_eigVdamp = []
                    for j in range(self.const.dim):
                        cov_eigVdamp.append(_cov_eigVdamp[:,j])
                    cov_eigVdamp = np.array(cov_eigVdamp)
                    covinvdamp = np.linalg.inv(covdamp)
                    covinv_eigVdamp = np.array(covinv_eigVdamp)
                    covinv_eigNdamp, _covinv_eigVdamp = np.linalg.eigh(covinvdamp)
                    covinv_eigVdamp = []
                    for j in range(self.const.dim):
                        covinv_eigVdamp.append(_covinv_eigVdamp[:,j])
                    covinv_eigVdamp = np.array(covinv_eigVdamp)
                    for k, eigVdamp in enumerate(covinv_eigVdamp):
                        minimumangle = 1.0e30
                        for l, eigV in enumerate(self.covinv_eigV):
                            eigangle = self.angle(eigVdamp, eigV)
                            if pi * 0.5 < eigangle:
                                eigangle = pi - eigangle
                            #print(eigangle)
                            if eigangle < minimumangle:
                                minimumN     = copy.copy(l)
                                minimumangle = copy.copy(eigangle)
                        #print("minimumangle = %s"%minimumangle)
                        minimumanglelist.append(minimumangle)
                        #self.sigmasq_traj[minimumN].append(1.0 / covinv_eigNdamp[k] / covinv_eigNdamp[k] )
                        self.sigmasq_traj[minimumN].append(1.0 / covinv_eigNdamp[k])
                        eigNdamp = np.array([ 1.0 / x for x in covinv_eigNdamp])
                        #sigma_r = UIstepCython.calcsigma(
                                #avedamp, self.covinv_eigV[minimumN], 1.0, 
                                #eigNdamp, covinv_eigVdamp,
                                #self.const.periodicmax, self.const.periodicmin, self.const.dim)
                        #print(sigma_r)
                        #P_traj[minimumN] = copy.copy(eigVdamp)
    #                P_traj = np.array(P_traj)
                    #eigNdamp = np.array([ 1.0 / x for x in covinv_eigNdamp])
                    #for k, eigV in enumerate(self.covinv_eigV):
                        ##print(k)
                        #sigma_r = UIstepCython.calcsigma(avedamp, eigV, 1.0, 
                                #eigNdamp, covinv_eigVdamp,
                                #self.const.periodicmax, self.const.periodicmin,self.const.dim)
                        #print(sigma_r)
                        #self.sigmasq_traj[k].append(1.0 / sigma_r / sigma_r)
                        #print(self.sigmasq_traj)
                    #Pavedamp = np.dot(P_traj, avedamp)
                    #print("%s  vs %s"%(Pavedamp, np.dot(P, avedamp)))
                    #for k, eigN in enumerate(self.covinv_eigN):
                        #minimumeigN = 1.0e30
                        #for eigNdamp in covinv_eigNdamp:
                            #if eigNdamp < minimumeigN:
                                #minimumeigN = copy.copy(eigNdamp)
                        #self.sigmasq_traj[k].append(1.0 / minimumeigN)
    
                    Pavedamp = np.dot(Pinv, avedamp)
                    for j in range(self.const.dim):
                        self.ave_traj[j].append(Pavedamp[j])
                    i += 1
                minimumangleMax  = max(minimumanglelist)
                #print(self.sigmasq_traj)
                self.ave_traj     = [np.var(np.array(li), ddof=1) 
                                      for li in self.ave_traj]
                self.sigmasq_traj = [np.var(np.array(li), ddof=1)
                                      for li in self.sigmasq_traj]
            #print(self.ave_traj)
            #print(self.sigmasq_traj)
        #print("minimumangleMax(%s, %s) = %s"%(segN, i, minimumangleMax))
        if self.const.nextstepconstQ is False:
            #self.nlist       = copy.copy(importedPolarPY)
            pass

        if whamQ:
            self.calchisto()
        if rmdataQ:
            del self.data, ldamp
            gc.collect()
        return True
    def readCOORD(self, filename):
        ldamp = [] # data which do not append periodic (for UI analysis)
        self.const.dimdamp = copy.copy(self.const.dim)
        try:
            refdamp = self.ref
        except AttributeError:
            pass
        fieldnames = []
        t = False
        for line in open(filename):
            if len(line.split()) == 0:
                continue
            if "FIELDS" in line:
                fieldnames = line.split()[3:]
                self.const.dimdamp = len(fieldnames)
                #print("self.const.dimdamp = %s"%self.const.dimdamp)
                if self.const.periodicQ:
                    try:
                        refdamp = self.ref
                    except AttributeError:
                        refdamp = np.zeros(self.const.dimdamp)
                        colname = filename.split("/")[-1]
                        for line in open(filename.replace(colname, "plumed.dat")):
                            if not "RESTRAINT" in line:
                                continue
                            includeQ = False
                            for con in line.split():
                                con = con.replace("\n", "")
                                if "ARG" in con:
                                    refname = con.replace("ARG=", "")
                                    if refname in fieldnames:
                                        refindex = fieldnames.index(refname)
                                        includeQ = True
                                elif "AT" in con:
                                    refpoint = float(con.replace("AT=", ""))
                            if includeQ:
                                refdamp[refindex] = copy.copy(refpoint)
                    #print("refdamp = %s"%refdamp)
                continue
            elif "#" in line.split()[0]:
                continue
            try:
                line   = line.split()
                t      = line[0]
                _point = line[1:self.const.dimdamp + 1]
            except:
                c = inspect.currentframe()
                print(
"""ERROR(%s): The calculation of %s is not finished.
***EXIT***"""%(c.f_lineno,filename))
                return False, False, False
            if len(_point) < self.const.dimdamp:
                c = inspect.currentframe()
                print(
"""ERROR(%s): The calculation of %s is not finished.
***EXIT***"""%(c.f_lineno,filename))
                return False, False, False
            try:
                _point = np.array(_point, dtype=float)
            except ValueError:
                with open("./error.txt", "a") as wf:
                    wf.write("ERROR in umbrellaINT.readCOORD: cannot read %s\n"%filename)
                return False, False, False
            if self.const.periodicQ:
                _point = UIstepCython.periodic(_point, 
                    self.const.periodicmax, self.const.periodicmin, refdamp, self.const.dimdamp)
            ldamp.append(_point)
        if t is False:
            with open("./error.txt", "a") as wf:
                wf.write("ERROR in umbrellaINT.readCOORD: cannot read %s\n"%filename)
            return False, False, False
        t = t.split(".")[0]
        return fieldnames, ldamp, t
    def calchisto(self):
        histomin = np.array([ 1.0e30 for _ in range(self.const.dim)])
        histomax = np.array([-1.0e30 for _ in range(self.const.dim)])
        for point in self.data:
            for i in range(self.const.dim):
                if point[i] < histomin[i]:
                    histomin[i] = copy.copy(point[i])
                elif histomax[i] < point[i]:
                    histomax[i] = copy.copy(point[i])
        #self.histodelta = (histomax - histomin) / Nbin
        self.histodelta = (histomax - histomin)
# This code uses Scott's choice to calculate the number of bins
# Scott, David W., and Stephan R. Sain. "9-Multiself.const.dimensional Density Estimation." Handbook of statistics 24 (2005): 229-261.
        Nbinlist = []
        for i in range(self.const.dim):
            sigma = np.sqrt(self.cov[i,i])
            #binL  = 3.5 * sigma / np.power(self.N, 1.0 / 3.0 / self.const.dim)
            binL  = 3.5 * sigma / np.power(self.N / 0.001, 1.0 / (4.0 + self.const.dim))
            Nbin  = int(self.histodelta[i] / binL)
            Nbinlist.append(Nbin)
            self.histodelta[i] /= Nbin
        print("Nbinlist = %s"%Nbinlist)
        #for i in range(self.const.dim):
            #self.histodelta[i] /= Nbin[i]
        self.histo = UIstepCython.histoC(
                self.data, histomin, self.histodelta, Nbinlist, self.const.dim)
        self.histo = np.array(self.histo)
    def findhispoint(self, xi):
        try:
            self.histo
        except:
            self.calchisto()
        if self.const.periodicQ:
            xidamp = UIstepCython.periodic(xi, 
                    self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
        else:
            xidamp = xi
        for histolist in self.histo:
            if all(histolist[i] <= xidamp[i] < histolist[i]  + self.histodelta[i] for i in range(self.const.dim)):
                #return histolist[-1]
                break
        else:
            return 0
        histopointpars = []
        for i in range(self.const.dim):
            if histolist[i] + self.histodelta[i] * 0.5 < xidamp[i]:
                histopointpars.append(
                        [histolist[i], histolist[i] + self.histodelta[i]])
            else:
                histopointpars.append(
                        [histolist[i], histolist[i] - self.histodelta[i]])
        histolistlist = []
        for prod in itertools.product(range(2),repeat=self.const.dim):
            xiprod = []
            for i in range(self.const.dim):
                xiprod.append(histopointpars[i][prod[i]])
            for histolist in self.histo:
                if all(histolist[i] == xiprod[i] for i in range(self.const.dim)):
                    histolistlist.append(histolist)
                    break
            else:
                histolistlist.append(xiprod + [0])
        returnH = LShisto(histolistlist, self.histodelta, xidamp)
        if returnH < 0:
            return 0.0
        else:
            return returnH
    def A_WHAM(self, xi):
        return UIstepCython.A_WHAM(xi, self.ref, self.ave, self.K,
                self.histo, self.histodelta, 
                self.const.periodicmax, self.const.periodicmin, self.const.betainv, self.const.dim, self.const.periodicQ)
#        avehisto = self.findhispoint(self.ave)
#        if self.const.periodicQ:
#            xidamp = UIstepCython.periodic(xi,
#                    self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
#        else:
#            xidamp = xi
#        xihisto  = self.findhispoint(xidamp)
#        if xihisto == 0:
#            return 1.0e30
#        vec_ave = self.ref - self.ave
#        vec_xi  = self.ref - xidamp
#        w_ave   = 0.0
#        w_xi    = 0.0
#        for i in range(self.const.dim):
#            for j in range(self.const.dim):
#                w_ave += vec_ave[i] * self.K[i,j] * vec_ave[j]
#                w_xi  += vec_xi[i]  * self.K[i,j] * vec_xi[j]
#        return -betainv * np.log(xihisto / avehisto) - 0.5 * (w_xi - w_ave)
    def exportdata(self, path, whamQ = False, equibliumstructure = False):
        """
        to export UI data as .npy file
        .npy is numpy data
        """
        exportnpzQ = True
        if self.const.WindowDataType == "hdf5":
            if "windows" in path:
                UIlistchunkmin = 1
                UIlistchunkmax = copy.copy(self.const.UIlistchunksize)
                while True:
                    if UIlistchunkmin <= self.stepN <= UIlistchunkmax:
                        break
                    UIlistchunkmin += self.const.UIlistchunksize
                    UIlistchunkmax += self.const.UIlistchunksize
                UIlistHDFpath = "%s/%s/windows/windowfile%s-%s.hdf5"%(self.const.pwdpath, self.const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
                lock = fasteners.InterProcessLock(self.const.lockfilepath_UIlist + str(UIlistchunkmax))
                exportnpzQ = False
                steppath = path.split("/")[-1]
                lock.acquire()
                #windowHDF = h5py.File("%s/jobfiles/windows/windowfile.hdf5"%pwdpath, "a")
                windowHDF = h5py.File(UIlistHDFpath, "a")
                if not steppath in windowHDF:
                    windowHDF.create_group(steppath)
                if "%s/ave"%steppath in windowHDF:
                    print("ERROR in hdf5 at exportdata")
                    print("UIlistHDFpath = %s"%UIlistHDFpath)
                    print("stepN = %s"%self.stepN)
                    lock.release()
                    #exit()
                    return False
                    windowHDF[      "%s/stepN"%steppath][...] = self.stepN
                    windowHDF[          "%s/N"%steppath][...] = self.N
                    windowHDF[          "%s/K"%steppath][...] = self.K
                    windowHDF[          "%s/A"%steppath][...] = self.A
                    windowHDF[        "%s/ref"%steppath][...] = self.ref
                    windowHDF[        "%s/ave"%steppath][...] = self.ave
                    windowHDF[       "%s/aveT"%steppath][...] = self.aveT
                    windowHDF["%s/aveTinitial"%steppath][...] = self.aveTinitial
                    windowHDF[        "%s/cov"%steppath][...] = self.cov
                    windowHDF[     "%s/covinv"%steppath][...] = self.covinv
                    windowHDF[        "%s/hes"%steppath][...] = self.hes
                    windowHDF[     "%s/hesinv"%steppath][...] = self.hesinv
                    windowHDF[       "%s/eigN"%steppath][...] = self.eigN
                    windowHDF[       "%s/eigV"%steppath][...] = self.eigV
                    windowHDF[   "%s/cov_eigN"%steppath][...] = self.cov_eigN
                    windowHDF[   "%s/cov_eigV"%steppath][...] = self.cov_eigV
                    windowHDF["%s/covinv_eigN"%steppath][...] = self.covinv_eigN
                    windowHDF["%s/covinv_eigV"%steppath][...] = self.covinv_eigV
                    windowHDF[   "%s/Pbiasave"%steppath][...] = self.Pbiasave
                    windowHDF["%s/connections"%steppath][...] = self.connections
                    windowHDF["%s/path"%steppath][...]        = path
                    windowHDF["%s/calculationpath"%steppath][...] = self.calculationpath
                    if not equibliumstructure is False:
                        windowHDF["%s/equibliumstructure"%steppath][...] = equibliumstructure
                else:
                    windowHDF.create_dataset(      "%s/stepN"%steppath, data = self.stepN)
                    windowHDF.create_dataset(          "%s/N"%steppath, data = self.N)
                    windowHDF.create_dataset(          "%s/K"%steppath, data = self.K)
                    windowHDF.create_dataset(          "%s/A"%steppath, data = self.A)
                    windowHDF.create_dataset(        "%s/ref"%steppath, data = self.ref)
                    windowHDF.create_dataset(       "%s/aveT"%steppath, data = self.aveT)
                    windowHDF.create_dataset("%s/aveTinitial"%steppath, data = self.aveTinitial)
                    windowHDF.create_dataset(        "%s/ave"%steppath, data = self.ave)
                    windowHDF.create_dataset(        "%s/cov"%steppath, data = self.cov)
                    windowHDF.create_dataset(     "%s/covinv"%steppath, data = self.covinv)
                    windowHDF.create_dataset(        "%s/hes"%steppath, data = self.hes)
                    windowHDF.create_dataset(     "%s/hesinv"%steppath, data = self.hesinv)
                    windowHDF.create_dataset(       "%s/eigN"%steppath, data = self.eigN)
                    windowHDF.create_dataset(       "%s/eigV"%steppath, data = self.eigV)
                    windowHDF.create_dataset(   "%s/cov_eigN"%steppath, data = self.cov_eigN)
                    windowHDF.create_dataset(   "%s/cov_eigV"%steppath, data = self.cov_eigV)
                    windowHDF.create_dataset("%s/covinv_eigN"%steppath, data = self.covinv_eigN)
                    windowHDF.create_dataset("%s/covinv_eigV"%steppath, data = self.covinv_eigV)
                    windowHDF.create_dataset(   "%s/Pbiasave"%steppath, data = self.Pbiasave)
                    windowHDF.create_dataset("%s/connections"%steppath, data = self.connections)
                    windowHDF.create_dataset("%s/path"%steppath, data = path)
                    windowHDF.create_dataset("%s/calculationpath"%steppath, data = self.calculationpath)
                if not equibliumstructure is False:
                    windowHDF.create_dataset("%s/equibliumstructure"%steppath, data = equibliumstructure)
                windowHDF.flush()
                windowHDF.close()
                lock.release()

        if exportnpzQ:
            np.savez("%s/UIstep.npz"%path,
                stepN        = self.stepN,
                N            = self.N,
                K            = self.K,
                A            = self.A,
                ref          = self.ref,
                aveT         = self.aveT,
                aveTinitial  = self.aveTinitial,
                ave          = self.ave,
                cov          = self.cov,
                covinv       = self.covinv,
                hes          = self.hes,
                hesinv       = self.hesinv,
                eigN         = self.eigN,
                eigV         = self.eigV,
                cov_eigN     = self.cov_eigN,
                cov_eigV     = self.cov_eigV,
                covinv_eigN  = self.covinv_eigN,
                covinv_eigV  = self.covinv_eigV,
                Pbiasave     = self.Pbiasave,
                connections    = self.connections,
                )
            if whamQ:
                np.save("%s/histo.npy"%path, self.histo)
                np.save("%s/histodelta.npy"%path, self.histodelta)
            self.calculationpath = str(self.calculationpath)
            open("%s/calculationpath.dat"%path, "w").write(self.calculationpath)
            self.path = copy.copy(path)
            if self.const.nextstepconstQ is False:
                #np.save("%s/nlist.npy"%path, self.nlist)
                pass
        return True
    def exportdata_ALL(self, path, whamQ = False):
        exportnpzQ = True
        #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
        if self.const.WindowDataType == "hdf5":
            if "windows" in path:
                UIlistchunkmin = 1
                UIlistchunkmax = self.const.UIlistchunksize
                while True:
                    if UIlistchunkmin <= self.stepN <= UIlistchunkmax:
                        break
                    UIlistchunkmin += self.const.UIlistchunksize
                    UIlistchunkmax += self.const.UIlistchunksize
                UIlistHDFpath = "%s/%s/windows/windowfile%s-%s.hdf5"%(self.const.pwdpath, self.const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
                lock = fasteners.InterProcessLock(lockfilepath_UIlist + str(UIlistchunkmax))
                exportnpzQ = False
                steppath = path.split("/")[-1]
                lock.acquire()
                windowHDF = h5py.File(UIlistHDFpath, "a")
                if not steppath in windowHDF:
                    windowHDF.create_group(steppath)
                #lock.acquire()
                #exportnpzQ = False
                #steppath = path.split("/")[-1]
                #with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "a") as windowHDF:
                if True:
                    #windowHDF = h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "a", libver='latest')
                    #windowHDF = h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "a")
                    if not steppath in windowHDF:
                        windowHDF.create_group(steppath)
                    if "%s/ave"%steppath in windowHDF:
                        print("ERROR in hdf5 at exportdata_ALL")
                        lock.release()
                        exit()
                        windowHDF[        "%s/refALL"%steppath][...] = self.refALL
                        windowHDF[        "%s/aveALL"%steppath][...] = self.aveALL
                        windowHDF[          "%s/KALL"%steppath][...] = self.KALL
                        windowHDF[        "%s/covALL"%steppath][...] = self.covALL
                        windowHDF[     "%s/covinvALL"%steppath][...] = self.covinvALL
                        windowHDF[        "%s/hesALL"%steppath][...] = self.hesALL
                        windowHDF[     "%s/hesinvALL"%steppath][...] = self.hesinvALL
                    else:
                        windowHDF.create_dataset(        "%s/refALL"%steppath, data = self.refALL)
                        windowHDF.create_dataset(        "%s/aveALL"%steppath, data = self.aveALL)
                        windowHDF.create_dataset(          "%s/KALL"%steppath, data = self.KALL)
                        windowHDF.create_dataset(        "%s/covALL"%steppath, data = self.covALL)
                        windowHDF.create_dataset(     "%s/covinvALL"%steppath, data = self.covinvALL)
                        windowHDF.create_dataset(        "%s/hesALL"%steppath, data = self.hesALL)
                        windowHDF.create_dataset(     "%s/hesinvALL"%steppath, data = self.hesinvALL)
                    windowHDF.flush()
                    windowHDF.close()
                lock.release()
        if exportnpzQ:
            np.savez("%s/UIstep_ALL.npz"%path,
                refALL       = self.refALL,
                aveALL       = self.aveALL,
                covALL       = self.covALL,
                KALL         = self.KALL,
                covinvALL    = self.covinvALL,
                hesALL       = self.hesALL,
                hesinvALL    = self.hesinvALL,
                )
    #def importdata(self, path):
    def importdata(self, path, windowHDF = False):
        """
        to import UI data as .npy file
        .npy is numpy data
        """
        importnpzQ = True
        if self.const.WindowDataType == "hdf5":
            if "windows" in path:
                importnpzQ = False
                steppath = path.split("/")[-1]
                stepN    = int(steppath.replace("step",""))
                #print(steppath)
                if windowHDF is False:
                    UIlistchunkmin = 1
                    UIlistchunkmax = copy.copy(UIlistchunksize)
                    while True:
                        if UIlistchunkmin <= stepN <= UIlistchunkmax:
                            break
                        UIlistchunkmin += UIlistchunksize
                        UIlistchunkmax += UIlistchunksize
                    UIlistHDFpath = "%s/jobfiles/windows/windowfile%s-%s.hdf5"%(pwdpath, UIlistchunkmin, UIlistchunkmax)
                    UIlistchunkmin += UIlistchunksize
                    UIlistchunkmax += UIlistchunksize
                    UIlistHDFpath_next = "%s/jobfiles/windows/windowfile%s-%s.hdf5"%(pwdpath, UIlistchunkmin, UIlistchunkmax)
                    if not os.path.exists(UIlistHDFpath_next):
                        lock = fasteners.InterProcessLock(lockfilepath_UIlist + str(UIlistchunkmax - UIlistchunksize))
                        lock.acquire()
                    windowHDF = h5py.File(UIlistHDFpath, "r")
                self.stepN          = windowHDF[         "%s/stepN"%steppath][...]
                self.stepN          = int(self.stepN)
                self.N              = windowHDF[             "%s/N"%steppath][...]
                self.K              = windowHDF[             "%s/K"%steppath][...]
                self.A              = windowHDF[             "%s/A"%steppath][...]
                self.ref            = windowHDF[           "%s/ref"%steppath][...]
                self.ave            = windowHDF[           "%s/ave"%steppath][...]
                self.aveT           = windowHDF[          "%s/aveT"%steppath][...]
                self.calculationpath = windowHDF["%s/calculationpath"%steppath][...]
                self.calculationpath = str(self.calculationpath)
                try:
                    self.aveTinitial = windowHDF["%s/aveTinitial"%steppath][...]
                except:
                    self.aveTinitial = copy.copy(self.aveT)
                self.ave         = windowHDF[        "%s/ave"%steppath][...]
                self.cov         = windowHDF[        "%s/cov"%steppath][...]
                self.covinv      = windowHDF[     "%s/covinv"%steppath][...]
                self.hes         = windowHDF[        "%s/hes"%steppath][...]
                self.hesinv      = windowHDF[     "%s/hesinv"%steppath][...]
                self.eigN        = windowHDF[       "%s/eigN"%steppath][...]
                self.eigV        = windowHDF[       "%s/eigV"%steppath][...]
                self.cov_eigN    = windowHDF[   "%s/cov_eigN"%steppath][...]
                self.cov_eigV    = windowHDF[   "%s/cov_eigV"%steppath][...]
                self.covinv_eigN = windowHDF["%s/covinv_eigN"%steppath][...]
                self.covinv_eigV = windowHDF["%s/covinv_eigV"%steppath][...]
                self.Pbiasave    = windowHDF[   "%s/Pbiasave"%steppath][...]
                try:
                    self.connections = list(windowHDF["%s/connections"%steppath][...])
                except:
                    self.connections = [0]
                self.calculationpath    = windowHDF[   "%s/calculationpath"%steppath][...]
                self.path = copy.copy(path)
                if partADDQ:
                    if os.path.exists("%s/UIstep_ALL.npz"%path):
                        self.refALL      = windowHDF[    "%s/refALL"%steppath][...]
                        self.ref = copy.copy(self.refALL)
                        self.aveALL      = windowHDF[    "%s/aveALL"%steppath][...]
                        self.ave = copy.copy(self.aveALL)
                        self.KALL        = windowHDF[      "%s/KALL"%steppath][...]
                        self.K = copy.copy(self.KALL)
                        self.covALL      = windowHDF[    "%s/covALL"%steppath][...]
                        self.cov = copy.copy(self.covALL)
                        self.covinvALL   = windowHDF[ "%s/covinvALL"%steppath][...]
                        self.covinv = copy.copy(self.covinvALL)
                        self.hesALL      = windowHDF[    "%s/hesALL"%steppath][...]
                        self.hes = copy.copy(self.hesALL)
                        self.hesinvALL   = windowHDF[ "%s/hesinvALL"%steppath][...]
                        self.hesinv = copy.copy(self.hesinvALL)
                windowHDF.close()
                if not os.path.exists(UIlistHDFpath_next):
                    lock.release()

        if importnpzQ:
            importnpz        = np.load("%s/UIstep.npz"%path)
            self.stepN       = importnpz[     "stepN"]
            self.N           = importnpz[         "N"]
            self.K           = importnpz[         "K"]
            self.A           = importnpz[         "A"]
            self.ref         = importnpz[       "ref"]
            self.aveT        = importnpz[      "aveT"]
            try:
                self.aveTinitial = importnpz["aveTinitial"]
            except:
                self.aveTinitial = copy.copy(self.aveT)
            self.ave         = importnpz[       "ave"]
            self.cov         = importnpz[       "cov"]
            self.covinv      = importnpz[    "covinv"]
            self.hes         = importnpz[       "hes"]
            self.hesinv      = importnpz[    "hesinv"]
            self.eigN        = importnpz[      "eigN"]
            self.eigV        = importnpz[      "eigV"]
            self.cov_eigN    = importnpz[  "cov_eigN"]
            self.cov_eigV    = importnpz[  "cov_eigV"]
            self.covinv_eigN = importnpz[  "covinv_eigN"]
            self.covinv_eigV = importnpz[  "covinv_eigV"]
            self.Pbiasave    = importnpz[  "Pbiasave"]
            try:
                self.connections = list(importnpz[  "connections"])
            except:
                self.connections = [0]
            #self.UInetwork   = importnpz[  "UInetwork"]
            #self.nearUIlist = importnpz["nearUIlist"]
            #self.ave_traj   = importnpz[  "ave_traj"]
            #self.sigmasq_traj   = importnpz[  "sigmasq_traj"]
            self.stepN      = int(self.stepN)
            #self.nearUIlist = list(self.nearUIlist)
            #self.ave_traj   = list(self.ave_traj)
            #self.sigmasq_traj   = list(self.sigmasq_traj)
            self.calculationpath = open("%s/calculationpath.dat"%path).read()
            #if os.path.exists("%s/histo.npy"%path):
                #self.histo      = np.load(     "%s/histo.npy"%path)
                #self.histodelta = np.load("%s/histodelta.npy"%path)
            self.path = copy.copy(path)
            if nextstepconstQ is False:
                #self.nlist = copy.copy(importedPolarPY)
                #self.nlist  = np.load(       "%s/nlist.npy"%path)
                pass
            if partADDQ:
                if os.path.exists("%s/UIstep_ALL.npz"%path):
                    importnpz        = np.load("%s/UIstep_ALL.npz"%path)
                    self.refALL      = importnpz[    "refALL"]
                    self.ref         = copy.copy(self.refALL)
                    self.aveALL      = importnpz[    "aveALL"]
                    self.ave         = copy.copy(self.aveALL)
                    self.KALL        = importnpz[      "KALL"]
                    self.K           = copy.copy(self.KALL)
                    self.covALL      = importnpz[    "covALL"]
                    self.cov         = copy.copy(self.covALL)
                    self.covinvALL   = importnpz[ "covinvALL"]
                    self.covinv      = copy.copy(self.covinvALL)
                    self.hesALL      = importnpz[    "hesALL"]
                    self.hes         = copy.copy(self.hesALL)
                    self.hesinvALL   = importnpz[ "hesinvALL"]
                    self.hesinv      = copy.copy(self.hesinvALL)
    def importdata_aveonly(self, path, windowHDF):
        importnpzQ = True
        if self.const.WindowDataType == "hdf5":
            #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
            #print(path)
            #print("windows" in path)
            if "windows" in path:
                importnpzQ = False
                steppath = path.split("/")[-1]
                #lock.acquire()
                #with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "r", libver='latest', swmr = True) as windowHDF:
                #with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "r") as windowHDF:
                if True:
                    self.stepN          = windowHDF[         "%s/stepN"%steppath][...]
                    self.stepN          = int(self.stepN)
                    self.ave            = windowHDF[           "%s/ave"%steppath][...]
                    try:
                        self.aveTinitial = windowHDF["%s/aveTinitial"%steppath][...]
                    except:
                        self.aveTinitial = copy.copy(self.aveT)
                    self.covinv      = windowHDF[     "%s/covinv"%steppath][...]
                    self.cov_eigN    = windowHDF[   "%s/cov_eigN"%steppath][...]
                    self.cov_eigV    = windowHDF[   "%s/cov_eigV"%steppath][...]
                    try:
                        self.connections = list(windowHDF["%s/connections"%steppath][...])
                    except:
                        self.connections = [0]
                #lock.release()

        if importnpzQ:
            importnpz       = np.load("%s/UIstep.npz"%path)
            self.stepN      =      int(importnpz[     "stepN"])
            self.ave        = np.array(importnpz[       "ave"])
            try:
                self.aveTinitial = np.array(importnpz["aveTinitial"])
            except:
                self.aveTinitial = copy.copy(self.ave)
            self.covinv     = np.array(importnpz[    "covinv"])
            self.cov_eigN   = np.array(importnpz[  "cov_eigN"])
            self.cov_eigV   = np.array(importnpz[  "cov_eigV"])
            try:
                self.connections = list(importnpz[  "connections"])
            except:
                self.connections = [0]
        self.path = copy.copy(path)
    def grad(self, xi):
        """
        gradient on xi
        """
        if self.const.periodicQ:
            xidamp = UIstepCython.periodic(xi, self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
        else:
            xidamp = xi
        grad = UIstepCython.cgrad(self.covinv, xidamp, self.ave, self.K, self.ref, self.const.dim, self.const.betainv )
        return grad
    def grad_periodic(self, xi):
        """
        gradient on xi
        """
        grad = UIstepCython.cgrad(self.covinv, xi, self.ave_periodic, self.K, self.ref_periodic, self.const.dim, self.const.betainv )
        return grad
    def gradsqrt(self,xi):
        """
        gradient ** 2
        """
        graddamp = self.grad(xi)
        gradsqrt = 0.0
        for x in range(self.const.dim):
            gradsqrt += graddamp[x] * graddamp[x]
        return gradsqrt
    def grad_on_vec(self, t, initialp, Evec):
        return UIstepCython.cgrad_on_vec(self.covinv, initialp + t * Evec,
                   self.ave, self.K, self.ref, Evec, self.const.dim, self.const.betainv)
    #def NRnextstep(self, UIlist):
    def NRnextstep(self, initialpoint, UIlist):
        """
        next step whth Newton-Raphson method
        """
        #nextstep   = copy.copy(self.ave)
        nextstep   = copy.copy(initialpoint)
        #graddamp   = gradUIall(UIlist, self.ave)
        graddamp   = gradUIall(UIlist, initialpoint)
        if self.const.partOPTQ:
            sortedgrad = sorted([graddamp[i] for i in range(self.const.dim)])[:self.const.partdim]
        hesdamp    = HessianUIall(UIlist, initialpoint)
        hesinvdamp = np.linalg.inv(hesdamp)
        for x in range(self.const.dim):
            if self.const.partOPTQ:
                if not graddamp[x] in sortedgrad:
                    continue
            for y in range(self.const.dim):
                if self.const.partOPTQ:
                    if not graddamp[y] in sortedgrad:
                        continue
                nextstep[x] -= hesinvdamp[x,y] * graddamp[y]
                #nextstep[x] -= self.hesinv[x,y] * graddamp[y]
        #functions.periodicpoint(nextstep, {})
        return nextstep
    def gradvector(self, xi, r):
        """
        gradient on the point(xi) along the vector(r)
        """
        return np.dot(self.grad(xi), r)
    def minimumgrad(self):
        gradlist    = []
        minimumlist = []
        #for n in importedPolarPY:
        for thetalist in importedPolarPY:
            n = functions.SperSphere_cartesian(self, 1.0, thetalist, self.const.dim)
            n = n / np.linalg.norm(n)
            gradlist.append((self.gradvector(self.ave, n), n))
        gmin = min(x[0] for x in gradlist)
        for grad, n in gradlist:
            if grad == gmin:
                break
        return n
    def gradonave(self, r):
        """
        gradient on the self.ave along the vector(r)
        """
        return np.dot(self.grad(self.ave), r)
    def deltaA_UI(self, initialp, finalp):
        """
        deltaA_UI
        free energy difference from InitialPoint to FinalPoint
        with Umberlla Integration
        initialp == (vec) ==> finalp
                   intdamp
        """
        if len(initialp) != self.ADDself.const.dim:
            initialpdamp = []
            for i, x in enumerate(initialp):
                if i in self.partlist:
                    initialpdamp.append(x)
            initialpdamp = np.array(initialpdamp)
        else:
            initialpdamp = initialp
        if len(finalp) != self.ADDself.const.dim:
            finalpdamp = []
            for i, x in enumerate(finalp):
                if i in self.partlist:
                    finalpdamp.append(x)
            finalpdamp = np.array(finalpdamp)
        else:
            finalpdamp   = finalp
        intdamp = UIstepCython.deltaA_UI(self.covinv, self.K, self.ave, self.ref,
                initialpdamp, finalpdamp, self.const.betainv, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin)
        return intdamp
    def deltaA_UImax(self, UI):
        if parallelPy != 0:
            p = mp.Pool(parallelPy)
            A_UIlist = p.map(self.parallelUI, 
                    [(UI, n) for n in importedPolarPY])
            p.close()
        else:
            A_UIlist = map(self.parallelUI, 
                    [(UI, n) for n in importedPolarPY])
        #for n in importedPolarPY:
        A_UImax = - 1.0e30
        for A_UI in A_UIlist:
            if A_UImax < A_UI:
                A_UImax = copy.copy(A_UI)
        return A_UImax
    def deltaA_UImin(self, UI):
        if parallelPy != 0:
            p = mp.Pool(parallelPy)
            A_UIlist = p.map(self.parallelUI, 
                    [(UI, n) for n in importedPolarPY])
            p.close()
        else:
            A_UIlist = map(self.parallelUI, 
                    [(UI, n) for n in importedPolarPY])
        #for n in importedPolarPY:
        A_UImin = 1.0e30
        for A_UI in A_UIlist:
            if A_UI < A_UImin:
                A_UImin = copy.copy(A_UI)
        return A_UImin
    def parallelUI(self, parallellist):
        UI, n = parallellist
        n      = np.array(n)
        d      = self.calcsigma(n, nextstepsigmamaxTH)
        _point = UI.ave + d * n
        if self.const.periodicQ:
            _point = UIstepCython.periodic(_point,
                self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
        A_UI = self.deltaA_UI(self.ave, _point)
        return A_UI
    def Pbias(self, xi):
        """
        distribution on xi
        """
        if self.const.periodicQ:
            return UIstepCython.cPbias_periodic(
                    xi, self, self.const.periodicmax, self.const.periodicmin, self.const.dim)
        else:
            return UIstepCython.cPbias(xi, self.ave, self.covinv, self.Pbiasave, self.const.dim)
    def delPbias(self, xi):
        if self.const.periodicQ:
            avedamp = UIstepCython.periodic(self.ave,
                        self.const.periodicmax, self.const.periodicmin, xi, self.const.dim)
        else:
            avedamp = self.ave
        return - np.dot(self.covinv, (xi - avedamp)) * self.Pbias(xi)
    def Dbias(self, xi):
        #if self.const.periodicQ:
            #D = UIstepCython.Dbias_periodic(xi, self.const.periodicmax, self.const.periodicmin,
                    #self.ave, self.covinv, self.const.dim)
        #else:
            #D = UIstepCython.Dbias(xi, self.ave, self.cov_eigN, self.cov_eigV, self.const.dim)
        if self.const.nextstepconstQ and len(xi) == self.const.partdim:
            D = UIstepCython.Dbias(xi, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin,
                self.ave, self.covinv, self.const.dim, self.const.partOPTQ, self.const.partdim)
        elif len(xi) == self.const.dim:
            if len(xi) != len(self.ave):
                D = UIstepCython.Dbias(xi, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin,
                self.aveAll, self.covinvAll, self.const.dim, self.const.partOPTQ, self.const.partdim)
                #D = UIstepCython.Dbias(xi, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin,
                    #self.aveALL, self.covinvALL, self.const.dim, partOPTQ, self.const.dim)
            else:
                D = UIstepCython.Dbias(xi, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin,
                    self.ave, self.covinv, self.const.dim, self.const.partOPTQ, self.const.partdim)
                #D = UIstepCython.Dbias(xi, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin,
                    #self.ave, self.covinv, self.const.dim, partOPTQ, partdim)
        return D
    def Dbiasave_UIlistmin(self, UIlist):
        Dbiasmin = 1.0e30
        for UI in UIlist:
            if self.Dbias(UI.ave) < Dbiasmin:
                Dbiasmin = copy.copy(self.Dbias(UI.ave))
        return Dbiasmin
    def Dbias_delta(self, nextref, Deltatarget, _nextK, targetpointdamp):
        _delta1   = UIstepCython.cdelta1(self.ave, self.K, self.ref, self.covinv, 
                              nextref, Deltatarget, self.const.dim, self.const.betainv)
        deltadamp = UIstepCython.cdelta(_delta1, self.K, _nextK, self.covinv, 
                              targetpointdamp, nextref, self.const.dim, self.const.betainv)
        _point = self.ave + deltadamp
        _point = functions.periodicpoint(_point, self.const)
        return self.Dbias(_point)
    def calcnextstep(self, UIlist, UIeq, ADDlist, nextstepD, IOEsphereAmax, rank, size, comm):
        """
        next step calculation in this algorithm
        vec         : the vector to the next step
        UIbefore    : the result of umbrella integration in the before point
        """
        if nextstepconstQ:
            nlist = itertools.product([-1,0,1], repeat=self.const.dim)
            parallellists = (
                    (UIlist, UIeq, n, ADDlist, nextstepD, IOEsphereAmax)
                    for n in nlist if sum(n) != 0)
            nextstepsP = map(self.parallelcalcnextstep, parallellists)
        else:
            #print("len(self.nlist)[%s] = %s"%(self.stepN, len(self.nlist)))
            nextstepsP = []
            endnums    = []
            nlistdamp = copy.copy(self.nlist)
            nlist = []
            nearUIlistdamp = []
            nextsteplist = []
            #if parallelPy == 0:
            if True:
                #for n in  nlistdamp:
                nextsteplist = self.parallelcalcnextstep((UIlist, UIeq, nlistdamp, ADDlist, nextstepD, IOEsphereAmax, rank, size, comm))
            else:
                nlistchunks = [[] for _ in range(parallelPy)]
                for i, n in enumerate(nlistdamp):
                    nlistchunks[i % parallelPy].append(n)
                parallellists = [(UIlist, UIeq, nlistchunk, ADDlist, nextstepD, sameAN)
                        for nlistchunk in nlistchunks]
                for P in nextsteplistP:
                    nextsteplist.extend(P)
            if rank == root: 
                for nsQ, nextstepdamp, nearestUIlistdamp, n in nextsteplist:
                    if nsQ:
                        nextstepsP.append([nsQ, nextstepdamp])
                        nlist.append(n)
                        for nearUINnew in nearestUIlistdamp:
                            for nearUINold in nearUIlistdamp:
                                if nearUINnew == nearUINold:
                                    break
                            else:
                                nearUIlistdamp.append(nearUINnew)
                #print(self.nearUIlist)
                #self.nearUIlist = copy.copy(nearUIlistdamp)
                #print(self.nearUIlist)
                self.nlist = copy.copy(nlist)
                self.exportdata(self.path)

        if rank == root:
            nextsteps = []
            nextstepQ = False
            for nsQ, nextstepdamp in nextstepsP:
                if nsQ is True:
                    nextstepQ = True
                    if nextstepdamp:
                        nextsteps.extend(nextstepdamp)
                if len(self.nearUIlist) == 0:
                    break
            if not nextsteps:
                return False, False
            #deltaA_minimum = 1.0e30
            nextstep = []
            for nextstepFOR, stepN, _Dmin in nextsteps:
                #if deltaA_UIeq < deltaA_minimum:
                    #deltaA_minimum = deltaA_UIeq
                    #nextstep       = [[nextstepFOR, deltaA_UIeq, stepN]]
                nextstep.append([nextstepFOR, stepN, _Dmin])
            #for nextstepFOR, deltaA_UIeq, stepN in nextsteps:
                #nextstep.append([nextstepFOR, deltaA_UIeq, stepN])
    
            if len(nextstep) == 0:
                return nextstepQ, []
            else:
                return True, nextstep
        else:
            return True, []
    def parallelcalcnextstep(self, parallellist):
        UIlist, UIeq, nlist, ADDlist, nextstepD, IOEsphereAmax, rank, size,comm = parallellist
        returnlistP = []
        #for n in nlist:
            #returnlist.append(self.parallelcalcnextstep_n(UIlist, UIeq, n, ADDlist, nextstepD, sameAN))
        for i, n in enumerate(nlist):
            if i % size != rank:
                continue
            returnlistP.append(self.parallelcalcnextstep_n(UIlist, UIeq, n, ADDlist, nextstepD, IOEsphereAmax))
        if self.const.K_mpiQ:
            returnlistP = comm.gather(returnlistP, root=0)
        if rank == root:
            returnlist = []
            for returnpoint in returnlistP:
                returnlist.extend(returnpoint)
        else:
            returnlist = []
        return returnlist
    def parallelcalcnextstep_n(self, UIlist, UIeq, n, ADDlist, nextstepD, IOEsphereAmax):
        if nextstepconstQ:
            if np.linalg.norm(n) == 0.0:
                return False, [], [], n
            nextstep = self.aveT + n * nextstepD
            if self.wallQ(nextstep):
                return False, [], [], n
            if self.const.periodicQ:
                nextstepdamp = UIstepCython.periodic(nextstep,
                           self.const.periodicmax, self.const.periodicmin, UIeq.ave, self.const.dim)
            else:
                nextstepdamp = nextstep
            #deltaA_UIeq = UIeq.deltaA_UI(UIeq.ave, nextstepdamp)
            #if IOEsphereAmax < deltaA_UIeq:
                #return False, [], [], n
            if reflengthQ(UIlist, nextstep, deltaA_UIeq, nextstepD):
                return False, [], [], n
            _p = self.Dbias(nextstep)
            while _p < didpointTH:
                nextstep += n * nextstepD
                if self.wallQ(nextstep):
                    return False, [], [], n
                if self.const.periodicQ:
                    nextstepdamp = UIstepCython.periodic(nextstep, 
                           self.const.periodicmax, self.const.periodicmin, UIeq.ave, self.const.dim)
                else:
                    nextstepdamp = nextstep
                _p = self.Pbias(nextstep)
            #deltaA_UIeq   = UIeq.deltaA_UI(UIeq.ave, nextstepdamp)
            #if Dmin(UIlist, nextstep) < didpointTH:
                #return False, [], [], n
            if reflengthQ(UIlist, nextstep, deltaA_UIeq, nextstepD):
                return False, [], [], n
        else:
            d1 = self.calcsigma(n, nextstepsigmamaxTH)
            points_on_n = [self.ave + d1 * n]
            chkDminQ = False
            nearestUIlistdamp = []
            for nextstep in points_on_n:
                if self.wallQ(nextstep):
                    continue
                #self.nearUIlist = []
                #for UI in UIlist:
                    #if self.stepN != UI.stepN:
                        #self.nearUIlist.append(UI.stepN)
                if len(self.nearUIlist) == 0:
                    grad     = self.grad(nextstep)
                    _Dmin    = np.dot(grad, grad)
                    chkDminQ = True
                    break
                else:
                    #print(self.nearUIlist)
                    _Dmin, nearestUIN = self.Dmin(UIlist, self.nearUIlist, nextstep)
                    for UIN in nearestUIlistdamp:
                        if UIN == nearestUIN:
                            break
                    else:
                        nearestUIlistdamp.append(nearestUIN)
                    if _Dmin > nextstepsigmaminTH:
                        if self.chkADDc(ADDlist, nextstep):
                            chkDminQ = True
                            #break
            #else:
                #return chkDminQ, [], nearestUIlistdamp, n
#            if self.const.periodicQ:
#                nextstepdamp = UIstepCython.periodic(nextstep,
#                           self.const.periodicmax, self.const.periodicmin, UIeq.ave, self.const.dim)
#            else:
#                nextstepdamp = nextstep
#            deltaA_UIeq   = UIeq.deltaA_UI(UIeq.ave, nextstepdamp)
            if chkDminQ:
                return True, [[nextstep, self.stepN, _Dmin]], nearestUIlistdamp , n
            else:
                return chkDminQ, [], nearestUIlistdamp, n
                #return True, [[nextstep, self.stepN, _Dmin]], nearestUIlistdamp , n
    def chkADDc(self, ADDlist, nextstep):
        if ADDlist is False:
            return True
        for ADDc in ADDlist:
            if self.const.periodicQ:
                nextstepdamp = UIstepCython.periodic(nextstep,
                       self.const.periodicmax, self.const.periodicmin, ADDc.nextstep, self.const.dim)
            else:
                nextstepdamp = nextstep
            vec = nextstepdamp - ADDc.nextstep
            if nextstepconstQ:
                sigma2 = nextstepDADD * 2.0
            else:
                sigma2 = self.calcsigma(vec, edgelistsigmaTH)
            if np.linalg.norm(vec) - sigma2 < ADDlengthTH:
                return True
        return False
    def calcsigma(self, _vec, D_maha):
        """
        return the variance of UIbefore 
        along _vec
        D_maha is Mahalanobis' distance
        if D_maha == 1, result is equal to sigma
        """
        #return UIstepCython.calcsigma(self.ave, _vec, NormalD,
                #self.cov_eigN, self.cov_eigV, self.const.periodicmax, self.const.periodicmin, self.const.dim)
        if len(_vec) != len(self.ave):
            print("ERROR in umbrellaINT.calcsigma: len(_vec) != len(self.ave)")
            return False

        return UIstepCython.calcsigma(_vec, D_maha, self.covinv)
    def angle(self, x, y):
        """
        angle between x vector and y vector
        0 < angle < 2 * pi in this calculation
        """
        dot_xy = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        _cos = dot_xy / (norm_x*norm_y)
        if _cos > 1:
            return 0.0
        elif _cos < -1:
            return pi
        return np.arccos(_cos)
    def nextK(self, targetpoint, refthresholdchune, Kminchune):
        if self.const.K_mpiQ:
            with open("./selfpath.dat",   "w") as wf:
                wf.write("%s"%self.path)
            writeline = ""
            for i in range(self.const.dim):
                writeline += " %s"%targetpoint[i]
            with open("./targetpoint.dat", "w") as wf:
                wf.write(writeline)
            with open("./refthresholdchune.dat", "w") as wf:
                wf.write("%s"%refthresholdchune)
            with open("./Kmin.dat", "w") as wf:
                wf.write("%s"%Kminchune)
            sp.call(mpirunlist + ["%s/Poppins.py"%pwdpath,"nextK_MPI"])
            nextK = []
            try:
                for line in open("./nextK.dat"):
                    line = line.split()
                    nextK.append(line)
                nextK      = np.array(nextK, dtype=float)
                nextref    = np.array(open("./nextref.dat").readline().split(), dtype=float)
                gradvector = np.array(open("./gradvector.dat").readline().split(), dtype=float)
            except:
                #nextK      = E * Kminchune
                nextK      = np.identity(self.const.dim) * Kmax * 100.0
                nextref    = targetpoint
                gradvector = 0.0
            if rmdatfilesQ:
                try:
                    os.remove("./selfpath.dat")
                    os.remove("./targetpoint.dat")
                    os.remove("./refthresholdchune.dat")
                    os.remove("./Kmin.dat")
                    if os.path.exists("./nextK.dat"):
                        os.remove("./nextK.dat")
                        os.remove("./nextref.dat")
                        os.remove("./gradvector.dat")
                except FileNotFoundError:
                    c = inspect.currentframe()
                    print("Debag(%s): FileNotFoundError in %s."%(c.f_lineno, os.getcwd()))

        else:
            comm = 0
            nextK, nextref, gradvector = self.nextK_parallel(targetpoint, comm, refthresholdchune, Kminchune)
        for i in range(self.const.dim):
            if nextK[i, i] > self.const.Kmax:
                nextK[i, i] = self.const.Kmax + 1.0
        nextK   = np.round(nextK, self.const.nextsteproundN)
        nextref = np.round(nextref, self.const.nextsteproundN)
        return nextK, nextref, gradvector
    def nextK_parallel(self, targetpoint, comm, 
            refthresholdchune = False, Kminchune = False):
        """
        next constrain 
        """
        if refthresholdchune is False:
            refthresholdchune = self.const.refthreshold
        if Kminchune is False:
            Kminchune = self.const.Kmin
        if self.const.K_mpiQ:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1
        #os.environ["OMP_NUM_THREADS"] = "1"
        root = 0
        if rank == root:
            nextref = np.array(targetpoint)
            targetpointdamp = np.array(targetpoint)
            if self.const.periodicQ:
                targetpointdamp = UIstepCython.periodic(targetpointdamp,
                        self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
            else:
                targetpointdamp = targetpoint
            gradvector = targetpointdamp - self.ave
            if np.linalg.norm(gradvector) != 0.0:
                gradvector = gradvector / np.linalg.norm(gradvector)
            if self.const.nextKconstQ:
                return nextKconst, nextref, gradvector
            thresholdchune  = self.const.nextK_threshold
            targetpointdamp = np.array(targetpoint)
            if self.const.periodicQ:
                selfaveTdamp = UIstepCython.periodic(self.aveT,
                        self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
                selfavedamp  = UIstepCython.periodic(self.ave,
                        self.const.periodicmax, self.const.periodicmin, self.ref, self.const.dim)
            else:
                selfaveTdamp = self.aveT
                selfavedamp  = self.ave
            Deltatarget = targetpointdamp - selfaveTdamp
            vec         = targetpointdamp - selfavedamp
            ############
            # calculate max constrain:
            # the result, which refpoint is targetpoint.
            #############
            kradmax = Kminchune
            nextref = copy.copy(targetpointdamp)
        else:
            nextref         = None
            targetpointdamp = None
            kradmax         = None
            Deltatarget     = None
        if self.const.K_mpiQ:
            nextref         = comm.bcast(nextref,         root=0)
            targetpointdamp = comm.bcast(targetpointdamp, root=0)
            kradmax         = comm.bcast(kradmax,         root=0)
            Deltatarget     = comm.bcast(Deltatarget,     root=0)
        krad      = 0.0
        movevec   = np.zeros(self.const.dim) 
        Kdelta    = self.const.Kmin
        #Kdelta    = copy.copy(Kmax)
        KmaxQ     = False
        thetalist = None
        while True:
            if self.const.Kmax < Kminchune + krad:
                krad -= Kdelta
                Kdelta *= 0.5
                if Kdelta < self.const.Kmin * 0.01:
                    KmaxQ  = True
                    _nextK = 2.0 * self.const.Kmax * np.identity(self.const.dim)
                    break
                continue
            thetalist = None
            movevec, delta, _nextK, thetalist = self.optref(
                    nextref, targetpointdamp, Deltatarget,
                    krad, movevec, refthresholdchune, Kminchune, thetalist, comm)
            _point  = self.ave + delta
            _point2 = self.ave + movevec
            _point  = functions.periodicpoint(_point, self.const)
            _point2 = functions.periodicpoint(_point2, self.const)
            print("%s"%datetime.datetime.now())
            print("krad = %s"%krad)
            print("(%s, %s)"%(self.Dbias(_point),self.Dbias(_point2)))
            if self.Dbias(_point) < self.const.threshold and self.Dbias(_point2) < refthresholdchune:
                if 0.0 < krad - Kdelta:
                    krad -= Kdelta
                Kdelta *= 0.5
                if Kdelta < self.const.Kmin * 0.1:
                    break
                if krad == 0.0:
                    break
            else:
                krad += Kdelta
        nextref    = targetpointdamp - movevec
        nextref    = functions.periodicpoint(nextref, self.const)
        gradvector = targetpointdamp - self.ave
        if np.linalg.norm(gradvector) == 0.0:
            gradvector = np.zeros(self.const.dim)
        else:
            gradvector = gradvector / np.linalg.norm(gradvector)
        return _nextK, nextref, gradvector
    def optref(self, nextref, targetpointdamp, Deltatarget, krad,
                    movevec, refthresholdchune, Kminchune, thetalist, comm):
        movevec = np.zeros(self.const.dim)
        deltaD  = 0.5
        whileN  = 0
        if self.const.K_mpiQ:
            rank = comm.Get_rank()
        else:
            rank = 0
        root = 0
        movevecbefore = np.array([1.0e30 for _ in range(self.const.dim)])
        #deltabefore = copy.copy(delta)
        deltabefore = np.array([1.0e30 for _ in range(self.const.dim)])
        #while True:
        if True:
            whileN += 1
            if 100 <  whileN:
                deltaD *= 0.5
                whileN  = 0
#                if rank == root:
#                    with open("./nextKerror.txt", "a") as wf:
#                        c = inspect.currentframe()
#                        wf.write("Error(%s): whileN over 100\n"%(c.f_lineno))
#                        wf.write("Debag: optref: deltaD = %s\n"%deltaD)
                return movevec, delta, _nextK, thetalist

                #if deltaD < 0.01:
                    #return movevec, delta, _nextK, thetalist
                movevec = np.zeros(self.const.dim)
            ##if krad == 0.0 and KoffdiagonalQ is False:
            #if False:
            if krad == 0.0:
                #if KoffdiagonalQ:
                if False:
                    thetalist = np.array([0.0 for _ in range(self.const.dim - 1)])
                    _nextK = self.calcK_theta(krad, thetalist, Kminchune)
                else:
                    _nextK  = np.identity(self.const.dim) * Kminchune
                _delta1 = UIstepCython.cdelta1(self.ave, self.K, self.ref, self.covinv, 
                              nextref + movevec, Deltatarget, self.const.dim, self.const.betainv)
                delta   = UIstepCython.cdelta(_delta1, self.K, _nextK, self.covinv, 
                              targetpointdamp, nextref + movevec, self.const.dim, self.const.betainv)
            else:
                delta, _nextK, thetalist = self.nextK_rangenextK(nextref + movevec, 
                    targetpointdamp, krad, Deltatarget, Kminchune, thetalist, comm)
            #_point = self.ave + delta
            #functions.periodicpoint(_point, {})
            #if self.Dbias(_point) < threshold:
                #break
#            result = minimize(lambda x:
#                        np.linalg.norm(
#                          UIstepCython.cdelta(
#                            UIstepCython.cdelta1(self.ave, self.K, self.ref, self.covinv,
#                              nextref + movevec - delta * x, Deltatarget, self.const.dim, self.const.betainv),
#                            self.K, _nextK, self.covinv,
#                            targetpointdamp, nextref + movevec - delta * x, self.const.dim, self.const.betainv
#                          )
#                        ),
#                        x0 = 0.0,
#                        method = "L-BFGS-B")
#            vec = delta / np.linalg.norm(delta)
            if rank == root:
                refdelta = copy.copy(refthresholdchune)
                refthresholdchunedamp = copy.copy(refthresholdchune)
                movevecbefore = None
                Dbiasbefore   = 1.e30
                whileN = 0
                while True:
                    whileN += 1
                    cons = (
                        {"type": "ineq", "fun": lambda x: self.consF(x, movevec, refthresholdchunedamp)}
#in ths rextriction, the x (moveing of ref point) is constrainted as smaller than refthresholdchune
                        )
                    #print("start: %s"%datetime.datetime.now(), flush = True)
                    result = minimize(lambda x: 
                                self.Dbias_delta(nextref + movevec - x,
                                #self.Dbias_delta(self.ave + movevec - vec * x,
                                    Deltatarget, _nextK, targetpointdamp),
                        x0 = np.zeros(self.const.dim), constraints=cons,
                        method = "SLSQP")
                    #print("end: %s"%datetime.datetime.now(), flush = True)
                    movevecdamp = movevec - result.x
                    _point2 = self.ave + movevecdamp
                    _point2 = functions.periodicpoint(_point2, self.const)
                    _delta1 = UIstepCython.cdelta1(self.ave, self.K, self.ref, self.covinv, 
                              nextref + movevecdamp, Deltatarget, self.const.dim, self.const.betainv)
                    deltadamp = UIstepCython.cdelta(_delta1, self.K, _nextK, self.covinv, 
                              targetpointdamp, nextref + movevecdamp, self.const.dim, self.const.betainv)
                    _point  = self.ave + deltadamp
                    _point  = functions.periodicpoint(_point, self.const)
                    refdelta *= 0.5
                    #print("(%s, %s)"%(self.Dbias(_point),self.Dbias(_point2)), flush = True)
                    if self.const.threshold < self.Dbias(_point):
                        if not movevecbefore is None:
                            refthresholdchunedamp += refdelta
                        else:
                            #movevecbefore = copy.copy(movevecdamp)
                            break
                    else:
                        refthresholdchunedamp -= refdelta
                        if self.Dbias(_point2) < Dbiasbefore:
                            movevecbefore = copy.copy(movevecdamp)
                            Dbiasbefore   = self.Dbias(_point2)
                    if refdelta < 0.0001:
                        movevecdamp = copy.copy(movevecbefore)
                        break
                    if self.Dbias(_point) < self.const.threshold and self.Dbias(_point2) < refthresholdchune:
                        movevecdamp = copy.copy(movevecbefore)
                        break
                    #if refthresholdchunedamp < 0.0001 or \
                            #refthresholdchune < abs(refthresholdchunedamp):
                        #movevecdamp = copy.copy(movevecbefore)
                        #break
                    #if 1000 < whileN:
                        #movevecdamp = copy.copy(movevecbefore)
                        #break
            else:
                movevecdamp = None
            if self.const.K_mpiQ:
                movevecdamp = comm.bcast(movevecdamp, root=0)
            _point2 = self.ave + movevecdamp
            _point2 = functions.periodicpoint(_point2, self.const)
            _delta1 = UIstepCython.cdelta1(self.ave, self.K, self.ref, self.covinv, 
                      nextref + movevecdamp, Deltatarget, self.const.dim, self.const.betainv)
            deltadamp = UIstepCython.cdelta(_delta1, self.K, _nextK, self.covinv, 
                      targetpointdamp, nextref + movevecdamp, self.const.dim, self.const.betainv)
            _point  = self.ave + deltadamp
            _point  = functions.periodicpoint(_point, self.const)
            #if self.Dbias(_point) < self.const.threshold and self.Dbias(_point2) < refthresholdchune:
            if True:
                movevec = copy.copy(movevecdamp)
                delta       = copy.copy(deltadamp)
#                break
#            if np.linalg.norm(movevecdamp - movevec) < 1.0e-4:
#                #if rank == root:
#                    #print("movevec is not moved", flush = True)
#                break
#            _point = self.ave + movevecdamp
#            functions.periodicpoint(_point, {})
#            movevec = copy.copy(movevecdamp)
#            delta       = copy.copy(deltadamp)
        _delta1 = UIstepCython.cdelta1(self.ave, self.K, self.ref, self.covinv, 
                              nextref + movevec, Deltatarget, self.const.dim, self.const.betainv)
        delta   = UIstepCython.cdelta(_delta1, self.K, _nextK, self.covinv, 
                              targetpointdamp, nextref + movevec, self.const.dim, self.const.betainv)
        return movevec, delta, _nextK, thetalist
    def nextK_rangenextK(self, nextref, targetpointdamp, krad, Deltatarget, Kminchune, thetalist, comm):
        if self.const.K_mpiQ:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1
        #os.environ["OMP_NUM_THREADS"] = "1"
        root = 0
        _delta1 = UIstepCython.cdelta1(self.ave, self.K, self.ref, 
                    self.covinv, nextref, Deltatarget, self.const.dim, self.const.betainv)
        deltaNormMinbefore = 1.0e30
        deltaNormMin       = 1.0e30
        boundlist = [[0.0, 0.5 * np.pi] for _ in range(self.const.dim - 1)]
        thetalist          = None
        if not thetalist is None:
            if rank == root:
                result = minimize(lambda x: 
                        self.calcdelta_optNorm(nextref, targetpointdamp, _delta1, krad, x, Kminchune), 
                        x0 = np.array(thetalist), bounds = (boundlist),
                        method = "L-BFGS-B")
                thetalist = copy.copy(result.x)
            if self.const.K_mpiQ:
                thetalist = comm.bcast(thetalist, root=0)

        else:
            whileN = 0
            while True:
                deltaNormMin_mp = 1.0e30
                randomnumber = int(self.const.Krandommax / self.const.parallelPy) + 1
                whileN += randomnumber * self.const.parallelPy
                #if rank == root:
                    #print("%s:whileN = %s"%(datetime.datetime.now(), whileN), flush = True)
                for n in random_hyper_sphere(randomnumber, self.const, positiveQ = True):
                    if n is False:
                        break
                    thetalistdamp = calctheta(n)
                    #result = minimize(lambda x: 
                        #self.calcdelta_optNorm(nextref, targetpointdamp, _delta1, krad, x, Kminchune), 
                        #x0 = np.array(thetalistdamp), bounds = (boundlist),
                        #method = "L-BFGS-B")
                    #thetalistdamp = copy.copy(result.x)
                    deltaNorm_mp = self.calcdelta_optNorm(
                        nextref, targetpointdamp, _delta1, krad, thetalistdamp, Kminchune)
                    if deltaNorm_mp < deltaNormMin_mp:
                        #result = minimize(lambda x: 
                            #self.calcdelta_optNorm(nextref, targetpointdamp, _delta1, krad, x, Kminchune), 
                            #x0 = np.array(thetalistdamp), bounds = (boundlist),
                            #method = "L-BFGS-B")
                        #thetalistdamp = copy.copy(result.x)
                        #deltaNorm_mp = self.calcdelta_optNorm(
                            #nextref, targetpointdamp, _delta1, krad, thetalistdamp, Kminchune)
                        deltaNormMin_mp = copy.copy(deltaNorm_mp)
                        thetalist_mp    = copy.copy(thetalistdamp)
                        #deltaNormMin = copy.copy(deltaNorm_mp)
                        #thetalist    = copy.copy(thetalistdamp)
    #            if deltaNormMin_mp < deltaNormMin:
    #                deltaNormMin = copy.copy(deltaNormMin_mp)
    #                thetalist    = copy.copy(thetalist_mp)
    #                #thetalist    = copy.copy(result.x)
                if self.const.K_mpiQ:
                    result = minimize(lambda x: 
                        self.calcdelta_optNorm(nextref, targetpointdamp, _delta1, krad, x, Kminchune), 
                        x0 = np.array(thetalist_mp), bounds = (boundlist),
                        method = "L-BFGS-B")
                    thetalist_mp = copy.copy(result.x)
                    deltaNormMin_mp = self.calcdelta_optNorm(
                        nextref, targetpointdamp, _delta1, krad, thetalist_mp, Kminchune)
                    #returnlist = [deltaNormMin, thetalist]
                    returnlist = [deltaNormMin_mp, thetalist_mp]
                    teturnlist = comm.gather(returnlist, root=0)
                else:
                    #returnlist = [[deltaNormMin, thetalist]]
                    returnlist = [[deltaNormMin_mp, thetalist_mp]]
                if rank == root:
                    if type(returnlist[0]) is float:
                        #print("ERROR: returnlist = %s"%returnlist)
                        returnlist = [returnlist]
                    for deltaNorm, thetalistdamp in returnlist:
                        if deltaNorm < deltaNormMin:
                            deltaNormMin = copy.copy(deltaNorm)
                            thetalist    = copy.copy(thetalistdamp)
                    #result = minimize(lambda x: 
                        #self.calcdelta_optNorm(nextref, targetpointdamp, _delta1, krad, x, Kminchune), 
                        #x0 = np.array(thetalist), bounds = (boundlist), 
                        #method = "L-BFGS-B")
                    #thetalist = copy.copy(result.x)
                    #print("deltaNormMin = %s"%deltaNormMin)
                if self.const.K_mpiQ:
                    deltaNormMin       = comm.bcast(deltaNormMin,       root=0)
                    deltaNormMinbefore = comm.bcast(deltaNormMinbefore, root=0)
                    thetalist          = comm.bcast(thetalist,          root=0)
    
                if abs(deltaNormMin - deltaNormMinbefore) < 0.0001:
                    #if rank == root:
                        #with open("./deltaNorm.dat", "a") as wf:
                            #wf.write("thetalist = %s\n"%thetalist)
                    break
                else:
                    deltaNormMinbefore = copy.copy(deltaNormMin)
                    #deltaNormMinbefore = comm.bcast(deltaNormMinbefore, root=0)
        if rank == root:
#            result = minimize(lambda x: 
#                    self.calcdelta_optNorm(nextref, targetpointdamp, _delta1, krad, x), 
#                    x0 = np.array(thetalist), bounds = (boundlist), 
#                    method = "L-BFGS-B")
#            thetalist = copy.copy(result.x)
            _nextK = self.calcK_theta(krad, thetalist, Kminchune)
            delta  = UIstepCython.cdelta(_delta1, self.K, _nextK, 
                    self.covinv, targetpointdamp, nextref, self.const.dim, self.const.betainv)
            #print("deltaNormMin = %s"%deltaNormMin)
            #print("norm(delta)  = %s"%np.linalg.norm(delta))
        else:
            delta  = None
            _nextK = None
        if self.const.K_mpiQ:
            delta  = comm.bcast(delta,  root=0)
            _nextK = comm.bcast(_nextK, root=0)
        return delta, _nextK, thetalist
    def calcdelta_opt(self, nextref, targetpointdamp, _delta1, krad, x, Kminchune):
        _nextK = self.calcK_theta(krad, x, Kminchune)
        _delta = UIstepCython.cdelta(
                _delta1, self.K, _nextK, self.covinv, targetpointdamp, nextref, self.const.dim, self.const.betainv)
        return _delta
    def calcdelta_optNorm(self, nextref, targetpointdamp, _delta1, krad, x, Kminchune):
        _nextK     = self.calcK_theta(krad, x, Kminchune)
        #delta_Norm = UIstepCython.cdelta_Norm(_delta1, self.K, _nextK, 
                #self.covinv, self.const.dim, self.const.betainv, targetpointdamp, nextref)
        #print(delta_Norm)
        #return delta_Norm
        _delta = UIstepCython.cdelta(
                _delta1, self.K, _nextK, self.covinv, targetpointdamp, nextref, self.const.dim, self.const.betainv)
        _point  = self.ave + _delta
        _point = functions.periodicpoint(_point, self.const)
        return self.Dbias(_point)
    def calcK_theta(self, krad, x, Kminchune):
        eigN, eigVlist = np.linalg.eigh(self.hes)
        #eigN, eigVlist = np.linalg.eigh(self.hesinv)
        #_eigV = []
        #for i in range(self.const.dim):
            #_eigV.append(eigVlist[:,i])
        #return UIstepCython.calcK_theta(krad, x, _eigV, KoffdiagonalQ, Kminchune, self.const.dim)
        KoffdiagonalQ = False
        return UIstepCython.calcK_theta(krad, x, eigVlist, KoffdiagonalQ, Kminchune, self.const.dim)
    def SQaxes(self):
        SQaxes = np.array([
            np.sqrt(1.0 / self.eigN[i]) * self.eigV[i]
            for i in range(self.const.dim)
            ])
        return SQaxes
    def wallQ(self, point):
        """
        outside of wall or not
        """
        for _j in range(self.const.dim):
            if  wallmax[_j] < point[_j]:
                return True
            if point[_j] < wallmin[_j]:
                return True
        return False
    def Dmin(self, UIlist, nearUIlist, xi):
        minD = 1.0e30
        nearestUIN = False
        #print("%s : %s"%(len(UIlist), len(nearUIlist)))
        #print(nearUIlist)
        for UI in UIlist:
            if UI.stepN < 0:
                continue
            if UI.stepN == self.stepN:
                continue
            if not UI.stepN in nearUIlist:
                continue
            UID = UI.Dbias(xi)
            if UID < minD:
                minD      = copy.copy(UID)
                nearestUIN = copy.copy(UI.stepN)
        return minD, nearestUIN
    def Shapiro_Maharanobis(self, data):
        _ave  = data.mean(0)
        _cov  = np.cov(data.T)
        _covinv = np.linalg.inv(_cov)
        _cov_eigN, cov_eigVdamp = np.linalg.eigh(_cov)
        _cov_eigV = []
        for i in range(self.const.dim):
            _cov_eigV.append(cov_eigVdamp[:,i])
        _cov_eigV = np.array(_cov_eigV)
        Dlist = []
        for xi in data:
#            if self.const.periodicQ:
#                D = UIstepCython.Dbias_periodic(xi, self.const.periodicmax, self.const.periodicmin,
#                    _ave, _cov_eigN, _cov_eigV, self.const.dim)
#            else:
#                D = UIstepCython.Dbias_periodic(xi, _ave, 
#                        _cov_eigN, _cov_eigV, self.const.dim)
            D = UIstepCython.Dbias(xi, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin,
                _ave, _covinv, self.const.dim, partOPTQ, partdim)
            Dlist.append(D)
        Dlist = np.array(Dlist)
        W, p  = stats.shapiro(Dlist)
        return W, p
    def calcvarA_w(self, Evec, point, t):
        #sigma_rsqinv = 0.0
        #for i in range(self.const.dim):
            #for j in range(self.const.dim):
                #sigma_rsqinv += Evec[i] * self.covinv[i, j] * Evec[j]
        sigma_r = self.calcsigma(Evec, 1.0)
        #print(sigma_r)
        #sigma_r = np.sqrt()
        if self.Dbias(point) > integratesigmaTH:
            return 0.0, sigma_r
        #covinv_eigN, covinv_eigVdamp = np.linalg.eigh(self.covinv)
        #covinv_eigV = []
        #for i in range(self.const.dim):
            #covinv_eigV.append(covinv_eigVdamp[:,i])
        #P      = self.cov_eigV
        P      = np.array(self.covinv_eigV)
        Pinv   = np.linalg.inv(P)
        cinvdash = np.zeros((self.const.dim,self.const.dim))
        for i in range(self.const.dim):
            cinvdash[i,i] = self.covinv_eigN[i]
        varA_w = 0.0
        if self.const.periodicQ:
            avedamp = UIstepCython.periodic(self.ave,
                            self.const.periodicmax, self.const.periodicmin, point, self.const.dim)
        else:
            avedamp = self.ave
        #dotvec = np.dot(Pinv, point + t * Evec - avedamp)
        #ErdotP = np.dot(Evec, P)
        dotvec = np.zeros(self.const.dim)
        vecdamp = point + t * Evec - avedamp
        #vecrefdamp = point + t * Evec - refdamp
        ErdotP = np.zeros(self.const.dim)
        #gradtest = 0.0
        cinvtest = np.zeros((self.const.dim,self.const.dim))
        for i in range(self.const.dim):
            for j in range(self.const.dim):
                ErdotP[j] += Evec[i]    * P[i, j]
                dotvec[j] += Pinv[j, i] * vecdamp[i]
                #gradtest -= Evec[i] * self.K[i,j] * vecrefdamp[j]
        ErdotP = np.dot(Evec, P)
        dotvec = np.dot(Pinv, vecdamp)
        #gradbefore  = np.dot(Evec, self.covinv)
        #gradbefore  = np.dot(gradbefore, vecdamp) * self.const.betainv
        #gradbefore += copy.copy(gradtest)
        for j in range(self.const.dim):
            #print("eigN = %s"%(self.covinv_eigN[j]))
            #print("sigmar_sq, sigma_jsq = %s, %s"%(1.0/sigma_rsqinv, self.cov_eigN[j]))
            #sigma_jsq = 1.0 / self.cov_eigN[j]
            sigma_jsq = self.covinv_eigN[j]
            #sigmatest = self.calcsigma(self.cov_eigV[j], 1.0)
            #print("sigmatest(%s) = %s"%(j, 1.0 / sigmatest / sigmatest))
            #print("sigmadiff = %s"%(sigma_jsq - 1.0/sigmatest /sigmatest))
            #sigma_jsq = 1.0 / sigmatest /sigmatest
            varA_j    = dotvec[j] * sigma_jsq
            #print(self.sigmasq_traj[j])
            #print("(%s + %s * %s)"%(self.ave_traj[j], varA_j * varA_j, self.sigmasq_traj[j]))
            varA_j    = self.ave_traj[j] + varA_j * varA_j * self.sigmasq_traj[j]
            #print("varA_j = %s"%varA_j)
            varA_w   += varA_j * ErdotP[j] * ErdotP[j] * sigma_jsq * sigma_jsq
            #print("EdotP * sigma_jsq = %s"%(ErdotP[j] * sigma_jsq))
            #gradtest += ErdotP[j] * dotvec[j] * sigma_jsq * self.const.betainv
        #print("gradtest    = %s"%gradtest)
        #print("gradbefore  = %s"%gradbefore)
        #print("grad        = %s"%np.dot(Evec, self.grad(point + t * Evec)))
        #print("grad_on_vec = %s"%self.grad_on_vec(t, point, Evec))
        #print((gradtest - np.dot(Evec, self.grad(point + t*Evec))) / gradtest)
        varA_w *= self.const.betainv * betainv
        #print("varA_w(%s) = %s"%(self.Dbias(point), varA_w))
        return varA_w, sigma_r
    def consF(self, _x, movevec, refthresholdchune):
        _point2 = self.ave + movevec - _x
        _point2 = functions.periodicpoint(_point2, self.const)
        returnF = refthresholdchune - 0.0001 - self.Dbias(_point2)
        #print("%s: %s"%(_point2, returnF))
        return returnF
    def MahaSphere(self, UIeq, thetalist, targetsigma):
        _vec    = np.array([1.0 for i in range(self.const.dim)])
        thetalist = list(thetalist)
        thetalist.reverse()
        for i, theta in enumerate(thetalist):
            cosS = np.cos(theta)
            for n in range(self.const.dim - i - 1):
                _vec[n] *= cosS
            _vec[self.const.dim - i - 1] *= np.sin(theta)
        _vec = _vec * self.calcsigma(_vec, targetsigma)
        return UIeq.ave + _vec
    def deltaA_sigma(self, UIeq, thetalist, targetsigma, periodiclist):
        _x  = self.MahaSphere(UIeq, thetalist, targetsigma)
        _x += periodiclist
        #functions.periodicpoint(_x,  {})
        return UIeq.deltaA_UI(UIeq.ave, _x)
    def importPCA(self, filename):
        from sklearn.decomposition import PCA
        fieldnames, ldamp, t = self.readCOORD(filename)
        if fieldnames is False:
            return False
        pca = PCA(n_components = len(ldamp[0]), whiten = False)
        pca.fit(ldamp)
        self.pca_Vratio     = pca.explained_variance_ratio_
        self.pca_components = pca.components_
    def mkplumed(self, nextstep, Kmat, currentstep):
        plumeddic = copy.copy(self.const.plumeddic_def)
        CVlist = []
        for i in range(self.const.dim):
            for pldic in plumeddic:
                if "CV%s"%i in pldic["comments"]:
                    CVlist.append(pldic['LABEL'])
                    break
        for i, CVlabel in enumerate(CVlist):
            pldic = {"options":["RESTRAINT"], "comments":[], "linefeedQ": True}
            pldic["ARG"]   = CVlabel
            pldic["AT"]    = nextstep[i]
            pldic["KAPPA"] = Kmat[i,i]
            pldic["LABEL"] = "restraint%s"%i
            plumeddic.append(pldic)
        
        plname   = "plumed.dat"
        printdic = {"options":["PRINT"], "comments":[], "linefeedQ": False}
        printdic["STRIDE"] = "500"
        printdic["ARG"] = ",".join(CVlist)
        printdic["FILE"] = "COLVAR"
        functions.exportplumed(plname,plumeddic+[printdic])
        printdic["FILE"] = "COLVAR_npt"
        plname   = "plumed_npt.dat"
        functions.exportplumed(plname,plumeddic+[printdic])
        dirname = "./"
        if self.copynpt(dirname, currentstep) is False:
            return False
        return True
    def copynpt(self, dirname, currentstep):
        if "window" in self.path and self.const.WindowDataType == "hdf5":
            if self.stepN < 1:
                formatline = "{}/%s"%self.const.grofilename
                nptfiles   = formatline.format(self.calculationpath)
                if os.path.exists(nptfiles):
                    shutil.copy(nptfiles, "./min.gro")
                else:
                    c = inspect.currentframe()
                    print("ERROR({}): npt can not copy {}".format(c.f_lineno, nptfiles))
                    print("os.getcwd() = %s"%os.getcwd())
                    #exit()
                    return False
                return True
    
            steppath = self.path.split("/")[-1]
            #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
            #lock.acquire()
            #with h5py.File("%s/jobfiles/windows/windowfile.hdf5"%pwdpath, "r", libver='latest', swmr=True) as windowHDF:
                #grotext = windowHDF["%s/equibliumstructure"%steppath][...]
            #lock.release()
            UIlistchunkmin = 1
            UIlistchunkmax = self.const.UIlistchunksize
            while True:
                if UIlistchunkmin <= self.stepN <= UIlistchunkmax:
                    break
                UIlistchunkmin += self.const.UIlistchunksize
                UIlistchunkmax += self.const.UIlistchunksize
            UIlistHDFpath = "%s/%s/windows/windowfile%s-%s.hdf5"%(self.const.pwdpath, self.const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
            UIlistchunkmin += self.const.UIlistchunksize
            UIlistchunkmax += self.const.UIlistchunksize
            UIlistHDFpath_next = "%s/%s/windows/windowfile%s-%s.hdf5"%(self.const.pwdpath, self.const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
            if not os.path.exists(UIlistHDFpath_next):
                lock = fasteners.InterProcessLock(self.const.lockfilepath_UIlist + str(UIlistchunkmax - self.const.UIlistchunksize))
                lock.acquire()
            windowHDF = h5py.File(UIlistHDFpath, "r")
            grotext = windowHDF["%s/equibliumstructure"%steppath][...]
            windowHDF.close()
            if not os.path.exists(UIlistHDFpath_next):
                lock.release()
            grotext = str(grotext)
            open("./min.gro", "w").write(grotext)
            return True
        else:
            formatline = "{}/%s"%self.const.grofilename
            nptfiles   = formatline.format(self.path)
            if os.path.exists(nptfiles):
                shutil.copy(nptfiles, "./min.gro")
            else:
                c = inspect.currentframe()
                print("ERROR({}): npt can not copy {}".format(c.f_lineno, nptfiles))
                print("os.getcwd() = %s"%os.getcwd())
                #exit()
                return False
            return True



### end class UIstep() ###
class savenpzclass():
    pass

def parallelchkUI(parallellist):
    UI, ADDlist = parallellist
    #return [UI]
    for ADDc in ADDlist:
        vec = ADDc.nADDnext - ADDc.nADDbefore
        #print(ADDc.nADDnext)
        for i in range(6):
            _point = ADDc.beforestep + vec * i * 0.2
            if UI.Dbias(_point) < neighborsigmaTH:
                return [UI]
    return []

#def updateUIlist_exclusion(UIeq, UIlist, edgelist, endlist, edgeADDQ, IOEsphereA_before, IOEsphereA):
##    with open(lockfilepath) as oLockFile:
##        #print('EXCLUSION CONTROL: START updateUIlist_exclusion"')
##        fcntl.flock(oLockFile.fileno(), fcntl.LOCK_EX)
##        try:
##            #print('EXCLUSION CONTROL: It gains the lock file.')
##            UIlist, edgelist = updateUIlist(UIeq, UIlist, edgelist, endlist, edgeADDQ, IOEsphereA_before, IOEsphereA)
##        finally:
##            fcntl.flock(oLockFile.fileno(), fcntl.LOCK_UN)
##    #print('EXCLUSION CONTROL: END updateUIlist_exclusion"')
#    UIlist, edgelist = updateUIlist(UIeq, UIlist, edgelist, endlist, edgeADDQ, IOEsphereA_before, IOEsphereA)
#    if len(UIlist) == 0:
#        print("ERROR: len(UIlist) = 0")
#
#    return UIlist, edgelist
#def updateUIlist(UIeq, UIlist, edgelist, endlist, edgeADDQ, IOEsphereA_before, IOEsphereA):
#    num = 0
#    dirkind = "step"
#    UIlist     = []
#    edgelist   = []
#    #UIpathlist = [UI.calculationpath for UI in UIlist]
#    UInumlist  = [UI.stepN for UI in UIlist]
#    dirpathlistlist = [[] for _ in range(parallelPy)]
#    turnN = 0
#    for dirpath in glob.glob("{0}/jobfiles/windows/{1}*".format(pwdpath, dirkind)):
#        if os.path.exists("%s/calculationpath.dat"%dirpath) is False:
#            continue
#        #beforepath = open("%s/calculationpath.dat"%dirpath).read()
#        #if beforepath in UIpathlist:
#            #continue
#        dirpathlistlist[turnN % parallelPy].append(dirpath)
#        turnN += 1
#    parallellists = []
#    for dirpathlist in dirpathlistlist:
#        parallellists.append(
#                [UIeq, UIlist, edgelist, endlist, edgeADDQ, IOEsphereA_before, 
#                    IOEsphereA, dirpathlist])
#
#    p = mp.Pool(parallelPy)
#    UIlistPlist = p.map(updateUIlistP, parallellists)
#    p.close()
#
#    edgelist = [[UIeq.stepN] + list(UIeq.ave) + [0.0, 0.0]]
#    UIlist     = [UIeq]
#    for UIlistP in UIlistPlist:
#        for UI in UIlistP:
#            didpointQ = False
#            for UIbefore in UIlist:
#                if UIbefore.Dbias(UI.ave) < didpointsigmaTH:
#                    didpointQ = True
#                    break
#            if didpointQ:
#                continue
#            if np.linalg.norm(UIeq.ave - UI.ave_periodic) != 0.0:
#                deltaA_UIeq  = UIeq.deltaA_UI(UIeq.ave, UI.ave_periodic)
#                ADDfe        = 0.0
#                UIlist.append(UI)
#                edgelist.append([int(UI.stepN)] + list(UI.ave) + [UI.A, ADDfe])
#    return UIlist, edgelist
#def updateUIlistP(parallellist):
#    UIeq, UIlist, edgelist, endlist, edgeADDQ, IOEsphereA_before, IOEsphereA, dirpathlist = parallellist
#    UIlistP = []
#    for dirpath in dirpathlist:
#        UI = UIstep()
#        UI.importdata(dirpath)
#        if UI.stepN in [x[0] for x in endlist]:
#            continue
#        if self.const.periodicQ:
#            UI.ave_periodic = UIstepCython.periodic(UI.ave,
#                    self.const.periodicmax, self.const.periodicmin, UIeq.ave, self.const.dim)
#            UI.ref_periodic = UIstepCython.periodic(UI.ref,
#                    self.const.periodicmax, self.const.periodicmin, UIeq.ave, self.const.dim)
#        else:
#            UI.ave_periodic = UI.ave
#            UI.ref_periodic = UI.ref
#        deltaA_UIeq  = UIeq.deltaA_UI(UIeq.ave, UI.ave_periodic)
#        UI.A = copy.copy(deltaA_UIeq)
#        if edgeADDQ:
#            vec     = UI.ave_periodic - UIeq.ave
#            vecnorm = np.linalg.norm(vec)
#            if vecnorm == 0.0:
#                continue
#            Evec = vec / vecnorm
#            point = UIeq.ave + functions.SperSphere_cartesian_n(UIeq, IOEsphereA, Evec, self.const.dim)
#            #functions.periodicpoint(point, {})
#            if self.const.periodicQ:
#                point = UIstepCython.periodic(point,
#                    self.const.periodicmax, self.const.periodicmin, np.zeros(self.const.dim), self.const.dim)
#            if IOEsphereA_before == IOEsphereA:
#                if integratesigmaTH < UI.Dbias(point):
#                    continue
#            else:
#                if IOEsphereA_before != 0.0:
#                    point_before = UIeq.ave + functions.SperSphere_cartesian_n(UIeq, IOEsphereA_before, Evec, self.const.dim)
#                    #functions.periodicpoint(point_before, {})
#                    if self.const.periodicQ:
#                        point_before = UIstepCython.periodic(point_before,
#                            self.const.periodicmax, self.const.periodicmin, np.zeros(self.const.dim), self.const.dim)
#                    if integratesigmaTH < UI.Dbias(point) and \
#                       integratesigmaTH < UI.Dbias(point_before):
#                        continue
#                elif IOEsphereA < deltaA_UIeq:
#                    if integratesigmaTH < UI.Dbias(point):
#                        continue
#        UIlistP.append(UI)
#    return UIlistP

#def IOE(nextstep, nextstepM, ADDfeM, UIeq):
#    if (nextstep == nextstepM).all():
#    #if th == psi:
#        return ADDfeM
#    UI = UIstep()
#    if partADDQ:
#        deltaTH = UI.angle(nextstep - UIeq.aveALL, nextstepM - UIeq.aveALL)
#    else:
#        deltaTH = UI.angle(nextstep - UIeq.ave, nextstepM - UIeq.ave)
#    if deltaTH < pi * 0.5:
#        cosdamp = np.cos(deltaTH)
#        return (ADDfeM)* cosdamp * cosdamp * cosdamp
#    else:
#        return 0.0
def IOE(nADD, nADDneibor, ADDfeM):
    if (nADD == nADDneibor).all():
        return ADDfeM
    deltaTH = functions.angle(nADD, nADDneibor)
    if deltaTH < pi * 0.5:
        cosdamp = np.cos(deltaTH)
        return ADDfeM * cosdamp * cosdamp * cosdamp
    else:
        return 0.0
#def IOE(ADDth, ADDbeforethetas):
#    returnIOE = 0.0
#    for neiborADDth in ADDbeforethetas:
#        if ADDth.IDnum == neiborADDth.IDnum:
#            continue
#            #return 0.0
#        if 0.0 < neiborADDth.ADDfeIOE:
#            continue
#            #return 0.0
#        if neiborADDth.ADDoptQ:
#            continue
#        if ADDth.ADDfe < neiborADDth.ADDfe:
#            continue
#        if np.allclose(ADDth.nADD, neiborADDth.nADD):
#            returnIOE += neiborADDth.ADDfeIOE
#        deltaTH = functions.angle(ADDth.nADD, neiborADDth.nADD)
#        if deltaTH < pi * 0.5:
#            cosdamp = np.cos(deltaTH)
#            #return neiborADDth.ADDfeIOE * cosdamp * cosdamp * cosdamp
#            returnIOE += neiborADDth.ADDfeIOE * cosdamp * cosdamp * cosdamp
#        #else:
#            #return 0.0
#    return returnIOE
    
def IOE_forADDths(UIeq, ADDths):
    ADDths = sorted(ADDths, key = lambda ADDth: ADDth.ADDfe)
    for i in range(len(ADDths)):
        ADDths[i].ADDfeIOE = copy.copy(ADDths[i].ADDfe)
    for i in range(len(ADDths)):
        ADDths[i].deltaTHmax = 1.0e30
        for j in range(i):
            #if 1.0e10 < ADDths[j].resultbefore:
            if ADDths[j].ADDoptQ:
                continue
            deltaTH = functions.angle(ADDths[i].nADD, ADDths[j].nADD)
            if pi * 0.5 < deltaTH:
                continue
            if minimizeTH <= ADDths[j].resultbefore:
                ADDths[i].deltaTHmax = copy.copy(deltaTH)
            ADDfeM    = ADDths[j].ADDfe
            ADDfeIOEM = ADDths[j].ADDfeIOE
            if 0.0 < ADDfeIOEM:
                continue
            if ADDfeM < ADDths[i].ADDfe:
                    cosdamp = np.cos(deltaTH)
                    ADDths[i].ADDfeIOE  -= ADDfeIOEM * cosdamp * cosdamp * cosdamp
    return ADDths
def IOE_forADD(deltaTH, ADDfeM):
    if deltaTH == 0.0:
        return ADDfeM
    #if min((abs(deltaTH), pi2 - abs(deltaTH))) < pi * 0.5:
    if deltaTH < pi * 0.5:
        cosdamp = np.cos(deltaTH)
        return (ADDfeM)* cosdamp * cosdamp * cosdamp
    else:
       return 0.0
def calcUIall(UIlist, initialpoint, finishpoint, const, calcVarQ = False, UIdataClass = False):
    if not UIdataClass:
        UIdataClass = functions.mkUIdataClass(UIlist)
    avelist      = UIdataClass.avelist
    reflist      = UIdataClass.reflist
    covinvlist   = UIdataClass.covinvlist
    Pbiasavelist = UIdataClass.Pbiasavelist
    Klist        = UIdataClass.Klist
    Nlist        = UIdataClass.Nlist
    #if Dmin(UIlist, finishpoint) > ADDstepsigmaTH:
        #print("ERROR: Dmin(UIlist, finishpoint) = %s"%Dmin(UIlist, finishpoint))
        #return None, None

    deltaA = UIstepCython.calcUIallC(avelist, reflist, covinvlist, Pbiasavelist, Klist, Nlist, 
            initialpoint, finishpoint, const.periodicQ,
            const.periodicmin, const.periodicmax, const.betainv, const.dim)
    #if calcVarQ:
    if False:
        varA = calcVarA(UIlist, initialpoint, finishpoint, Evec, vecnorm)
    else:
        varA = 0.0
    return deltaA, varA
def calcUIall_nearW(UIlist, finishpoint, UIself = False, calcVarQ = False, ipoint = False, NgradQ = False, UIdataClass = False):
    if isinstance(finishpoint, list):
        print("finishpoint is list")
        finishpoint = np.array(finishpoint)
    #deltaA = 0.0
    #initialpointdamp = copy.copy(initialpoint)
    #functions.periodicpoint(initialpointdamp, {})
    finishpointdamp = copy.copy(finishpoint)
    finishpointdamp = functions.periodicpoint(finishpointdamp)
    if not ipoint is False:
        ipointdamp = copy.copy(ipoint)
        ipointdamp = functions.periodicpoint(ipointdamp)
#    for UI in UIlist:
#        if UI.Dbias(initialpointdamp) < ADDstepsigmaTH:
#            avedamp = copy.copy(UI.ave)
#            functions.periodicpoint(avedamp, {})
#            if self.const.periodicQ:
#                initialpointdamp = UIstepCython.periodic(initialpointdamp, 
#                    self.const.periodicmax, self.const.periodicmin, avedamp, self.const.dim)
#            deltaA += UIstepCython.deltaA_UI(UI.covinv, UI.K, UI.ave, UI.ref,
#                initialpointdamp, avedamp, self.const.betainv)
#            deltaA -= UI.A
#            break
#    else:
#        return False, 0.0
    Dminimum    = 1.0e30
    Dminimum_ip = 1.0e30
    UIlist_calc = []
    for UI in UIlist:
        if UI.A is False:
            continue
        Ddamp = UI.Dbias(finishpointdamp)
        #print("%s <> %s -> %s"%(UIself.stepN, UI.stepN, Ddamp))
        #if Ddamp < ADDstepsigmaTH:
        if Ddamp < neighborwindowTH:
            UIlist_calc.append(UI)
        elif not ipoint is False:
        #if False:
            Ddamp_ip = UI.Dbias(ipointdamp)
            if Ddamp_ip < neighborwindowTH:
                UIlist_calc.append(UI)
        if not ipoint is False:
        #if False:
            Ddamp_ip = UI.Dbias(ipointdamp)
            if Ddamp_ip < Dminimum_ip:
                Dminimum_ip = copy.copy(Ddamp_ip)
                UInear_ip = copy.copy(UI)

        if Ddamp < Dminimum:
            Dminimum = copy.copy(Ddamp)
            UInear = copy.copy(UI)
    if UIself:
        for UI in UIlist:
            if UI.A is False:
                continue
            Ddamp = UIself.Dbias(UI.ave)
            #print("%s <> %s -> %s"%(UIself.stepN, UI.stepN, Ddamp))
            #if Ddamp < neighborwindowTH:
                #UIlist_calc.append(UI)
            if Ddamp < Dminimum:
                Dminimum = copy.copy(Ddamp)
                UInear = copy.copy(UI)
    #print(Dminimum)
    #print("%s <> %s"%(UInear.stepN, UIself.stepN))
    #if ADDstepsigmaTH < Dminimum:
    if neighborwindowTH < Dminimum:
        #with open("./UIlistdata.txt", "a")  as wf:
            #wf.write("Dminimum = %s\n"%Dminimum)
        print("Dminimum = %s"%Dminimum)
        #return False, 0.0, UInear
        return False, 0.0, None

    if not ipoint is False:
    #if False:
        avedamp = copy.copy(UInear_ip.ave)
    else:
        avedamp = copy.copy(UInear.ave)
    avedamp = functions.periodicpoint(avedamp)
    if self.const.periodicQ:
        finishpointdamp = UIstepCython.periodic(finishpointdamp, 
            self.const.periodicmax, self.const.periodicmin, avedamp, self.const.dim)
    if UIself is False:
        #if ADDstepsigmaTH < Dminimum:
            #with open("./UIlistdata.txt", "a")  as wf:
                #wf.write("Dminimum = %s\n"%Dminimum)
            #return False, 0.0, None
        #if NgradQ:
            #print("UInear.stepN    = %s"%UInear.stepN)
            #print(finishpointdamp)
        if not ipoint is False:
            A  = 0.0
            A += UInear_ip.A
            deltaA_UIall, varA = calcUIall(UIlist_calc, avedamp, finishpointdamp)
            #deltaA_UIall, varA = calcUIall(UIlist_calc, UInear_ip.ave, finishpointdamp)
            #deltaA_UIall, varA = calcUIall(UIlist_calc, avedamp, finishpointdamp)
            #deltaA_UIall, varA = calcUIall(UIlist, UInear_ip.ave, finishpointdamp)
            A += deltaA_UIall
        else:
            #if NgradQ:
                #print("UInear.stepN = %s"%UInear.stepN)
                ##print("finishpointdamp = %s"%finishpointdamp)
                #print("Dminimum = %s\n"%Dminimum)
            A = 0.0
            A += UInear.A
            #if ADDstepsigmaTH < Dminimum:
            #if True:
            if not UIdataClass is False:
                deltaA_UIall, varA = calcUIall(UIlist_calc, avedamp, finishpointdamp)
                #deltaA_UIall, varA = calcUIall(UIlist, avedamp, finishpointdamp, UIdataClass)
                A += deltaA_UIall
            else:
                A += UIstepCython.deltaA_UI(UInear.covinv, UInear.K, UInear.ave, UInear.ref,
                avedamp, finishpointdamp, self.const.betainv, self.const.periodicQ, self.const.periodicmax, self.const.periodicmin)
            #deltaA_UIall, varA = calcUIall(UIlist_calc, UInear.ave, finishpointdamp)
            #deltaA_UIall, varA = calcUIall(UIlist_calc, avedamp, finishpointdamp)

    else:
        deltaA_UIall, varA = calcUIall([UInear, UIself], UInear.ave, UIself.ave)
        A = UInear.A + deltaA_UIall
    if False:
        varA = calcVarA(UIlist, initialpoint, finishpoint, Evec, vecnorm)
    else:
        varA = 0.0
    return A, varA, UInear
def calcUIall_nearW_connection(UIlist, WGraph, UIlistall, UItarget, calcVarQ = False):
    lenSpath_minimum = 1.0e30
    Spath_minimum    = []
    for UI in UIlist:
        if WGraph.has_edge(UI.stepN, UItarget.stepN) is False:
            continue
        Spath = nx.shortest_path(WGraph, UI.stepN, UItarget.stepN)
        lenSpath = len(Spath)
        if lenSpath < lenSpath_minimum:
            lenSpath_minimum = copy.copy(lenSpath)
            Spath_minimum    = copy.copy(Spath)
    if len(Spath_minimum) == 0:
        return False, UIlist

    for beforeUI in UIlist:
        if beforeUI.stepN == Spath_minimum[0]:
            break
    for i in range(len(Spath_minimum) - 1):
        insideQ = False
        for newUI in UIlist:
            if newUI.stepN == Spath_minimum[i + 1]:
                insideQ = True
                break
        if insideQ:
            beforeUI = copy.copy(newUI)
            continue
            
        for newUI in UIlistall:
            if newUI.stepN == Spath_minimum[i + 1]:
                break
        else:
            print("ERROR: window cannot added")
            exit()
        deltaA_UIall, varA = calcUIall([beforeUI, newUI], beforeUI.ave, newUI.ave)
        newUI.A = beforeUI.A + deltaA_UIall
        UIlist.append(newUI)
        beforeUI = copy.copy(newUI)
    return True, UIlist
def calcVarA(UIlist, initialpoint, finishpoint, Evec, vecnorm):
    #vecinv = calcvecinv(Evec)
    sigma_rlist  = []
    varAalls = [0.0, 0.0]
    Dmin    = 1.0e10
    Pallinv = [1.0 / pUIall(UIlist, initialpoint),
               1.0 / pUIall(UIlist, finishpoint)]
    for UI in UIlist:
        Dbiasinitial =  UI.Dbias(initialpoint)
        t = 0.0
        varA_w, sigma_r = UI.calcvarA_w(Evec, initialpoint, t)
        if Dbiasinitial < Dmin:
            Dmin = copy.copy(Dbiasinitial)
            sigma_rave = copy.copy(sigma_r)
        if varA_w == varA_w:
            Pbias = UI.N * UI.Pbias(initialpoint) * Pallinv[1]
            #print("Pbias = %s"%Pbias)
            varAalls[0] += Pbias * Pbias * varA_w
        t = vecnorm
        varA_w, sigma_r = UI.calcvarA_w(Evec, initialpoint, t)
        if varA_w == varA_w:
            Pbias = UI.N * UI.Pbias(finishpoint) * Pallinv[1]
            varAalls[1] += Pbias * Pbias * varA_w
    varA =  vecnorm / sigma_rave
    varA = np.exp(- 0.5 * varA * varA) 
    varA = 0.25 * vecnorm * vecnorm * (
            varAalls[0] + varAalls[1] + 2.0 * varAalls[0] * varA)
    #print("varA = %s"%varA)
    #print("="*40)
    return varA
def calccovlist(UIlist, mfepstep, mfepstepdamp, initialpoint):
    return [0.0, 0.0, mfepstep, 0.0]
    Pallinv    = 1.0 / pUIall(UIlist, initialpoint)
    Dminimum   = 1.0e30
    sigma_rmin = 0.0
    vec        = mfepstepdamp - initialpoint
    #vecnorm    = 0.0
    #for i in range(self.const.dim):
        #vecnorm += vec[i] * vec[i]
    vecnorm = np.sqrt(sum(x * x for x in vec))
    if vecnorm == 0.0:
        return [0.0, 0.0, mfepstep, 0.0]
    #vecnorm = np.sqrt(vecnorm)
    Evec    = vec / vecnorm
    #vecinv  = calcvecinv(Evec)
    varAall = 0.0
    for UI in UIlist:
        t = 0.0
        varA_w, sigma_r = UI.calcvarA_w(Evec, initialpoint, t)
        if varA_w == 0.0:
            continue
        if varA_w == varA_w:
            D = UI.Dbias(initialpoint)
            if D < Dminimum:
                Dminimum   = copy.copy(D)
                sigma_rmin = copy.copy(sigma_r)
            Pbias    = UI.N * UI.Pbias(initialpoint) * Pallinv
            varAall += Pbias * Pbias * varA_w
            #if Pbias * Pbias * varA_w > 1.0:
                #print("P^2 * varA_w(%s) = %s"%(D, Pbias * Pbias * varA_w))
                #print("varA_w(%s) = %s"%(D, varA_w))
    #print("varAall(%s,%s) = %s"%(mfepstep, Dminimum, varAall))
    return [varAall, sigma_rmin, mfepstep, vecnorm]
def calcvarall(covlist):
    varall = 0.0
    for i, [varAall_i, sigma_rmin_i, mfepstep_i, vecnorm_i] in enumerate(covlist):
        #print(varAall_i)
        varAalldamp = 0.0
        if varAall_i == 0.0:
            continue
        if i == 0 or i == len(covlist) - 1:
            c_i = 0.5
        else:
            c_i = 1.0
        for j, [varAall_j, sigma_rmin_j, mfepstep_j, vecnorm_j] in enumerate(covlist):
            if varAall_j == 0.0:
                continue
            if j == 0 or j == len(covlist) - 1:
                c_j = 0.5
            else:
                c_j = 1.0
            if i == j:
                varall      += c_i * c_j * vecnorm_i * vecnorm_j * varAall_i
                varAalldamp += c_i * c_j * vecnorm_i * vecnorm_j * varAall_i
                #if i == 1:
                    #print("(%s, %s)"%(i,j))
                    #print(c_i * c_j * vecnorm_i * vecnorm_j * varAall_i)
            else:
                if self.const.periodicQ:
                    mfepstepJdamp = UIstepCython.periodic(mfepstep_j,
                                self.const.periodicmax, self.const.periodicmin, mfepstep_i, self.const.dim)
                else:
                    mfepstepJdamp = mfepstep_j
                cov_ij  = np.linalg.norm(mfepstep_i - mfepstepJdamp)/sigma_rmin_i 
                cov_ij  = varAall_i * np.exp(-0.5 * cov_ij * cov_ij)
                varall += vecnorm_i * vecnorm_j * c_i * c_j * cov_ij
                varAalldamp += vecnorm_i * vecnorm_j * c_i * c_j * cov_ij
                #if i == 1:
                    #print("(%s, %s)"%(i,j))
                    #print(vecnorm_i * vecnorm_j * c_i * c_j * cov_ij)
        #print("varAalldamp(%s) = %s"%(mfepstep_i, varAalldamp))
    #print("="* 50)
    return varall

def calcvecinv(Evec):
    R = np.identity(self.const.dim)
    e_x = np.array([1.0] + [0.0 for _ in range(self.const.dim - 1)])
    for i in range(1, self.const.dim)[::-1]:
        th = np.arctan2(Evec[i], Evec[i-1])
        Rdamp           =  np.identity(self.const.dim)
        Rdamp[i,   i  ] =  np.cos(th)
        Rdamp[i-1, i-1] =  np.cos(th)
        Rdamp[i  , i-1] = -np.sin(th)
        Rdamp[i-1, i  ] =  np.sin(th)
        R = np.dot(Rdamp, R)
    vecinv = np.dot(np.transpose(e_x), R)
    return vecinv
def gradUIall(UIlist, xi):
    #if ADDstepsigmaTH < Dmin(UIlist, xi):
        #return 1.0e30
    eps = np.sqrt(np.finfo(float).eps)
    if self.const.periodicQ:
        return UIstepCython.cgradUIall_periodic(UIlist, xi, 
                self.const.periodicmax, self.const.periodicmin, self.const.dim, self.const.betainv, eps)
    else:
        return UIstepCython.cgradUIall(UIlist, xi, self.const.dim, self.const.betainv, eps)
def gradUIall_minimum(UIlist, xi, UIeq, beforeAmin):
    vec = xi - UIeq.ave
    vec = vec / np.linalg.norm(vec)
    gradlist = []
    for j in range(200):
        xidamp = xi - vec * j / 200.0
        if UIeq.deltaA_UI(UIeq.ave, xidamp) < beforeAmin:
            if len(gradlist) == 0:
                graddamp = gradUIall(UIlist, xidamp)
                return xidamp, np.dot(graddamp, graddamp)
            return sorted(gradlist, key = lambda x: x[1])[0]
        graddamp = gradUIall(UIlist, xidamp)
        gradlist.append([xidamp, np.dot(graddamp, graddamp)])
    else:
        c = inspect.currentframe()
        print("ERROR(%s): too long distance of windows"%(c.f_lineno))
        functions.TARGZandexit()
def gradUIall_HS(UIlist, thetalist, UIeq, minimumA):
    _grad_HS = np.zeros(self.const.dim - 1)
    nADD     = functions.SperSphere_cartesian(UIeq, minimumA, thetalist, self.const.dim)
    _xi      = UIeq.ave + nADD
    #_thetalist = UIstepCython.SperSphere_polar(UIeq.ave, xi, self.const.dim)
    _grad     = gradUIall(UIlist, _xi)
    for k in range(self.const.dim - 1):
        Del_x = UIstepCython.Del_x(nADD, thetalist, k, self.const.dim)
        #print(Del_x)
        for i in range(self.const.dim):
            _grad_HS[k] += _grad[i] * Del_x[i]
    #print(_grad_HS)
    return _grad_HS
def gradUIall_HS_IOE(UIlist, thetalist, UIeq, minimumA, neiborADDthsL, ADDfe):
    _grad_HS = np.zeros(self.const.dim - 1)
    #_xi      = UIeq.ave + functions.SperSphere_cartesian(UIeq, minimumA, thetalist, self.const.dim)
    nADD     = functions.SperSphere_cartesian(UIeq, minimumA, thetalist, self.const.dim)
    _xi      = UIeq.ave + nADD
    _grad    = gradUIall(UIlist, _xi)
    for k in range(self.const.dim - 1):
        Del_x = UIstepCython.Del_x(_xi, thetalist, k, self.const.dim)
        for i in range(self.const.dim):
            _grad_HS[k] += _grad[i] * Del_x[i]
    for neiborADDth in neiborADDthsL:
        ADDfeM    = neiborADDth.ADDfe
        ADDfeIOEM = neiborADDth.ADDfeIOE
        if ADDfeIOEM > 0.0:
            continue
        if ADDfeM < ADDfe:
            _grad_HS -= gradIOE(UIeq, _xi, thetalist, neiborADDth) 
    return _grad_HS
def gradIOE(UIeq, _xi, thetalist, neiborADDth):
    h = 0.01
    _gradioe = np.zeros(self.const.dim - 1)
    vec = _xi - UIeq.ave
    #for ADDthsL in neiborADDthsL:
    if True:
        delTH = np.zeros(self.const.dim)
        th = functions.angle(neiborADDth.nADD, vec)
        if th > pi * 0.5:
            #continue
            return _gradioe
        for i in range(self.const.dim):
            vec_di = copy.copy(vec)
            vec_di[i] += h
            delTH[i] = (functions.angle(neiborADDth.nADD, vec_di) - th)
        delTH /= h
        for k in range(self.const.dim-1):
            Del_x = UIstepCython.Del_x(_xi, thetalist, k, self.const.dim)
            for i in range(self.const.dim):
                _gradioe[k] += -3.0 * neiborADDth.ADDfe * np.sin(th) * np.cos(th) * np.cos(th) * delTH[i] * Del_x[i]
    return _gradioe
def pUIall(UIlist, xi):
    allp = 0.0
    for UI in UIlist:
        allp += UI.N * UI.Pbias(xi)
    return allp
def Ptotal(UIlist, xi):
    allp = 0.0
    for UI in UIlist:
        allp += UI.Pbias(xi) / UI.Pbias(UI.ave)
    return allp
def Dmin(UIlist, xi):
    #allp = []
    minD = 1.0e30
    for UI in UIlist:
        if UI.stepN < 0:
            print(UI.stepN)
            continue
        #UIp = UI.Pbias(xi) / UI.Pbiasave
        UID = UI.Dbias(xi)
        if UID < minD:
            minD = copy.copy(UID)
    return minD
    #return UIstepCython.cDmin(UIlist, xi, self.const.dim)
def Dmin_theta(UIlist, UIeq, minimumA, thetalist):
    nADD = functions.SperSphere_cartesian(UIeq, minimumA, thetalist, self.const.dim)
    nextpoint = UIeq.ave + nADD
    nextpoint = functions.periodicpoint(nextpoint)
    return Dmin(UIlist, nextpoint)

def HessianUIall(UIlist, xi, const):
    Ndamp = UIlist[0].N
    allP = 0.0
    allDelP = np.zeros(const.dim)
    for UI in UIlist:
        _N = UI.N / Ndamp
        #allp += UI.N * UI.Pbias(xi)
        allP += _N * UI.Pbias(xi)
        #allDelP += UI.N * UI.delPbias(xi)/Ndamp
        allDelP += _N * UI.delPbias(xi)
    hes = np.zeros((const.dim,const.dim))
    for UI in UIlist:
        _N = UI.N / Ndamp
        delNP = np.zeros((const.dim,const.dim))
        #delNP += UI.N * UI.delPbias(xi) - UI.N * UI.Pbias(xi) * allP
        delNP += _N * UI.delPbias(xi) * allP - _N * UI.Pbias(xi) * allDelP
        delNP /= allP * allP
        #hes   += UI.hes * UI.N * UI.Pbias(xi) / allP
        hes   += UI.hes * _N * UI.Pbias(xi) / allP
        hes   += np.dot(UI.grad(xi), delNP)
    for i in range(const.dim):
        for j in range(const.dim):
            if not hes[i,j] == hes[i,j]:
                c = inspect.currentframe()
                print("ERROR(%s): there is nan in hes[%s, %s]"%(c.f_lineno, i, j), flush = True)
    return hes

def reflengthQ(UIlist, nextstep, deltaA_UIeq, nextstepD, const):
    #reflengthM = 999.99
    UIlistdamp = sorted(UIlist, key = lambda x: np.abs(x.A - deltaA_UIeq))
    for UI in UIlistdamp:
        if UI.stepN < 0:
            continue
        nextstepdamp = copy.copy(nextstep)
        nextstepdamp = functions.periodicpoint(nextstepdamp)
        #UIrefdamp    = copy.copy(UI.ref)
        UIrefdamp    = copy.copy(UI.aveT)
        UIrefdamp = functions.periodicpoint(UIrefdamp)
        oldpointQ = [False for i in range(const.dim)]
        for i in range(const.dim):
            reflength = nextstepdamp[i] - UIrefdamp[i]
            if reflength > 0:
                if reflength < nextstepD * 0.9:
                    #break
                    oldpointQ[i] = True
            else:
                if reflength > - nextstepD * 0.9:
                    #return True
                    #break
                    oldpointQ[i] = True
        if all(oldpointQ):
            return True
    return False
def LShisto(histolistlist, histodelta, xi, const):
    LSmat = np.zeros((const.dim + 1, const.dim + 1))
    Zvec  = np.zeros(const.dim + 1)
    for histolist in histolistlist:
        histopoint = np.zeros(self.const.dim)
        for i in range(const.dim):
            histopoint[i] = histolist[i] + 0.5 * histodelta[i]
        LSmat[0, 0] += 1.0
        Zvec[0] += histolist[-1]
        for i in range(const.dim):
            LSmat[0, i + 1] += histopoint[i]
            LSmat[i + 1, 0] += histopoint[i]
            Zvec[i + 1] += histolist[-1] * histopoint[i]
            for j in range(const.dim):
                LSmat[i + 1, j + 1] += histopoint[i] * histopoint[j]
    xvec = np.dot(np.linalg.inv(LSmat), Zvec)
    returnH = xvec[0]
    for i in range(const.dim):
        returnH += xi[i] * xvec[i+1]
    return returnH
def importUIlistall_exclusion(WGraph, UIlistall, const):
#    if usefcntlQ:
#        with open(lockfilepath) as oLockFile:
#            #print('EXCLUSION CONTROL: This job is waitting until it gains the lock file.')
#            fcntl.flock(oLockFile.fileno(), fcntl.LOCK_EX)
#            try:
#                #print('EXCLUSION CONTROL: It gains the lock file.')
#                if UIlistall:
#                    stepNmax = max(UI.stepN for UI in UIlistall)
#                else:
#                    stepNmax = 0
#                if WindowDataType == "hdf5":
#                    dirpath = "step{0:0>5}".format(stepNmax + 1)
#                    importQ = False
#                    with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "r") as windowHDF:
#                        if dirmath in windowHDF:
#                            importQ = True
#                    if importQ:
#                        WGraph, UIlistall = importUIlistall(WGraph, UIlistall)
#                else:
#                    dirpath = "{0}/jobfiles/windows/step{1:0>5}".format(pwdpath, stepNmax + 1)
#                    if os.path.exists(dirpath):
#                        WGraph, UIlistall = importUIlistall(WGraph, UIlistall)
#            finally:
#                fcntl.flock(oLockFile.fileno(), fcntl.LOCK_UN)
#    else:
        #with fasteners.InterProcessLock(lockfilepath_UIlist):
            #pass
        #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
    if True:
        if True:
            if UIlistall:
                stepNmax = max(UI.stepN for UI in UIlistall)
            else:
                stepNmax = 0
                UIlistall = []
            if const.WindowDataType == "hdf5":
                #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
                #dirpath = "step{0:0>5}".format(stepNmax + 1)
                #importQ = False
                #lock.acquire()
                ##with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "r", libver='latest', swmr = True) as windowHDF:
                #with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "r") as windowHDF:
                    #if dirpath in windowHDF:
                        #importQ = True
                #lock.release()
                #if importQ:
                if True:
                    WGraph, UIlistall = importUIlistall(WGraph, UIlistall, const)
            else:
                dirpath = "{0}/jobfiles/windows/step{1:0>5}".format(pwdpath, stepNmax + 1)
                if os.path.exists(dirpath):
                    WGraph, UIlistall = importUIlistall(WGraph, UIlistall)
    return WGraph, UIlistall
def importUIlistall(WGraph, UIlistall, const):
    lock = fasteners.InterProcessLock(const.lockfilepath_UIlist)
    if WGraph is False:
        #with fasteners.InterProcessLock(lockfilepath_UIlist):
        if True:
            #lock.acquire()
            if os.path.exists("%s/jobfiles/windows/network.txt"%const.pwdpath):
                WGraph = nx.read_edgelist("%s/jobfiles/windows/network.txt"%pwdpath, nodetype=int)
            else:
                WGraph = nx.DiGraph()
            #lock.release()
    with open("./UIlistdata.txt", "a")  as wf:
        c = inspect.currentframe()
        wf.write("Debag(%s) %s: start calculation of importUIlistall\n"%(
            c.f_lineno, datetime.datetime.now()))
    mkUIlistallQ = False

    stepNlistbeforeset = set([UI.stepN for UI in UIlistall])
    if len(stepNlistbeforeset) == 0:
        stepNbeforemax = 0
    else:
        stepNbeforemax = max(stepNlistbeforeset)
    #with fasteners.InterProcessLock(lockfilepath_UIlist):
    if True:
        #lock.acquire()
        if const.WindowDataType == "hdf5":
            #if not os.path.exists("%s/jobfiles/windows/windowfile.hdf5"%pwdpath):
                #with open("./UIlistdata.txt", "a")  as wf:
                    #wf.write("Debag(%s) %s: there is not jobfiles/windows/UIlistall*.npz: start mkuilistall\n"%(
                        #c.f_lineno, datetime.datetime.now()))
                #print("in importUIlistall: there is not UIlistall:make it", flush = True)
                #print("lockfilepath_UIlist = %s"%lockfilepath_UIlist, flush = True)
                #WGraph, UIlistall = mkuilistall_hdf5(WGraph)
                #print("end mkuilistall", flush = True)
                #lock.release()
                #return WGraph, UIlistall
            #lock.acquire()
            #with h5py.File("%s//jobfiles/windows/windowfile.hdf5"%pwdpath, "r", libver='latest', swmr = True) as windowHDF:
            UIlistchunkmin = 1
            UIlistchunkmax = copy.copy(const.UIlistchunksize)
            if stepNbeforemax != 0:
                while True:
                    if UIlistchunkmin <= stepNbeforemax <= UIlistchunkmax:
                        break
                    UIlistchunkmin += UIlistchunksize
                    UIlistchunkmax += UIlistchunksize
            stepNmax = stepNbeforemax
            while True:
                UIlistHDFpath = "%s/%s/windows/windowfile%s-%s.hdf5"%(const.pwdpath, const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
                UIlistchunkmin += const.UIlistchunksize
                UIlistchunkmax += const.UIlistchunksize
                UIlistHDFpath_next = "%s/%s/windows/windowfile%s-%s.hdf5"%(const.pwdpath, const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
                if not os.path.exists(UIlistHDFpath):
                    #print("there is not %s: stop"%UIlistHDFpath)
                    break
                if not os.path.exists(UIlistHDFpath_next):
                    lock = fasteners.InterProcessLock(const.lockfilepath_UIlist + str(UIlistchunkmax - const.UIlistchunksize))
                    lock.acquire()
                windowHDF = h5py.File(UIlistHDFpath, "r")
                while True:
                    stepNmax += 1
                    if UIlistchunkmin <= stepNmax:
                        break
                    steppath = "step{0:0>5}".format(stepNmax)
                    if not steppath in windowHDF:
                        break
                    UI = UIstep(const)
                    try:
                        UI.stepN = windowHDF["%s/stepN"%steppath][...]
                    except KeyError:
                        print("KeyError: there is not %s/stepN"%steppath)
                        print("UIlistHDFpath = %s"%UIlistHDFpath)
                        continue
                    UI.stepN = int(UI.stepN)
                    if not UI.stepN in WGraph.nodes():
                        WGraph.add_node(UI.stepN)
                    path    = windowHDF["%s/path"%steppath][...]
                    UI.path = str(path)
                    UI.ave  = windowHDF["%s/ave"%steppath][...]
                    try:
                        UI.aveTinitial = windowHDF["%s/aveTinitial"%steppath][...]
                    except:
                        UI.aveTinitial = copy.copy(aveT)
                    UI.covinv   = windowHDF[  "%s/covinv"%steppath][...]
                    UI.cov_eigN = windowHDF["%s/cov_eigN"%steppath][...]
                    UI.cov_eigV = windowHDF["%s/cov_eigV"%steppath][...]
                    UI.N = windowHDF["%s/N"%steppath][...]
                    UI.ref = windowHDF["%s/ref"%steppath][...]
                    UI.Pbiasave = windowHDF["%s/Pbiasave"%steppath][...]
                    UI.K = windowHDF["%s/K"%steppath][...]
                    UI.A = windowHDF["%s/A"%steppath][...]
                    UI.hes = windowHDF["%s/hes"%steppath][...]
                    UI.aveT = windowHDF["%s/aveT"%steppath][...]
                    try:
                        UI.connections = list(windowHDF["%s/connections"%steppath][...])
                    except:
                        connections = [0]
                    #print(UI.stepN)
                    UIlistall.append(UI)
                windowHDF.close()
                if not os.path.exists(UIlistHDFpath_next):
                    lock.release()

#            with h5py.File("%s/jobfiles/windows/windowfile.hdf5"%pwdpath, "r") as windowHDF:
#                #for steppath in windowHDF:
#                stepNmax = stepNbeforemax
#                while True:
#                    stepNmax += 1
#                    #try:
#                    if True:
#                        steppath = "step{0:0>5}".format(stepNmax)
#                        if not steppath in windowHDF:
#                            break
#                        UI = UIstep()
#                        try:
#                            UI.stepN = windowHDF["%s/stepN"%steppath][...]
#                        except KeyError:
#                            print("KeyError: there is not %s/stepN"%steppath)
#                            continue
#                        UI.stepN = int(UI.stepN)
#                        #print(UI.stepN)
#                        #if newstepN in stepNlistbeforeset:
#                            #continue
#                        if not UI.stepN in WGraph.nodes():
#                            WGraph.add_node(UI.stepN)
#                        #path = windowHDF[ "%s/path"%steppath][...]
#                        #path = str(path)
#                        #UI.importdata_aveonly(path, windowHDF)
#                        path    = windowHDF["%s/path"%steppath][...]
#                        UI.path = str(path)
#                        UI.ave  = windowHDF["%s/ave"%steppath][...]
#                        try:
#                            UI.aveTinitial = windowHDF["%s/aveTinitial"%steppath][...]
#                        except:
#                            UI.aveTinitial = copy.copy(aveT)
#                        UI.covinv   = windowHDF[  "%s/covinv"%steppath][...]
#                        UI.cov_eigN = windowHDF["%s/cov_eigN"%steppath][...]
#                        UI.cov_eigV = windowHDF["%s/cov_eigV"%steppath][...]
#                        try:
#                            UI.connections = list(windowHDF["%s/connections"%steppath][...])
#                        except:
#                            connections = [0]
#                        UIlistall.append(UI)
#                    #except KeyError:
#                        #continue
#                        #break
#            lock.release()
            with open("./UIlistdata.txt", "a")  as wf:
                c = inspect.currentframe()
                wf.write("Debag(%s) %s: end   calculation of importUIlistall\n"%(
                    c.f_lineno, datetime.datetime.now()))
            return WGraph, UIlistall
        else:
            UIlistalllist = glob.glob("%s/jobfiles/windows/UIlistall*.npz"%pwdpath)
            #if not os.path.exists("%s/jobfiles/windows/UIlistall.npz"%pwdpath):
            if len(UIlistalllist) == 0:
                with open("./UIlistdata.txt", "a")  as wf:
                    wf.write("Debag(%s) %s: there is not jobfiles/windows/UIlistall*.npz: start mkuilistall\n"%(
                        c.f_lineno, datetime.datetime.now()))
                print("in importUIlistall: there is not UIlistall:make it", flush = True)
                print("lockfilepath_UIlist = %s"%lockfilepath_UIlist, flush = True)
                WGraph, UIlistall = mkuilistall(WGraph)
                print("end mkuilistall", flush = True)
                lock.release()
                return WGraph, UIlistall

            #lock.release()
            importwhileN = 0
            importQ = True
            UIlistchunkmax = 0
            firstQ = True
            UIlistallpathlist = []
            while True:
                UIlistchunkmax += 1000
                UIlistallpath = "%s/jobfiles/windows/UIlistall%s-%s.npz"%(pwdpath, UIlistchunkmax - 999, UIlistchunkmax)
                #if UIlistchunkmax - 999 < len(stepNlistbeforeset):
                    #continue
                if not os.path.exists(UIlistallpath):
                    #print("there is not %s"%UIlistallpath)
                    break
                UIlistallpathlist.append(UIlistallpath)
            for UIlistallpath in UIlistallpathlist:
                #lockchunk = fasteners.InterProcessLock(lockfilepath_UIlist + str(UIlistchunkmax))
                importwhileN += 1
                #lockchunk.acquire()
                lock.acquire()
                importnpz = np.load(UIlistallpath)
                #if UIlistchunkmax == 1000:
                if firstQ:
                    stepNlist      = importnpz["stepN"]
                    avelist        = importnpz["ave"  ]
                    importpathlist = importnpz["path"]
                else:
                    stepNlist       = np.r_[stepNlist, importnpz["stepN"]]
                    avelist         = np.r_[avelist,   importnpz["ave"]]
                    importpathlist  = np.r_[importpathlist, importnpz["path"]]
                try:
                    aveTinitiallistdamp =     importnpz["aveTinitial"]
                except:
                    aveTinitiallistdamp = importnpz["ave"]
                if firstQ:
                    aveTinitiallist = copy.copy(aveTinitiallistdamp)
                    covinvlist      = importnpz["covinv"     ]
                    cov_eigNlist    = importnpz["cov_eigN"   ]
                    cov_eigVlist    = importnpz["cov_eigV"   ]
                    connectionslist = importnpz["connections"]
                    connectionslist = list(connectionslist)
                    firstQ          = False
                else:
                    aveTinitiallist     = np.r_[aveTinitiallist, aveTinitiallistdamp]
                    covinvlist          = np.r_[covinvlist,      importnpz["covinv"]] 
                    cov_eigNlist        = np.r_[cov_eigNlist,    importnpz["cov_eigN"]]
                    cov_eigVlist        = np.r_[cov_eigVlist,    importnpz["cov_eigV"]]
                    #connectionslist    = np.r_[connectionslist, importnpz["connections"]]
                    connectionslistdamp =  importnpz["connections"]
                    connectionslistdamp = list(connectionslistdamp)
                    connectionslist.extend(connectionslistdamp)
                #lockchunk.release()
                stepNlistset = set(stepNlist)
            lock.release()


    if importQ is False:
        print("Fatal ERROR!!!: windows/UIlistall.npz cannnot be read!\n ***EXIT***", flush = True)
        exit()
    if mkUIlistallQ:
        return WGraph, UIlistall
    with open("./UIlistdata.txt", "a")  as wf:
        c = inspect.currentframe()
        wf.write("Debag(%s) %s: UIlistall.npz was read.\n"%(
            c.f_lineno, datetime.datetime.now()))
    #stepNlist  = importnpz["stepN"]
    #stepNlistset = set(stepNlist)
    #pathlist     =     importnpz["path"    ]
    #avelist      =     importnpz["ave"     ]
    #try:
        #aveTinitiallist  =     importnpz["aveTinitial"]
    #except:
        #aveTinitiallist = copy.copy(avelist)
    #covinvlist   =     importnpz["covinv"  ]
    #cov_eigNlist =     importnpz["cov_eigN"]
    #cov_eigVlist =     importnpz["cov_eigV"]
#    try:
#        connectionslist =     importnpz["connections"]
#    except:
#        c = inspect.currentframe()
#        with open("./UIlistdata.txt", "a")  as wf:
#            wf.write("Debag(%s) %s: connectionslist cannot be imported.\n"%(
#                c.f_lineno, datetime.datetime.now()))
#        connectionslist = [[] for _ in range(len(stepNlist))]
    #stepNlistbeforeset = set([UI.stepN for UI in UIlistall])
    pathlist = []
    #for path in importnpz["path"]:
    for path in importpathlist:
        if not pwdpath in path:
            path = path.split("/")[-1]
            path = "%s/jobfiles/windows/%s"%(pwdpath, path)
        pathlist.append(path)
    for stepN in stepNlistset:
        if not stepN in WGraph.nodes():
            WGraph.add_node(stepN)
    if len(connectionslist) != len(stepNlist):
        print("len(stepNlist)       = %s"%len(stepNlist))
        print("len(connectionslist) = %s"%len(connectionslist))
        connectionslist = [list(nx.all_neighbors(WGraph, stepN)) for stepN in stepNlist]
        #connectionslist = [[] + x for x in connectionslist]
    addconnectionQ = False
    newNlist       = list(stepNlistset - stepNlistbeforeset)
    if newNlist:
        with open("./UIlistdata.txt", "a")  as wf:
            c = inspect.currentframe()
            wf.write("Debag(%s) %s: len(newNlist) = %s.\n"%(
                c.f_lineno, datetime.datetime.now(), len(newNlist)))
    changeconnectionlistQ = False
    for newN in newNlist:
        if newN is None:
            break
        UI = UIstep()
        for i, beforestepN in enumerate(stepNlist):
            if beforestepN == newN:
                UI.stepN       = int(newN)
                UI.path        = str(            pathlist[i])
                UI.ave         = np.array(        avelist[i])
                UI.aveTinitial = np.array(aveTinitiallist[i])
                UI.covinv      = np.array(     covinvlist[i])
                UI.cov_eigN    = np.array(   cov_eigNlist[i])
                UI.cov_eigV    = np.array(   cov_eigVlist[i])
                #UI.connections = [0]
                UI.connections = connectionslist[i]
                UI.npzNumber   = copy.copy(i)
                break
        else:
            print("ERROR: cannot found newN = %s"%newN)
            eixt()
        #if len(stepNlist) != len(connectionslist):
            #print("len(stepNlist)       = %s"%len(stepNlist))
            #print("len(connectionslist) = %s"%len(connectionslist))
#        for i, beforestepN in enumerate(stepNlist):
#            if beforestepN == newN:
#                if len(connectionslist) < i + 1:
#                    connectionslist = list(connectionslist)
#                    connectionslist.append([])
#                else:
#                    UI.connections = list(connectionslist[i])
#                break
        if len(UI.connections) == 0:
            changeconnectionlistQ = True
            c = inspect.currentframe()
            with open("./UIlistdata.txt", "a")  as wf:
                wf.write("Debag(%s) %s: start addWindowNetwork on window %s\n"%(
                    c.f_lineno, datetime.datetime.now(), UI.stepN))
            addconnectionQ = True
            WGraph, UI, UIlistall = IOpack.addWindowNetwork(WGraph, UI, UIlistall)
            for i, beforestepN in enumerate(stepNlist):
                if beforestepN == newN:
                    connectionslist[i] = copy.copy(UI.connections)
                    break
            for nearUI in UIlistall:
                if nearUI.stepN in UI.connections:
                    if not UI.stepN in nearUI.connections:
                        nearUI.connections.append(UI.stepN)
                        for i, beforestepN in enumerate(stepNlist):
                            if beforestepN == nearUI.stepN:
                                connectionslist[i] = copy.copy(nearUI.connections)
                                break
        #else:
            #WGraph = IOpack.importWindowNetwork(WGraph, UI)
        #print(WGraph.nodes())
        UIlistall.append(UI)
    #with open("./UIlistdata.txt", "a")  as wf:
        #wf.write("Debag(%s) %s: chk the connections of UIlistall %s\n"%(
            #c.f_lineno, datetime.datetime.now(), UI.stepN))
    #if len(newNlist) != 0:
    if changeconnectionlistQ:
        for UI in UIlistall:
            UI.connections = copy.copy(connectionslist[UI.npzNumber])
            #for i, stepN in enumerate(stepNlist):
                #if UI.stepN == stepN:
                    #UI.connections = copy.copy(connectionslist[i])
                    #break
    #if addconnectionQ:
    if False:
        #for UI in UIlistall:
            #if len(UI.connections) == 1:
                #WGraph, UI = IOpack.addWindowNetwork(WGraph, UI, UIlistall)

        connectionslist = np.array(connectionslist)
        np.savez("%s/jobfiles/windows/UIlistall.npz"%pwdpath,
                path         = pathlist,
                stepN        = stepNlist,
                ave          = avelist,
                aveTinitial  = aveTinitiallist,
                covinv       = covinvlist,
                cov_eigN     = cov_eigNlist,
                cov_eigV     = cov_eigVlist,
                connections  = connectionslist,
                )
    with open("./UIlistdata.txt", "a")  as wf:
        c = inspect.currentframe()
        wf.write("Debag(%s) %s: end   calculation of importUIlistall\n"%(
            c.f_lineno, datetime.datetime.now()))
    return WGraph, UIlistall
def mkuilistall_hdf5(WGraph):
    print("Fatal ERROR: thre is not windowfile.hdf5: all of window data is removed!")
    TARGZandexit()
def mkuilistall(WGraph):
    #os.environ["OMP_NUM_THREADS"] = "1"
    if WGraph is False:
        WGraph = nx.DiGraph()
    UIlistall = []
    #UIlistall_chunk = []
    firstQ = True
    #i = 0
    #for line in open("%s/jobfiles/windows/pathlist.dat"%pwdpath):
    #for path in glob.glob( "%s/jobfiles/windows/step*"%pwdpath):
        #print(path, flush = True)
        #i += 1
        #if line[0] == "#":
            #continue
        #path = line.split()[-1]
        #path = path.replace("\n","")
        #path = path.split("/")[-1]
        #path = "%s/jobfiles/windows/%s"%(pwdpath, path)
    #for i in range(len(glob.glob("%s/jobfiles/windows/step*"%pwdpath))):
        #i += 1
        #path =  "{0}/jobfiles/windows/step{1:0>5}".format(pwdpath, i)
    #for i in range(len(glob.glob("%s/jobfiles/windows/step*"%pwdpath))):
    i = 0
    windowN = 0
    windowNmax = len(glob.glob("%s/jobfiles/windows/step*"%pwdpath))
    print("windowNmax = %s"%windowNmax, flush = True)
    savezClist = []
    while True:
        i += 1
        path =  "{0}/jobfiles/windows/step{1:0>5}".format(pwdpath, i)
        if not os.path.exists("%s/UIstep.npz"%path):
            print("mkuilistall: %s: ERROR: there is not UIstep.npz in path = %s"%(
                datetime.datetime.now(), path), flush = True)
            continue
        windowN += 1
        UI = UIstep()
        UI.importdata_aveonly(path)
        UIlistall.append(UI)
        #UIlistall_chunk.append(UI)
        UI.stepN = copy.copy(i)
        WGraph.add_node(UI.stepN)
        if windowNmax <= windowN:
            break
    #p.map(lambda UI: IOpack.addWindowNetwork(WGraph, UI, UIlistall), UIlistall)
    #parallellist = [(WGraph, UI, UIlistall) for UI in UIlistall]
    parallellist = []
    for UI in UIlistall:
        if not os.path.exists("%s/network.txt"%UI.path):
            parallellist.append((WGraph, UI, UIlistall))
    if len(parallellist) != 0:
        p = mp.Pool(32)
        p.map(IOpack.addWindowNetwork_parallel, parallellist)
        p.close()
    for UI in UIlistall:
        WGraph, UI, UIlistall = IOpack.addWindowNetwork(WGraph, UI, UIlistall)
    #WGraph, UIlistall = IOpack.mkWindowNetwork(UIlistall)
    if not os.path.exists("%s/jobfiles/windows/network.txt"%pwdpath):
        writeline = ""
        for UI in UIlistall:
            #print(UI.connections)
            for connectionN in UI.connections:
                if UI.stepN < connectionN:
                    writeline += "%s %s\n"%(UI.stepN, connectionN)
        with open("%s/jobfiles/windows/network.txt"%pwdpath, "w") as wf:
            wf.write(writeline)
    #exit()
    UIlistchunkmax = 0
    UIlistall = sorted(UIlistall, key = lambda UI: UI.stepN)
    while True:
        UIlistchunkmax += 1000
        UIlistall_chunk = UIlistall[UIlistchunkmax - 1000: UIlistchunkmax]
        stepNlist       = np.array([UI.stepN       for UI in UIlistall_chunk])
        pathlist        = np.array([UI.path        for UI in UIlistall_chunk])
        avelist         = np.array([UI.ave         for UI in UIlistall_chunk])
        aveTinitiallist = np.array([UI.aveTinitial for UI in UIlistall_chunk])
        covinvlist      = np.array([UI.covinv      for UI in UIlistall_chunk])
        cov_eigNlist    = np.array([UI.cov_eigN    for UI in UIlistall_chunk])
        cov_eigVlist    = np.array([UI.cov_eigV    for UI in UIlistall_chunk])
        connectionslist = np.array([UI.connections for UI in UIlistall_chunk])

        np.savez("%s/jobfiles/windows/UIlistall%s-%s.npz"%(pwdpath, UIlistchunkmax - 999, UIlistchunkmax),
                        path         = pathlist,
                        stepN        = stepNlist,
                        ave          = avelist,
                        aveTinitial  = aveTinitiallist,
                        covinv       = covinvlist,
                        cov_eigN     = cov_eigNlist,
                        cov_eigV     = cov_eigVlist,
                        connections  = connectionslist,
                        )
        if not os.path.exists(lockfilepath_UIlist + str(UIlistchunkmax)):
            with open(lockfilepath_UIlist + str(UIlistchunkmax), "w") as wf:
                wf.write("")
        if windowNmax <= UIlistchunkmax:
            break


#        if firstQ:
#            stepNlist       = np.array([UI.stepN       ])
#            pathlist        = np.array([UI.path        ])
#            avelist         = np.array([UI.ave         ])
#            aveTinitiallist = np.array([UI.aveTinitial ])
#            covinvlist      = np.array([UI.covinv      ])
#            cov_eigNlist    = np.array([UI.cov_eigN    ])
#            cov_eigVlist    = np.array([UI.cov_eigV    ])
#            #connectionslist = np.array([UI.connections ])
#            #connectionslist = [UI.connections ]
#            firstQ = False
#            #continue
#        else:
#            stepNlist       = np.r_[stepNlist,       np.array([UI.stepN       ])]
#            pathlist        = np.r_[pathlist,        np.array([UI.path        ])]
#            avelist         = np.r_[avelist,         np.array([UI.ave         ])]
#            aveTinitiallist = np.r_[aveTinitiallist, np.array([UI.aveTinitial ])]
#            covinvlist      = np.r_[covinvlist,      np.array([UI.covinv      ])]
#            cov_eigNlist    = np.r_[cov_eigNlist,    np.array([UI.cov_eigN    ])]
#            cov_eigVlist    = np.r_[cov_eigVlist,    np.array([UI.cov_eigV    ])]
#        #connectionslist = np.r_[connectionslist, np.array([UI.connections ])]
#        print("%s, %s"%(windowN, path), flush = True)
#        if windowN == UIlistchunkmax or windowNmax <= windowN:
#            for UI in UIlistall_chunk:
#                WGraph, UI, UIlistall = IOpack.addWindowNetwork(WGraph, UI, UIlistall)
#                connectionslist.append(UI.connections)
#            connectionslist = np.array(connectionslist)
#            np.savez("%s/jobfiles/windows/UIlistall%s-%s.npz"%(pwdpath, UIlistchunkmax - 999, UIlistchunkmax),
#                        path         = pathlist,
#                        stepN        = stepNlist,
#                        ave          = avelist,
#                        aveTinitial  = aveTinitiallist,
#                        covinv       = covinvlist,
#                        cov_eigN     = cov_eigNlist,
#                        cov_eigV     = cov_eigVlist,
#                        connections  = connectionslist,
#                        )
#            firstQ = True
#            if not os.path.exists(lockfilepath_UIlist + str(UIlistchunkmax)):
#                with open(lockfilepath_UIlist + str(UIlistchunkmax), "w") as wf:
#                    wf.write("")
#            UIlistchunkmax += 1000
#            UIlistall_chunk = []
#        if windowNmax <= windowN:
#            break



#    connectionslist = []
#    #if 1000 <  len(UIlistall):
#    #if False:
#    if 1000 <  len(stepNlist):
#        networkpathlist = []
#        if os.path.exists("%s/jobfiles/windows/networkpath.txt"%pwdpath):
#            for line in open("%s/jobfiles/windows/networkpath.txt"%pwdpath):
#                networkpathlist.append(line.replace("\n",""))
#        parallellists = []
#        for UI in UIlistall:
#            netpath = UI.path.split("jobfiles")[-1]
#            netpath = "%s/%s"%(pwdpath, netpath)
#            if not netpath in networkpathlist:
#                parallellists.append((WGraph, UI, UIlistall, networkpathlist))
#
#
#        #p = mp.Pool(parallelPy)
#        #parallellists = [(WGraph, UI, UIlistall, networkpathlist) for UI in UIlistall]
#        #UIlistalldamp = p.map(parallelcalcnetwork, parallellists)
#        #p.close()
#        UIlistalldamp = map(parallelcalcnetwork, parallellists)
#        #for UI in UIlistall:
#            #connectionslist.append(UI.connections)
#    #else:
#    if True:
#        for UI in UIlistall:
#            WGraph, UI, UIlistall = IOpack.addWindowNetwork(WGraph, UI, UIlistall)
#            connectionslist.append(UI.connections)
#    connectionslist = np.array(connectionslist)
#    np.savez("%s/jobfiles/windows/UIlistall.npz"%pwdpath,
#                path         = pathlist,
#                stepN        = stepNlist,
#                ave          = avelist,
#                aveTinitial  = aveTinitiallist,
#                covinv       = covinvlist,
#                cov_eigN     = cov_eigNlist,
#                cov_eigV     = cov_eigVlist,
#                connections  = connectionslist,
#                )
    return WGraph, UIlistall
#def parallelcalcnetwork(parallellist):
#    WGraph, UI, UIlistall = IOpack.addWindowNetwork(*parallellist)
#    return UI
#
def exportUI_exclusion(UI, const, WGraph = False, UIlistall_damp = False):
    #IOpack.chklockfilepath()
    if const.usefcntlQ:
    #if True:
        with open(const.lockfilepath) as oLockFile:
            fcntl.flock(oLockFile.fileno(), fcntl.LOCK_EX)
            try:
                UI, WGraph = exportUI(UI, UIlistall_damp, WGraph, const)
            finally:
                fcntl.flock(oLockFile.fileno(), fcntl.LOCK_UN)
    else:
        #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
        #with fasteners.InterProcessLock(lockfilepath_UIlist):
        if True:
            #lock.acquire()
            if const.WindowDataType == "hdf5":
                UI, WGraph = exportUI_hdf5(UI, UIlistall_damp, WGraph,const)
            else:
                UI, WGraph = exportUI(UI, UIlistall_damp, WGraph,const)
            #lock.release()
    return UI, WGraph
def exportUI_hdf5(UI, UIlistall_damp, WGraph,const):
    exportwhleN = 0
    while exportwhleN < 1000:
        exportwhleN += 1
        num = 0
        dirkind = "step"
        UIlistchunkmin = 1
        UIlistchunkmax = const.UIlistchunksize
        #UIlistHDFpath       = "%s/jobfiles/windows/windowfile%s-%s.hdf5"%(pwdpath, UIlistchunkmin, UIlistchunkmax)
        UIlistHDFpathbefore = "%s/%s/windows/windowfile%s-%s.hdf5"%(const.pwdpath, const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
        if not os.path.exists(const.lockfilepath_UIlist + str(UIlistchunkmax)):
            with open(const.lockfilepath_UIlist + str(UIlistchunkmax), "w") as wf:
                wf.write("")
        lock = fasteners.InterProcessLock(const.lockfilepath_UIlist + str(UIlistchunkmax))
        lock.acquire()
        if not os.path.exists(UIlistHDFpathbefore):
            num = 1
            dirpath = "{0}{1:0>5}".format(dirkind, num)
            dirfullpath = "%s/%s/windows/%s"%(const.pwdpath, const.jobfilepath, dirpath)
            UIlistHDFpath = UIlistHDFpathbefore
            windowHDF     = h5py.File(UIlistHDFpath, "w")
            windowHDF.create_group(dirpath)
            windowHDF.flush()
            windowHDF.close()
            lock.release()
        else:
            lock.release()
            while True:
                UIlistchunkmin += const.UIlistchunksize
                UIlistchunkmax += const.UIlistchunksize
                UIlistHDFpath = "%s/%s/windows/windowfile%s-%s.hdf5"%(const.pwdpath, const.jobfilepath, UIlistchunkmin, UIlistchunkmax)
                if not os.path.exists(const.lockfilepath_UIlist + str(UIlistchunkmax)):
                    with open(const.lockfilepath_UIlist + str(UIlistchunkmax), "w") as wf:
                        wf.write("")
                lock = fasteners.InterProcessLock(const.lockfilepath_UIlist + str(UIlistchunkmax))
                lock.acquire()
                if not os.path.exists(UIlistHDFpath):
                    break
                UIlistHDFpathbefore = copy.copy(UIlistHDFpath)
                lock.release()
            lock_before = fasteners.InterProcessLock(const.lockfilepath_UIlist + str(UIlistchunkmin - 1))
            lock_before.acquire()
            windowHDF = h5py.File(UIlistHDFpathbefore, "r")
            dirpath   = "{0}{1:0>5}".format(dirkind, UIlistchunkmin - 1)
            if dirpath in windowHDF:
                #print("in %s: found %s: make next HDF"%(UIlistHDFpathbefore, dirpath))
                windowHDF.close()
                dirpath = "{0}{1:0>5}".format(dirkind, UIlistchunkmin)
                dirfullpath = "%s/jobfiles/windows/%s"%(pwdpath, dirpath)
                windowHDF = h5py.File(UIlistHDFpath, "w")
                windowHDF.create_group(dirpath)
                windowHDF.flush()
                windowHDF.close()
                num = UIlistchunkmin
            else:
                #print("in %s: not find %s"%(UIlistHDFpathbefore, dirpath))
                UIlistchunkmin -= const.UIlistchunksize
                UIlistchunkmax -= const.UIlistchunksize
                num = UIlistchunkmin - 1
                while True:
                    num += 1
                    dirpath     = "{0}{1:0>5}".format(dirkind, num)
                    if not dirpath in windowHDF:
                        dirfullpath = "%s/%s/windows/%s"%(const.pwdpath, const.jobfilepath, dirpath)
                        #windowHDF.create_group(dirpath)
                        break
                windowHDF.close()
                UIlistHDFpath   = copy.copy(UIlistHDFpathbefore)
            lock.release()
            lock_before.release()
        if not os.path.exists(const.lockfilepath_UIlist + str(UIlistchunkmax)):
            with open(cosnt.lockfilepath_UIlist + str(UIlistchunkmax), "w") as wf:
                wf.write("")
        #lock = fasteners.InterProcessLock(lockfilepath_UIlist + "_hdf5")
        lock = fasteners.InterProcessLock(const.lockfilepath_UIlist + str(UIlistchunkmax))
        lock.acquire()
        windowHDF = h5py.File(UIlistHDFpath, "a")
        windowHDF.flush()
        windowHDF.close()
        lock.release()
        UI.stepN = copy.copy(num)
        UI.path = copy.copy(dirfullpath)
        if UIlistall_damp:
            WGraph, UI, UIlistall_damp = IOpack.addWindowNetwork(WGraph, UI, UIlistall_damp, const)
        EQgro = open("%s/%s"%(UI.calculationpath, const.grofilename)).read()
        exportQ = UI.exportdata(dirfullpath, equibliumstructure = EQgro)
        if exportQ:
            return UI, WGraph
        else:
            print("wait 1 second")
            time.sleep(1.0)
    print("error: turn over 1000: cannot exportUI for hdf5!=>exit")
    exit()
def exportUI(UI, UIlistall_damp, WGraph, const):
    num = 0
    dirkind = "step"
    while True:
        num += 1
        dirpath = "{0}/jobfiles/windows/{1}{2:0>5}".format(const.pwdpath, dirkind, num)
        if False:
        #if not os.path.exists(dirpath):
            try:
                os.mkdir(dirpath)
            except FileExistsError:
                print('FileExistsError: there is %s'%dirpath, flush = True)
                continue
            break
    #addUInetwork = []
    #if UIlistall_damp:
        #for nearUI in UIlistall_damp:
            #if UI.Dbias(UInear.ave) < neighborwindowTH or UInear.:
    UIlistchunkmax = 0
    while True:
        UIlistchunkmax += 1000
        if UIlistchunkmax - 999 <= num <= UIlistchunkmax:
            break
    UIlistallpath = "%s/jobfiles/windows/UIlistall%s-%s.npz"%(pwdpath, UIlistchunkmax - 999, UIlistchunkmax)
    if not os.path.exists(lockfilepath_UIlist + str(UIlistchunkmax)):
        with open(lockfilepath_UIlist + str(UIlistchunkmax), "w") as wf:
            wf.write("")
    #lockchunk = fasteners.InterProcessLock(lockfilepath_UIlist + str(UIlistchunkmax))
    lock = fasteners.InterProcessLock(lockfilepath_UIlist)
    UI.stepN = copy.copy(num)
    UI.path = copy.copy(dirpath)
    if UIlistall_damp:
        WGraph, UI, UIlistall_damp = IOpack.addWindowNetwork(WGraph, UI, UIlistall_damp)
    lock.acquire()
    with open("%s/jobfiles/windows/pathlist.dat"%pwdpath,"a") as wf:
        wf.write("%s -> %s\n"%(UI.calculationpath, UI.path))
    shutil.copy("%s/%s"%(UI.calculationpath, const.grofilename), "%s/%s"%(UI.path, const.grofilename))
    UI.exportdata(dirpath)
    #lockchunk.acquire()
    if not os.path.exists(UIlistallpath):
        print("in exportUI: there is not %s:make it"%UIlistallpath, flush = True)
        #UIlistall = mkuilistall(WGraph)
        stepNlist       = np.array([UI.stepN       ])
        pathlist        = np.array([UI.path        ])
        avelist         = np.array([UI.ave         ])
        aveTinitiallist = np.array([UI.aveTinitial ])
        covinvlist      = np.array([UI.covinv      ])
        cov_eigNlist    = np.array([UI.cov_eigN    ])
        cov_eigVlist    = np.array([UI.cov_eigV    ])
        #connectionslist = np.r_[importnpz["connections"],    np.array([UI.connections  ])]
        connectionslist = []
        connectionslist.append(UI.connections)
        connectionslist = np.array(connectionslist)
    else:
        #importnpz = np.load("%s/jobfiles/windows/UIlistall.npz"%pwdpath)
        importnpz = np.load(UIlistallpath)
        stepNlist       = np.r_[importnpz["stepN"],       np.array([UI.stepN       ])]
        pathlist        = np.r_[importnpz["path"],        np.array([UI.path        ])]
        avelist         = np.r_[importnpz["ave"],         np.array([UI.ave         ])]
        aveTinitiallist = np.r_[importnpz["aveTinitial"], np.array([UI.aveTinitial ])]
        covinvlist      = np.r_[importnpz["covinv"],      np.array([UI.covinv      ])]
        cov_eigNlist    = np.r_[importnpz["cov_eigN"],    np.array([UI.cov_eigN    ])]
        cov_eigVlist    = np.r_[importnpz["cov_eigV"],    np.array([UI.cov_eigV    ])]
        #connectionslist = np.r_[importnpz["connections"],    np.array([UI.connections  ])]
        connectionslist = list(importnpz["connections"])
        connectionslist.append(UI.connections)
        connectionslist = np.array(connectionslist)
    #np.savez("%s/jobfiles/windows/UIlistall.npz"%pwdpath,
    np.savez(UIlistallpath,
                    path         = pathlist,
                    stepN        = stepNlist,
                    ave          = avelist,
                    aveTinitial  = aveTinitiallist,
                    covinv       = covinvlist,
                    cov_eigN     = cov_eigNlist,
                    cov_eigV     = cov_eigVlist,
                    connections  = connectionslist,
                    )
    #lockchunk.release()
    lock.release()
    return UI, WGraph
def is_in_eqlist(eqlist, avepoint, eqsteppath):
    for eqpoint in eqlist:
        pathlist = eqpoint[0].split("/")
        samepathQ = False
        for path in pathlist:
            if "EQ" in path:
                if path in eqsteppath:
                    samepathQ = True
                break
        if samepathQ:
            continue
        path = "%s/jobfiles/%s"%(pwdpath, eqpoint[0])
        UI   = UIstep()
        UI.importdata(path)
        if partADDQ:
            if not os.path.exists("%s/UIstep_ALL.npz"%path):
                UI.ref   = np.zeros(const.dim)
                UI.K     = np.zeros((const.dim,self.const.dim))
                i = -1
                for line in open("%s/plumed.dat"%path):
                    if line[0] == "#":
                        continue
                    if "restraint" in line:
                        if "BIASVALUE" in line:
                            continue
                        i += 1
                        line = line.split()
                        if i < const.dim:
                            UI.K[i,i] = float(line[3].replace("KAPPA=",""))
                            UI.ref[i] = float(line[4].replace("AT=",""))
                colverlist = glob.glob("%s/COLVAR.*"%path) \
                                    + glob.glob("%s/COLVAR"%path)
                if len(colverlist) != 0:
                    UI.readdat(colverlist[0], maxtimehere = maxtimeEQ)
                    UI.aveALL = UI.ave
            if np.linalg.norm(avepoint - UI.aveALL) == 0.0:
                continue
        else:
            if np.linalg.norm(avepoint - UI.ave) == 0.0:
                continue
        Dbias_eqtarget = UI.Dbias(avepoint)
        if Dbias_eqtarget < ADDstepsigmaTH:
            optedpath       = "%s/jobfiles/%s"%(pwdpath, eqpoint[0])
            c = inspect.currentframe()
            with open("./optlist.dat","a") as wf:
                wf.write("Debag(%s): %s is same with %s\n"%(
                    c.f_lineno, eqsteppath, optedpath))
            return True, UI
    return False, None
#def is_in_tslist(tslist, avepoint, tssteppath):
def is_in_tslist(tslist, avepoint):
    for tspoint in tslist:
        if len(tspoint) == 0:
            continue
        pathlist      = tspoint[0].split("/")
        if isinstance(tspoint[-1], float):
            beforeADDname = "calculatedMeta"
        else:
            beforeADDname = tspoint[-1].replace("\n", "")
        samepathQ = False
#        for path in pathlist:
#            if "TS" in path:
#                if path in tssteppath:
#                    samepathQ = True
#                break
#        if samepathQ:
#            continue
        path = "%s/jobfiles/%s"%(pwdpath, tspoint[0])
        UI   = UIstep()
        UI.importdata(path)
        if np.linalg.norm(avepoint - UI.ave) == 0.0:
            continue
        Dbias_target = UI.Dbias(avepoint)
        if Dbias_target < ADDstepsigmaTH:
            optedpath = "%s/jobfiles/%s"%(pwdpath, tspoint[0])
            #c = inspect.currentframe()
            #with open("./optlist.dat","a") as wf:
                #wf.write("Debag(%s): %s is same with %s\n"%(
                    #c.f_lineno, tssteppath, optedpath))
            #for name in tspoint[0].split():
            for name in pathlist:
                if "TS" in name:
                    UI.tsN = int(name.replace("TS", ""))
                    break

            return True, UI, beforeADDname
    return False, None, None
def multihess(UIlist, UI):
    if len(UI.ave) == const.dim:
        UIdamp = UI
    else:
        UIdamp = UIstep()
        UIdamp.importdata(UI.path)
        if len(UIdamp.ave) != const.dim:
            UIdamp.ave    = UIdamp.aveALL
            UIdamp.ref    = UIdamp.refALL
            UIdamp.cov    = UIdamp.covALL
            UIdamp.covinv = UIdamp.covinvALL
            UIdamp.hes    = UIdamp.hesALL
            UIdamp.hesinv = UIdamp.hesinvALL
            UIdamp.K      = UIdamp.KALL
    UIlistdamp = []
    for UIbefore in UIlist:
        if UIbefore.Dbias(UIdamp.ave) < neighborwindowTH:
            UIlistdamp.append(UIbefore)
    #UIlistdamp = UIlist
    #c = inspect.currentframe()
    #print("Debag(%s): len(UIlistdamp) = %s"%(c.f_lineno, len(UIlistdamp)), flush = True)
    UIdamp.hes         = HessianUIall(UIlistdamp, UIdamp.ave)
    UIdamp.hesinv      = np.linalg.inv(UIdamp.hes)
    UIdamp.covinv      = (UIdamp.hes + UIdamp.K) / const.betainv
    UIdamp.cov         = np.linalg.inv(UIdamp.covinv)
    UIdamp.Pbiasave    = Pbiasconst / np.sqrt(np.linalg.det(UIdamp.cov))
    UIdamp.eigN, _eigV = np.linalg.eigh(UIdamp.hes)
    UIdamp.eigV = []
    for i in range(const.dim):
        UIdamp.eigV.append(_eigV[:,i])
    UIdamp.eigV = np.array(UIdamp.eigV)
    UIdamp.cov_eigN, _cov_eigV = np.linalg.eigh(UIdamp.cov)
    UIdamp.cov_eigV = []
    for i in range(const.dim):
        UIdamp.cov_eigV.append(_cov_eigV[:,i])
    UIdamp.cov_eigV = np.array(UIdamp.cov_eigV)
    if len(UI.ave) == const.dim:
        UI = UIdamp
    else:
        UI = functions.mkUIeqshort(UIdamp)
    return UI
def getUIlistSurface(UIeq, UIlist, IOEsphereA):
    UIlistSurface = []
    for UI in UIlist:
        if const.periodicQ:
            productlist = []
            for i, x in enumerate(UI.ave):
                if x < const.periodicmax[i] + self.const.periodicmin[i]:
                    productlist.append([0.0, const.periodicmax[i] - self.const.periodicmin[i]])
                else:
                    productlist.append([0.0, const.periodicmin[i] - self.const.periodicmax[i]])
        else:
            periodiclist = [[0.0] for _ in range(const.dim)]
        SurfaceQ = False
        for periodiclist in itertools.product(*productlist):
            if chkSurface(UIeq, UI, IOEsphereA, periodiclist):
                #UIlistSurface.append(UI)
                #continue
                SurfaceQ = True
                break
        if SurfaceQ:
            UIlistSurface.append(UI)
    return UIlistSurface
def chkSurface(UIeq, UI, IOEsphereA, periodiclist):
    surfacesigma = 3.0
    vec  = UI.ave - UIeq.ave
    if np.linalg.norm(vec) == 0.0:
        vec = UIeq.eigV[0]
    else:
        vec  = vec / np.linalg.norm(vec)
    thetalist = calctheta(vec)
    if UIeq.Dbias(UI.ave) < surfacesigma:
        deltaAmin = 0.0
    else:
        result = minimize(lambda x:
            UI.deltaA_sigma(UIeq, thetalist + x, surfacesigma, periodiclist),
            x0 = np.zeros(const.dim - 1),
            method = "L-BFGS-B")
        deltaAmin1 = UI.deltaA_sigma(UIeq, thetalist + result.x, surfacesigma, periodiclist)
    result = minimize(lambda x:
            -UI.deltaA_sigma(UIeq, thetalist + x, surfacesigma, periodiclist),
            x0 = np.zeros(const.dim - 1),
            method = "L-BFGS-B")
    deltaAmax1 =  UI.deltaA_sigma(UIeq, thetalist + result.x, surfacesigma, periodiclist)

    thetalist = calctheta(-vec)
    if not UIeq.Dbias(UI.ave) < surfacesigma:
        result = minimize(lambda x:
            UI.deltaA_sigma(UIeq, thetalist + x, surfacesigma, periodiclist),
            x0 = np.zeros(const.dim - 1),
            method = "L-BFGS-B")
        deltaAmin2 =  UI.deltaA_sigma(UIeq, thetalist + result.x, surfacesigma, periodiclist)
        if deltaAmin1 < deltaAmin2:
            deltaAmin = deltaAmin1
        else:
            deltaAmin = deltaAmin2

    result = minimize(lambda x:
            -UI.deltaA_sigma(UIeq, thetalist + x, surfacesigma, periodiclist),
            x0 = np.zeros(const.dim - 1),
            method = "L-BFGS-B")
    deltaAmax2 =  UI.deltaA_sigma(UIeq, thetalist + result.x, surfacesigma, periodiclist)
    if deltaAmax1 < deltaAmax2:
        deltaAmax = deltaAmax2
    else:
        deltaAmax = deltaAmax1
    if deltaAmin <= IOEsphereA <= deltaAmax:
        return True
    else:
        return False
def random_hyper_sphere(pointN, const, positiveQ = False):
    returnlist = []
    turnN = 0
    while True:
        if turnN >= pointN:
            #print("now StopIteration")
            #raise StopIteration
            return False
        point = UIstepCython.random_hyper_sphereC(const.dim)
        if positiveQ:
            for i, x in enumerate(point):
                if x < 0:
                    point[i] = -x
            if min(point) < 0:
                continue
        turnN += 1
        yield point
def calctheta(eigV):
    if eigV is False:
        return False
    eigVdim  = len(eigV)
    eigVdamp = list(eigV)
    eigVdamp.reverse()
    _thetalist = [1.0 for _ in range(eigVdim - 1)]

    r = eigV[0] * eigV[0] + eigV[1] * eigV[1]
    if r == 0.0:
        #eigV[1] += 1.0e-4
        #r = eigV[0] * eigV[0] + eigV[1] * eigV[1]
        _thetalist[0] = 0.0
    else:
    #if True:
        _thetalist[0] = np.arccos(eigV[1] / np.sqrt(r))
    if eigV[0] < 0:
        _thetalist[0] = np.pi * 2.0 - _thetalist[0]
    for i in range(1, eigVdim - 1):
        r += eigV[i + 1] * eigV[i + 1]
        if r == 0.0:
            #eigV[i + 1] += 1.0e-4
            #r += eigV[i + 1] * eigV[i + 1]
            _thetalist[i] = 0.0
        else:
        #if True:
            _thetalist[i] = np.arccos(eigV[i + 1] / np.sqrt(r))
    _thetalist.reverse()
    return np.array(_thetalist)
