#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2018 mitsutay13 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, re
import time, copy, inspect
import fcntl, datetime
import subprocess as sp
import numpy      as np
from   scipy.optimize import minimize
import h5py

#from .const       import *
from . import functions
from . import IOpack
from . import umbrellaINT as UIint

import pyximport  ## for cython
pyximport.install()
try:
    from . import UIstepCython
except ImportError:
    import UIstepCython
include_dirs = [np.get_include()]

import fasteners

class JobClass():
    pass
def chkneedwindow(parallelC):
    #print("start chkneedwindow")
    UIeq                = parallelC.UIeq
    minimumA            = parallelC.minimumA
    UIlist              = parallelC.UIlist
    UIlistdamp          = parallelC.UIlistdamp
    UIlist_initial      = parallelC.UIlist_initial
    WGraph              = parallelC.WGraph
    UIlistall           = parallelC.UIlistall
    edgeN               = parallelC.edgeN
    edgelist            = parallelC.edgelist
    endlist             = parallelC.endlist
    #nextsteps_global    = parallelC.nextsteps_global
    endsteps_global     = parallelC.endsteps_global
    errorsteps_global   = parallelC.errorsteps_global
    processturnN_global = parallelC.processturnN_global
    mpID                = parallelC.mpID
    finishpoint         = parallelC.finishpoint
    initialpoint        = None
    const = parallelC.const
    #if const.allperiodicQ:
        #finishpointdamp = UIstepCython.periodic(finishpoint,
                    #const.periodicmax, const.periodicmin, initialpoint, const.dim)
    #else:
        #finishpointdamp = finishpoint
    #nADD_needwindow = finishpointdamp - initialpoint
    whileN = -1
    runNglobal_exclusion(const,parallelC.runN_global, whileN + 1, +1)
    while True:
        whileN += 1
        if 100 < whileN:
            with open("./UIlistdata.txt", "a")  as wf:
                c = inspect.currentframe()
                wf.write("ERROR(%s): whileN over 100.\n"%(c.f_lineno))
            calcQ_global = False
            runNglobal_exclusion(const,parallelC.runN_global, whileN, -1)
            return False
        if const.edgeNmax < edgeN:
            with open("./UIlistdata.txt", "a")  as wf:
                c = inspect.currentframe()
                wf.write("ERROR(%s): edgeN over edgeNmax(%s).\n"%(c.f_lineno, const.edgeNmax))
            calcQ_global = False
            runNglobal_exclusion(const,parallelC.runN_global, whileN, -1)
            return False
        for UI_g in parallelC.UIlist_global:
            for UI in UIlist:
                #if UI.stepN == UI_g.stepN:
                #if np.linalg.norm(UI.stepN - UI_g.stepN) == 0.0:
                #if np.linalg.norm(UI.ref - UI_g.ref) == 0.0:
                if UI.path == UI_g.path:
                    break
            else:
                UIlist.append(UI_g)
                UIlistdamp.append(UI_g)


        if parallelC.forcecallwindowQ:
            with open("./UIlistdata.txt", "a")  as wf:
                wf.write("in forcecallwindow; whileN = %s\n"%(whileN))
            if whileN != 0:
                #print("accept 93")
                return UIlistdamp
            acceptQ = False
            needwindowpoint = copy.copy(finishpoint)
            Dmin_minimum = 1.0e30
            UIbefore = False
            for UI in UIlist:
                #_Dbias = UI.Dbias(needwindowpoint)
                _Dbias = UI.Dbias(finishpoint)
                if _Dbias < Dmin_minimum:
                    Dmin_minimum = copy.copy(_Dbias)
                    UIbefore     = copy.copy(UI)
        else:
            acceptQ, needwindowpoint, UIlistdamp, UIbefore = getneedwindow(
                const,UIlist, UIlistdamp, initialpoint, finishpoint)
        #print("needwindowpoint = %s"%needwindowpoint)
        #print(UIbefore.ave)
        if len(needwindowpoint) == 0:
            with open("./UIlistdata.txt", "a") as wf:
                c = inspect.currentframe()
                wf.write("ERROR(%s): getneedwindow\n"%(c.f_lineno))
            runNglobal_exclusion(const,parallelC.runN_global, whileN, -1)
            calcQ_global = False
            return False
        if acceptQ:
            runNglobal_exclusion(const,parallelC.runN_global, whileN, -1)
            #return UIlist
            #print("accept 119")
            return UIlistdamp
            #break
#        with open("./UIlistdata.txt", "a")  as wf:
#            c = inspect.currentframe()
#            wf.write("Debag(%s):(whileN, len(UIlistdamp)) = (%s, %s)\n"%(
#                c.f_lineno, whileN, len(UIlistdamp)))

        errorlist = []
        errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                "#(reaction coordinate...)\n")
        is_in_endlist = chkerrorlist(const,parallelC.UIlist_global, needwindowpoint, errorlist)
        if is_in_endlist:
            with open("./UIlistdata.txt", "a")  as wf:
                c = inspect.currentframe()
                wf.write("Debag(%s): rm by errorlist\n"%(c.f_lineno))
            #return False
            runNglobal_exclusion(const,parallelC.runN_global, whileN, -1)
            #return UIlist
            return False
        try:
            UIbefore
        except:
            with open("./UIlistdata.txt", "a")  as wf:
                c = inspect.currentframe()
                wf.write("ERROR(%s):there is not UIbefore\n"%(c.f_lineno))
                wf.write("UIlist = %s\n"%(UIlist))
            runNglobal_exclusion(const,parallelC.runN_global, whileN, -1)
            return False
        lastpath = os.getcwd().split("/")[-1]
        if "edge" in lastpath:
            lockfilesearchpath = "../"
        else:
            lockfilesearchpath = "./"

        nextsteps = [(needwindowpoint, UIbefore)]
        addQ      = True
        edgeADDQ  = True
        os.chdir(lockfilesearchpath)
        with open("./pwdpath.dat", "w") as wf:
            wf.write(const.pwdpath)
        errorsteplist, UIlist, WGraph, UIlistall = calc_edgewindows(
            const,edgeN, WGraph, UIlistall, UIlist, UIeq, nextsteps, finishpoint,parallelC.forcecallwindowQ) 
        runNglobal_exclusion(const,parallelC.runN_global, whileN,     -1)
        runNglobal_exclusion(const,parallelC.runN_global, whileN + 1, +1)
        waitN = 0
        while True:
            if parallelC.runN_global[whileN]  <= 0:
                break
            waitN += 1
            if const.calltime < waitN:
                c = inspect.currentframe()
                with open("./UIlistdata.txt", "a")  as wf:
                    wf.write("Debag(%s): waittime over calltime = %s\n"%(c.f_lineno,const.calltime))
                parallelC.runN_global[whileN] = 0
                runNglobal_exclusion(const,parallelC.runN_global, whileN + 1, -1)
                return False
            time.sleep(1.0)
            print("wait %s: runN = %s"%(waitN,parallelC.runN_global[whileN]))
        if UIlist is False:
            runNglobal_exclusion(const,parallelC.runN_global, whileN + 1, -1)
            return False
        #UIlist = list(parallelC.UIlist_global)
        if errorsteplist:
            for errorstep in errorsteplist:
                errorstepstr = ""
                for x in errorstep:
                    errorstepstr += " % 4.2f"%x
                with open("./UIlistdata.txt", "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s): (%s) cannot calculated next steps and will be removed it.\n"%(c.f_lineno, errorstepstr))
                errorlist = []
                errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                "#path  (reaction coordinate...)  A\n")
                newerrorlist = []
                adderrorlistQ = True
                #for errorpoint in errorlist:
                    #e = np.array(errorpoint[1:-1])
                    #e = np.array(errorpoint)
                    #if np.linalg.norm(e - errorstep) < 0.001:
                        #adderrorlistQ = False
                        #break
                if adderrorlistQ:
                    newerrorlist.append(errorstep)
                errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, newerrorlist,
                "#path  (reaction coordinate...)  A\n")
            with open("./UIlistdata.txt", "a")  as wf:
                wf.write("mpID = %s: ERROR: error window\n"%(mpID))
            calcQ_global = False
            ADDfe = 0.0
            #return ADDfe
            runNglobal_exclusion(const,parallelC.runN_global, whileN + 1, -1)
            return False
    #return UIlist
def getneedwindow(const,UIlist, UIlistdamp, initialpoint, finishpoint):
    #print("start getneedwindow")
    if not all(x == x for x in finishpoint):
        with open("./UIlistdata.txt", "a")  as wf:
            c = inspect.currentframe()
            wf.write("ERROR(%s): there is nan in finishpoint(%s).\n"%(c.f_lineno, finishpoint))
        return False, [], UIlistdamp, None
    for i in range(len(finishpoint)):
        wallQ = False
        if const.wallmax[i] < finishpoint[i]:
            wallQ = True
        elif finishpoint[i] < const.wallmin[i]:
            wallQ = True
        if wallQ:
            with open("./UIlistdata.txt", "a")  as wf:
                c = inspect.currentframe()
                wf.write("ERROR(%s): finishpoint(%s) is out of wall.\n"%(c.f_lineno, finishpoint))
            return False, [], UIlistdamp, None
    needwindowpoint = copy.copy(finishpoint)
    whileN = 0
    while True:
        if whileN > 1000:
            print("ERROR in getneedwindow")
            #exit()
            return False, [], UIlistdamp, None
        needwindowQ = True
        Dmin_minimum = 1.0e30
        UIbefore = False
        for UI in UIlist:
            #_Dbias = UI.Dbias(needwindowpoint)
            _Dbias = UI.Dbias(finishpoint)
            if _Dbias < Dmin_minimum:
                Dmin_minimum = copy.copy(_Dbias)
                UIbefore     = copy.copy(UI)
            if _Dbias < const.ADDstepsigmaTH:
                if const.nearestWindowQ:
                    #print("line 319")
                    #print("_Dbias = ",_Dbias)
                    acceptQ = True
                    return acceptQ, needwindowpoint, UIlistdamp, UIbefore
            if UI.stepN in [UIdamp.stepN for UIdamp in UIlistdamp]:
                continue
            #if UI.Dbias(needwindowpoint) < const.ADDstepsigmaTH:
            if UI.Dbias(finishpoint) < const.ADDstepsigmaTH:
                UIlistdamp.append(UI)
                needwindowQ = False
                if const.nearestWindowQ:
                    #print("line 329")
                    #print("_Dbias = ",UI.Dbias(finishpoint))
                    acceptQ = True
                    return acceptQ, needwindowpoint, UIlistdamp, UIbefore
        if chkintegration(const,UIlist,UIbefore,finishpoint):
            #print("_Dbias = ",UIbifore.Dbias(finishpoint))
            acceptQ = True
            print("line 265: accept")
            return acceptQ, needwindowpoint, UIlistdamp, UIbefore
        #with open("./UIlistdata.txt", "a")  as wf:
            #wf.write("Dmin(UIlist, finishpoint) = %s\n"%Dmin(UIlist, finishpoint))
        if const.nearestWindowQ:
            if UIbefore is False:
                with open("./UIlistdata.txt", "a")  as wf:
                    wf.write("ERROR: cannot found UIbefore\n")
                    wf.write("len(UIlist) = %s\n"%len(UIlist))
                print("ERROR: cannot found UIbefore", flush = True)
                print("len(UIlist) = %s"%len(UIlist), flush = True)
                print("finishpoint = %s"%finishpoint, flush = True)
                for UI in UIlist:
                    print("UI.Dbias = %s"%UI.Dbias(finishpoint), flush = True)
                UIbefore = copy.copy(UIlist[0])
            initialpoint = copy.copy(UIbefore.ave)
            initialpoint = functions.periodicpoint(initialpoint,const)
        if const.allperiodicQ:
            finishpointdamp = UIstepCython.periodic(finishpoint,
                    const.periodicmax, const.periodicmin, initialpoint, const.dim)
        else:
            finishpointdamp = finishpoint
        nADD = finishpointdamp - initialpoint
        #print("nADD = %s"%nADD)
        nADD = nADD/np.linalg.norm(nADD)
        sig = UIbefore.calcsigma(nADD,const.nextstepsigmaminTH)
        #print("sig = %s"%sig)
        nADD *= sig
        #print("nADD = %s"%nADD)

        print("len(UIlist) = ",len(UIlist))
        acceptQ, needwindowpoint = UIstepCython.getneedwindowC(UIlist, UIlist, UIbefore,
            initialpoint, finishpoint, nADD,
            const.ADDstepsigmaTH, const.minimizeTH, const.nextstepsigmaminTH,
            const.allperiodicQ, const.periodicmin, const.periodicmax, const.dim)

        if acceptQ:
            errorlist = []
            errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                    "#(reaction coordinate...)\n")
            is_in_endlist = chkerrorlist(const,UIlist, needwindowpoint, errorlist)
            if is_in_endlist:
                with open("./UIlistdata.txt", "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s): point %s is removed because of errorlist.dat\n"%(c.f_lineno, needwindowpoint))
                needwindowpoint = []
            print("line 304: accept")
            return acceptQ, needwindowpoint, UIlistdamp, UIbefore
        else:
            if needwindowQ:
                return acceptQ, needwindowpoint, UIlistdamp, UIbefore
def runNglobal_exclusion(const,runN_global, whileN, i):
    """
    set exclusion control
    export data list to file(fname)
    """

    if const.usefcntlQ:
        with open(lockfilepath) as oLockFile:
            fcntl.flock(oLockFile.fileno(), fcntl.LOCK_EX)
            try:
                if len(runN_global) < whileN + 1:
                    runN_global.append(0)
                runN_global[whileN] += i
            finally:
                fcntl.flock(oLockFile.fileno(), fcntl.LOCK_UN)
    else:
        #with fasteners.InterProcessLock(lockfilepath):
        if True:
            #lock = fasteners.InterProcessLock(lockfilepath)
            #lock.acquire()
            if len(runN_global) < whileN + 1:
                runN_global.append(0)
            runN_global[whileN] += i
            #lock.release()
def calc_edgewindows(const,edgeN, WGraph, UIlistall, UIlist, UIeq, nextsteps, finishpoint, forcecallwindowQ):
    with open("./UIlistdata.txt", "a")  as wf:
        wf.write("%s: start calc_edgewindows on edge%s\n"%(datetime.datetime.now(), edgeN))
    #dirname       = "edge{0:0>4}".format(edgeN)
    dirname       = "%s/jobfiles/windows"%const.pwdpath
    errorsteplist = []
    if len(nextsteps) == 0:
        return errorsteplist, UIlist, WGraph, UIlistall
    #print(os.getcwd())
    #print(dirname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    #os.chdir(dirname) ## $tmp/jobfiles/[EQ/TI]/edge$edgeN
    with open("./pwdpath.dat", "w") as wf:
        wf.write(const.pwdpath)
    WGraph, UIlistall = UIint.importUIlistall_exclusion(const,WGraph, UIlistall)

    lenUIlistbefore = len(UIlist)

    UIlistpathset = set(UI.path for UI in UIlist)
    nextstepsdamp = []
    UIlist_append  = []
    for newUI in UIlistall:
        if newUI.path in UIlistpathset:
            continue
        if newUI.Dbias(finishpoint) < const.integratesigmaTH:
            print("newUI.Dbias(finishpoint) = ",newUI.Dbias(finishpoint))
            UIlist_append.append((newUI, finishpoint, newUI))
            UIlistpathset.add(newUI.path)
            addwindowQ = True
    #print("len(UIlistall) = %s"%len(UIlistall), flush = True )
    for nextstep, UIbefore in nextsteps:
        Dmin = 1.0e30
        addwindowQ = False
        UInearest = False
        for newUI in UIlistall:
            if newUI.path in UIlistpathset:
                continue
            #if newUI.Dbias(np.array(nextstep)) < neighborsigmaTH:
            #if newUI.Dbias(nextstep) < neighborsigmaTH:
            #if newUI.Dbias(nextstep) < const.ADDstepsigmaTH:
            dist = newUI.Dbias(nextstep)
            if dist < Dmin:
                Dmin = dist
                UInearest = newUI
            if newUI.Dbias(nextstep) < const.integratesigmaTH:
                print("newUI.Dbias(nextstep) = ",newUI.Dbias(nextstep))
                newUI.importdata(newUI.path)
                #print("DEBAG: newUI.Dbias(nextstep) = ",newUI.Dbias(nextstep))
                if const.nearestWindowQ:
                    UIlist_append.append((newUI, nextstep, UIbefore))
                    UIlistpathset.add(newUI.path)
                    addwindowQ = True
                else:
                    UIlist.append(newUI)
                    UIlistpathset.add(newUI.path)
                    addwindowQ = True
        if addwindowQ is False:
            if not UInearest is False and UInearest.path in UIlistpathset:
                if const.nearestWindowQ:
                    UIlist_append.append((UInearest, nextstep, UIbefore))
                    UIlistpathset.add(UInearest.path)
                    addwindowQ = True
                else:
                    UIlist.append(UInearest)
                    UIlistpathset.add(UInearest.path)
                    addwindowQ = True

                #break
        #if len(needwindowpoint) == 0:
            #break
        #else:
        if forcecallwindowQ:
            nextstepsdamp.append((nextstep, UIbefore))
            break
        if addwindowQ is False:
            nextstepsdamp.append((nextstep, UIbefore))
    with open("./UIlistdata.txt", "a")  as wf:
        wf.write("%s:len(UIlist_append) = %s\n"%(datetime.datetime.now(), len(UIlist_append)))
        wf.write("%s:len(nextstepsdamp) = %s\n"%(datetime.datetime.now(), len(nextstepsdamp)))
    falseUIlist = []
    aveTlist = []
    #print("%s:len(UIlist_append) = %s\n"%(datetime.datetime.now(), len(UIlist_append)))
    if len(UIlist_append) != 0:
        if const.nearestWindowQ:
            beforeUIlistlen = len(UIlist_append) + 1
            while len(UIlist_append) - beforeUIlistlen != 0:
                beforeUIlistlen = len(UIlist_append)
                UIlist_appenddamp = copy.copy(UIlist_append)
                UIlist_append     = []
                nextstepsdamp     = []
                addUIQ = False
                aveTlist = []
                falseUIlist = []
                for newUI, nextstep, UIbefore in UIlist_appenddamp:
                    #print(newUI.path)
                    UIlistpathset = set(UI.path for UI in UIlist)
                    if newUI.path in UIlistpathset:
                        continue

                    #A_nearW, varA, UInear = UIint.calcUIall_nearW(const, UIlist, newUI.ave, UIself = newUI)
                    #newUI.A = copy.copy(A_nearW)
                    #print("newUI.A = ",newUI.A)
                    if newUI.A:
                        UIlistpathset = set(UI.path for UI in UIlist)
                        if newUI.path in UIlistpathset:
                            continue
                        with open("./addUIdata.txt", "a")  as wf:
                            c = inspect.currentframe()
                            wf.write(
        "Debag(%s):(newUI.A, Dmin, newUI.aveTinitial = (% 4.2f, %4.2f, %s)\n"%(
            c.f_lineno, newUI.A, UIint.Dmin(UIlist, newUI.ave), newUI.aveTinitial))
                        addUIQ = True
                        UIlist.append(newUI)
                    else:
                        with open("./addUIdata.txt", "a")  as wf:
                            c = inspect.currentframe()
                            wf.write(
        "Debag(%s):(newUI.A, Dmin, newUI.aveTinitial = (False, %4.2f, %s)\n"%(
            c.f_lineno, UIint.Dmin(UIlist, newUI.ave), newUI.aveTinitial))
                            wf.write("newUI.Dbias(nextstep) = %s\n"%newUI.Dbias(nextstep))
                        #addUIlistQ = False
                        addUIlistQ, UIlist = UIint.calcUIall_nearW_connection(UIlist, WGraph, UIlistall,  newUI)
                        #newUI.A = copy.copy(A_nearW)
                        if addUIlistQ:
                            addUIQ = True
                        else:
                            UIlist_append.append((newUI, nextstep, UIbefore))
                            for nextstepbefore, UIbefore in nextstepsdamp:
                                if np.allclose(nextstepbefore, nextstep):
                                    break
                            else:
                                nextstepsdamp.append([nextstep, UIbefore])
#                                addUIlistQ = True
                if addUIQ:
                    with open("./UIlistdata.txt", "a")  as wf:
                        wf.write("len(UIlist) = %s\n"%len(UIlist))
                else:
                    break
    with open("./UIlistdata.txt", "a")  as wf:
        wf.write("%s: len(UIlist) %s -> %s\n"%(datetime.datetime.now(), lenUIlistbefore, len(UIlist)))
        wf.write("%s:len(nextstepsdamp) = %s\n"%(datetime.datetime.now(), len(nextstepsdamp)))
    nextsteps = []
    for nextstep, UIbefore in nextstepsdamp:
        with open("./UIlistdata.txt", "a")  as wf:
            wf.write("Dmin(UIlist, nextstep) = %s\n"%UIint.Dmin(UIlist, nextstep))
        if forcecallwindowQ or const.ADDstepsigmaTH < UIint.Dmin(UIlist, nextstep):
            nextsteps.append([nextstep, UIbefore])
            break
    nextstepsdamp = []

    is_in_errorlist = False
    for i in range(len(nextsteps)):
        if forcecallwindowQ:
            nextstepsdamp.append(nextsteps[i])
            continue
        nextstepsinitial = copy.copy(nextsteps[i][0])
        there_is_aveT = False
        whileN = 0
        delta = 0.5
        while whileN < 5:
            strpoint = ""
            for x in nextsteps[i][0]:
                strpoint += "% 3.2f,"%x
            with open("./UIlistdata.txt", "a")  as wf:
                wf.write("whileN(%s): nextstep = %s\n"%(whileN, strpoint))
            whileN += 1
            there_is_aveT = False
            #if 2 <= whileN :
            if True:
                errorlist = []
                errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                    "#(reaction coordinate...)\n")
                needwindowpoint = nextsteps[i][0]
                is_in_errorlist = chkerrorlist(const,UIlist, needwindowpoint, errorlist)
                if is_in_errorlist:
                    with open("./UIlistdata.txt", "a")  as wf:
                        c = inspect.currentframe()
                        wf.write("Debag(%s): rm by errorlist\n"%(c.f_lineno))
                    errorsteplist.append(nextstepsinitial)
                    errorsteplist.append(needwindowpoint)
                    whileN = 999
                    break
            for nearUI in UIlistall:
                #if np.linalg.norm(nextsteps[i][0] - nearUI.aveTinitial) == 0.0:
                if np.allclose(nextsteps[i][0], nearUI.aveTinitial):
                    with open("./UIlistdata.txt", "a")  as wf:
                        wf.write("A calculated point is found!\n")
                    UIbefore = nextsteps[i][1]
                    if const.allperiodicQ:
                        ave = UIstepCython.periodic(UIbefore.ave,
                            const.periodicmax, const.periodicmin, nextsteps[i][0], const.dim)
                    else:
                        ave = UIbefore.ave
                    #if whileN == 1:
                    if True:
                        vec             = nextstepsinitial - ave
                    #else:
                        #vec             =  - ave
                    nextsteps[i][0] = ave + vec * delta
                    nextsteps[i][0] = functions.periodicpoint(nextsteps[i][0],const)
                    delta  *= 0.5
                    there_is_aveT = True
                    break
            #else:
            if there_is_aveT is False:
                break
            if np.linalg.norm(vec) == 0.0:
                break

        if is_in_errorlist is False:
            if whileN < 5:
                nextstepsdamp.append(nextsteps[i])
            else:
                with open("./UIlistdata.txt", "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Error%s): whieN over 5: the step is added to errorsteplist.\n"%c.f_lineno)
                errorsteplist.append( nextstepsinitial)
    nextsteps = []
    for nextstep, UIbefore in nextstepsdamp:
        Dmin_minimum = 1.0e30
        for UI in UIlist:
            _Dbias = UI.Dbias(nextstep)
            if _Dbias < Dmin_minimum:
                Dmin_minimum = copy.copy(_Dbias)
                UIbefore = copy.copy(UI)
        #with open("./UIlistdata.txt", "a")  as wf:
            #wf.write("UIbefore.Dbias(nextstep) = %s\n"%UIbefore.Dbias(nextstep))
        nextsteps.append([nextstep, UIbefore])

    #copy.copy(nextstepsdamp)

    #if addwindowQ:
    if is_in_errorlist:
        with open("./UIlistdata.txt", "a")  as wf:
            c = inspect.currentframe()
            wf.write("Debag(%s): rm by errorlist.\n"%c.f_lineno)
        return errorsteplist, UIlist, WGraph, UIlistall
    elif len(nextsteps) == 0:
        with open("./UIlistdata.txt", "a")  as wf:
            c = inspect.currentframe()
            wf.write("Debag(%s): windows are added from UIlistall.\n"%c.f_lineno)
        return errorsteplist, UIlist, WGraph, UIlistall
    stepNpathlist = glob.glob("step*")
    if stepNpathlist:
        turnN = max(int(stepNpath.replace("step","")) for stepNpath in stepNpathlist)
    else:
        turnN = 1
    jobNdic        = {}
    for nextstep, UIbefore in nextsteps:
        UIbefore.recallN = 0
        if nextstep is False:
            c = inspect.currentframe()
            print("ERROR({}): nextstep is False".format(c.f_lineno))
            functions.TARGZandexit()

    UIbeforedic_stepN  = {}
    periodicQdic_stepN = {}
    nextset        = set()
    nextlist       = []
    currentdic     = {}
    periodicQdic   = {}
    Kdic           = {}
    newrefdic      = {}
    UIbeforedic    = {}
    currentendlist = []
    parallellists  = []
    currentstep    = 99999
    currentstepN   = 99999
    readKQ         = False
    refthresholdchune = copy.copy(const.refthreshold)
    const.Kminchune         = copy.copy(const.Kmin)
    joblistall   = []
    for nextstep, UIbefore in nextsteps:
        jobC          = JobClass()
        jobC.nextstep = copy.copy(nextstep)
        jobC.aveTinitial = copy.copy(nextstep)
        setpoint        = nextstep
        setpointdamp    = copy.copy(nextstep)
        setpoint        = functions.periodicpoint(setpoint,const)
        jobC.UIbefore   = copy.copy(UIbefore)
        jobC.UIbefore_initial   = copy.copy(UIbefore)
        setpoint        = tuple(setpoint)
        currentnextsetD = copy.copy(setpoint)
        currentdicD     = copy.copy(currentstep)
        KparallelQ = True
        #jobC.K = jobC.UIbefore.K
        #jobC.K = const.Kmin * np.identity(const.dim)
        jobC.K = const.Kminchune * np.identity(const.dim)
        joblistall.append(jobC)
    while joblistall:
        joblist = []
        parallelMD = 1
        for parallelMD in sorted(const.parallelMDs)[::-1]:
            if parallelMD <= len(joblistall):
                break
        for _ in range(parallelMD):
            joblist.append(joblistall.pop(0))
        callstepN = len(joblist)
        print("callstepN =",callstepN)
        i = 0
        for j in range(len(joblist)):
            while True:
                i += 1
                stepdirname = "%s/jobfiles/windows/step%s"%(const.pwdpath, i)
                if not os.path.exists(stepdirname):
                    os.mkdir(stepdirname)
                    joblist[j].stepdirname = copy.copy(stepdirname)
                    joblist[j].stepN       = copy.copy(i)
                    break
        recallturnN  = 0
        refthresholdchunedelta = refthresholdchune * 0.5
        nextstepD = const.nextstepsigmaminTH
        stepsize_kappa = const.stepsize_kappa
        stepsize_cv    = const.stepsize_cv
        #jobC.newref = UIbefore.ref
        jobC.newref = jobC.nextstep
        while recallturnN < const.recallturnmax:
            joblistEquil = []
            recallturnN += 1
            callstepNdamp = 0
            dirnames = ""
            for jobC in joblist:
                #print(jobC)
                os.chdir(jobC.stepdirname)
                UIbefore     = jobC.UIbefore
                currentstep  = UIbefore.stepN
                if mkplumed_opt(const,jobC.nextstep, jobC.K, jobC.newref, UIbefore, stepsize_cv, stepsize_kappa):
                    dirnames += " step%s"%jobC.stepN
                    callstepNdamp += 1
                os.chdir("../")
            #try:
            if True:
                #if moveQ:
                if const.shellname == "csh":
                    sp.call(["%s/calljob.csh"%const.pwdpath, str(callstepNdamp), dirnames, "equiblium", str(const.parallelPy)],
                        timeout=const.calltime)
                elif const.shellname == "bash":
                    sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepNdamp), dirnames, "equiblium", str(const.parallelPy)],
                        timeout=const.calltime)
                else:
                    c = inspect.currentframe()
                    print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, const.shellname), flush = True)

            #except:
            else:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno, const.calltime))
            for jobC in joblist:
                dirname = jobC.stepdirname
                COL_kappa          = "%s/COLVAR_kappa"%dirname 
                if not os.path.exists(COL_kappa):
                    c = inspect.currentframe()
                    print("ERROR(%s in callwindows.py): %s cannot be found."%(c.f_lineno, COL_kappa), flush = True)
                    exit()
                for line in open(COL_kappa):
                    pass
                jobC.K = np.zeros((const.dim, const.dim))
                kappalist = line.split()[1:]
                errorstepQ = False
                for i in range(const.dim):
                    for j in range(const.dim):
                        try:
                            jobC.K[i][j]= float(kappalist[i*const.dim+j])
                        except:
                            errorstepQ = True
                            break
                COL_cntr = "%s/COLVAR_cntr"%dirname 
                if not os.path.exists(COL_cntr):
                    c = inspect.currentframe()
                    print("ERROR(%s in callwindows.py): %s cannot be found."%(c.f_lineno, COL_cntr), flush = True)
                    exit()
                for line in open(COL_cntr):
                    pass
                reflist = line.split()[1:]
                jobC.newref = np.array(reflist, dtype=float)
                jobC.newref = functions.periodicpoint(jobC.newref,const)
                COL_mean          = "%s/COLVAR_mean"%dirname 
                if not os.path.exists(COL_mean):
                    c = inspect.currentframe()
                    print("ERROR(%s in callwindows.py): %s cannot be found."%(c.f_lineno, COL_mean), flush = True)
                    exit()
                for line in open(COL_mean):
                    pass
                meanpoint = np.array(line.split()[1:], dtype=float)
                meanpoint = functions.periodicpoint(meanpoint,const)
                Dmean = UIbefore.Dbias(meanpoint) 
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s) %s: step%s: UIbefore.Dbias(meanpoint) = % 3.2f\n"%(
                        c.f_lineno, datetime.datetime.now(), jobC.stepN, Dmean))
                #if  const.nextstepsigmamaxTH <  Dmean:
                if False:
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        wf.write("Debag(%s) UIbefore.Dbias(meanpoint) > nextstepsigmamaxTH; retry.\n"%(c.f_lineno))
                    recalldirpath = "%s/recall%s"%(dirname,recallturnN)
                    os.mkdir(recalldirpath)
                    for fname in glob.glob("%s/*"%dirname):
                        if "recall" in fname:
                            continue
                        shutil.move(fname, recalldirpath)
                    nptgro = "%s/run.gro"%recalldirpath
                    shutil.copy(nptgro, "%s/min.gro"%dirname)
                    #stepsize_kappa *= 10
                    continue
                joblistEquil.append(jobC)
            if len(joblistEquil) == 0:
                continue
            
            callstepNdamp = 0
            dirnames = ""
            for jobC in joblistEquil:
                os.chdir(jobC.stepdirname)
                mkplumed(const,jobC.newref, currentstep, jobC.K, UIbefore)
                os.chdir("../")
                dirnames += " step%s"%jobC.stepN
                callstepNdamp += 1
            #try:
            if True:
                if const.shellname == "csh":
                    sp.call(["%s/calljob.csh"%const.tmppath, str(callstepNdamp), dirnames, "", str(const.parallelPy)],
                        timeout=const.calltime)
                elif const.shellname == "bash":
                    sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepNdamp), dirnames, "def", str(const.parallelPy)],
                        timeout=const.calltime)
                else:
                    c = inspect.currentframe()
                    print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
            #except:
            else:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno, const.calltime))
    
            for jobC in joblistEquil:
                dirname        = jobC.stepdirname
                UI             = UIint.UIstep(const)
                UI.K           = jobC.K
                #periodicQ      = jobC.periodicQdic
                refpoint       = jobC.newref
                UI.ref         = np.array(refpoint, dtype=float)
                UI.aveT        = jobC.nextstep 
                UI.aveTinitial = jobC.aveTinitial 
                UI.stepN       = jobC.stepN
                UIbefore       = jobC.UIbefore 
                UIbefore_initial = jobC.UIbefore_initial
                UI.A = 0.0
                COLlist = glob.glob("%s/COLVAR"%dirname) 
                if not COLlist:
                    COLlist += glob.glob("%s/COLVAR.*"%dirname)
                errorstepQ = False
                if len(COLlist) == 0:
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        c = inspect.currentframe()
                        wf.write("Debag(%s): there is not COLVAR in %s\n"%(c.f_lineno, dirname))
                    refthresholdchunedelta *= 0.5
                    if refthresholdchunedelta < const.threshold:
                        errorsteplist.append(jobC.aveTinitial)
                        errorsteplist.append(jobC.nextstep)
                    errorstepQ = True
                    refthresholdchune += refthresholdchunedelta
                elif UI.readdat(COLlist[0]) is False:
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        c = inspect.currentframe()
                        wf.write("Debag(%s): In %s, this calculation is not finished.\n"%(c.f_lineno, dirname))
                    refthresholdchunedelta *= 0.5
                    if refthresholdchunedelta < const.threshold:
                        errorsteplist.append(jobC.aveTinitial)
                        errorsteplist.append(jobC.nextstep)
                    refthresholdchune += refthresholdchunedelta
                    errorstepQ = True
                if const.chkDihedralQ:
                    dihedlist = [[] for _ in range(Ndihedral)]
                    omeganame = glob.glob("%s/OMEGA*"%(dirname))
                    if len(omeganame) == 0:
                        errorstepQ = True
                    else:
                        for line in open(omeganame[0]):
                            if line[0] == "#":
                                continue
                            for i, x in enumerate(line.split()):
                                if i == 0:
                                    continue
                                dihedlist[i - 1].append(abs(float(x)))
                        for diheds in dihedlist:
                            if pi < sum(diheds) / len(diheds):
                                with open("./optlist.dat", "a") as wf:
                                    wf.write("#ERROR: dihedral = %s: this is cis structure.\n"%(sum(diheds) / len(diheds)))
                                errorsteplist.append(jobC.aveTinitial)
                                errorsteplist.append(jobC.nextstep)
                                errorstepQ = True
                if errorstepQ:
                    continue
                D_before_new = UIbefore_initial.Dbias(UI.ave)
                D_new_before = UI.Dbias(UIbefore.ave)
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s) %s: step%s: UI.Dbias(UI.aveTinitial) = % 3.2f\n"%(
                        c.f_lineno, datetime.datetime.now(), jobC.stepN, UI.Dbias(UI.aveTinitial)))
                    wf.write("Debag(%s) %s: step%s: UIbefore_initial.Dbias(UI.ave) = % 3.2f\n"%(
                        c.f_lineno, datetime.datetime.now(), jobC.stepN, UIbefore_initial.Dbias(UI.ave)))
                    wf.write("Debag(%s) %s: step%s: UI.Dbias(UIbefore_initial.ave) = % 3.2f\n"%(
                        c.f_lineno, datetime.datetime.now(), jobC.stepN, UI.Dbias(UIbefore_initial.ave)))
            if errorstepQ:
                const.Kminchune += const.Kmin
                nextstepD -= 0.5
                initialpoint = UIbefore.ave
                finishpoint = jobC.nextstep
                if const.allperiodicQ:
                    finishpointdamp = UIstepCython.periodic(finishpoint,
                        const.periodicmax, const.periodicmin, initialpoint, const.dim)
                else:
                    finishpointdamp = finishpoint
                nADD = finishpointdamp - initialpoint
                #nADD = nADD/np.linalg.norm(nADD)
                nADD = nADD/np.linalg.norm(nADD)
                sig = UIbefore.calcsigma(nADD,const.nextstepsigmaminTH)
                nADD *= sig
                acceptQ, needwindowpoint = UIstepCython.getneedwindowC(UIlist, UIlist, UIbefore,
                    initialpoint, finishpoint, nADD,
                    const.ADDstepsigmaTH, const.minimizeTH, nextstepD,
                    const.allperiodicQ, const.periodicmin, const.periodicmax, const.dim)
                jobC.nextstep = needwindowpoint
                jobC.aveTinitial = needwindowpoint
                jobC.newref = jobC.nextstep
                jobC.K = const.Kminchune * np.identity(const.dim)
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s) %s: step%s: kminchune is changed to % 3.2f\n"%(
                        c.f_lineno, datetime.datetime.now(), jobC.stepN, const.Kminchune))
                    wf.write("Debag(%s) nextstepD is changed to %s\n"%(c.f_lineno, nextstepD))
                jobC.K = const.Kminchune * np.identity(const.dim)
                stepsize_kappa = const.stepsize_kappa
            elif const.didpointsigmaTH <  UI.Dbias(UI.aveTinitial):
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    wf.write("Debag(%s) didpointsigmaTH <  UI.Dbias(UI.aveTinitial); retry.\n"%(c.f_lineno))
                stepsize_kappa *= 10
            elif chkintegration(const,UIlist+[UI],UIbefore,UI.ave):
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    wf.write("Debag(%s) this point is accepted.\n"%(c.f_lineno))
                break
            else:
                nextstepD -= 0.5
                initialpoint = UIbefore.ave
                finishpoint = jobC.nextstep
                if const.allperiodicQ:
                    finishpointdamp = UIstepCython.periodic(finishpoint,
                        const.periodicmax, const.periodicmin, initialpoint, const.dim)
                else:
                    finishpointdamp = finishpoint
                nADD = finishpointdamp - initialpoint
                #nADD = nADD/np.linalg.norm(nADD)
                acceptQ, needwindowpoint = UIstepCython.getneedwindowC(UIlist, UIlist, UIbefore,
                    initialpoint, finishpoint, nADD,
                    const.ADDstepsigmaTH, const.minimizeTH, nextstepD,
                    const.allperiodicQ, const.periodicmin, const.periodicmax, const.dim)
                jobC.nextstep = needwindowpoint
                jobC.aveTinitial = needwindowpoint
                jobC.newref = jobC.nextstep
                jobC.K = const.Kminchune * np.identity(const.dim)
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    wf.write("Debag(%s) nextstepD is changed to %s\n"%(c.f_lineno, nextstepD))
            #elif  const.nextstepsigmamaxTH <  D_before_new:
                #with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    #wf.write("Debag(%s) UIbefore_initial.Dbias(UI.ave) > nextstepsigmamaxTH; retry.\n"%(c.f_lineno))
            #elif  const.nextstepsigmamaxTH <  D_new_before:
                #with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    #wf.write("Debag(%s) UIDbias(UIbefore_initial.ave) > nextstepsigmamaxTH; retry.\n"%(c.f_lineno))
            recalldirpath = "%s/recall%s"%(dirname,recallturnN)
            os.mkdir(recalldirpath)
            for fname in glob.glob("%s/*"%dirname):
                if "recall" in fname:
                    continue
                shutil.move(fname, recalldirpath)
            rungro = "%s/run.gro"%recalldirpath
            nptgro = "%s/npt.gro"%recalldirpath
            mingro = "%s/min.gro"%recalldirpath
            if os.path.exists(rungro):
                shutil.copy(rungro, "%s/min.gro"%dirname)
            elif os.path.exists(nptgro):
                shutil.copy(nptgro, "%s/npt.gro"%dirname)
            elif os.path.exists(mingro):
                shutil.copy(mingro, "%s/min.gro"%dirname)
            else:
                print("ERROR: there is not %s or %s or %s"%(rungro,nptgro,mingro))
                exit()

        if const.allperiodicQ:
            if const.partADDQ:
                UI.ave_periodic = UIstepCython.periodic(UI.ave,
                 const.periodicmax, const.periodicmin, UIeq.aveALL, const.dim)
                UI.ref_periodic = UIstepCython.periodic(UI.ref,
                 const.periodicmax, const.periodicmin, UIeq.aveALL, const.dim)
            else:
                UI.ave_periodic = UIstepCython.periodic(UI.ave,
                 const.periodicmax, const.periodicmin, UIeq.ave, const.dim)
                UI.ref_periodic = UIstepCython.periodic(UI.ref,
                 const.periodicmax, const.periodicmin, UIeq.ave, const.dim)
        else:
            UI.ave_periodic = UI.ave
            UI.ref_periodic = UI.ref

        UI.nearUIlist = []
        if const.nearestWindowQ:
            A_nearW, varA, UInear = UIint.calcUIall_nearW(const,UIlist, UI.ave, UIself = UI)
            UI.A = copy.copy(A_nearW)
            c = inspect.currentframe()
            with open("./addUIdata.txt", "a")  as wf:
                wf.write("Debag(%s):UI.A, Dmin = (%s, % 4.2f)\n"%(c.f_lineno, UI.A, UIint.Dmin(UIlist, UI.ave)))
            writeline = ""
            for x in UI.ave:
                writeline += "%s, "%x
            writeline = writeline.rstrip(", ") + "\n"
            with open("./avepoint.csv", "a")  as wf:
                wf.write(writeline)
        else:
            UI.A = 0.0
        UI.calculationpath = copy.copy(UI.path)
        WGraph, UIlistall = UIint.importUIlistall_exclusion(const,WGraph, UIlistall)
        UI, WGraph = UIint.exportUI_exclusion(const,UI, WGraph, UIlistall)
        UIlistall.append(UI)
        if not UI.A is False:
            UIlist.append(UI)

    return errorsteplist, UIlist, WGraph, UIlistall
### end calc_edgewindows ###
def mkplumed(const,newref, currentstep, Kmat,UIbefore):
    """
    make plumed file
    """
    adict = {}
    adict["@pwd@"] = const.pwdpath
    for i in range(len(newref)):
        for j in range(len(newref)):
            if i == j:
                adict["@r%s@"%i] = "%s,0.0"%str(newref[i])
                adict["@K%s@"%i] = "%s,0.0"%(str(abs(Kmat[i,i])))
                #adict["@K%s@"%i] = "%s"%(str(Kmat[i,i]))
                if const.KoffdiagonalQ:
                    pstr = "%s,%s"%(str(const.periodicmin[i] + newref[i]), str(const.periodicmax[i] + newref[i]))
                    adict["@p%s@"%i] = pstr
            elif i < j:
                adict["@K%s-%s@"%(i, j)] = "%s"%(str(Kmat[i,j]))
    dirname = "./"
    #if copynpt(const,dirname, UIbefore) is False:
        #return False
    KoffdiagonalQ = True

    #plname   = "plumed_npt.dat"
    #if const.moveQ:
        #readname = "%s/plumed_npt.dat"%const.tmppath
    #else:
        #readname = "%s/plumed_npt.dat"%const.pwdpath
    #mdpsed(readname, plname, dirname, adict, nextstep, Kmat)

    plname   = "plumed.dat"
    if const.moveQ:
        readname = "%s/plumed.dat"%const.tmppath
    else:
        readname = "%s/plumed.dat"%const.pwdpath
    mdpsed(readname, plname, dirname, adict, newref, Kmat,KoffdiagonalQ)
    plname   = "plumedR.dat"
    if const.moveQ:
        readname = "%s/plumedR.dat"%const.tmppath
    else:
        readname = "%s/plumedR.dat"%const.pwdpath
    mdpsed(readname, plname, dirname, adict, newref, Kmat,KoffdiagonalQ)
    plname   = "plumedEQ.dat"
    if const.moveQ:
        readname = "%s/plumedEQ.dat"%const.tmppath
    else:
        readname = "%s/plumedEQ.dat"%const.pwdpath
    if os.path.exists(readname):
        mdpsed(readname, plname, dirname, adict, newref, Kmat, KoffdiagonalQ)

    return True
def mkplumed_opt(const,nextstep, Kmat, refpoint, UIbefore, stepsize_cv, stepsize_kappa):
    adict = {}
    adict["@pwd@"] = const.pwdpath
    for i in range(len(nextstep)):
        for j in range(len(nextstep)):
            if i == j:
                adict["@r%s@"%i] = "%s,0.0"%str(nextstep[i])
                adict["@K%s@"%i] = "%s,0.0"%(str(abs(Kmat[i,i])))
                #adict["@K%s@"%i] = "%s"%(str(Kmat[i,i]))
                if const.KoffdiagonalQ:
                    pstr = "%s,%s"%(str(const.periodicmin[i] + nextstep[i]), str(const.periodicmax[i] + nextstep[i]))
                    adict["@p%s@"%i] = pstr
            elif i < j:
                adict["@K%s-%s@"%(i, j)] = "%s"%(str(Kmat[i,j]))
    dirname = "./"
    if copynpt(const,dirname, UIbefore) is False:
        return False
    plname   = "plumed_npt.dat"
    if const.moveQ:
        readname = "%s/plumed_npt.dat"%const.tmppath
    else:
        readname = "%s/plumed_npt.dat"%const.pwdpath
    beforestep = refpoint
    mdpsed_opt(const, readname, plname, dirname, beforestep, nextstep, Kmat, stepsize_cv, stepsize_kappa)
    return True
    
def mdpsed(readname, mdpfname, dirname, adict, newref, Kmat,KoffdiagonalQ):
    pldiclist = functions.importplumed(readname)
    if pldiclist is False:
        exit()
    for pldic in pldiclist:
        if "RESTRAINT" in pldic["options"]:
            for k, v in adict.items():
                if pldic["AT"] == k:
                    pldic["AT"] = copy.copy(v)
                elif pldic["KAPPA"] == k:
                    pldic["KAPPA"] = copy.copy(v)

    arglist = []
    for pldic in pldiclist:
        #if "TORSION" in pldic["options"]:
            #arglist.append(pldic["LABEL"])
        if any("CV" in x for x in pldic["comments"]):
            arglist.append(pldic["LABEL"])
        elif "MOLINFO" in pldic["options"]:
            pldic["STRUCTURE"] = pldic["STRUCTURE"].replace("@pwd@", adict["@pwd@"])
    if KoffdiagonalQ:
        pldic          = {"options":["RESTRAINTMATRIX"], "comments":[], "linefeedQ" : True}
        pldic["LABEL"]    = "Rest-MAT"
        pldic["ARG"] = ",".join(arglist)
        pldic["AT"] = ",".join([str(x) for x in newref])
        for i in range(len(newref)):
            kappalist = [Kmat[i,j] for j in range(len(newref))]
            pldic["KAPPA%s"%i]    = ",".join([str(x) for x in kappalist])
        pldiclist.append(pldic)
    else:
        for i in range(len(newref)):
            pldic          = {"options":["RESTRAINT"], "comments":[], "linefeedQ" : False}
            pldic["LABEL"]    = "Rest-%s"%(arglist[i])
            atlist = [str(newref[i])]
            kappalist = [str(Kmat[i,i])]
            pldic["ARG"] = "%s"%arglist[i]
            for j in range(i + 1, len(newref)):
                if Kmat[i,j] == 0.0:
                    continue
                kappalist.append(str(Kmat[i, j] * 2.0))
                atlist.append(str(newref[j]))
                pldic["ARG"] += ",%s"%arglist[j]
            pldic["AT"]       = ",".join([str(x) for x in atlist])
            pldic["KAPPA"]    = ",".join([str(x) for x in kappalist])
            pldiclist.append(pldic)
    functions.exportplumed("%s/%s"%(dirname, mdpfname), pldiclist)
def mdpsed_opt(const,readname, mdpfname, dirname, beforestep, nextstep, Kmat, stepsize_cv, stepsize_kappa):
    pldiclist = functions.importplumed(readname)
    if pldiclist is False:
        exit()
    arglist = []
    for pldic in pldiclist:
        if any("CV" in x for x in pldic["comments"]):
            arglist.append(pldic["LABEL"])
        #if "TORSION" in pldic["options"]:
            #arglist.append(pldic["LABEL"])
    pldic          = {"options":["OPTIMIZERESTRAINTADAM","OFFDIAGONAL"], "comments":[], "linefeedQ" : True}
    pldic["LABEL"]    = "OptRest"
    pldic["ARG"] = ",".join(arglist)
    pldic["AT"] = ",".join([str(x) for x in beforestep])
    pldic["TARGET"] = ",".join([str(x) for x in nextstep])
    for i in range(len(nextstep)):
        kappalist = [str(Kmat[i,j]) for j in range(len(nextstep))]
        pldic["KAPPA%s"%i]    = ",".join(kappalist)
    pldic["OPTSTRIDE"] = "5000"
    pldic["CVSTEPSIZE"] = "%s"%stepsize_cv
    pldic["KAPPASTEPSIZE"] = "%s"%stepsize_kappa
    pldic["OPTMETHOD"] = const.UIoptMethod
    pldic["BETA1"] = "0.9"
    pldic["BETA2"] = "0.999"
    pldic["EPSILON"] = "1.0e-8"

    pldiclist.append(pldic)
    pldic          = {"options":["PRINT"], "comments":[], "linefeedQ" : True}
    pldic["STRIDE"] = "500"
    pldic["FILE"] = "COLVAR_kappa"
    kappaarglist = []
    for i in range(len(arglist)):
        for j in range(len(arglist)):
            kappaarglist.append("OptRest.%s_%s_kappa"%(arglist[i],arglist[j]))
    pldic["ARG"] = ",".join(kappaarglist)
    pldiclist.append(pldic)
    pldic          = {"options":["PRINT"], "comments":[], "linefeedQ" : True}
    pldic["STRIDE"] = "500"
    pldic["FILE"] = "COLVAR_cntr"
    cntrarglist = []
    for i in range(len(arglist)):
        cntrarglist.append("OptRest.%s_cntr"%(arglist[i]))
    pldic["ARG"] = ",".join(cntrarglist)
    pldiclist.append(pldic)
    pldic          = {"options":["PRINT"], "comments":[], "linefeedQ" : True}
    pldic["STRIDE"] = "500"
    pldic["FILE"] = "COLVAR_mean"
    cntrarglist = []
    for i in range(len(arglist)):
        cntrarglist.append("OptRest.%s_mean"%(arglist[i]))
    pldic["ARG"] = ",".join(cntrarglist)
    pldiclist.append(pldic)

    functions.exportplumed("%s/%s"%(dirname, mdpfname), pldiclist)
    return True

def multiple_replace(text, adict):
    rx = re.compile("|".join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text)
def copynpt(const,dirname, UIbefore):
    """ 
    searching npt*.gro around the point which is selected
    """
    if "window" in UIbefore.path and const.WindowDataType == "hdf5":
        if UIbefore.stepN < 1:
            formatline = "{}/%s"%const.grofilename
            nptfiles   = formatline.format(UIbefore.calculationpath)
            if os.path.exists(nptfiles):
                print(nptfiles)
                shutil.copy(nptfiles, "./min.gro")
            else:
                c = inspect.currentframe()
                print("ERROR({}): npt can not copy {}".format(c.f_lineno, nptfiles))
                print("os.getcwd() = %s"%os.getcwd())
                #exit()
                return False
            return True

        steppath = UIbefore.path.split("/")[-1]
        #lock = fasteners.InterProcessLock(lockfilepath_UIlist)
        #lock.acquire()
        #with h5py.File("%s/jobfiles/windows/windowfile.hdf5"%const.pwdpath, "r", libver='latest', swmr=True) as windowHDF:
            #grotext = windowHDF["%s/equibliumstructure"%steppath][...]
        #lock.release()
        UIlistchunkmin = 1
        UIlistchunkmax = copy.copy(const.UIlistchunksize)
        while True:
            if UIlistchunkmin <= UIbefore.stepN <= UIlistchunkmax:
                break
            UIlistchunkmin += const.UIlistchunksize
            UIlistchunkmax += const.UIlistchunksize
        UIlistHDFpath = "%s/jobfiles/windows/windowfile%s-%s.hdf5"%(const.pwdpath, UIlistchunkmin, UIlistchunkmax)
        UIlistchunkmin += const.UIlistchunksize
        UIlistchunkmax += const.UIlistchunksize
        UIlistHDFpath_next = "%s/jobfiles/windows/windowfile%s-%s.hdf5"%(const.pwdpath, UIlistchunkmin, UIlistchunkmax)
        if not os.path.exists(UIlistHDFpath_next):
            lock = fasteners.InterProcessLock(const.lockfilepath_UIlist + str(UIlistchunkmax - const.UIlistchunksize))
            lock.acquire()
        #print("1282: UIbefore.path = ",UIbefore.path,flush=True)
        #print("1283: steppath = ",steppath,flush=True)
        #exit()
        if os.path.exists("%s/%s"%(UIbefore.path, const.grofilename)):
            shutil.copy("%s/%s"%(UIbefore.path, const.grofilename),"./min.gro")
        else:
            windowHDF = h5py.File(UIlistHDFpath, "r")
            grotext = windowHDF["%s/equibliumstructure"%steppath][0]
            if isinstance(grotext, bytes):
                grotext = grotext.decode("utf-8")
            windowHDF.close()
            if not os.path.exists(UIlistHDFpath_next):
                lock.release()
            #grotext = str(grotext)
            if isinstance(grotext, bytes):
                grotext = grotext.decode('utf-8')
            open("./min.gro", "w").write(grotext)
        return True
    else:
        formatline = "{}/%s"%const.grofilename
        nptfiles   = formatline.format(UIbefore.path)
        print(UIbefore.path)
        if os.path.exists(nptfiles):
            shutil.copy(nptfiles, "./min.gro")
        else:
            c = inspect.currentframe()
            print("ERROR({}): npt can not copy {}".format(c.f_lineno, nptfiles))
            print("os.getcwd() = %s"%os.getcwd())
            exit()
            return False
        return True
def calljobADD(_Cnextref, turnADDN, _nextK, UIbefore):
    """
    call the job
    """
    stepN       = copy.copy(turnADDN)
    callstepN   = 1
    os.chdir("step%s"%stepN)
    currentstep = stepN - 1
    refpoint    = _Cnextref
    if mkplumed(refpoint, _nextK, currentstep, UIbefore):
        dirnames    = " step%s"%stepN
    else:
        dirnames = ""
    os.chdir("../")
    stepN       = copy.copy(turnADDN)
    if shellname == "csh":
        try:
            sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "", str(parallelPy)],
                        timeout=calltime)
        except:
            print("in calljobADD")
            c = inspect.currentframe()
            print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
    elif shellname == "bash":
        gpuid = "0"
        for i in range(callstepN - 1):
            gpuid += "0"
        try:
            sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "def", str(parallelPy)],
                        timeout=calltime)
        except:
            c = inspect.currentframe()
            print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
    else:
        c = inspect.currentframe()
        print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
def calljobopt_eq(nextstep, turnN, UIbefore, callK):
    """
    call the job
    """
    #if KoffdiagonalQ:
        #thetalist = np.array([0.0 for _ in range(const.dim - 1)])
        #krad = 0.0
        #_nextK = self.calcK_theta(krad, thetalist, callK)
    #else:
        #_nextK = callK
    stepN       = copy.copy(turnN)
    callstepN   = 1
    os.chdir("step%s"%stepN)
    currentstep = stepN - 1
    refpoint    = nextstep
    if mkplumed(refpoint, callK, currentstep, UIbefore):
        dirnames    = " step%s"%stepN
    else:
        dernames = ""
    os.chdir("../")

    if needgpuid:
        gpuid = "0"
        for i in range(callstepN - 1):
            gpuid += "0"
        if shellname == "csh":
            sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "EQ", str(parallelPy), gpuid],
                timeout=calltime)
        elif shellname == "bash":
            sp.call(["%s/calljoboptEQ.sh"%const.pwdpath, str(callstepN), dirnames, const.pwdpath, gpuid],
                timeout=calltime)
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
    else:
        if shellname == "csh":
            try:
                sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "EQ", str(parallelPy)],
                    timeout=calltime)
            except:
                print("in calljobopt_eq")
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
        elif shellname == "bash":
            try:
                sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "EQ", str(parallelPy)],
                        timeout=calltime)
            except:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
def calljobopt_ts(UItslist):
    """
    call the job
    """
    callstepN = 0
    dirnamelist = []
    for UIts in UItslist:
        nextstep    = UIts.ref
        turnN       = UIts.stepN
        UIbefore    = UIts.UIbefore
        callK       = UIts.K
    
        stepN       = copy.copy(turnN)
        os.chdir("step%s"%stepN)
        currentstep = stepN - 1
        refpoint    = nextstep
        if mkplumed(refpoint, callK, currentstep, UIbefore):
            dirnamelist.append(" step%s"%stepN)
            callstepN  += 1
        os.chdir("../")
    allstepN = len(dirnamelist)
    while allstepN > 0:
        for parallelMD in sorted(const.parallelMDs):
            if allstepN >=  parallelMD:
                callstepN = copy.copy(parallelMD)
            else:
                break
        dirnames  = " "
        for _ in range(callstepN):
            dirnames += dirnamelist.pop()

        if shellname == "csh":
            sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "OPT", str(parallelPy)],
                        timeout=calltime)
        elif shellname == "bash":
            gpuid = "0"
            for i in range(callstepN - 1):
                gpuid += "0"
            try:
                sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "OPT", str(parallelPy)],
                #sp.call(["%s/calljobopt.sh"%const.pwdpath, 
                        #str(callstepN), dirnames, const.pwdpath, gpuid],
                        timeout=calltime)
            except:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
        dirnames  = " "
        allstepN = len(dirnamelist)
        
    if shellname == "csh":
        sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "OPT", str(parallelPy)],
                        timeout=calltime)
    elif shellname == "bash":
        gpuid = "0"
        for i in range(callstepN - 1):
            gpuid += "0"
        try:
            sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "OPT", str(parallelPy)],
            #sp.call(["%s/calljobopt.sh"%const.pwdpath, 
                    #str(callstepN), dirnames, const.pwdpath, gpuid],
                    timeout=calltime)
        except:
            c = inspect.currentframe()
            print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
    else:
        c = inspect.currentframe()
        print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
def calljobopt(nextstep, turnN, UIbefore, callK):
    """
    call the job
    """
    stepN       = copy.copy(turnN)
    callstepN   = 1
    os.chdir("step%s"%stepN)
    currentstep = stepN - 1
    refpoint    = nextstep
    if mkplumed(refpoint, callK, currentstep, UIbefore):
        dirnames    = " step%s"%stepN
    else:
        dirnames    = ""
    os.chdir("../")

    if shellname == "csh":
        sp.call(["%s/calljob.csh"%const.pwdpath, str(callstepN), dirnames, "OPT", str(parallelPy)],
                        timeout=calltime)
    elif shellname == "bash":
        gpuid = "0"
        for i in range(callstepN - 1):
            gpuid += "0"
        try:
            #sp.call(["%s/calljobopt.sh"%const.pwdpath, str(callstepN), dirnames, const.pwdpath, gpuid],
            sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "OPT", str(parallelPy)],
                        timeout=calltime)
        except:
            c = inspect.currentframe()
            print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
    else:
        c = inspect.currentframe()
        print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
def calljobMFEP(MFEPClist):
    """
    call the job
    """
    recallturnN = 0
    refthresholdchune = copy.copy(refthreshold)
    const.Kminchune         = copy.copy(const.Kmin)
    for MFEPC in MFEPClist:
        MFEPC.recallQs = []
        for j in range(2):
            if MFEPC.endnextstepQs[j]:
                MFEPC.recallQs.append(False)
            else:
                MFEPC.recallQs.append(True)
        MFEPC.dirnames  = ["", ""]
        MFEPC.Klist     = [0.0, 0.0]
        MFEPC.refpoints = [0.0, 0.0]
    flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') else (y,))]
    chuneupQ = False
    while recallturnN < const.recallturnmax:
        if not any(flatten([MFEPC.recallQs for MFEPC in MFEPClist])):
            break
        dirnamelist  = []
        recallturnN += 1
        if recallturnN != 1:
        #if chuneupQ:
            refthresholdchune  *= 0.5
            #const.Kminchune          = const.Kminchunemax - (const.Kminchunemax - const.Kmin) * (const.recallturnmax - recallturnN) / recallturnmax
            const.Kminchune          = copy.copy(const.Kmin)
        callstepNdamp = 0
        dirnames = ""
        for MFEPC in MFEPClist:
            if moveQ:
                MFEPC.mfeppathPWD = MFEPC.mfeppath.split("jobfiles")[-1]
                MFEPC.mfeppathPWD = "%s/jobfiles/%s"%(const.pwdpath, MFEPC.mfeppathPWD)
            for j in range(2):
                if MFEPC.endnextstepQs[j]:
                    continue
                if MFEPC.recallQs[j] is False:
                    continue
                if recallturnN == 1:
                    while True:
                        MFEPC.dirnames[j] = "%s/step%s"%(MFEPC.mfeppath, MFEPC.edgeNs[j])
                        if moveQ:
                            dirnamePWD = "%s/step%s"%(MFEPC.mfeppathPWD, MFEPC.edgeNs[j])
                            if os.path.exists(dirnamePWD):
                                if not os.path.exists(MFEPC.dirnames[j]):
                                    shutil.copytree(dirnamePWD, MFEPC.dirnames[j])
                        if os.path.exists(MFEPC.dirnames[j]):
                            if MFEPC.edgeNs[j] < 0:
                                MFEPC.edgeNs[j] -= 1
                            else:
                                MFEPC.edgeNs[j] += 1
                            #edgeN = MFEPC.edgeNs[j]
                        else:
                            os.mkdir(MFEPC.dirnames[j])
                            break
                os.chdir(MFEPC.dirnames[j])
                if 0 < MFEPC.edgeNs[j]:
                    currentstep = MFEPC.edgeNs[j] - 1
                else:
                    currentstep = MFEPC.edgeNs[j] + 1
                KdicD, newref, periodicQ, gradvector = MFEPC.UIbefores[j].nextK(
                    MFEPC.calcsteps[j], {}, False, refthresholdchune, const.Kminchune)
                MFEPC.Klist[j]     = copy.copy(KdicD)
                MFEPC.refpoints[j] = copy.copy(newref)
                if all([const.Kmax <= KdicD[i][i] for i in range(const.dim)]):
                    c = inspect.currentframe()
                    print("ERROR(%s): nextK over const.Kmax."%(c.f_lineno))
                    #refthresholdchune *= 2.0
                    #continue
                    MFEPC.endnextstepQs[j] = True
                    MFEPC.recallQs[j]      = True
                    continue
                UIbefore   = MFEPC.UIbefores[j]
                formatline = "{}/%s"%const.grofilename
                nptfiles   = formatline.format(UIbefore.path)
                refpoint   = MFEPC.refpoints[j]
                Kmat       = MFEPC.Klist[j]
                if mkplumed(refpoint, Kmat, currentstep, UIbefore):
                    dirnamelist.append(" %s"%MFEPC.dirnames[j])
                else:
                    MFEPC.endnextstepQs[j] = True
        dirnamelistEQ = copy.copy(dirnamelist)
        allstepN      = len(dirnamelistEQ)
        while 0 < len(dirnamelistEQ):
            #callstepN = 0
            #while 0 < allstepN:
                #for parallelMD in sorted(const.parallelMDs):
                    #if allstepN >=  parallelMD:
                        #callstepN = copy.copy(parallelMD)
                #else:
                    #break
            for parallelMD in sorted(const.parallelMDs):
                if parallelMD <= len(dirnamelistEQ):
                    callstepN = copy.copy(parallelMD)

            dirnames  = " "
            for _ in range(callstepN):
                dirnames += dirnamelistEQ.pop()
                #callstepN += 1
            #with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                #wf.write("dirnames = %s\n"%dirnames)
            try:
                if shellname == "csh":
                    sp.call(["%s/calljob.csh"%tmppath, str(callstepN),
                        dirnames, "equibliumMFEP", str(parallelPy)],
                        timeout=calltime)
                elif shellname == "bash":
                    sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN),
                        dirnames, "equibliumMFEP", str(parallelPy)],
                        timeout=calltime)
                else:
                    c = inspect.currentframe()
                    print("ERROR(%s): shellname( %s ) is not csh or bash."%(
                        c.f_lineno, shellname), flush = True)
            except:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(
                    c.f_lineno,const.calltime))
        for MFEPC in MFEPClist:
            for j in range(2):
                if MFEPC.endnextstepQs[j]:
                    continue
                if MFEPC.recallQs[j] is False:
                    continue
                dirname        = MFEPC.dirnames[j]
                UI             = UIstep()
                UI.K           = MFEPC.Klist[j]
                UI.ref         = MFEPC.refpoints[j]
                UI.aveT        = copy.copy(MFEPC.calcsteps[j])
                UI.aveTinitial = copy.copy(MFEPC.calcsteps[j])
                UI.stepN       = MFEPC.edgeNs[j]
                UIbefore       = MFEPC.UIbefores[j]
                UIbefore_initial = MFEPC.UIbefores_initial[j]
                COLlist        = glob.glob("%s/COLVAR_npt"%dirname) 
                if not COLlist:
                    COLlist += glob.glob("%s/COLVAR_npt.*"%dirname)
                if len(COLlist) == 0:
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        c = inspect.currentframe()
                        wf.write("Debag(%s): there is not COLVAR in %s\n"%(c.f_lineno, dirname))
                    MFEPC.endnextstepQs[j] = True
                    MFEPC.recallQs[j]      = False
                    continue
                elif UI.readdat(COLlist[0], maxtimehere = maxtimeEQUI) is False:
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        c = inspect.currentframe()
                        wf.write(
                        "Debag(%s): In %s, this calculation is not finished.\n"%(c.f_lineno, dirname))
                    MFEPC.endnextstepQs[j] = True
                    MFEPC.recallQs[j]      = False
                    continue
                with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                    c = inspect.currentframe()
                    wf.write(
                    "step%s (recallturnN = %s): UI.Dbias(UI.aveTinitial)       = % 3.2f\n"%(MFEPC.edgeNs[j], recallturnN, UI.Dbias(UI.aveTinitial)))
                    wf.write(
                    "step%s (recallturnN = %s): UIbefore_initial.Dbias(UI.ave) = % 3.2f\n"%(MFEPC.edgeNs[j], recallturnN, UIbefore_initial.Dbias(UI.ave)))
                #if UI.Dbias(UI.aveTinitial) < didpointsigmaTH:
                #if UIbefore.Dbias(UI.ave) < nextstepsigmamaxTH:
                if UI.Dbias(UI.aveTinitial) < didpointsigmaTH and UIbefore.Dbias(UI.ave) < nextstepsigmamaxTH:
                    MFEPC.recallQs[j]      = False
                    continue
                if recallturnN == const.recallturnmax:
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        c = inspect.currentframe()
                        wf.write("ERROR(%s): recallturn is over\n"%(c.f_lineno))
                    MFEPC.endnextstepQs[j] = True
                    MFEPC.recallQs[j]      = False
                calcpath = copy.copy(UI.path)
                UI.path += "/recall%s"%recallturnN
                os.mkdir(UI.path)
                shutil.move("%s/%s"%(calcpath, const.grofilename), "%s/%s"%(UI.path, const.grofilename))
                COLname = COLlist[0].split("/")[-1]
                shutil.move("%s/%s"%(calcpath, COLname),    "%s/%s"%(UI.path, COLname))
                shutil.move("%s/plumed.dat"%(calcpath),     "%s/plumed.dat"%(UI.path))
                shutil.move("%s/plumedR.dat"%(calcpath),    "%s/plumedR.dat"%(UI.path))
                shutil.move("%s/plumed_npt.dat"%(calcpath), "%s/plumed_npt.dat"%(UI.path))
                if nextstepsigmamaxTH < UI.Dbias(UI.aveTinitial):
                    with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
                        c = inspect.currentframe()
                        wf.write( "Error(%s): nextstepsigmamaxTH < UI.Dbias(UI.aveTinitial)\n"%c.f_lineno)
                else:
                    UI.A = 0.0
                    UI.nearUIlist = []
                    UI.calculationpath = copy.copy(UI.path)
                    UI.exportdata(UI.path)
                    MFEPC.UIbefores[j] = copy.copy(UI)
    dirnamelist = []
    for MFEPC in MFEPClist:
        for j in range(2):
            if MFEPC.endnextstepQs[j] is False:
                dirnamelist.append(" %s"%MFEPC.dirnames[j])
    while 0 < len(dirnamelist):
        #callstepN = 0
        #while 0 < allstepN:
            #for parallelMD in sorted(const.parallelMDs):
                #if allstepN >=  parallelMD:
                    #callstepN = copy.copy(parallelMD)
            #else:
                #break
        for parallelMD in sorted(const.parallelMDs):
            if parallelMD <= len(dirnamelist):
                callstepN = copy.copy(parallelMD)
        dirnames  = " "
        for _ in range(callstepN):
            dirnames += dirnamelist.pop()
            #callstepN += 1
        if shellname == "csh":
            try:
                sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "MFEP", str(parallelPy)],
                        timeout=calltime)
            except:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
        elif shellname == "bash":
            gpuid = "0"
            for i in range(callstepN - 1):
                gpuid += "0"
            try:
                sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "MFEP", str(parallelPy)],
                        timeout=calltime)
            except:
                c = inspect.currentframe()
                print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
    return MFEPClist
def calljob(joblist, UIbeforedic, newrefdic, Kdic, turnN, callstepN):
    """
    call the job
    """
    stepN    = copy.copy(turnN)
    #stepN = 0
    dirnames = ""
    for job in joblist:
        stepN += 1
        os.chdir("step%s"%stepN)
        _targetpoint = tuple(job[:const.dim])
        UIbefore     = UIbeforedic[_targetpoint]
        currentstep  = UIbefore.stepN
        refpoint     = newrefdic[_targetpoint]
        Kmat         = Kdic[_targetpoint]
        if mkplumed(refpoint, Kmat, currentstep, UIbefore):
            dirnames += " step%s"%stepN
        else:
            callstepN -= 1
        os.chdir("../")
    try:
        if shellname == "csh":
            sp.call(["%s/calljob.csh"%tmppath, str(callstepN), dirnames, "", str(parallelPy)],
                        timeout=calltime)
        elif shellname == "bash":
            gpuid = "0"
            for i in range(callstepN - 1):
                gpuid += "0"
            sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepN), dirnames, "def", str(parallelPy)],
                        timeout=calltime)
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
    except:
        c = inspect.currentframe()
        print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
def mkFElist(FElist, UIlist, WGraph, UIlistall, UIeq, FEveclist, edgelist, endlist, IOEsphereA, edgeN, mkFEpath):
    with open("./UIlistdata.txt", "a")  as wf:
        wf.write("%s: start mkFElist\n"%(datetime.datetime.now()))
    if const.partADDQ:
        initialpoint = copy.copy(UIeq.aveALL)
    else:
        initialpoint = copy.copy(UIeq.ave)
    #FEveclistTF = [[FEvec, False] for FEvec in FEveclist]
    nADDlist  = []
    FEveclistturn = copy.copy(FEveclist)
    while True:
        UIlistnumall_before = len(UIlistall)
        UIlistnum_before    = len(UIlist)
        with open("./UIlistdata.txt", "a")  as wf:
            wf.write("len(UIlist) = %s\n"%len(UIlist))
        nextsteps = []
        #for FEvec, endQ in FEveclistTF:
        FEveclistdamp = copy.copy(FEveclistturn)
        FEveclistturn = []
        for FEvec in FEveclistdamp:
            thetalist = functions.calctheta(FEvec)
            if const.partADDQ:
                nADD_min = functions.SperSphere_cartesian(UIeq, IOEsphereA, thetalist, UIeq.ADDconst.dim)
                if nADD_min is False:
                    c = inspect.currentframe()
                    with open("./UIlistdata.txt", "a")  as wf:
                        wf.write("Error(%s):nADD_min cannot calculated\n FEvec = %s\n"%(c.f_lineno, FEvec))
                    nADDlist.append((nADD_min, False))
                    continue
                nADD        = functions.mkADD(UIeq, nADD_min)
                finishpoint = UIeq.aveALL + nADD
            else:
                nADD     = functions.SperSphere_cartesian(UIeq, IOEsphereA, thetalist, const.dim)
                if nADD is False:
                    c = inspect.currentframe()
                    with open("./UIlistdata.txt", "a")  as wf:
                        wf.write("Error(%s):nADD cannot calculated\n FEvec = %s\n"%(c.f_lineno, FEvec))
                    nADDlist.append((nADD, False))
                    continue
                finishpoint = UIeq.ave + nADD
            UIlist_initial  = []
            for UI in UIlist:
                Dmindamp = UI.Dbias(initialpoint)
                #if Dmindamp < const.ADDstepsigmaTH:
                if Dmindamp < neighborwindowTH:
                    UIlist_initial.append(UI) 
                    finishpoint = functions.periodicpoint(finishpoint,const)
            with open("./UIlistdata.txt", "a")  as wf:
                wf.write("%s: start getneedwindow in mkFElist\n"%(datetime.datetime.now()))
            acceptQ, needwindowpoint, UIlistdamp, UIbefore = getneedwindow(UIlist, UIlist_initial, 
                    UIlist, UIeq, initialpoint, finishpoint)
            if acceptQ:
                nADDlist.append((nADD, UIbefore))
                continue
            if len(needwindowpoint) == 0:
                with open("./UIlistdata.txt", "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s): needwindowpoint cannot calculated\n"%(c.f_lineno))
                    wf.write("initialpoint = %s\n"%initialpoint)
                    wf.write("finishpoint  = %s\n"%finishpoint)
                    wf.write("nADD  = %s\n"%nADD)
                    wf.write("IOEsphereA  = %s\n"%IOEsphereA)
                    wf.write("thetalist  = %s\n"%thetalist)
                nADDlist.append((nADD, False))
                continue
            if 1 < len(UIlist):
                errorlist = []
                errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                    "#(reaction coordinate...)\n")
                is_in_endlist = chkerrorlist(const,UIlist, needwindowpoint, errorlist)
                if is_in_endlist:
                    with open("./UIlistdata.txt", "a")  as wf:
                        c = inspect.currentframe()
                        wf.write("Debag(%s): point %s is removed because of errorlist.dat\n"%(c.f_lineno, needwindowpoint))
                    nADDlist.append((nADD, False))
                    continue
            errorlist = []
            errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                    "#(reaction coordinate...)\n")
            is_in_endlist = chkerrorlist(const,UIlist, needwindowpoint, errorlist)
            if is_in_endlist:
                with open("./UIlistdata.txt", "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s): point %s is removed because of errorlist.dat\n"%(c.f_lineno, needwindowpoint))
                continue
            os.chdir(mkFEpath)
            with open("./pwdpath.dat", "w") as wf:
                wf.write(const.pwdpath)
            nextsteps.append((needwindowpoint, UIbefore))
            FEveclistturn.append(FEvec)
        if len(nextsteps) == 0:
            break
        nextstepsdamp = copy.copy(nextsteps)
        nextstepsdamp = sorted(nextstepsdamp, 
                key = lambda x: UIeq.deltaA_UI(UIeq.ave, x[0]))
        nextsteps = []
        for needwindowpoint, UIbefore in nextstepsdamp:
            addwindowQ = True
            for nextstep, UIbefore2 in nextsteps:
                if const.allperiodicQ:
                    nextstepdamp = UIstepCython.periodic(nextstep,
                        const.periodicmax, const.periodicmin, needwindowpoint, const.dim)
                else:
                    nextstepdamp = nextstep
                vec = needwindowpoint - nextstepdamp
                Vpoint = UIbefore2.ave + vec
                Vpoint = functions.periodicpoint(Vpoint,const)
                if UIbefore2.Dbias(Vpoint) < const.ADDstepsigmaTH:
                    addwindowQ = False
                    break
            if addwindowQ:
                nextsteps.append((needwindowpoint, UIbefore))
        with open("./UIlistdata.txt", "a")  as wf:
            c = inspect.currentframe()
            wf.write("Debag(%s): len(nextsteps) = %s\n"%(c.f_lineno, len(nextsteps)))
        if len(nextsteps) == 0:
            break
        addQ = True
        forcecallwindowQ = False
        errorsteplist, UIlist, WGraph, UIlistall = calc_edgewindows(
                const,edgeN, WGraph, UIlistall, UIlist, UIeq, nextsteps, forcecallwindowQ) 
        if errorsteplist:
            for errorstep in errorsteplist:
                with open("./UIlistdata.txt", "a")  as wf:
                    c = inspect.currentframe()
                    wf.write("Debag(%s): %s cannot calculated next steps and will be removed it.\n"%(c.f_lineno, errorstep))
                errorlist = []
                errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, errorlist,
                "#(reaction coordinate...)\n")
                newerrorlist = []
                adderrorlistQ = True
#                for errorpoint in errorlist:
#                    #e = np.array(errorpoint[1:-1])
#                    e = np.array(errorpoint)
#                    if np.linalg.norm(e - errorstep) < 0.001:
#                        adderrorlistQ = False
#                        break
                if adderrorlistQ:
                    newerrorlist.append(errorstep)
            errorlist = IOpack.exportlist_exclusion(const,"%s/jobfiles/errorlist.dat"%const.pwdpath, newerrorlist,
                "#path  (reaction coordinate...)  A\n")
            continue
        if len(UIlistall) == UIlistnumall_before and len(UIlist) == UIlistnum_before:
            os.chdir(mkFEpath)
            break
    #for nADD, UIbefore in nADDlist:
    for FEvec in FEveclist:
        thetalist = functions.calctheta(FEvec)
        if const.partADDQ:
            nADD_min = functions.SperSphere_cartesian(UIeq, IOEsphereA, thetalist, UIeq.ADDconst.dim)
            if nADD_min is False:
                nADDlist.append((nADD_min, False))
                continue
            nADD        = functions.mkADD(UIeq, nADD_min)
            finishpoint = UIeq.aveALL + nADD
            finishpoint = functions.periodicpoint(const,finishpoint)
            if UIbefore is False:
                FElist.append((finishpoint, False))
                continue
            if const.nearestWindowQ:
                A_nearW, varA, UInear = UIint.calcUIall_nearW(const,UIlist, finishpoint)
                if A_nearW is False:
                    FElist.append((finishpoint, False))
                    continue
                deltaA_UIall = copy.copy(A_nearW)
            else:
                deltaA_UIall, varA = UIint.calcUIall(UIlist, UIeq.aveALL, finishpoint)
        else:
            nADD     = functions.SperSphere_cartesian(UIeq, IOEsphereA, thetalist, const.dim)
            if nADD is False:
                continue
            finishpoint = UIeq.ave + nADD
            finishpoint = functions.periodicpoint(const,finishpoint)
            if UIbefore is False:
                FElist.append((finishpoint, False))
                continue
            if const.nearestWindowQ:
                deltaA_UIall, varA, UInear = UIint.calcUIall_nearW(const,UIlist, finishpoint)
                if deltaA_UIall is False:
                    FElist.append((finishpoint, False))
                    continue
            else:
                if const.partADDQ:
                    ave = UIeq.aveALL
                else:
                    ave = UIeq.ave
                deltaA_UIall, varA = UIint.calcUIall(UIlist, ave, finishpoint)
        finishstr = "[% 3.2f"%finishpoint[0]
        for x in finishpoint[1:]:
            finishstr += ", %3.2f"%x
        finishstr += "]"
        #with open("./FEdata.txt", "a")  as wf:
        with open("./UIlistdata.txt", "a")  as wf:
            wf.write("(len(UIlist), deltaA_UIall, ADD) = (%s, % 5.3f, % 5.3f)\n"%(len(UIlist), deltaA_UIall, deltaA_UIall - IOEsphereA))
        FElist.append((finishpoint, deltaA_UIall))
    with open("./UIlistdata.txt", "a")  as wf:
        c = inspect.currentframe()
        wf.write("Debag(%s): end mkFElist\n"%(c.f_lineno))
    return UIlist, WGraph, UIlistall, FElist
def chkerrorlist(const,UIlist, needwindowpoint, errorlist):
    minimumD = 1.0e30
    for UI in UIlist:
        if UI.Dbias(needwindowpoint) < minimumD:
            minimumD = copy.copy(UI.Dbias(needwindowpoint))
            UIbefore = copy.copy(UI)
    is_in_endlist = False
    for endpointlist in errorlist:
        endpoint   = np.array(endpointlist)
        if const.allperiodicQ:
            endpointdamp = UIstepCython.periodic(endpoint,
                    const.periodicmax, const.periodicmin, needwindowpoint, const.dim)
        else:
            endpointdamp = endpoint
        vec = UIbefore.ave + needwindowpoint - endpointdamp
        vec = functions.periodicpoint(vec,const)
        #if UIbefore.Dbias(UIbefore.ave + vec) < didpointsigmaTH:
        if UIbefore.Dbias(vec) < const.errorwindowsigmaTH:
            is_in_endlist = True
            return is_in_endlist
            #break
    return is_in_endlist
def getwindow_DC(UIlist, nextstep):
    Dbias_minimum = 1.0e30
    for UI in UIlist:
        Dbias_damp = UI.Dbias(nextstep)
        if Dbias_damp < Dbias_minimum:
            Dbias_minimum = copy.copy(Dbias_damp)
            UIbefore      = copy.copy(UI)
    const.Kminchune = const.Kmin
    refthresholdchune = refthreshold
    KparallelQ = True
    K, newref, periodicQdic, gradvector =\
        UIbefore.nextK(nextstep, {}, KparallelQ, refthresholdchune, Kminchune)
    i = 0
    while True:
        i += 1
        stepdirname = "%s/step%s"%(os.getcwd(), i)
        if not os.path.exists(stepdirname):
            os.mkdir(stepdirname)
            stepdirname = copy.copy(stepdirname)
            stepN       = copy.copy(i)
            break
    os.chdir(stepdirname)
    currentstep  = UIbefore.stepN
    callstepNdamp = 0
    dirnames = ""
    if mkplumed(newref, K, currentstep, UIbefore):
        dirnames += " step%s"%stepN
        callstepNdamp += 1
    os.chdir("../")
    try:
        if shellname == "csh":
            sp.call(["%s/calljob.csh"%tmppath, str(callstepNdamp), dirnames, "equiblium", str(parallelPy)],
                timeout=calltime)
        elif shellname == "bash":
            sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepNdamp), dirnames, "equiblium", str(parallelPy)],
                timeout=calltime)
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)

    except:
        c = inspect.currentframe()
        print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))
#####
##### now the recallturn is ignored in DC calculation: future task
#####

    try:
        if shellname == "csh":
            sp.call(["%s/calljob.csh"%tmppath, str(callstepNdamp), dirnames, "MFEP", str(parallelPy)],
                timeout=calltime)
        elif shellname == "bash":
            sp.call(["%s/calljob.sh"%const.pwdpath, str(callstepNdamp), dirnames, "MFEP", str(parallelPy)],
                timeout=calltime)
        else:
            c = inspect.currentframe()
            print("ERROR(%s): shellname( %s ) is not csh or bash."%(c.f_lineno, shellname), flush = True)
    except:
        c = inspect.currentframe()
        print("ERROR(%s):subprocess.TimeoutExpired: over %s second"%(c.f_lineno,const.calltime))

    dirname        = stepdirname
    UI             = UIstep()
    UI.K           = K
    periodicQ      = periodicQdic
    refpoint       = newref
    UI.ref         = np.array(refpoint, dtype=float)
    UI.aveT        = nextstep 
    #UI.aveTinitial = aveTinitial 
    UI.aveTinitial = nextstep 
    UI.stepN       = stepN
    #UIbefore_initial = UIbefore_initial
    UI.A = 0.0
    COLlist = glob.glob("%s/COLVAR"%dirname) 
    if not COLlist:
        COLlist += glob.glob("%s/COLVAR.*"%dirname)
    errorstepQ = False
    if len(COLlist) == 0:
        with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
            c = inspect.currentframe()
            wf.write("Debag(%s): there is not COLVAR in %s\n"%(c.f_lineno, dirname))
        #errorsteplist.append(jobC.aveTinitial)
        #errorsteplist.append(jobC.nextstep)
        #errorstepQ = True
    elif UI.readdat(COLlist[0]) is False:
        with open("%s/../UIlistdata.txt"%dirname, "a")  as wf:
            c = inspect.currentframe()
            wf.write("Debag(%s): In %s, this calculation is not finished.\n"%(c.f_lineno, dirname))
        #errorsteplist.append(jobC.aveTinitial)
        #errorsteplist.append(jobC.nextstep)
        #errorstepQ = True
    UI.calculationpath = UI.path
    UI.exportdata(UI.path)
    UIlist.append(UI)
    return UIlist
def chkintegration(const,UIlist,UIbefore,finishpoint):
    initialpoint = UIbefore.ave
    #finishpoint = UI.ave
    if const.allperiodicQ:
        finishpointdamp = UIstepCython.periodic(finishpoint,
                        const.periodicmax, const.periodicmin, initialpoint, const.dim)
    else:
        finishpointdamp = finishpoint
    nADD = finishpointdamp - initialpoint
    for i in range(1,21):
        point = initialpoint + i*0.05*nADD
        if UIint.Dmin(UIlist,point) < const.ADDstepsigmaTH:
            #print("%s: UI.Dbias = %s"%(i,UI.Dbias(point)))
            continue
        #elif UIbefore.Dbias(point) < const.ADDstepsigmaTH:
            #print("%s: UIbefore.Dbias = %s"%(i,UIbefore.Dbias(point)))
            #continue
        else:
            return False
    return True
