#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
main part of SHS4py

Available functions:
    SHSearch: main part of SHS4py calculation
    getdiTH: the calculation of digthreshold * hill (if you want to know the constant of digthreshold from out of SHS4py)
"""
import os, glob, shutil, copy
import numpy as np
from . import mkconst, functions, IOpack
from . import OPT, ADD, MinimumPath
class constClass():
    pass
def SHSearch(f, grad, hessian, 
        importinitialpointQ = True, initialpoints = None, initialTSpoints = [],
        SHSrank = 0, SHSroot = 0, SHSsize = 1, SHScomm = None, optdigTH = False,
        eigNth  = - 1.0e30, const = False, metaDclass = False):
    """
    SHSearch: main part of SHS4py calculation

    Args:
        f                   : function to calculate potential as f(x)
        grad                : function to calculate gradient as grad(x)
        hessian             : function to calculate hessian as hessian(x)
        importinitialpointQ : if true, initial points are imported from "%s/%s/initialpoint.csv"%(const.pwdpath, const.jobfilepath)
        initialpoints       : initial points of optimizations
        SHSrank             : rank of MPI
        SHSroot             : root rank of MPI
        SHScomm             : communicate class of mpi4py
        optdigTH            : the threshold of potential 
                                (in metadynamics calculation, the area f(x) < optdigTH as confidence)
        eigNth              : threshold of eigen value of hessian on EQ and TS point  
                                because the points that have small eigen valule, which mean the plaine area, cannot apply SHS
        const               : class of constants
    """
    global IOpack
    if const is False:
        const = constClass()
    const = mkconst.main(const)

    if SHSrank == SHSroot:
        print("start SHSearch", flush = True)
        if not os.path.exists(const.jobfilepath):
            os.mkdir(const.jobfilepath)
        if importinitialpointQ:
            importname = "%s/%s/initialpoint.csv"%(const.pwdpath, const.jobfilepath)
            print("importinitialpointQ = True: try read %s"%importname)
            initialpoints = IOpack.importlist(importname)
        else:
            if initialpoints is None:
                print("Error in SHSearch: initialpoints is not defined.", flush = True)
                #functions.TARGZandexit()
        eqlistpath = "%s/%s/eqlist.csv"%(const.pwdpath, const.jobfilepath)
        tslistpath = "%s/%s/tslist.csv"%(const.pwdpath, const.jobfilepath)
        if not os.path.exists(eqlistpath):
            with open(eqlistpath, "w") as wf:
                wf.write("")
        #eqlist = IOpack.importlist(eqlistpath)
        eqlist = IOpack.importlist_exclusion(eqlistpath, const)
        if not os.path.exists(tslistpath):
            with open(tslistpath, "w") as wf:
                wf.write("")
        #tslist = IOpack.importlist(tslistpath)
        tslist = IOpack.importlist_exclusion(tslistpath, const)
    else:
        eqlist        = None
        initialpoints = None
    if const.calc_mpiQ:
        eqlist        = SHScomm.bcast(eqlist, root = 0)
        initialpoints = SHScomm.bcast(initialpoints, root = 0)

    for initialpoint in initialpoints:
        eqpoint, eqhess_inv = OPT.PoppinsMinimize(initialpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)

        EQhess = hessian(eqpoint)
        eigNlist, _eigV = np.linalg.eigh(EQhess)
        f_point = f(eqpoint)
        chkdigQ = True
        if optdigTH is not False:
            if optdigTH < f_point:
                if SHSrank == SHSroot:
                    print("f_eqpoint(%s) is larger  than optdigTH(%s)"%(f_point, optdigTH), flush  = True)
                chkdigQ = False
            if min(eigNlist) <= eigNth:
                if SHSrank == SHSroot:
                    print("min(eigNlist) = %s < eigNth = %s: this EQ point is too shallow"%(min(eigNlist), eigNth),flush = True)
                chkdigQ = False
        if chkdigQ:
            if SHSrank == SHSroot:
                if 0.0 < min(eigNlist):
                    eqlist, tslist, pointname = IOpack.chksamepoint_exportlist("EQ", eqlist, tslist, eqpoint, f_point, const)
                elif 0.0 < eigNlist[1]:
                    eqlist, tslist, pointname = IOpack.chksamepoint_exportlist("TS", eqlist, tslist, eqpoint, f_point, const)
                else:
                    print("%s is not EQ or TS point: eigN = %s"%(eqpoint, eigNlist), flush = True)
            else:
                eqlist = None
                tslist = None
        if const.calc_mpiQ:
            eqlist        = SHScomm.bcast(eqlist, root = 0)
            tslist        = SHScomm.bcast(tslist, root = 0)
        if eqlist is False:
            exit()
    initialTSpointsdamp = []
    for TSinitialpoint in initialTSpoints:
        tspoint = OPT.PoppinsDimer(TSinitialpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
        if tspoint is False:
            continue
        tspoint = np.array(tspoint)
        TShess  = hessian(tspoint)
        eigNlist, _eigV = np.linalg.eigh(TShess)
        if 0.0 < eigNlist[0]:
            if SHSrank == SHSroot:
                print("this point is not ts point", flush = True)
                print(eigNlist, flush = True)
            continue
        f_ts = f(tspoint)
        if SHSrank == SHSroot:
            print("f_ts = %s"%f_ts, flush = True)
        chkdigQ = True
        if optdigTH is not False:
            if optdigTH < f_ts:
                if SHSrank == SHSroot:
                    print("f_ts(%s) is larger than optdigTH(%s)"%(
                        f_ts, optdigTH), flush = True)
                chkdigQ = False
        if chkdigQ:
            if SHSrank == SHSroot:
                eqlist, tslist, pointname = IOpack.chksamepoint_exportlist(
                            "TS", eqlist, tslist, tspoint, f_ts, const)
            else:
                eqlist = None
                tslist = None
            if const.calc_mpiQ:
                eqlist = SHScomm.bcast(eqlist, root = 0)
                tslist = SHScomm.bcast(tslist, root = 0)
            if tslist is False:
                exit()

    if SHSrank == SHSroot:
        headline = "#EQname, "
        headline += "CV, ..., "
        headline += "FE (kJ/mol)\n"
        eqlist = IOpack.exportlist_exclusion(eqlistpath, eqlist, headline, const)
        headline = "#TSname, "
        headline += "CV, ..., "
        headline += "FE (kJ/mol)\n"
        tslist = IOpack.exportlist_exclusion(tslistpath, tslist, headline, const)
    else:
        eqlist = None
        tslist = None
    if const.calc_mpiQ:
        eqlist = SHScomm.bcast(eqlist, root = 0)
        tslist = SHScomm.bcast(tslist, root = 0)

    while True:
        eqpoint = None
        if SHSrank == SHSroot:
            tslistpath = "%s/%s/tslist.csv"%(const.pwdpath, const.jobfilepath)
            if not os.path.exists(tslistpath):
                with open(tslistpath, "w") as wf:
                    wf.write("")
            #tslist = IOpack.importlist(tslistpath)
            tslist = IOpack.importlist_exclusion(tslistpath, const)
            #eqlist = sorted(eqlist, key = lambda x:x[-1])
            eqpoint, dirname = IOpack.findEQpath_exclusion(eqlist, const)

            #for eqpointlist in eqlist:
                #eqpoint = eqpointlist[1:-1]
                #dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, eqpointlist[0])
                #if os.path.exists("%s/end.txt"%dirname):
                    #continue
                #else:
                    #break
        else:
            dirname = None
        if const.calc_mpiQ:
            eqpoint = SHScomm.bcast(eqpoint, root = 0)
            dirname = SHScomm.bcast(dirname, root = 0)
        if dirname is False:
            break
        eqpoint = np.array(eqpoint)
        TSinitialpoints = ADD.main(eqpoint, f, grad, hessian, dirname, optdigTH, SHSrank, SHSroot, SHScomm, SHSsize, const, metaDclass)
        if SHSrank == SHSroot:
            print("find %s TS initial points"%len(TSinitialpoints), flush = True)
        initialTSpointsdamp = []
        for TSinitialpoint in TSinitialpoints:
            initialTSpointsdamp.append(TSinitialpoint)

            if SHSrank == SHSroot:
                print("TSinitialpoint = %s"%TSinitialpoint, flush = True)
            #print(SHSrank, flush = True)
            tspoint = OPT.PoppinsDimer(TSinitialpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
            #print("%s -> pass"%SHSrank, flush = True)
            if tspoint is False:
                continue
            tspoint = np.array(tspoint)
            TShess  = hessian(tspoint)
            eigNlist, _eigV = np.linalg.eigh(TShess)
            if 0.0 < eigNlist[0]:
                if SHSrank == SHSroot:
                    print("this point is not ts point", flush = True)
                    print(eigNlist, flush = True)
                eqpoint, eqhess_inv = OPT.PoppinsMinimize(tspoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
                EQhess = hessian(eqpoint)
                eigNlist, _eigV = np.linalg.eigh(EQhess)
                f_eqpoint = f(eqpoint)
                chkdigQ = True
                if optdigTH is not False:
                    if optdigTH < f_eqpoint:
                        if SHSrank == SHSroot:
                            print("f_eqpoint(%s) is larger  than optdigTH(%s)"%(
                                f_eqpoint, optdigTH), flush  = True)
                        chkdigQ = False
                    if min(eigNlist) <= eigNth:
                        if SHSrank == SHSroot:
                            print("min(eigNlist) = %s < eigNth = %s: this EQ point is too shallow"%(
                                min(eigNlist), eigNth),flush = True)
                        chkdigQ = False
                if chkdigQ:
                    if SHSrank == SHSroot:
                        if 0.0 < min(eigNlist):
                            eqlist, tslist, pointname = IOpack.chksamepoint_exportlist(
                                    "EQ", eqlist, tslist, eqpoint, f_eqpoint, const)
        
                    elif eigNlist[1] < 0.0:
                         if SHSrank == SHSroot:
                             print("this point is not ts point", flush = True)
                             print(eigNlist, flush = True)

                continue

            #if const.calc_mpiQ:
                #tspoint = SHScomm.bcast(tspoint, root = 0)
            f_ts = f(tspoint)
            #if const.calc_mpiQ:
                #f_ts = SHScomm.bcast(f_ts, root = 0)
            if SHSrank == SHSroot:
                print("f_ts = %s"%f_ts, flush = True)
            chkdigQ = True
            if optdigTH is not False:
                if optdigTH < f_ts:
                    if SHSrank == SHSroot:
                        print("f_ts(%s) is larger than optdigTH(%s)"%(f_ts, optdigTH), flush = True)
                    chkdigQ = False

            if chkdigQ:
                if SHSrank == SHSroot:
                    eqlist, tslist, pointname = IOpack.chksamepoint_exportlist(
                            "TS", eqlist, tslist, tspoint, f_ts, const)
                else:
                    eqlist = None
                    tslist = None
                if const.calc_mpiQ:
                    #print(SHSrank, flush = True)
                    eqlist = SHScomm.bcast(eqlist, root = 0)
                    tslist = SHScomm.bcast(tslist, root = 0)
                    #print("%s -> pass"%SHSrank, flush = True)
                if tslist is False:
                    exit()
        if SHSrank == SHSroot:
            IOpack.writeEND_exclusion(dirname, "TS", const)
            tslist = IOpack.importlist_exclusion(tslistpath, const)
        else:
            tslist = None
        if const.calc_mpiQ:
            tslist = SHScomm.bcast(tslist, root = 0)

        #for tspointlist in tslist:
            #tspoint = tspointlist[1:-1]
            #dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, tspointlist[0])
            #if os.path.exists("%s/end.txt"%dirname):
                #continue
        while True:
            if SHSrank == SHSroot:
                tspointlist = IOpack.chkTSpath_exclusion(tslist, const)
            else:
                tspointlist = None
            if const.calc_mpiQ:
                tspointlist = SHScomm.bcast(tspointlist, root = 0)
            if tspointlist is False:
                break
            tspoint = tspointlist[1:-1]
            dirname = "{0}/{1}/{2}".format(const.pwdpath, const.jobfilepath, tspointlist[0])
            tspoint = np.array(tspoint)
            nearEQpoints = MinimumPath.main(tspoint, f, grad, hessian, dirname, SHSrank, SHSroot, SHScomm, const)
            for nearEQpoint in nearEQpoints:
                #if const.calc_mpiQ:
                    #nearEQpoint = SHScomm.bcast(nearEQpoint, root = 0)
                eqpoint, eqhess_inv = OPT.PoppinsMinimize(nearEQpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
                beforeeqpoint = functions.periodicpoint(nearEQpoint, const, eqpoint)
                dis = beforeeqpoint - eqpoint
                if type(const.sameEQthreshold) is float:
                    disMax = max([abs(x) for x in dis])
                    if const.sameEQthreshold < disMax:
                        samepointQ = False
                    else:
                        samepointQ = True
                elif type(const.sameEQthreshold) is list:
                    samepointQ = all([ abs(x) < const.sameEQthreshold[i] for i,x in enumerate(dis)])
                else:
                    print("ERROR; const.sameEQthreshold is not float or list")
                    eqlist  = False
                    samepointQ = True
                if samepointQ is False:
                    print("ERROR; the optimized point is too far from nearEQpoint! We propose you to change const.deltas.")
                    print("eqpoint = %s"%eqpoint)
                    print("nearEQpoint = %s"%nearEQpoint)
                    print("dis = %s"%dis)
                    continue

                EQhess = hessian(eqpoint)
                eigNlist, _eigV = np.linalg.eigh(EQhess)
                if min(eigNlist) < 0.0:
                    if SHSrank == SHSroot:
                        print("%s is not eq point: eigN = %s"%(eqpoint, eigNlist), flush = True)
                    continue
                f_eqpoint = f(eqpoint)
                chkdigQ = True
                if optdigTH is not False:
                    if optdigTH < f_eqpoint:
                        if SHSrank == SHSroot:
                            print("f_eqpoint is larger than optdigTH(%s)"%(f_eqpoint, optdigTH), flush = True)
                        chkdigQ = False
                    if min(eigNlist) <= eigNth:
                        if SHSrank == SHSroot:
                            print("min(eigNlist) = %s < eigNth = %s: this EQ point is too shallow"%(min(eigNlist), eigNth), flush = True)
                        chkdigQ = False
                if chkdigQ:
                    if SHSrank == SHSroot:
                        eqlist, tslist, pointname = IOpack.chksamepoint_exportlist(
                            "EQ", eqlist, tslist, eqpoint, f_eqpoint, const)
                        IOpack.exportconnectionlist_exclusion(tspointlist[0], pointname, const)
                    else:
                        eqlist = None
                if const.calc_mpiQ:
                    eqlist        = SHScomm.bcast(eqlist, root = 0)
                if eqlist is False:
                    exit()

            if SHSrank == SHSroot:
                #if os.path.exists("%s/running.txt"%dirname):
                    #os.remove("%s/running.txt"%dirname)
                #with open("%s/end.txt"%dirname, "w") as wf:
                    #wf.write("calculated")
                IOpack.writeEND_exclusion(dirname, "EQ", const)
            if const.calc_mpiQ:
                eqlist        = SHScomm.bcast(eqlist, root = 0)
        if SHSrank == SHSroot:
            notbreakQ = False
            for dirname in glob.glob("EQ*"):
                if not os.path.exists("%s/end.txt"%dirname):
                    notbreakQ = True
                    break
            for dirname in glob.glob("TS*"):
                if not os.path.exists("%s/end.txt"%dirname):
                    notbreakQ = True
                    break
        else:
            notbreakQ = None
        if const.calc_mpiQ:
            notbreakQ = SHScomm.bcast(notbreakQ, root = 0)
        if notbreakQ:
            continue
        break
    if SHSrank == SHSroot:
        print("ALL ADDs on EQ and TS points are calculated.")

def getdiTH(hill, const):
    """
    getdiTH: the calculation of digthreshold * hill (if you want to know the constant of digthreshold from out of SHS4py)
    """
    return - const.digThreshold * hill
