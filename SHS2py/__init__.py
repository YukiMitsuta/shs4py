#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
main part of SHS2py

Available functions:
    SHSearch: main part of SHS2py calculation
    getdiTH: the calculation of digthreshold * hill (if you want to know the constant of digthreshold from out of SHS2py)
"""
import os, glob, shutil, copy
import numpy as np
from . import mkconst, functions, IOpack
from . import OPT, ADD, MinimumPath
class constClass():
    pass
def SHSearch(f, grad, hessian, 
        importinitialpointQ = True, initialpoints = None, 
        SHSrank = 0, SHSroot = 0, SHSsize = 1, SHScomm = None, optdigTH = False,
        eigNth  = - 1.0e30, const = False):
    """
    SHSearch: main part of SHS2py calculation

    Args:
        f                   : function to calculate potential as f(x)
        grad                : function to calculate gradient as grad(x)
        hessian             : function to calculate hessian as hessian(x)
        importinitialpointQ : if true, initial points are imported from "%s/jobfiles_meta/initialpoint.csv"%const.pwdpath
        initialpoints       : initial points of optimizations
        SHSrank             : rank of MPI
        SHSroot             : root rank of MPI
        SHScomm             : communicate class of mpi4py
        optdigTH            : the threshold of potential 
                                (in metadynamics calculation, the area f(x) < optdigTH as confidence)
        eigNth              : threshold of eigen value of hessian on EQ and TS point  
                                because the points that have small eigen valule, which mean the plaine aria, cannot apply SHS
        const               : class of constants
    """
    global IOpack
    if const is False:
        const = constClass()
    const = mkconst.main(const)

    if SHSrank == SHSroot:
        print("start SHSearch", flush = True)
        if not os.path.exists("jobfiles_meta"):
            os.mkdir("jobfiles_meta")
        if importinitialpointQ:
            importname = "%s/jobfiles_meta/initialpoint.csv"%const.pwdpath
            print("importinitialpointQ = True: try read %s"%importname)
            initialpoints = IOpack.importlist(importname)
        else:
            if initialpoints is None:
                print("Error in SHSearch: initialpoints is not defined.", flush = True)
                #functions.TARGZandexit()
        eqlistpath = "%s/jobfiles_meta/eqlist.csv"%const.pwdpath
        if not os.path.exists(eqlistpath):
            with open(eqlistpath, "w") as wf:
                wf.write("")
        eqlist = IOpack.importlist(eqlistpath)
    else:
        eqlist        = None
        initialpoints = None
    if const.calc_mpiQ:
        eqlist        = SHScomm.bcast(eqlist, root = 0)
        initialpoints = SHScomm.bcast(initialpoints, root = 0)

    for initialpoint in initialpoints:
        if const.calc_mpiQ:
            initialpoint = SHScomm.bcast(initialpoint, root = 0)
        eqpoint, eqhess_inv = OPT.PoppinsMinimize(initialpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
        if const.calc_mpiQ:
            eqpoint        = SHScomm.bcast(eqpoint, root = 0)

        EQhess = hessian(eqpoint)
        eigNlist, _eigV = np.linalg.eigh(EQhess)
        f_eqpoint = f(eqpoint)
        chkdigQ = True
        if optdigTH is not False:
            if optdigTH < f_eqpoint:
                if SHSrank == SHSroot:
                    print("f_eqpoint(%s) is larger  than optdigTH(%s)"%(f_eqpoint, optdigTH), flush  = True)
                chkdigQ = False
            if min(eigNlist) <= eigNth:
                if SHSrank == SHSroot:
                    print("min(eigNlist) = %s < eigNth = %s: this EQ point is too shallow"%(min(eigNlist), eigNth),flush = True)
                chkdigQ = False
        if chkdigQ:
            if SHSrank == SHSroot:
                if 0.0 < min(eigNlist):
                    dmin = 1.0e30
                    for beforeeqpointlist in eqlist:
                        beforeeqpoint = beforeeqpointlist[1:-1]
                        dis = np.linalg.norm(beforeeqpoint - eqpoint)
                        #if np.linalg.norm(beforeeqpoint - eqpoint) < const.sameEQthreshold:
                        if dis < dmin:
                            dmin = copy.copy(dis)
                            #break
                    #else:
                    if const.sameEQthreshold < dmin:
                        EQnum  = IOpack.mkdir_exclusion("EQ", 1, const)
                        pointname = "EQ{0:0>4}".format(EQnum)
                        eqlist.append([pointname] + list(eqpoint) + [f_eqpoint])
                else:
                    print("%s is not eq point: eigN = %s"%(eqpoint, eigNlist), flush = True)
            else:
                eqlist = None
        if const.calc_mpiQ:
            eqlist        = SHScomm.bcast(eqlist, root = 0)
        if SHSrank == SHSroot:
            IOpack.exportlist(eqlistpath, eqlist)

    if SHSrank == SHSroot:
        eqlist = sorted(eqlist, key = lambda x:x[-1])
        IOpack.exportlist(eqlistpath, eqlist)
    else:
        eqlist        = None
    if const.calc_mpiQ:
        eqlist        = SHScomm.bcast(eqlist, root = 0)
    tslist = []

    while True:
        eqpoint = None
        if SHSrank == SHSroot:
            tslistpath = "%s/jobfiles_meta/tslist.csv"%const.pwdpath
            if not os.path.exists(tslistpath):
                with open(tslistpath, "w") as wf:
                    wf.write("")
            tslist = IOpack.importlist(tslistpath)
            eqlist = sorted(eqlist, key = lambda x:x[-1])
            dirname = False
            for eqpointlist in eqlist:
                eqpoint = eqpointlist[1:-1]
                dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, eqpointlist[0])
                if os.path.exists("%s/end.txt"%dirname):
                    continue
                else:
                    break
        else:
            dirname = None
        if const.calc_mpiQ:
            eqpoint = SHScomm.bcast(eqpoint, root = 0)
            dirname = SHScomm.bcast(dirname, root = 0)
        if dirname is False:
            break
        eqpoint = np.array(eqpoint)
        TSinitialpoints = ADD.main(eqpoint, f, grad, hessian, dirname, optdigTH, SHSrank, SHSroot, SHScomm, SHSsize, const)
        if SHSrank == SHSroot:
            print("find %s TS initial points"%len(TSinitialpoints), flush = True)
        for TSinitialpoint in TSinitialpoints:
            #if const.calc_mpiQ:
                #TSinitialpoint = SHScomm.bcast(TSinitialpoint, root = 0)
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
                continue
            elif eigNlist[1] < 0.0:
                if SHSrank == SHSroot:
                    print("this point is not ts point", flush = True)
                    print(eigNlist, flush = True)
                continue

            if const.calc_mpiQ:
                tspoint = SHScomm.bcast(tspoint, root = 0)
            f_ts = f(tspoint)
            if const.calc_mpiQ:
                f_ts = SHScomm.bcast(f_ts, root = 0)
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
                    print("tspoint = %s"%tspoint)
                    dmin = 1.0e30
                    for beforetspointlist in tslist:
                        beforetspoint = beforetspointlist[1:-1]
                        #print("beforetspoint = %s"%beforetspoint)
                        dis = np.linalg.norm(beforetspoint - tspoint)
                        #if np.linalg.norm(beforetspoint - tspoint) < const.sameEQthreshold:
                            #break
                        if dis < dmin:
                            dmin = copy.copy(dis)
                    #else:
                    if const.sameEQthreshold < dmin:
                        EQnum  = IOpack.mkdir_exclusion("TS", 1, const)
                        pointname = "TS{0:0>4}".format(EQnum)
                        #print("find %s"%pointname)
                        tslist.append([pointname] + list(tspoint) + [f_ts])
                        IOpack.exportlist(tslistpath, tslist)
                        print("%s is found"%pointname, flush = True)
                else:
                    tslist = None
                if const.calc_mpiQ:
                    #print(SHSrank, flush = True)
                    tslist = SHScomm.bcast(tslist, root = 0)
                    #print("%s -> pass"%SHSrank, flush = True)
        if SHSrank == SHSroot:
            with open("%s/end.txt"%dirname, "w") as wf:
                wf.write("calculated")
            IOpack.exportlist(tslistpath, tslist)
        else:
            tslist = None
        if const.calc_mpiQ:
            tslist = SHScomm.bcast(tslist, root = 0)

        for tspointlist in tslist:
            tspoint = tspointlist[1:-1]
            dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, tspointlist[0])
            if os.path.exists("%s/end.txt"%dirname):
                continue
            tspoint = np.array(tspoint)
            nearEQpoints = MinimumPath.main(tspoint, f, grad, hessian, dirname, SHSrank, SHSroot, SHScomm, const)
            for nearEQpoint in nearEQpoints:
                if const.calc_mpiQ:
                    nearEQpoint = SHScomm.bcast(nearEQpoint, root = 0)
                eqpoint, eqhess_inv = OPT.PoppinsMinimize(nearEQpoint, f, grad, hessian, SHSrank, SHSroot, optdigTH, const)
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
                        dmin = 1.0e30
                        for beforeeqpointlist in eqlist:
                            beforeeqpoint = beforeeqpointlist[1:-1]
                            dis = np.linalg.norm(beforeeqpoint - eqpoint)
                            #if np.linalg.norm(beforeeqpoint - eqpoint) < const.sameEQthreshold:
                                #with open("%s/jobfiles_meta/connections.csv"%const.pwdpath, "a") as wf:
                                    #wf.write("%s, %s\n"%(tspointlist[0], beforeeqpointlist[0]))
                                #break
                            if dis < dmin:
                                dmin = copy.copy(dis)
                                nearestbeforeEQlist = copy.copy(beforeeqpointlist)
                        print("dmin = %s"%dmin, flush = True)
                        #else:
                        if const.sameEQthreshold < dmin:
                            print("new point!", flush = True)
                            EQnum  = IOpack.mkdir_exclusion("EQ", 1, const)
                            pointname = "EQ{0:0>4}".format(EQnum)
                            #print("find %s"%pointname)
                            with open("%s/jobfiles_meta/connections.csv"%const.pwdpath, "a") as wf:
                                wf.write("%s, %s\n"%(tspointlist[0], pointname))
                            #eqlist.append([pointname] + list(eqpoint) + [f(eqpoint)])
                            eqlist.append([pointname] + list(eqpoint) + [f_eqpoint])
                            eqlist = sorted(eqlist, key = lambda x:x[-1])
                            IOpack.exportlist(eqlistpath, eqlist)
                        else:
                            with open("%s/jobfiles_meta/connections.csv"%const.pwdpath, "a") as wf:
                                wf.write("%s, %s\n"%(tspointlist[0], nearestbeforeEQlist[0]))
                    else:
                        eqlist = None
                if const.calc_mpiQ:
                    eqlist        = SHScomm.bcast(eqlist, root = 0)

            if SHSrank == SHSroot:
                with open("%s/end.txt"%dirname, "w") as wf:
                    wf.write("calculated")
            if const.calc_mpiQ:
                eqlist        = SHScomm.bcast(eqlist, root = 0)
        if SHSrank == SHSroot:
            notbreakQ = False
            for dirname in glob.glob("EQ*"):
                if not os.path.exists("%s/end.txt"%dirname):
                    notbreakQ = True
            for dirname in glob.glob("TS*"):
                if not os.path.exists("%s/end.txt"%dirname):
                    notbreakQ = True
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
    getdiTH: the calculation of digthreshold * hill (if you want to know the constant of digthreshold from out of SHS2py)
    """
    return - const.digThreshold * hill
