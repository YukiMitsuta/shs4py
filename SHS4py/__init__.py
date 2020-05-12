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
        importinitialpointQ : if true, initial points are imported from "%s/jobfiles_meta/initialpoint.csv"%const.pwdpath
        initialpoints       : initial points of optimizations
        SHSrank             : rank of MPI
        SHSroot             : root rank of MPI
        SHScomm             : communicate class of mpi4py
        optdigTH            : the threshold of potential
                                (in metadynamics calculation, the area f(x) < optdigTH as confidence)
        eigNth              : threshold of eigen value of hessian on EQ and TS point
                                because the points that have small eigen valule, which mean the plain area, cannot apply SHS
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
        tslistpath = "%s/jobfiles_meta/tslist.csv"%const.pwdpath
        if not os.path.exists(tslistpath):
            with open(tslistpath, "w") as wf:
                wf.write("")
        tslist = IOpack.importlist(tslistpath)
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
                    eqlist, tslist = IOpack.chksamepoint_exportlist("EQ", eqlist, tslist, eqpoint, f_point, const)
                elif 0.0 < eigNlist[1]:
                    eqlist, tslist = IOpack.chksamepoint_exportlist("TS", eqlist, tslist, eqpoint, f_point, const)
#                    tspoint = eqpoint
#                    f_ts    = f_eqpoint
#                    dmin    = 1.0e30
#                    disQ    = [False]
#                    tslistpath = "%s/jobfiles_meta/tslist.csv"%const.pwdpath
#                    headline = "#TSname, "
#                    headline += "CV, ..., "
#                    headline += "FE (kJ/mol)\n"
#                    tslist = IOpack.exportlist_exclusion(tslistpath, tslist, headline, const)
#                    for beforetspointlist in tslist:
#                        beforetspoint = beforetspointlist[1:-1]
#                        beforetspoint = functions.periodicpoint(beforetspoint, const, tspoint)
#                        dis = beforetspoint - tspoint
#                        if type(const.sameEQthreshold) is float:
#                            dis = max([abs(x) for x in dis])
#                            if dis < dmin:
#                                dmin = copy.copy(dis)
#                        elif type(const.sameEQthreshold) is list:
#                            disQ = [ abs(x) < const.sameEQthreshold[i] for i,x in enumerate(dis)]
#                            if all(disQ):
#                                break
#                    if type(const.sameEQthreshold) is float:
#                        if const.sameEQthreshold < dmin:
#                            disQ = True
#                        else:
#                            disQ = False
#                    elif type(const.sameEQthreshold) is list:
#                        if all(disQ):
#                            disQ = False
#                        else:
#                            disQ = True
#                    else:
#                        print("ERROR; const.sameEQthreshold is not float or list")
#                        eqlist  = False
#                        disQ = False
#                    if disQ:
#                        TSnum  = IOpack.mkdir_exclusion("TS", 1, const)
#                        pointname = "TS{0:0>4}".format(TSnum)
#                        #print("find %s"%pointname)
#                        tslist = IOpack.importlist(tslistpath)
#                        tslist.append([pointname] + list(tspoint) + [f_ts])
#                        #IOpack.exportlist(tslistpath, tslist)
#                        headline = "#TSname, "
#                        headline += "CV, ..., "
#                        headline += "FE (kJ/mol)\n"
#                        tslist = IOpack.exportlist_exclusion(tslistpath, tslist, headline, const)
#                        print("%s is found"%pointname, flush = True)
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
#        if SHSrank == SHSroot:
#            #IOpack.exportlist(eqlistpath, eqlist)
#            headline = "#EQname, "
#            #for i in range(len(eqpoint)):
#                #headline += "CV%s, "%i
#            headline += "CV, ..., "
#            headline += "FE (kJ/mol)\n"
#            eqlist = IOpack.exportlist_exclusion(eqlistpath, eqlist, headline, const)
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
                eqlist, tslist = IOpack.chksamepoint_exportlist(
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
            tslistpath = "%s/jobfiles_meta/tslist.csv"%const.pwdpath
            if not os.path.exists(tslistpath):
                with open(tslistpath, "w") as wf:
                    wf.write("")
            tslist = IOpack.importlist(tslistpath)
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
                            eqlist, tslist = IOpack.chksamepoint_exportlist(
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
                    eqlist, tslist = IOpack.chksamepoint_exportlist(
                            "TS", eqlist, tslist, tspoint, f_ts, const)
                    #print("tspoint = %s"%tspoint)
#                    dmin = 1.0e30
#                    tslist = IOpack.importlist(tslistpath)
#                    disQ = [False]
#                    tslist = IOpack.exportlist_exclusion(tslistpath, tslist, headline, const)
#                    for beforetspointlist in tslist:
#                        beforetspoint = beforetspointlist[1:-1]
#                        beforetspoint = functions.periodicpoint(beforetspoint, const, tspoint)
#                        dis = beforetspoint - tspoint
#                        if type(const.sameEQthreshold) is float:
#                            dis = max([abs(x) for x in dis])
#                            if dis < dmin:
#                                dmin = copy.copy(dis)
#                        elif type(const.sameEQthreshold) is list:
#                            disQ = [ abs(x) < const.sameEQthreshold[i] for i,x in enumerate(dis)]
#                            if all(disQ):
#                                break
#                    if type(const.sameEQthreshold) is float:
#                        if const.sameEQthreshold < dmin:
#                            disQ = True
#                        else:
#                            disQ = False
#                    elif type(const.sameEQthreshold) is list:
#                        if all(disQ):
#                            disQ = False
#                        else:
#                            disQ = True
#                    else:
#                        print("ERROR; const.sameEQthreshold is not float or list")
#                        tslist  = False
#                        disQ = False
#                    if disQ:
#                        EQnum  = IOpack.mkdir_exclusion("TS", 1, const)
#                        pointname = "TS{0:0>4}".format(EQnum)
#                        #print("find %s"%pointname)
#                        tslist = IOpack.importlist(tslistpath)
#                        tslist.append([pointname] + list(tspoint) + [f_ts])
#                        headline = "#TSname, "
#                        #for i in range(len(tspoint)):
#                            #headline += "CV%s, "%i
#                        headline += "CV, ..., "
#                        headline += "FE (kJ/mol)\n"
#                        tslist = IOpack.exportlist_exclusion(tslistpath, tslist, headline, const)
#                        #IOpack.exportlist(tslistpath, tslist)
#                        print("%s is found"%pointname, flush = True)
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
            #if os.path.exists("%s/running.txt"%dirname):
                #os.remove("%s/running.txt"%dirname)
            #with open("%s/end.txt"%dirname, "w") as wf:
                #wf.write("calculated")
            IOpack.writeEND_exclusion(dirname, "TS", const)
            tslist = IOpack.importlist(tslistpath)
            #IOpack.exportlist(tslistpath, tslist)
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
            dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, tspointlist[0])
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
                        disQ = [False]
                        eqlist = IOpack.exportlist_exclusion(eqlistpath, eqlist, headline, const)
                        for beforeeqpointlist in eqlist:
                            beforeeqpoint = beforeeqpointlist[1:-1]
                            beforeeqpoint = functions.periodicpoint(beforeeqpoint, const, eqpoint)
                            dis = beforeeqpoint - eqpoint
                            if type(const.sameEQthreshold) is float:
                                dis = max([abs(x) for x in dis])
                                if dis < dmin:
                                    dmin = copy.copy(dis)
                                    nearestbeforeEQlist = copy.copy(beforeeqpointlist)
                            elif type(const.sameEQthreshold) is list:
                                disQ = [ abs(x) < const.sameEQthreshold[i] for i,x in enumerate(dis)]
                                if all(disQ):
                                    nearestbeforeEQlist = copy.copy(beforeeqpointlist)
                                    break
                        if type(const.sameEQthreshold) is float:
                            if const.sameEQthreshold < dmin:
                                disQ = True
                            else:
                                disQ = False
                        elif type(const.sameEQthreshold) is list:
                            if all(disQ):
                                disQ = False
                            else:
                                disQ = True
                        else:
                            print("ERROR; const.sameEQthreshold is not float or list")
                            eqlist  = False
                            disQ = False
                        if disQ:
                            print("new point!", flush = True)
                            EQnum  = IOpack.mkdir_exclusion("EQ", 1, const)
                            pointname = "EQ{0:0>4}".format(EQnum)
                            #with open("%s/jobfiles_meta/connections.csv"%const.pwdpath, "a") as wf:
                                #wf.write("%s, %s\n"%(tspointlist[0], pointname))
                            IOpack.exportconnectionlist_exclusion(tspointlist[0], pointname, const)
                            eqlist = IOpack.importlist(eqlistpath)
                            eqlist.append([pointname] + list(eqpoint) + [f_eqpoint])
                            #eqlist = sorted(eqlist, key = lambda x:x[-1])
                            #IOpack.exportlist(eqlistpath, eqlist)
                            headline = "#EQname, "
                            #for i in range(len(eqpoint)):
                                #headline += "CV%s, "%i
                            headline += "CV, ..., "
                            headline += "FE (kJ/mol)\n"
                            eqlist = IOpack.exportlist_exclusion(eqlistpath, eqlist, headline, const)
                            if SHSrank == SHSroot:
                                print("%s is found"%pointname, flush = True)
                        else:
                            #with open("%s/jobfiles_meta/connections.csv"%const.pwdpath, "a") as wf:
                                #wf.write("%s, %s\n"%(tspointlist[0], nearestbeforeEQlist[0]))
                            IOpack.exportconnectionlist_exclusion(tspointlist[0], nearestbeforeEQlist[0], const)
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
