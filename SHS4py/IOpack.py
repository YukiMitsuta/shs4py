#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
the functions of input/output processes

Available functions:
    importlist: import data file(fname) and return the list of data
    exportlist: export data list to file(fname)
    exportlist_exclusion: set exclusion control of exportlist
    mkdir_exclusion: Directry Booking System
    exportlist: exclusioncontrol to export data list to file(fname)

"""

import os, glob, shutil, sys, re
import time, datetime, copy, inspect
import numpy      as np

#from . import const
from . import functions

import fasteners

def importlist_exclusion(fname, const):
    lockfilepath_list = const.lockfilepath + "_" + fname.split("/")[-1]
    if not os.path.exists(lockfilepath_list):
        with open(lockfilepath_list, "w") as wf:
            wf.write("")
    lock = fasteners.InterProcessLock(lockfilepath_list)
    lock.acquire()
    returnlist = importlist(fname)
    lock.release()
    return returnlist
def importlist(fname):
    """
    import data file(fname) and return the list of data
    """
    if not os.path.exists(fname):
        raise FileNotFoundError("There is not %s."%fname)
    returnlist = []
    with open(fname) as f:
        for line in f:
            appendlist = []
            if {line[0]} & {"#", "\n"}:
                continue
            try:
                for x in line.split(","):
                    try:
                        y = float(x)
                    except:
                        y = x
                    appendlist.append(y)
                returnlist.append(appendlist)
            except:
                c       = inspect.currentframe()
                print("ERROR(%s): %s can not read."%(c.f_lineno, fname))
                print("***EXIT***")
                raise FileNotFoundError("ERROR: there is not %s"%fname)
                functions.TARGZandexit()
    return returnlist
def exportlist(fname,flist):
    """
    export data list to file(fname)
    """
    formatlinelist = []
    if flist:
        for i, x in enumerate(flist[0]):
            if i == 0:
                if isinstance(x, str):
                    formatlinelist.append("{0[%s]}"%i)
                elif isinstance(x, float):
                    formatlinelist.append("{0[%s]: #3.8f}"%i)
            else:
                formatlinelist.append("{0[%s]:< #3.8f}"%i)
    formatline = ",".join(formatlinelist)
    formatline += "\n"
    with open(fname, "w") as wf:
        #wf.write("#path   (reaction coordinate....)           A\n")
        if len(flist) != 0:
            for line in flist:
                if isinstance(line[0], float):
                    line[0] = int(line[0])
                wf.write(formatline.format(line))
def exportlist_exclusion(fname, newlist, headline, const):
    """
    set exclusion control
    export data list to file(fname)
    """
    lockfilepath_list = const.lockfilepath + "_" + fname.split("/")[-1]
    if not os.path.exists(lockfilepath_list):
        with open(lockfilepath_list, "w") as wf:
            wf.write("")
    #print("Waiting exportlist_exclusion", flush = True)
    lock = fasteners.InterProcessLock(lockfilepath_list)
    lock.acquire()
    returnlist = exportlist_share(fname, newlist, headline)
    lock.release()
    #print("End exportlist_exclusion", flush = True)
    return returnlist
def mkdir_exclusion(dirkind, fileN, const):
    """
    set exclusion control
    Directry Booking System
    """
    lockfilepath_mkdir = const.lockfilepath + "_mkdir"
    if not os.path.exists(lockfilepath_mkdir):
        with open(lockfilepath_mkdir, "w") as wf:
            wf.write("")
    #print("Waiting mkdir_exclusion", flush = True)
    lock = fasteners.InterProcessLock(lockfilepath_mkdir)
    lock.acquire()
    for num in range(1,100000):
        if os.path.exists("{0}/jobfiles_meta/{1}{2:0>4}".format(const.pwdpath, dirkind, num)) is False:
            for i in range(fileN):
                os.mkdir("{0}/jobfiles_meta/{1}{2:0>4}".format(const.pwdpath, dirkind, num + i))
                if const.moveQ:
                    os.mkdir("{0}/jobfiles_meta/{1}{2:0>4}".format(const.tmppath, dirkind, num + i))
            break
    lock.release()
    #print("End mkdir_exclusion", flush = True)
    return num
def exportlist_share(fname, newlist, headline):
    """
    exportlist: exclusioncontrol to export data list to file(fname)
    """
    returnlist = []
    windowdic  = {}
    if "error" in fname:
        if os.path.exists(fname):
            sharedlist = open(fname).readlines()
            for line in sharedlist:
                if line[0] == "#":
                    continue
                if "," in line:
                    line = line.replace("\n", "").split(",")
                else:
                    line = line.replace("\n", "").split()
                l = []
                for a in line:
                    try:
                        a = float(a)
                    except:
                        pass
                    l.append(a)
                a = np.array(l[1:-1])
                addQ = True
                for rlist in returnlist:
                    b = np.array(rlist[1:-1])
                    if np.linalg.norm(a - b) < 0.01:
                        addQ = False
                        break
                if addQ:
                    returnlist.append(l)
        #returnlistdamp = copy.copy(returnlist)
        for line in newlist:
            l = []
            for a in line:
                try:
                    a = float(a)
                except:
                    pass
                l.append(a)
            a = np.array(l[1:-1])
            addQ = True
            for rlist in returnlist:
                b = np.array(rlist[1:-1])
                if np.linalg.norm(a - b) < 0.01:
                    addQ = False
                    break
            if addQ:
                returnlist.append(l)
        if len(returnlist) != 0:
            formatline = "  ".join( "{0[%s]}"%i for i in range(len(returnlist[0])))
            formatline += "\n"
            with open(fname, "w") as wf:
                wf.write(headline)
                for returnline in returnlist:
                    wf.write(formatline.format(returnline))
        else:
            with open(fname, "w") as wf:
                wf.write(headline)
        return returnlist
    returnlist = newlist
    beforelistLength = 0
    if os.path.exists(fname):
        #print("Import %s"%fname, flush = True)
#        sharedlist = open(fname).readlines()
#        for line in sharedlist:
#            if line[0] == "#":
#                continue
#            beforelistLength += 1
#            samepointQ = False
#            for returnpoint in returnlist:
#                if returnpoint[0] in line:
#                    samepointQ = True
#                    break
#            if samepointQ:
#                continue
#            if "," in line:
#                line = line.replace("\n", "").split(",")
#            else:
#                line = line.replace("\n", "").split()
#            l = []
#            for a in line:
#                try:
#                    a = float(a)
#                except:
#                    pass
#                    #if "window" in a:
#                        #a = windowdic[a]
#                l.append(a)
        sharedlist = []
        sharedNAMEset = set()
        for line in open(fname):
            if line[0] == "#":
                continue
            if "," in line:
                line = line.replace("\n", "").split(",")
            else:
                line = line.replace("\n", "").split()
            l = []
            for a in line:
                try:
                    a = float(a)
                except:
                    pass
                l.append(a)
            sharedlist.append(l)
            sharedNAMEset.add(l[0])
        #print("sharedlist, returnlist = %s, %s"%(len(sharedlist), len(returnlist)))
        beforelistLength = len(sharedlist)

        NAMEset = sharedNAMEset - set([ a[0] for a in returnlist])
        addNAMEset = set([ a[0] for a in returnlist]) - sharedNAMEset
        #print("len(NAMEset) = %s"%len(NAMEset))
        if len(NAMEset) != 0:
            for l in sharedlist:
                if not l[0] in NAMEset:
                    continue
                #if len(returunlist) == 0:
                if True:
                    returnlist.append(l)
                else:
                    for i, returnpoint in enumerate(returnlist):
                        if l[-1] < returnpoint[-1]:
                            returnlist.insert(i, l)
                            break
        #print("end Import %s"%fname, flush = True)
#    returnlistdamp = copy.copy(returnlist)
#    for line in newlist:
#        l = []
#        for a in line:
#            try:
#                a = float(a)
#            except:
#                #if "window" in a:
#                    #a = windowdic[a]
#                pass
#            l.append(a)
#        for returnline in returnlist:
#            if returnline[0] == l[0]:
#                if "MFEPconnections.csv" in fname:
#                    continue
#                break
#        else:
#            returnlistdamp.append(l)
    #if "MFEP" in fname or "ts" in fname:
        #returnlist = returnlistdamp
    #else:
        #returnlist = networkFE(returnlistdamp)
    #if "eqlist.dat" in fname:
    #if False:
        #returnlist = networkFE(returnlistdamp)
    #else:
        #returnlist = returnlistdamp
    else:
        print("there is not %s"%fname)
    if returnlist is False:
        return False
    if beforelistLength == len(returnlist):
        return returnlist
    if len(returnlist) != 0:
        if "csv" in fname:
            formatlinelist = []
            for i, x in enumerate(returnlist[0]):
                if i == 0:
                    if isinstance(x, str):
                        formatlinelist.append("{0[%s]}"%i)
                    elif isinstance(x, float):
                        formatlinelist.append("{0[%s]: #3.8f}"%i)
                else:
                    formatlinelist.append("{0[%s]:< #3.8f}"%i)
            formatline = ",".join(formatlinelist)
            formatline += "\n"
        else:
            formatline = ""
            for i in range(len(returnlist[0])):
                if isinstance(returnlist[0][i], float):
                    formatline += "{0[%s]: #3.8f}  "%i
                elif isinstance(returnlist[0][i], int):
                    formatline += "{0[%s]: 5d}  "%i
                else:
                    formatline += "{0[%s]}  "%i
            formatline += "\n"
    returnlist = sorted(returnlist, key = lambda x:x[-1])
    if len(returnlist) < 10:
        writeline  = ""
        writeline += headline
        for returnline in returnlist:
            if "csv" in fname:
                formatlinelist = []
                for i, x in enumerate(returnline):
                    if i == 0:
                        if isinstance(x, str):
                            formatlinelist.append("{0[%s]}"%i)
                        elif isinstance(x, float):
                            formatlinelist.append("{0[%s]: #3.8f}"%i)
                    else:
                        formatlinelist.append("{0[%s]:< #3.8f}"%i)
                formatline = ",".join(formatlinelist)
                formatline += "\n"
                #print(returnlist)
                #print(formatline)
            else:
                formatline = ""
                for i in range(len(returnline)):
                    if isinstance(returnline[i], float):
                        formatline += "{0[%s]: #3.8f}  "%i
                    elif isinstance(returnline[i], int):
                        formatline += "{0[%s]: 5d}  "%i
                    else:
                        formatline += "{0[%s]}  "%i
                formatline += "\n"
    
            writeline += formatline.format(returnline)
        with open(fname, "w") as wf:
            wf.write(writeline)
    else:
        for returnline in returnlist:
            if not returnline[0] in addNAMEset:
                continue
            if "csv" in fname:
                formatlinelist = []
                for i, x in enumerate(returnline):
                    if i == 0:
                        if isinstance(x, str):
                            formatlinelist.append("{0[%s]}"%i)
                        elif isinstance(x, float):
                            formatlinelist.append("{0[%s]: #3.8f}"%i)
                    else:
                        formatlinelist.append("{0[%s]:< #3.8f}"%i)
                formatline = ",".join(formatlinelist)
                formatline += "\n"
                #print(returnlist)
                #print(formatline)
            else:
                formatline = ""
                for i in range(len(returnline)):
                    if isinstance(returnline[i], float):
                        formatline += "{0[%s]: #3.8f}  "%i
                    elif isinstance(returnline[i], int):
                        formatline += "{0[%s]: 5d}  "%i
                    else:
                        formatline += "{0[%s]}  "%i
                formatline += "\n"
            with open(fname, "a") as wf:
                wf.write(formatline.format(returnline))
    return returnlist
def findEQpath_exclusion(eqlist, const):
    lockfilepath = const.lockfilepath + "_eqpath"
    if not os.path.exists(lockfilepath):
        with open(lockfilepath, "w") as wf:
            wf.write("")
    print("Waiting findEQpath_exclusion", flush = True)
    lock = fasteners.InterProcessLock(lockfilepath)
    lock.acquire()
    dirname = False
    eqpoint = False
    eqlist = sorted(eqlist, key = lambda x:x[-1])
    for eqpointlist in eqlist:
        eqpoint = eqpointlist[1:-1]
        walloutQ = False
        for i in range(len(eqpoint)):
            if eqpoint[i] < const.EQwallmin[i] or const.EQwallmax[i] < eqpoint[i]:
                walloutQ = True
                break
        if walloutQ:
            continue
        dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, eqpointlist[0])
        if os.path.exists("%s/end.txt"%dirname):
            continue
        elif os.path.exists("%s/running.txt"%dirname):
            continue
        else:
            break
    else:
        dirname = False
        eqpoint = False
    if not dirname is False:
        with open("%s/running.txt"%dirname, "w") as wf:
            wf.write("running")
    else:
        eqpoint = False
    lock.release()
    print("End findEQpath_exclusion", flush = True)
    return eqpoint, dirname
def chkTSpath_exclusion(tslist, const):
    lockfilepath = const.lockfilepath + "_tspath"
    if not os.path.exists(lockfilepath):
        with open(lockfilepath, "w") as wf:
            wf.write("")
    print("Waiting chkTSpath_exclusion", flush = True)
    lock = fasteners.InterProcessLock(lockfilepath)
    lock.acquire()
    returntspointlist = False
    for tspointlist in tslist:
        tspoint = tspointlist[1:-1]
        dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, tspointlist[0])
        if not os.path.exists("%s/end.txt"%dirname):
            if not os.path.exists("%s/running.txt"%dirname):
                returntspointlist = tspointlist
                break
    if not dirname is False:
        with open("%s/running.txt"%dirname, "w") as wf:
            wf.write("running")
    lock.release()
    print("End chkTSpath_exclusion", flush = True)
    return returntspointlist
def writeEND_exclusion(dirname, dirtype, const):
    if dirtype == "TS":
        lockfilepath = const.lockfilepath + "_tspath"
    elif dirtype == "EQ":
        lockfilepath = const.lockfilepath + "_eqpath"
    if not os.path.exists(lockfilepath):
        with open(lockfilepath, "w") as wf:
            wf.write("")
    lock = fasteners.InterProcessLock(lockfilepath)
    lock.acquire()
    if os.path.exists("%s/running.txt"%dirname):
        os.remove("%s/running.txt"%dirname)
    with open("%s/end.txt"%dirname, "w") as wf:
        wf.write("calculated")
    lock.release()
def exportconnectionlist_exclusion(tspointname, eqpointname, const):
    lockfilepath_list = const.lockfilepath + "_connection"
    with open(lockfilepath_list, "w") as wf:
        wf.write("")
    lock = fasteners.InterProcessLock(lockfilepath_list)
    lock.acquire()
    with open("%s/jobfiles_meta/connections.csv"%const.pwdpath, "a") as wf:
        wf.write("%s, %s\n"%(tspointname, eqpointname))
    lock.release()
def chksamepoint_exportlist(pointtype, eqlist, tslist, point, f_point, const):
    lockfilepath = const.lockfilepath + "_chksamepoint"
    if not os.path.exists(lockfilepath):
        with open(lockfilepath, "w") as wf:
            wf.write("")
    #print("Waiting chkTSpath_exclusion", flush = True)
    lock = fasteners.InterProcessLock(lockfilepath)
    lock.acquire()
    dmin = 1.0e30
    disQ = [False]
    headline = "#%sname, "%pointtype
    headline += "CV, ..., "
    headline += "FE (kJ/mol)\n"
    if pointtype == "EQ":
        eqlistpath = "%s/jobfiles_meta/eqlist.csv"%const.pwdpath
        eqlist = exportlist_exclusion(eqlistpath, eqlist, headline, const)
        beforepointlist = eqlist
    elif pointtype == "TS":
        tslistpath = "%s/jobfiles_meta/tslist.csv"%const.pwdpath
        tslist = exportlist_exclusion(tslistpath, tslist, headline, const)
        beforepointlist = tslist
    else:
        print("ERROR; pointtype(%s) is not EQ or TS"%pointtype)
        exit()
    for beforepoint in beforepointlist:
        beforepointname = beforepoint[0]
        beforepoint = beforepoint[1:-1]
        beforepoint = functions.periodicpoint(beforepoint, const, point)
        dis = beforepoint - point
        if type(const.sameEQthreshold) is float:
            dis = max([abs(x) for x in dis])
            if dis < dmin:
                dmin = copy.copy(dis)
                pointname = copy.copy(beforepointname)
        elif type(const.sameEQthreshold) is list:
            disQ = [ abs(x) < const.sameEQthreshold[i] for i,x in enumerate(dis)]
            if all(disQ):
                pointname = copy.copy(beforepointname)
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
        eqlist = False
        disQ   = False
    if disQ:
        Pnum  = mkdir_exclusion(pointtype, 1, const)
        pointname = "{0}{1:0>4}".format(pointtype, Pnum)
        if pointtype == "EQ":
            #if len(eqlist) == 0:
            if True:
                eqlist.append([pointname] + list(point) + [f_point])
            #else:
                #for i, eqlistpoint in enumerate(eqlist):
                    #if f_point < eqlistpoint[-1]:
                        #eqlist.insert(i, [pointname] + list(point) + [f_point])
                        #break
            eqlistpath = "%s/jobfiles_meta/eqlist.csv"%const.pwdpath
            eqlist = exportlist_exclusion(eqlistpath, eqlist, headline, const)
        elif pointtype == "TS":
            #if len(tslist) == 0:
            if True:
                tslist.append([pointname] + list(point) + [f_point])
            else:
                for i, tslistpoint in enumerate(eqlist):
                    if f_point < tslistpoint[-1]:
                        tslist.insert(i, [pointname] + list(point) + [f_point])
                        break
            tslistpath = "%s/jobfiles_meta/tslist.csv"%const.pwdpath
            tslist = exportlist_exclusion(tslistpath, tslist, headline, const)
        print("%s is found"%pointname, flush=True)
    else:
        print("calculated point")
    lock.release()
    return eqlist, tslist, pointname
