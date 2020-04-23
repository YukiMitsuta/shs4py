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
    print("Waiting mkdir_exclusion", flush = True)
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
    print("End mkdir_exclusion", flush = True)
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
    if os.path.exists(fname):
        sharedlist = open(fname).readlines()
        for line in sharedlist:
            if line[0] == "#":
                continue
            #line = line.replace("\n", "").split()
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
                    #if "window" in a:
                        #a = windowdic[a]
                l.append(a)
            returnlist.append(l)
    returnlistdamp = copy.copy(returnlist)
    for line in newlist:
        l = []
        for a in line:
            try:
                a = float(a)
            except:
                #if "window" in a:
                    #a = windowdic[a]
                pass
            l.append(a)
        for returnline in returnlist:
            if returnline[0] == l[0]:
                if "MFEPconnections.csv" in fname:
                    continue
                break
        else:
            returnlistdamp.append(l)
    #if "MFEP" in fname or "ts" in fname:
        #returnlist = returnlistdamp
    #else:
        #returnlist = networkFE(returnlistdamp)
    #if "eqlist.dat" in fname:
    if False:
        returnlist = networkFE(returnlistdamp)
    else:
        returnlist = returnlistdamp
    if returnlist is False:
        return False
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
    for eqpointlist in eqlist:
        eqpoint = eqpointlist[1:-1]
        dirname = "{0}/jobfiles_meta/{1}".format(const.pwdpath, eqpointlist[0])
        if os.path.exists("%s/end.txt"%dirname):
            continue
        elif os.path.exists("%s/running.txt"%dirname):
            continue
        else:
            break
    if not dirname is False:
        with open("%s/running.txt"%dirname, "w") as wf:
            wf.write("running")
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
