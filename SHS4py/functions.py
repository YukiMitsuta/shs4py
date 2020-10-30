#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
the effective functions for SHS4py

Available Class:
    PlumedDatClass : CLass structure of Plumed infomation (plumed.dat)
Available functions:
    TARGZandexit: tar and gzip the result file and exit
    SQaxes      : the matrix of scaled hypersphere (q = v * Sqrt(1 / lambda))
    SQaxes_inv  : the inverse matrix of scaled hypersphere (q = v * Sqrt(lambda))
    calctheta   : the basis transformathon from cartesian coordinates to polar one.
    SuperSphere_cartesian : the vector of super sphere by cartetian 
    calcsphereA: calculation of harmonic potential of the EQ point on nADD
    IOE: calculation of the procedure iterative optimization and elimination
    angle_SHS:   angle of x and y on supersphere surface
    angle:   angle of x and y
    wallQ:       chk wallmin < x < wallmax or not
    periodicpoint: periodic calculation of x
    importplumed : To import a plumed input file to the list of pldic
    exportplumed : To export a plumed input file from the list of pldic

"""
import os, glob, shutil, sys, re
import time, copy, inspect
import tarfile, zipfile, random
import subprocess      as sp
import numpy           as np
import multiprocessing as mp

class PlumedDatClass(object):
    def __init__(self, plumedpath):
        self.pldiclist = []
        if not os.path.exists(plumedpath):
            return
        contentsfieldQ = False
        for line in open(plumedpath):
            if line[0] is "#":
                continue
            if "..." in line:
                if contentsfieldQ:
                    self.pldiclist.append(pldic)
                    contentsfieldQ = False
                else:
                    contentsfieldQ = True
                    pldic = {"options":[], "comments":[], "linefeedQ": True}
            elif not contentsfieldQ:
                pldic = {"options":[], "comments":[], "linefeedQ": False}
            for a in line.split():
                #print(a)
                if "#" in a:
                    pldic["comments"].append(a.replace("#",""))
                    break
                elif "=" in a:
                    a = a.split("=")
                    pldic[a[0]] = copy.copy(a[1]).replace("\n", "")
                elif ":" in a:
                    pldic["LABEL"] = copy.copy(a.replace(":","").replace("\n", ""))
                else:
                    pldic["options"].append(a.replace("\n",""))
            if not contentsfieldQ:
                if not pldic["linefeedQ"]:
                    self.pldiclist.append(pldic)
    def exportplumed(self, filename, equibliumQ = False):
        writeline       = ""
        option_Main_names = ["PRINT", "RESTRAINT", "TORSION", "MOLINFO",
                "VES_LINEAR_EXPANSION", "BF_FOURIER", "TD_UNIFORM", "TD_WELLTEMPERED",
                "BF_CHEBYSHEV", "OPT_AVERAGED_SGD", "COM", "UNITS", "RESTART"]
        optionMainName = False
        for pldic in self.pldiclist:
            notwriteQ = False
            if pldic["linefeedQ"]:
                for k, v in pldic.items():
                    if k is "options":
                        for a in v:
                            if equibliumQ:
                                if "METAD" in a:
                                    #notwriteQ = True
                                    pldic["PACE"] = "500000"
                                    pldic["HEIGHT"] = "0.0"
                                    #break
                            if a in option_Main_names:
                                optionMainName = copy.copy(a)
                                writeline += "%s ...\n"%optionMainName
                                break
                if notwriteQ:
                    continue
                for k, v in pldic.items():
                    if equibliumQ and "FILE" in k:
                        FILEname = copy.copy(v)
                    if k is "options":
                        for a in v:
                            if "..." in a or optionMainName in a:
                                continue
                            writeline += "%s \n"%a
                for k, v in pldic.items():
                    if equibliumQ and k is "FILE":
                        FILEname = copy.copy(v)
                    if not k is "comments" and not k is "options" and not k is "linefeedQ":
                        if equibliumQ and k is "FILE":
                            writeline += "%s=%s_npt \n"%(k, v)
                        else:
                            writeline += "%s=%s \n"%(k, v)

                if len(pldic["comments"]) != 0:
                    writeline += " #%s\n"%(" ".join(pldic["comments"]))
                writeline += "... %s\n"%optionMainName
            else:
                #renamefileQ = False
                for k, v in pldic.items():
                    #if equibliumQ and k is "FILE":
                        #FILEname = copy.copy(v)
                    if k is "options":
                        for a in v:
                            if a in option_Main_names:
                                optionMainName = copy.copy(a)
                            if equibliumQ:
                                if "METAD" in a:
                                    #notwriteQ = True
                                    pldic["PACE"] = "500000"
                                    pldic["HEIGHT"] = "0.0"
                                    #break
                            writeline += "%s "%a
                if notwriteQ:
                    continue
                for k, v in pldic.items():
                    if k is "options":
                        continue
                    if not k is "comments" and not k is "linefeedQ":
                        if equibliumQ and k is "FILE":
                            writeline += "%s=%s_npt "%(k, v)
                        else:
                            writeline += "%s=%s "%(k, v)
                if len(pldic["comments"]) != 0:
                    writeline += " #%s\n"%(" ".join(pldic["comments"]))
                else:
                    writeline += "\n"
            if optionMainName is False:
                print("ERROR; there is not optionMainName")
                print("       please add Main Name of option in the optionMainNames")
                print(pldic["options"])
                exit()
#            if equibliumQ:
#                if "METAD" in writeline:
#                    continue
#                if "PRINT" in writeline:
#                    #print(writeline)
#                    writeline.replace(FILEname, FILEname + "_npt")


        with open(filename, "w") as wf:
            wf.write(writeline)
    def rewritepwd(self, pwdpath):
        for pldic in self.pldiclist:
            if "STRUCTURE" in pldic.keys():
                if "@pwd@" in pldic["STRUCTURE"]:
                    pldic["STRUCTURE"] = pldic["STRUCTURE"].replace("@pwd@", pwdpath)
    def addwall(self, eqpoint):
        arglist = []
        for pldic in self.pldiclist:
            for comment in pldic["comments"]:
                if "CV" in comment:
                    arglist.append(pldic["LABEL"])


#            pldic               = {"options":["COMBINE"], "comments":[], "linefeedQ": True}
#            pldic["PERIODIC"]   = "%s,%s"%(- np.pi, np.pi)
#            pldic["ARG"]        = ",".join(arglist)
#            pldic["PARAMETERS"] = ",".join([str(x) for x in eqpoint])
#            pldic["POWERS"]     = ",".join(["1" for _ in range(len(arglist))])
#            pldic["LABEL"]      = "combD"
#            self.pldiclist.append(pldic)
#            distarglist.append(arg + "_comb")
        distarglist = []
        for i, arg in enumerate(arglist):
            pldic               = {"options":["COMBINE"], "comments":[], "linefeedQ": True}
            pldic["PERIODIC"]   = "%s,%s"%(- np.pi, np.pi)
            pldic["ARG"]        = arg
            pldic["PARAMETERS"] = eqpoint[i]
            pldic["POWERS"]     = "1"
            pldic["LABEL"]      = arg + "_combD"
            self.pldiclist.append(pldic)
            distarglist.append(arg + "_combD")
        if True:
            pldic               = {"options":["COMBINE"], "comments":[], "linefeedQ": True}
            pldic["PERIODIC"]   = "NO"
            #pldic["ARG"]        = arg + "_combD"
            #pldic["ARG"]        = "combD"
            pldic["ARG"]        = ",".join(distarglist)
            #pldic["POWERS"]     = "2"
            pldic["POWERS"]     = ",".join(["2"] * len(distarglist))
            #pldic["LABEL"]      = distarglist[-1]
            pldic["LABEL"]      = "comb"
            self.pldiclist.append(pldic)

        pldic = {"options":["PRINT"], "comments":[], "linefeedQ": True}
        #pldic["ARG"]        = ",".join(distarglist)
        pldic["ARG"]        = "comb"
        pldic["STRIDE"]     = "100"
        pldic["FILE"]       = "COMBINE"
        self.pldiclist.append(pldic)


        pldic = {"options":["UPPER_WALLS"], "comments":[], "linefeedQ": True}
        #pldic["ARG"]    = ",".join(arglist)
        #pldic["ARG"]    = ",".join(distarglist)
        pldic["ARG"]    = "comb"
        #print(const.mpirunlist)
        #at_uwall        = eqpoint + const.wallDist * 0.5
        #at_uwall        = periodicpoint(at_uwall)
        at_uwall        = str(const.wallDist ** 2)
        #pldic["AT"]     = ",".join([str(x)  for x in at_uwall])
        #pldic["AT"]     = ",".join([at_uwall for _ in range(len(arglist))])
        #pldic["KAPPA"]  = ",".join(["100.0" for _ in range(len(arglist))])
        #pldic["EXP"]    = ",".join(["1"     for _ in range(len(arglist))])
        #pldic["EPS"]    = ",".join(["1"     for _ in range(len(arglist))])
        #pldic["OFFSET"] = ",".join(["0"     for _ in range(len(arglist))])
        pldic["AT"]     = at_uwall
        pldic["KAPPA"]  = const.wallF
        pldic["EXP"]    = "1"
        pldic["EPS"]    = "1"
        pldic["OFFSET"] = "0"
        pldic["LABEL"]  = "uwall"
        self.pldiclist.append(pldic)

#        pldic = {"options":["LOWER_WALLS"], "comments":[], "linefeedQ": True}
#        #pldic["ARG"]    = ",".join(arglist)
#        pldic["ARG"]    = ",".join(distarglist)
#        #print(const.mpirunlist)
#        #at_uwall        = eqpoint + const.wallDist * 0.5
#        #at_uwall        = periodicpoint(at_uwall)
#        at_uwall        = str(-const.wallDist * 0.5)
#        #pldic["AT"]     = ",".join([str(x)  for x in at_uwall])
#        pldic["AT"]     = ",".join([at_uwall for _ in range(len(arglist))])
#        pldic["KAPPA"]  = ",".join(["00.0" for _ in range(len(arglist))])
#        pldic["EXP"]    = ",".join(["1"     for _ in range(len(arglist))])
#        pldic["EPS"]    = ",".join(["1"     for _ in range(len(arglist))])
#        pldic["OFFSET"] = ",".join(["0"     for _ in range(len(arglist))])
#        pldic["LABEL"]  = "lwall"
#        self.pldiclist.append(pldic)

        pldic = {"options":["PRINT"], "comments":[], "linefeedQ": True}
        #pldic["ARG"]        = "uwall.bias,lwall.bias"
        pldic["ARG"]        = "uwall.bias"
        pldic["STRIDE"]     = "100"
        pldic["FILE"]       = "WALL"
        self.pldiclist.append(pldic)

#        pldic = {"options":["LOWER_WALLS"], "comments":[], "linefeedQ": True}
#        pldic["ARG"]    = ",".join(arglist)
#        at_uwall        = eqpoint - const.wallDist * 0.5
#        at_uwall        = periodicpoint(at_uwall)
#        pldic["AT"]     = ",".join([str(x)  for x in at_uwall])
#        pldic["KAPPA"]  = ",".join(["500.0" for _ in range(len(arglist))])
#        pldic["EXP"]    = ",".join(["2"     for _ in range(len(arglist))])
#        pldic["EPS"]    = ",".join(["1"     for _ in range(len(arglist))])
#        pldic["OFFSET"] = ",".join(["0"     for _ in range(len(arglist))])
#        pldic["LABEL"]  = "lwall"
#        self.pldiclist.append(pldic)



def TARGZandexit(const):
    """
    tar and gzip the result file and exit
    the .tar.gz file will move to pwdpath 
       if calculation is tried on temp dir
    """
    #exit()  ##debag
    if const.moveQ:
        os.chdir(tmppath) # cd from jobfiles
        tarfilename = "./jobfiles%s.tar.gz"%time.time()
        targzf = tarfile.open(tarfilename, 'w:gz')
        tar_write_dir(tmppath, "./jobfiles", targzf)
        targzf.close()
        print("Data files were saved in %s."%tarfilename) 
    print("""

He who fights with animals should look to it
that he himself does not become an animal.
And when you gaze long into an zoo of your graduate school of science
the zoo of your graduate school of science also gazes into you.

""")
    if const.moveQ:
        hostname = callhostname()[0]
        print("%s is in %s on %s"%(tarfilename, tmppath, hostname.decode()))
    #sys.exit()
def SQaxes(eigNlist, eigVlist, dim):
    _SQaxes = np.array([
            1.0 / np.sqrt(eigNlist[i]) * eigVlist[i]
            #np.sqrt(eigNlist[i]) * eigVlist[i]
            for i in range(dim)
            ])
    _SQaxes = np.transpose(_SQaxes)
    #print("SQaxes = %s"%SQaxes)
    #print("SQaxes^inv = %s"%np.linalg.inv(SQaxes))
    return _SQaxes
def SQaxes_inv(eigNlist, eigVlist, dim):
    SQaxes = np.array([
            #1.0 / np.sqrt(eigNlist[i]) * eigVlist[i]
            np.sqrt(eigNlist[i]) * eigVlist[i]
            for i in range(dim)
            ])
    #SQaxes = np.transpose(SQaxes)
    #print("SQaxes_inv = %s"%SQaxes)
    #print("SQaxes_inv^inv = %s"%np.linalg.inv(SQaxes))
    return SQaxes
def calctheta(xlist, eigVlist, eigNlist):
    """
    the basis transformathon from cartesian coordinates to polar one.
    xlist = {x_1, x_2,..., x_n} -> qlist = {q_1, q_2,..., q_n} -> _thetalist = {theta_n-1,..., theta_1}
    this function only calculate thetalist: r (length) is ignored.
    """
    if xlist is False:
        return False
    SQ    = SQaxes_inv(eigNlist, eigVlist, len(xlist))
    qlist = list(np.dot(SQ, xlist))
    #qlist = list(xlist)
    qlist.reverse()
    _thetalist = [1.0 for _ in range(len(qlist) - 1)]

    r = qlist[0] * qlist[0] + qlist[1] * qlist[1]
    if r == 0.0:
        _thetalist[0] = 0.0
    else:
        _thetalist[0] = np.arccos(qlist[1] / np.sqrt(r))
    if qlist[0] <= 0.0:
        _thetalist[0] = np.pi * 2.0 - _thetalist[0]
    for i in range(1, len(qlist) - 1):
        r += qlist[i + 1] * qlist[i + 1]
        if r == 0.0:
            _thetalist[i] = 0.0
        else:
            _thetalist[i] = np.arccos(qlist[i + 1] / np.sqrt(r))
    _thetalist.reverse()
    return np.array(_thetalist)
def SuperSphere_cartesian(eigNlist, eigVlist, A, thetalist, dim):
    """
    vector of super sphere by cartetian 
    the basis transformathon from polar coordinates to cartesian coordinates
    {sqrt(2*A), theta_1,..., theta_n-1} ->  {q_1,..., q_n} -> {x_1, x_2,..., x_n}
    cf: thetalist = {theta_n-1,..., theta_1}
    """
    #qlist = [A * A  for i in range(dim)]
    #qlist = [1.0 for i in range(dim)]
    #qlist = [2.0 * A * A  for i in range(dim)]
    qlist = [np.sqrt(2.0 * A) for i in range(dim)]
    #qlist = [ A / 2.0 for i in range(dim)]
    a_k = 1.0
    for i, theta in enumerate(thetalist):
        qlist[i] *= a_k * np.cos(theta)
        a_k *= np.sin(theta)
    qlist[-1] *= a_k
    #qlist.reverse()

#    a = 0.0
#    for i in range(len(eigNlist)):
#        a += (np.dot(eigVlist[i], qlist) * np.sqrt(eigNlist[i])) ** 2
#    SSvec = np.sqrt(2.0 * A / a)
#    SSvec = np.array(qlist) * SSvec

    SQ    = SQaxes(eigNlist ,eigVlist, len(qlist))
    #SQ    = np.linalg.inv(SQ)
    SSvec =  np.dot(SQ, qlist)
    return SSvec
def calcsphereA(nADD, eigNlist, eigVlist):
    """
    harmonic potential of the EQ point
    """
    SQ   = SQaxes_inv(eigNlist, eigVlist, len(eigNlist))
    qvec = np.dot(SQ, nADD)
    r    = np.linalg.norm(qvec)
    return 0.5 * r * r
    #return r
    #return np.sqrt(2.0 * r)

#    Evec  = nADD / np.linalg.norm(nADD)
#    a= 0.0
#    for i in range(len(eigNlist)):
#        a += (np.dot(nADD, eigVlist[i]) * np.sqrt(eigNlist[i])) ** 2
#    A    = 0.5 * a
#    #print(A)
#    #print(0.5 * r * r)
#    return  A
#def IOE(nADD, nADDneibor, ADDfeM, eigNlist, eigVlist, const):
def IOE(nADD, nADDneibor, ADDfeM, SQ_inv, const):
    if (nADD == nADDneibor).all():
        return ADDfeM
    if const.cythonQ:
        #return  const.calcgau.IOE(nADD, nADDneibor, eigNlist, eigVlist, ADDfeM)
        return  const.calcgau.IOE(nADD, nADDneibor, SQ_inv, ADDfeM)
    #deltaTH = angle(nADD, nADDneibor)
    #deltaTH = angle_SHS(nADD, nADDneibor, eigNlist ,eigVlist, const)
    deltaTH = angle_SHS(nADD, nADDneibor, SQ_inv, const)
    if deltaTH <= np.pi * 0.5:
        cosdamp = np.cos(deltaTH)
        return ADDfeM * cosdamp * cosdamp * cosdamp
    else:
        return 0.0
#def angle_SHS(x, y, eigNlist, eigVlist, const):
def angle_SHS(x, y, SQ_inv, const):
    #SQ  = SQaxes_inv(eigNlist, eigVlist, len(eigNlist))
    q_x = np.dot(SQ_inv, x)
    q_y = np.dot(SQ_inv, y)
    return angle(q_x, q_y)
def angle(x, y):
    """
    angle between x vector and y vector
    0 < angle < pi in this calculation
    """
    dot_xy = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(len(x)):
        dot_xy += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]
    _cos = dot_xy / np.sqrt(norm_x * norm_y)
    if _cos > 1:
        return 0.0
    elif _cos < -1:
        return np.pi
    return np.arccos(_cos)
def wallQ(x, const):
    dim = len(x)
    for i in range(dim):
        if x[i] < const.wallmin[i] or const.wallmax[i] < x[i]:
            return True
    return False
def periodicpoint(x, const, beforepoint = False):
    """
    periodicpoint: periodic calculation of x
    """
    bdamp = copy.copy(x)
    if const.periodicQ:
        if type(const.periodicmax) is float:
            for i in range(len(x)):
                if beforepoint is False:
                    if x[i] < const.periodicmin or const.periodicmax < x[i]:
                        bdamp[i]  = (x[i] - const.periodicmax) % (const.periodicmin - const.periodicmax)
                        bdamp[i] += const.periodicmax
                else:
                    if bdamp[i] < const.periodicmin + beforepoint[i] or const.periodicmax + beforepoint[i] < bdamp[i]:
                        bdamp[i]  = (x[i] - const.periodicmax - beforepoint[i]) % (const.periodicmin - const.periodicmax)
                        bdamp[i] += const.periodicmax + beforepoint[i]
        else:
            for i in range(len(x)):
                if beforepoint is False:
                    if x[i] < const.periodicmin[i] or const.periodicmax[i] < x[i]:
                        bdamp[i]  = (x[i] - const.periodicmax[i]) % (const.periodicmin[i] - const.periodicmax[i])
                        bdamp[i] += const.periodicmax[i]
                else:
                    if bdamp[i] < const.periodicmin[i] + beforepoint[i] or const.periodicmax[i] + beforepoint[i] < bdamp[i]:
                        bdamp[i]  = (x[i] - const.periodicmax[i] - beforepoint[i]) % (const.periodicmin[i] - const.periodicmax[i])
                        bdamp[i] += const.periodicmax[i] + beforepoint[i]
    return bdamp
def importplumed(filename):
    if not os.path.exists(filename):
        return False
    contentsfieldQ = False
    returnlist = []
    for line in open(filename):
        if line[0] is "#":
            continue
        if "..." in line:
            if contentsfieldQ:
                returnlist.append(pldic)
                contentsfieldQ = False
            else:
                contentsfieldQ = True
                pldic = {"options":[], "comments":[], "linefeedQ": True}
        elif not contentsfieldQ:
            pldic = {"options":[], "comments":[], "linefeedQ": False}
        for a in line.split():
            #print(a)
            if "#" in a:
                pldic["comments"].append(a.replace("#",""))
                break
            elif "=" in a:
                a = a.split("=")
                pldic[a[0]] = copy.copy(a[1]).replace("\n", "")
            elif ":" in a:
                pldic["LABEL"] = copy.copy(a.replace(":","").replace("\n", ""))
                #print("function:pldic[Label]=%s"%pldic["LABEL"])
            else:
                pldic["options"].append(a.replace("\n",""))
        if not contentsfieldQ:
            #if not len(pldic) == 1:
            if not pldic["linefeedQ"]:
                returnlist.append(pldic)
    return returnlist
def exportplumed(filename, pldiclist):
    writeline = ""
    for pldic in pldiclist:
        if pldic["linefeedQ"]:
            option_Main_names = ["PRINT", "RESTRAINT", "TORSION", "MOLINFO",
                "VES_LINEAR_EXPANSION", "BF_FOURIER", "TD_UNIFORM", "TD_WELLTEMPERED",
                "OPT_AVERAGED_SGD"]
            for k, v in pldic.items():
                if k is "options":
                    for a in v:
                        if a in option_Main_names:
                            optionMainanme = copy.copy(a)
                            writeline += "%s ...\n"%optionMainanme
                            break
            for k, v in pldic.items():
                if k is "options":
                    for a in v:
                        if "..." in a or optionMainanme in a:
                            continue
                        writeline += "%s \n"%a
            for k, v in pldic.items():
                if not k is "comments" and not k is "options" and not k is "linefeedQ":
                    writeline += "%s=%s \n"%(k, v)

            if len(pldic["comments"]) != 0:
                writeline += " #%s\n"%(" ".join(pldic["comments"]))
            writeline += "... %s\n"%optionMainanme
        else:
            for k, v in pldic.items():
                if k is "options":
                    for a in v:
                        writeline += "%s "%a
            for k, v in pldic.items():
                if k is "options":
                    continue
                if not k is "comments" and not k is "linefeedQ":
                    writeline += "%s=%s "%(k, v)
            if len(pldic["comments"]) != 0:
                writeline += " #%s\n"%(" ".join(pldic["comments"]))
            else:
                writeline += "\n"
    with open(filename, "w") as wf:
        wf.write(writeline)
