#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2019/12/18 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import copy
import subprocess as sp
import numpy as np
#import functions
from mpi4py import MPI
if True:
    import pyximport  # for cython
    pyximport.install()
    try:
        from . import calcRCMC
    except ImportError:
        import calcRCMC

from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def main(const):
    """
    
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    #args = sys.argv
    #zpozition = int(sys.argv[1])
    #zpozition = 35

    #jobfilepath = "./jobfiles_%s"%zpozition
    if const.rank == const.root:
        eqlist = []
        eqpointdic = {}
        for line in open("./%s/eqlist.csv"%const.jobfilepath):
            if "#" in line:
                continue
            line = line.split(",")
            eqname = line[0]
            eqpoint = np.array(line[1:-1], dtype = float)
            eqlist.append([eqname, eqpoint])
            eqpointdic[eqname] = eqpoint
        tslist = []
        for line in open("./%s/tslist.csv"%const.jobfilepath):
            if "#" in line:
                continue
            line = line.split(",")
            tsname = line[0]
            tspoint = np.array(line[1:-1], dtype = float)
            #if 30.0 < abs(tspoint[-1]):
                #continue
            tslist.append([tsname, tspoint])
        connections = []
        for line in open("./%s/connections.csv"%const.jobfilepath):
            if "#" in line:
                continue
            line = line.replace("\n", "").replace(" ","").split(",")
            connections.append(line)
        diff = False
        if const.useZpozitionQ:
            if const.zpozition < 30.0:
                if os.path.exists("./diff_inner.csv"):
                    difflist = []
                    for line in open("./diff_inner.csv"):
                        difflist.append(float(line))
                    diff, diff_z = difflist
            else:
                if os.path.exists("./diff_outer.csv"):
                    difflist = []
                    for line in open("./diff_outer.csv"):
                        difflist.append(float(line))
                    diff, diff_z = difflist
        else:
            if os.path.exists("./diff.csv"):
                difflist = []
                for line in open("./diff.csv"):
                    difflist.append(float(line))
                diff, diff_z = difflist
        if diff is False:
            diff, diff_z = makediff(const)
    else:
        tslist        = None
        #velocitylists = None
        diff = None
        diff_z = None
    if const.size != 1:
        tslist = const.comm.bcast(tslist, root = 0)
        diff   = const.comm.bcast(diff,   root = 0)
        diff_z = const.comm.bcast(diff_z, root = 0)


    #v = np.average([x[-1] for x in velocitylist for velocitylist in velocitylists])
    #v = np.average([x[-1] for x in velocitylist])
    print("diff = %s"%diff)
    #exit()
    writeline = ""
    for tsN, (tsname, tspoint) in enumerate(tslist):
        #if tsN % const.size != const.rank:
            #continue
        #freqpath = "./%s/Frequency/%s.csv"%(const.jobfilepath, tsname)
        #if os.path.exists(freqpath):
            #continue
        #print(tsname, flush = True)
        initialpointlists = [[],[]]
        MFEPlengths = [0.0, 0.0]
        for i in range(2):
            if MFEPlengths is False:
                continue
            beforepoint = copy.copy(tspoint)
            pathlistpath = "./%s/%s/pathlist%s.csv"%(const.jobfilepath, tsname, i)
            if not os.path.exists(pathlistpath):
                MFEPlengths = False
                break
            for line in open(pathlistpath):
                line = line.split(",")
                initialpoint = np.array(line[:-1], dtype=float)
                #initialpointdamp = periodicpoint(initialpoint, const, beforepoint)
                if len(initialpoint) != len(beforepoint):
                    print("ERROR in %s"%tsname)
                    MFEPlengths = False
                    break
                dis = periodicpoint(initialpoint, const, beforepoint)
                #print(dis)
                #print(np.linalg.norm(VeloVec))
                if beforepoint is not False:
                    #MFEPlengths[i] += np.linalg.norm(beforepoint - initialpointdamp)
                    MFEPlengths[i] += dis

                initialpointlists[i].append(initialpoint)
                beforepoint = copy.copy(initialpoint)

        if MFEPlengths is False:
            continue
        MFEPlength = sum(MFEPlengths)
        #MFEPlength = periodicpoint(initialpointlists[0][0], const, initialpointlists[0][-1])
        eqconnectlist = []
        for tsconnect, eqconnect in connections:
            if tsconnect == tsname:
                eqconnectlist.append(eqconnect)
        for eqconnect in eqconnectlist:
            #MFEPlength = periodicpoint(tspoint, const, eqpointdic[eqconnect])
            freq = diff / MFEPlength / MFEPlength
            print("%s; freqency factor = % 10.8f"%(tsname, freq), flush =True)
            writeline += "%s, %s, %s\n"%(eqconnect, tsname, freq)
        #with open(freqpath, "w") as wf:
            #wf.write("%s"%freq)
        #MFEPlength = periodicpoint(initialpointlists[1][0], const, initialpointlists[1][-1])
        #freq = diff / MFEPlength / MFEPlength
        #print("%s; freqency factor = % 10.8f"%(tsname, freq), flush =True)
        #with open(freqpath, "a") as wf:
            #wf.write("\n%s"%freq)
    #writeline = ""
    #for tsN, (tsname, tspoint) in enumerate(tslist):
        #fricname = "./%s/%s/Frequency.csv"%(const.jobfilepath, tsname)
        #if not os.path.exists(fricname):
            #continue
        #friclist = []
        #for line in open(fricname):
            #friclist.append(float(line))
        #i = 0
        #for tsconnect, eqconnect in connections:
            #if tsconnect == tsname:
        #tsname = fricname.split("/")[3]
                #writeline += "%s, %s, %s\n"%(eqconnect, tsname,friclist[i])
                #i += 1
            #if 2 <= i:
                #break
        #print(tsname)
    with open("./%s/Frequencylist.csv"%const.jobfilepath, "w") as wf:
        wf.write(writeline)

    if const.useZpozitionQ:
        print("diff_z = %s"%diff_z)
        fric_z = diff_z / 1.0
        with open("./%s/fric_z.csv"%const.jobfilepath, "w") as wf:
            wf.write("%s"%fric_z)



def periodicpoint(x, const, beforepoint = False):
    """
    periodicpoint: periodic calculation of x
    """
    #print("x =%s"%x)
    #print("beforepoint =%s"%beforepoint)
    dis, bdamp = calcRCMC.periodicpoint(x, const.periodicmax, const.periodicmin, beforepoint)
    return dis
#    bdamp = copy.copy(x)
#    if const.periodicQ:
#        if type(const.periodicmax) is float:
#            print(beforepoint)
#            for i in range(len(x)):
#                if beforepoint is False:
#                    if x[i] < const.periodicmin or const.periodicmax < x[i]:
#                        bdamp[i]  = (x[i] - const.periodicmax) % (const.periodicmin - const.periodicmax)
#                        bdamp[i] += const.periodicmax
#                else:
#                    if bdamp[i] < const.periodicmin + beforepoint[i] or const.periodicmax + beforepoint[i] < bdamp[i]:
#                        bdamp[i]  = (x[i] - const.periodicmax - beforepoint[i]) % (const.periodicmin - const.periodicmax)
#                        bdamp[i] += const.periodicmax + beforepoint[i]
#        else:
#            for i in range(len(x)):
#                if beforepoint is False:
#                    if x[i] < const.periodicmin[i] or const.periodicmax[i] < x[i]:
#                        bdamp[i]  = (x[i] - const.periodicmax[i]) % (const.periodicmin[i] - const.periodicmax[i])
#                        bdamp[i] += const.periodicmax[i]
#                else:
#                    if bdamp[i] < const.periodicmin[i] + beforepoint[i] or const.periodicmax[i] + beforepoint[i] < bdamp[i]:
#                        bdamp[i]  = (x[i] - const.periodicmax[i] - beforepoint[i]) % (const.periodicmin[i] - const.periodicmax[i])
#                        bdamp[i] += const.periodicmax[i] + beforepoint[i]
#    return np.linalg.norm(bdamp)
def calcdiff(x, y):
    degree = 1
    x = np.array(x)
    y = np.array(y)
    model = make_pipeline(PolynomialFeatures(degree,include_bias=False),LinearRegression(fit_intercept=False))
    model.fit(x.reshape(-1,1),y)
    y_model=model.predict(x.reshape(-1,1))

    return model.steps[1][1].coef_[0]
def makediff(const):
    difflist   = []
    diff_zlist = []
    COVnames = glob.glob("%s/COLVAR.*"%const.colvarpath) 
    diff_trj = []
    for COVname in COVnames:
        print(COVname)
        initialpoint   = False
        tlist = []
        rsqlist = []
        MSDlist = []
        #rsqlist_z = []
        #MSDlist_z = []
        i = -1
        for line in open(COVname):
            if "#" in line:
                continue
            i += 1
            line = line.split()
            t    = float(line[0])
            #if t < 50000.0:
                #continue
            if const.useZpozitionQ:
                p    = np.array(line[1:-1], dtype = float)
                z    = float(line[-1])
            else:
                p    = np.array(line[1:], dtype = float)
            if initialpoint is False:
                initialpoint = p
                initial_t = t
                if const.useZpozitionQ:
                    initialpoint_z = z
            tlist.append(t - initial_t)
            #delta_t = t - before_t
            dis = periodicpoint(p, const, initialpoint)
            dis2 = dis * dis
            rsqlist.append(dis2)
            MSDlist.append(np.mean(rsqlist))
            if 10000 <= len(tlist):
                diff = calcdiff(tlist, MSDlist)
                diff = diff / 2.0 / len(p)
                print(diff)
                difflist.append(diff)
                initialpoint   = False
                if const.useZpozitionQ:
                    diff_trj.append([initialpoint_z,diff])
                else:
                    diff_trj.append(diff)
                tlist     = []
                rsqlist   = []
                MSDlist   = []
    if const.useZpozitionQ:
        diff_trj_z = []
        COVnames = glob.glob("%s/COLVAR_noabs.*"%const.colvarpath) 
        for COVname in COVnames:
            print(COVname)
            initialpoint_z = False
            tlist = []
            rsqlist_z = []
            MSDlist_z = []
            i = -1
            for line in open(COVname):
                if "#" in line:
                    continue
                i += 1
                #if i % 10 != 0:
                    #continue
                line = line.split()
                t    = float(line[0])
                z    = float(line[-1])
                if 35 < abs(z):
                    initialpoint_z = False
                    tlist = []
                    rsqlist_z = []
                    MSDlist_z = []
                    continue
                if initialpoint_z is False:
                    initialpoint_z = z
                    initial_t = t
                tlist.append(t - initial_t)
                dis = z - initialpoint_z
                dis2 = dis * dis
                rsqlist_z.append(dis2)
                MSDlist_z.append(np.mean(rsqlist_z))
                if 10000 <= len(tlist):
                    diff_z = calcdiff(tlist, MSDlist_z)
                    diff_z = diff_z / 2.0 
                    diff_zlist.append(diff_z)
                    diff_trj_z.append([initialpoint_z,diff_z])
                    initialpoint_z = False
                    tlist = []
                    rsqlist_z = []
                    MSDlist_z = []
        difflist   = [diff for z,diff in diff_trj if abs(z) < 30.0]
        difflist_z = [diff for z,diff in diff_trj_z if abs(z) < 30.0]
        diff   = np.mean(difflist)
        diff_z = np.mean(difflist_z)
        if const.zpozition < 30.0:
            returnlist = [diff, diff_z]
        with open("./diff_inner.csv", "w") as wf:
            wf.write("%s\n%s"%(diff, diff_z))
        std_error = np.std(difflist) / np.sqrt(len(difflist))
        std_error_z = np.std(difflist_z) / np.sqrt(len(difflist_z))
        with open("./diff_upmax_inner.csv", "w") as wf:
            wf.write("%s\n%s"%(diff+2.0*std_error, diff_z+2.0*std_error_z))
        with open("./diff_downmin_inner.csv", "w") as wf:
            wf.write("%s\n%s"%(diff-2.0*std_error, diff_z-2.0*std_error_z))
        with open("./std_error_inner.csv", "w") as wf:
            wf.write("%s\n%s"%(std_error, std_error_z))
    
        difflist   = [diff for z,diff in diff_trj if 30.0 <= abs(z) < 35.0]
        difflist_z = [diff for z,diff in diff_trj_z if 30.0 <= abs(z) < 35.0]
        diff   = np.mean(difflist)
        diff_z = np.mean(difflist_z)
        if 30.0 <= const.zpozition:
            returnlist = [diff, diff_z]
        with open("./diff_outer.csv", "w") as wf:
            wf.write("%s\n%s"%(diff, diff_z))
        std_error = np.std(difflist) / np.sqrt(len(difflist))
        std_error_z = np.std(difflist_z) / np.sqrt(len(difflist_z))
        with open("./diff_upmax_outer.csv", "w") as wf:
            wf.write("%s\n%s"%(diff+2.0*std_error, diff_z+2.0*std_error_z))
        with open("./diff_downmin_outer.csv", "w") as wf:
            wf.write("%s\n%s"%(diff-2.0*std_error, diff_z-2.0*std_error_z))
        with open("./std_error_outer.csv", "w") as wf:
            wf.write("%s\n%s"%(std_error, std_error_z))
        with open("./diff_trj.csv", "w") as wf:
            for z, diff in diff_trj:
                wf.write("%s, %s\n"%(z, diff))
        with open("./diff_trj_z.csv", "w") as wf:
            for z, diff in diff_trj_z:
                wf.write("%s, %s\n"%(z, diff))
    else:
        #difflist   = [diff for z,diff in diff_trj if 30.0 <= abs(z) < 35.0]
        #diff   = np.mean(difflist)
        diff   = np.mean(diff_trj)
        diff_z = 0.0
        returnlist = [diff, diff_z]
        with open("./diff.csv", "w") as wf:
            wf.write("%s\n0.0"%(diff))
        std_error = np.std(difflist) / np.sqrt(len(difflist))
        with open("./diff_upmax.csv", "w") as wf:
            wf.write("%s\n0.0"%(diff+2.0*std_error))
        with open("./diff_downmin.csv", "w") as wf:
            wf.write("%s\n0.0"%(diff-2.0*std_error))
        with open("./std_error.csv", "w") as wf:
            wf.write("%s\n0.0"%(std_error))
    #exit()
    return returnlist
if __name__ == "__main__":
    main()

