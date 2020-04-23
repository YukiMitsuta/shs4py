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
import functions
from mpi4py import MPI

def main():
    """
    
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    eqlist = []
    #for line in open("../gpu_7D400nsSHS2/jobfiles_meta/eqlist.csv"):
    for line in open("./jobfiles_meta/eqlist.csv"):
        if "#" in line:
            continue
        line = line.split(",")
        eqname = line[0]
        eqpoint = np.array(line[1:-1], dtype = float)
        eqlist.append([eqname, eqpoint])
    tslist = []
    #for line in open("../gpu_7D400nsSHS2/jobfiles_meta/tslist.csv"):
    for line in open("./jobfiles_meta/tslist.csv"):
        if "#" in line:
            continue
        line = line.split(",")
        tsname = line[0]
        tspoint = np.array(line[1:-1], dtype = float)
        tslist.append([tsname, tspoint])
    connections = []
    #for line in open("../gpu_7D400nsSHS2/jobfiles_meta/connections.csv"):
    for line in open("./jobfiles_meta/connections.csv"):
        if "#" in line:
            continue
        line = line.replace("\n", "").replace(" ","").split(",")
        connections.append(line)
    #hillslist = []
    #for line in open("./HILLS"):
        #if "#" in line:
            #continue
        #hillslist.append(line)

    covlists      = []
    velocitylists = []
    COVnames = glob.glob("../cpu_7D400nsTRJ*/run*/COLVAR.*")
    for COVname in COVnames:
        covlist      = []
        velocitylist = []
        before_point = False
        for line in open(COVname):
            if "#" in line:
                continue
            line = line.split()
            t    = float(line[0])
            p    = np.array(line[1:], dtype = float)
            covlist.append([t, p])
            if not before_point is False:
                delta_t = t - before_t
                veloVec = before_point - p
                dis     = np.linalg.norm(veloVec)
                velocitylist.append([before_t ,veloVec, dis / delta_t])
            before_t = copy.copy(t)
            before_point = copy.copy(p)
        covlists.append(covlist)
        velocitylists.append(velocitylist)

    for eqNcounter, (centerEQname, centerEQpoint) in enumerate(eqlist):
        if eqNcounter % size != rank:
            continue
        writeline = ""
        #COVname = "./%s/COLVAR"%centerEQname
        #if not os.path.exists(COVname):
            #continue
        #if os.path.exists("./%s/velocity_ave.csv"%centerEQname):
        if os.path.exists("./velocitydir/%s.csv"%centerEQname):
            continue
        print(centerEQname)
        #if not os.path.exists(centerEQname):
            #os.mkdir(centerEQname)
        center_connections = [x[0] for x in connections if x[-1] == centerEQname]
        print(center_connections)
        TSpathlist = []
        for center_connection in center_connections:
            TSpath = [x for x in connections if x[0] == center_connection]
            TSpathlist.extend(TSpath)
        if len(TSpathlist) == 0:
            continue
        for TSpath in TSpathlist:
            delta = 0.01
            TSpoint = [x[1] for x in tslist if x[0] == TSpath[0]][0]
            EQpoint = [x[1] for x in eqlist if x[0] == TSpath[1]][0]
            if TSpath[1] == centerEQname:
                vec = TSpoint - EQpoint
            else:
                vec = EQpoint - TSpoint
            initialpointlists = [[],[]]
            MFEPlengths = [0.0, 0.0]
            for i in range(2):
                beforepoint = False
                for line in open("./jobfiles_meta/%s/pathlist%s.csv"%(TSpath[0], i)):
                    line = line.split(",")
                    initialpoint = np.array(line[:-1], dtype=float)
                    if beforepoint is not False:
                        MFEPlengths[i] += np.linalg.norm(beforepoint - initialpoint)
                    initialpointlists[i].append(initialpoint)
                    beforepoint = copy.copy(initialpoint)
            #if np.linalg.norm(initialpointlists[0][-1] - EQpoint) < np.linalg.norm(initialpointlists[1][-1] - EQpoint):
                #initialpointlist = initialpointlists[0]
                #MFEPlength = MFEPlengths[0]
            #else:
                #initialpointlist = initialpointlists[1]
                #MFEPlength = MFEPlengths[1]
            MFEPlength = sum(MFEPlengths)
            initialpointlist = initialpointlists[0] + initialpointlists[1]

            #print(initialpointlist)
            #print("%s, %s"%(TSpath, np.linalg.norm(vec)))
            #initialpoint = copy.copy(EQpoint)
            if True:
            #while True:
                targettimes = []
                for covlistN in range(len(covlists)):
                    for i, (t, covpoint) in enumerate(covlists[covlistN][:-1]):
                        if i == 0:
                            continue
                        #for j in range(6):
                        beforepoint = False
                        for initialpoint in initialpointlist:
                            #if TSpath[1] == centerEQname:
                                #initialpoint = EQpoint + vec * j * 0.2
                            #else:
                                #initialpoint = TSpoint + vec * j * 0.2
                            if beforepoint is False:
                                beforepoint = copy.copy(initialpoint)
                                continue
                            vec = initialpoint - beforepoint
                            #dis = np.linalg.norm(initialpoint - covpoint)
                            dis = 0.0
                            v = initialpoint - covpoint
                            for x in v:
                                dis += x * x
                            #dis = np.sqrt(dis)
                            if len(targettimes) != 0:
                                dismax = targettimes[-1][0]
#                                dismax = max([x[0] for x in targettimes])
#                                for dismax_index, x in enumerate(targettimes):
#                                    if dismax == x[0]:
#                                        break
                            else:
                                dismax       = 1.0e30
#                                dismax_index = -1
                            if dis < dismax:
                                velocitypoint = velocitylists[covlistN][i]
    
                                velot, veloVec, v= velocitypoint
                                E1 = vec / np.linalg.norm(vec)
                                E2 = veloVec / np.linalg.norm(veloVec)
                                if  np.sqrt(3.0) / 2.0 < abs(np.dot(E1, E2)):
                                    #targettimes.append([dis, t, v / MFEPlength])
                                    for dis_index, x in enumerate(targettimes):
                                        if dis < x[0]:
                                            targettimes.insert(dis_index, [dis, t, v/ MFEPlength])
                                            break
                                    else:
                                        targettimes.append([dis, t, v / MFEPlength])
                                    break
                            beforepoint = copy.copy(initialpoint)
                        if 50 < len(targettimes):
                            #targettimes.pop(dismax_index)
                            targettimes = targettimes[:50]
                        #targettimes = sorted(targettimes, key = lambda x:x[0])[:50]

                #if len(targettimes) < 5:
                    #delta += 0.01
                    #delta *= 2.0
                    #print(delta)
                #else:
                    #break
                #if delta > 5:
                    #break
            #targettimes = sorted(targettimes, key = lambda x:x[0])[:50]
            if 5 < delta:
                print("ERROR:delta = %s"%delta)
                wr = "%s, %s, inf\n"%(TSpath[1], TSpath[0])
            elif TSpath[1] == centerEQname:
                wr = "%s, %s, %s\n"%(TSpath[1], TSpath[0], np.average([x[-1] for x in targettimes]))
            else:
                wr = "%s, %s, %s\n"%(TSpath[0], TSpath[1], np.average([x[-1] for x in targettimes]))
            print("delta = %s"%delta)
            print("len(targettimes) = %s"%len(targettimes))
            print(wr)
            writeline += wr
        with open("./velocitydir/%s.csv"%centerEQname, "w") as wf:
            wf.write(writeline)
    exit()

#        TSset = list(set([x[0] for x in TSpathlist]))
#        EQset = list(set([x[1] for x in TSpathlist]))
#        calclist = []
#        calcstr  = ""
#        for calcpoint in TSset + EQset:
#            point = [x for x in eqlist + tslist if x[0] == calcpoint]
#            calclist.extend(point)
#        for point in calclist:
#            calcstr += point[0]
#            for x in point[1]:
#                calcstr += ",%s"%x
#            calcstr += "\n"
#        with open("./%s/calclist.csv"%centerEQname, "w") as wf:
#            wf.write(calcstr)
#        pldicC = functions.PlumedDatClass("./plumed.dat")
#        for pldic in pldicC.pldiclist:
#            if "METAD" in pldic["options"]:
#                arglist = pldic["ARG"].split(",")
#                break
#        Uwall_dic           = {"options":["UPPER_WALLS"], "comments":[], "linefeedQ": False}
#        Lwall_dic           = {"options":["UPPER_WALLS"], "comments":[], "linefeedQ": False}
#        Uwall_dic["ARG"]    = ""
#        Lwall_dic["ARG"]    = ""
#        Uwall_dic["AT"]     = ""
#        Lwall_dic["AT"]     = ""
#        Uwall_dic["KAPPA"]  = ""
#        Lwall_dic["KAPPA"]  = ""
#        Uwall_dic["EXP"]    = ""
#        Lwall_dic["EXP"]    = ""
#        Uwall_dic["EPS"]    = ""
#        Lwall_dic["EPS"]    = ""
#        Uwall_dic["OFFSET"] = ""
#        Lwall_dic["OFFSET"] = ""
#        Uwall_dic["LABEL"]  = "uwall"
#        Lwall_dic["LABEL"]  = "lwall"
#        for i in range(7):
#            pointrange = [x[1][i] for x in calclist]
#            pointrange = [min(pointrange) , max(pointrange)]
#            Uwall_dic["ARG"]    += ",%s"%arglist[i]
#            Lwall_dic["ARG"]    += ",%s"%arglist[i]
#            Uwall_dic["AT"]     += ",%s"%(max(pointrange) + 0.05)
#            Lwall_dic["AT"]     += ",%s"%(min(pointrange) - 0.05)
#            Uwall_dic["KAPPA"]  += ",50.0"
#            Lwall_dic["KAPPA"]  += ",50.0"
#            Uwall_dic["EXP"]    += ",2"
#            Lwall_dic["EXP"]    += ",2"
#            Uwall_dic["EPS"]    += ",1"
#            Lwall_dic["EPS"]    += ",1"
#            Uwall_dic["OFFSET"] += ",0"
#            Lwall_dic["OFFSET"] += ",0"
#
#        Uwall_dic["ARG"]    = Uwall_dic["ARG"].lstrip(",")
#        Lwall_dic["ARG"]    = Lwall_dic["ARG"].lstrip(",")
#        Uwall_dic["AT"]    = Uwall_dic["AT"].lstrip(",")
#        Lwall_dic["AT"]    = Lwall_dic["AT"].lstrip(",")
#        Uwall_dic["KAPPA"]    = Uwall_dic["KAPPA"].lstrip(",")
#        Lwall_dic["KAPPA"]    = Lwall_dic["KAPPA"].lstrip(",")
#        Uwall_dic["EXP"]    = Uwall_dic["EXP"].lstrip(",")
#        Lwall_dic["EXP"]    = Lwall_dic["EXP"].lstrip(",")
#        Uwall_dic["EPS"]    = Uwall_dic["EPS"].lstrip(",")
#        Lwall_dic["EPS"]    = Lwall_dic["EPS"].lstrip(",")
#        Uwall_dic["OFFSET"]    = Uwall_dic["OFFSET"].lstrip(",")
#        Lwall_dic["OFFSET"]    = Lwall_dic["OFFSET"].lstrip(",")
#
#        pldicC.pldiclist.append(Uwall_dic)
#        pldicC.pldiclist.append(Lwall_dic)
#        pldicC.exportplumed("./%s/plumed.dat"%centerEQname, equibliumQ = False)
#
#        if os.path.exists("%s/trjfit.csv"%centerEQname):
#            continue
#
#        newhillslist = []
#        for hillsline in hillslist:
#            hillpoint = hillsline.split()
#            for i in range(7):
#                if float(hillpoint[i + 1]) < pointrange[0] - 0.30 \
#                        or pointrange[1] + 0.30 < float(hillpoint[i + 1]):
#                    break
#            else:
#                newhillslist.append(hillsline)
#        with open("./%s/HILLS"%centerEQname, "w") as wf:
#            for newhill in newhillslist:
#                wf.write(newhill)
#        os.chdir(centerEQname)
#        shutil.copy("/home/yuki/gromacs/metadynamics/1plx/meta_vanila/gpu_7D400ns/pdbfiles/%s.gro"%centerEQname, "./npt.gro")
#        sp.call(["../calljobTRJ.sh"])
#
#        COVname = "./COLVAR"
#        writeline = ""
#        for line in open(COVname):
#            if "#" in line:
#                continue
#            line = line.split()
#            t    = line[0]
#            p    = np.array(line[1:], dtype = float)
#            dis = 1.0e30
#            for eqname, eqpoint in eqlist:
#                eqdis = np.linalg.norm(p - eqpoint)
#                if eqdis < dis:
#                    dis = copy.copy(eqdis)
#                    nearestpoint = copy.copy(eqname)
#            for tsname, tspoint in tslist:
#                tsdis = np.linalg.norm(p - tspoint)
#                if tsdis < dis:
#                    dis = copy.copy(tsdis)
#                    nearestpoint = copy.copy(tsname)
#            writeline += "%s, %s, %s\n"%(t, nearestpoint, dis)
#        with open("./trjfit.csv", "w") as wf:
#            wf.write(writeline)
#        print("%s is analyzed"%centerEQname)
#
#        os.chdir("../")

if __name__ == "__main__":
    main()


