#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020/10/14 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import subprocess as sp

from SHS4py import functions

class Dihedral_info():
    def __init__(self, line):
        self.dihlist = line[:4]
        self.func = int(line[4])
        self.phi0 = float(line[5])
        self.Kphi = float(line[6])
        try:
            self.mult = int(line[7])
        except:
            pass

def main():
    """
    
    """
    dihClasslist = importforcefield()
    typedic, dihNumberlist = importtypedic()
    freedihlistdamp = []
    for dihNumber in dihNumberlist:
        dih_names = [typedic[x] for x in dihNumber]
        if "Nan" in dih_names:
            continue
        if "O" in dih_names:
            continue
        dih_Kphi = False
        for dihClass in dihClasslist:
            dihbool = [False, False, False, False]
            for i in range(4):
                if dihClass.dihlist[i] == "X" or dih_names[i] == dihClass.dihlist[i]:
                    dihbool[i] = True
            if all(dihbool):
                dih_Kphi = dihClass.Kphi
                break
            dihbool = [False, False, False, False]
            for i in range(4):
                if dihClass.dihlist[i] == "X" or dih_names[3-i] == dihClass.dihlist[i]:
                    dihbool[i] = True
            if all(dihbool):
                dih_Kphi = dihClass.Kphi
                break
        else:
            print("ERROR: There is not %s in forcefield"%dih_names)
            #exit()
            continue
        if 0.0 < dih_Kphi < 1.0:
            #print(dih_names)
            #print(dihNumber)
            #print("dih_Kphi = %s"%dih_Kphi)
            #print("dih_phi0 = %s"%dihClass.phi0)
            #print("+++")
            freedihlistdamp.append(dihNumber)
    #print(returndihlist)
    print("len(freedihlistdamp) = %s"%len(freedihlistdamp))
    freedihlist = []
    for freedih_b in freedihlistdamp:
        for freedih_f in freedihlist:
            if freedih_f[:3] == freedih_b[:3]:
                break
            elif freedih_f[-3:] == freedih_b[-3:]:
                break
            freedih_b.reverse()
            if freedih_f[:3] == freedih_b[:3]:
                break
            elif freedih_f[-3:] == freedih_b[-3:]:
                break
        else:
            if int(freedih_b[-1]) < int(freedih_b[0]):
                freedih_b.reverse()
            freedihlist.append(freedih_b)
    freedihlist.sort(key = lambda x: int(x[0]))
    pairlist = []
    for i, freedih_i in enumerate(freedihlist):
        for getpair_i in range(3):
            pair_i = set(freedih_i[getpair_i:getpair_i+2])
            for j, freedih_j in enumerate(freedihlist):
                if j <=i:
                    continue
                for getpair_j in range(3):
                    pair_j = set(freedih_j[getpair_j:getpair_j+2])
                    #print("%s, %s"%(pair_i, pair_j))
                    if len(pair_i & pair_j) == 2:
                        pairlist.append([i,j])
                        break
                else:
                    continue
                break
            else:
                continue
            break
    print("len(freedihlist) = %s"%len(freedihlist))
    print(freedihlist)
    print(pairlist)
    print("len(pairlist) = %s"%len(pairlist))

    plumedpath = ""
    pldicC = functions.PlumedDatClass(plumedpath)
    pldic_UNITS = {"options":["UNITS"], "comments":[], "linefeedQ": False}
    pldic_UNITS["LENGTH"] = "A"
    pldicC.pldiclist.append(pldic_UNITS)

    arglist = []
    for i, freedih in enumerate(freedihlist):
        pldic_TORSION = {"options":["TORSION"], "comments":["CV%s"%i], "linefeedQ": False}
        pldic_TORSION["ATOMS"] = ",".join(freedih)
        pldic_TORSION["LABEL"] = "tor%s"%i
        arglist.append("tor%s"%i)
        pldicC.pldiclist.append(pldic_TORSION)
    pldic = {"options":["CENTER"], "comments":[], "linefeedQ": False}
    pldic["LABEL"] = "c_pro"
    pldic["Atoms"] = "1-196"
    pldicC.pldiclist.append(pldic)
    pldic = {"options":["CENTER"], "comments":[], "linefeedQ": False}
    pldic["LABEL"] = "c_bil"
    pldic["Atoms"] = "196-8270"
    pldicC.pldiclist.append(pldic)
    pldic = {"options":["DISTANCE","COMPONENTS"], "comments":[], "linefeedQ": False}
    pldic["LABEL"] = "p"
    pldic["ATOMS"] = "c_pro,c_bil"
    pldicC.pldiclist.append(pldic)
    pldic = {"options":["COMBINE"], "comments":[], "linefeedQ": False}
    pldic["LABEL"]  = "pz2"
    pldic["ARG"]    = "p.z"
    pldic["POWERS"] = "2"
    pldic["PERIODIC"] = "NO"
    pldicC.pldiclist.append(pldic)
    pldic = {"options":["COMBINE"], "comments":[], "linefeedQ": False}
    pldic["LABEL"]  = "pz"
    pldic["ARG"]    = "pz2"
    pldic["POWERS"] = "0.5"
    pldic["PERIODIC"] = "NO"
    pldicC.pldiclist.append(pldic)
    pldic = {"options":["BF_FOURIER"], "comments":[], "linefeedQ": True}
    pldic["ORDER"]   = "15"
    pldic["MINIMUM"] = "-pi"
    pldic["MAXIMUM"] = "pi"
    pldic["LABEL"]   = "bfF"
    pldicC.pldiclist.append(pldic)
    pldic = {"options":["BF_FOURIER"], "comments":[], "linefeedQ": True}
    pldic["ORDER"]   = "50"
    pldic["MINIMUM"] = "0"
    pldic["MAXIMUM"] = "50.0"
    pldic["LABEL"]   = "bfC"
    pldicC.pldiclist.append(pldic)

    VESlabellist = []
    i = 0
    for pair in pairlist:
        pldic_TD = {"options":["TD_WELLTEMPERED"], "comments":[], "linefeedQ": False}
        pldic_TD["BIASFACTOR"] = "10"
        pldic_TD["LABEL"] = "td%s"%i
        pldicC.pldiclist.append(pldic_TD)
        pldic_VES = {"options":["VES_LINEAR_EXPANSION"], "comments":[], "linefeedQ": True}
        pldic_VES["BASIS_FUNCTIONS"] = "bfF,bfF"
        pldic_VES["TARGET_DISTRIBUTION"] = "td%s"%i
        pldic_VES["GRID_BINS"] = "100"
        pldic_VES["ARG"]   = "tor{0[0]},tor{0[1]}".format(pair)
        pldic_VES["LABEL"] = "b%s"%i
        VESlabellist.append("b%s"%i)
        pldicC.pldiclist.append(pldic_VES)
        i += 1
    pldic_VES["BASIS_FUNCTIONS"] = "bfF,bfC"
    for arg in arglist:
        pldic_TD = {"options":["TD_WELLTEMPERED"], "comments":[], "linefeedQ": False}
        pldic_TD["BIASFACTOR"] = "10"
        pldic_TD["LABEL"] = "td%s"%i
        pldicC.pldiclist.append(pldic_TD)
        pldic_VES = {"options":["VES_LINEAR_EXPANSION"], "comments":[], "linefeedQ": True}
        pldic_VES["BASIS_FUNCTIONS"] = "bfF,bfC"
        pldic_VES["GRID_BINS"] = "100"
        pldic_VES["ARG"]   = "tor{0[0]},pz".format(pair)
        pldic_VES["LABEL"] = "b%s"%i
        VESlabellist.append("b%s"%i)
        pldicC.pldiclist.append(pldic_VES)
        i += 1
    pldic_OPT = {"options":["OPT_AVERAGED_SGD"], "comments":[], "linefeedQ": True}
    pldic_OPT["BIAS"]              = ",".join(VESlabellist)
    pldic_OPT["STRIDE"]            = "500"
    pldic_OPT["LABEL"]             = "opt"
    pldic_OPT["STEPSIZE"]          = "0.001"
    pldic_OPT["FES_OUTPUT"]        = "50000000"
    pldic_OPT["BIAS_OUTPUT"]       = "50000000"
    pldic_OPT["COEFFS_OUTPUT"]     = "1000"
    pldic_OPT["COEFFS_FILE"]       = "coeffs.data"
    pldic_OPT["TARGETDIST_STRIDE"] = "500"
    pldic_OPT["TARGETDIST_OUTPUT"] = "5000000"
    pldicC.pldiclist.append(pldic_OPT)

    pldic_PRINT = {"options":["PRINT"], "comments":[], "linefeedQ": False}
    pldic_PRINT["STRIDE"] = "500"
    pldic_PRINT["ARG"] = "p.z"
    pldic_PRINT["FILE"] = "Z_noabs"
    pldicC.pldiclist.append(pldic_PRINT)
    pldic_PRINT = {"options":["PRINT"], "comments":[], "linefeedQ": False}
    pldic_PRINT["ARG"] = ",".join(arglist) + ",pz"
    pldic_PRINT["FILE"] = "COLVAR"
    pldicC.pldiclist.append(pldic_PRINT)

    pldic_WALLS = {"options":["UPPER_WALLS"], "comments":[], "linefeedQ": False}
    pldic_WALLS["ARG"] = "pz"
    pldic_WALLS["AT"] = "50.0"
    pldic_WALLS["KAPPA"] = "1000.0"
    pldic_WALLS["EXP"] = "2"
    pldic_WALLS["EPS"] = "1"
    pldic_WALLS["OFFSET"] = "0"
    pldic_WALLS["LABEL"] = "uwall"
    pldicC.pldiclist.append(pldic_WALLS)



    pldicC.exportplumed("./plumedVES.dat")



def importforcefield():
    dihedralQ = False
    dihClasslist = []
    for line in open("./forcefield.itp"):
        if "dihedraltypes" in line:
            dihedralQ = True
        if not dihedralQ:
            continue
        if line[0] == ";":
            continue
        line = line.split()
        if len(line) < 5:
            continue
        #print(line)
        dihC = Dihedral_info(line)
        dihClasslist.append(dihC)
    dihClasslist.sort(key = lambda x:x.Kphi)
    #for dihC in dihClasslist:
        #print(dihC.dihlist)
        #print("phi0, Kphi = %s, %s"%(dihC.phi0, dihC.Kphi))
    return dihClasslist
def importtypedic():
    atomsQ = False
    dihedralQ = False
    typedic = {}
    dihNumberlist = []
    for line in open("./HETA.itp"):
        if "atoms" in line:
            atomsQ = True
            continue
        elif "bonds" in line:
            atomsQ = False
            continue
        elif "dihedrals" in line:
            dihedralQ = True
            continue
        elif "POSRES" in line:
            dihedralQ = False

        if atomsQ is False and dihedralQ is False:
            continue
        if line[0] == ";":
            continue
        line = line.split()
        if len(line) < 5:
            continue
        if atomsQ:
            mass = float(line[7])
            if mass < 2.0:
                typedic[line[5]] = "Nan"
            else:
                typedic[line[5]] = line[1]
        elif dihedralQ:
            dihNumber = line[:-1]
            if int(dihNumber[-1]) < int(dihNumber[0]):
                dihNumber.reverse()
            dihNumberlist.append(dihNumber)
    return typedic, dihNumberlist

if __name__ == "__main__":
    main()


