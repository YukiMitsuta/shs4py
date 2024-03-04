#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021/06/29 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, shutil, glob, copy
import subprocess as sp
import numpy as np

import curveNEB
from curveNEB import NEB, functions
class CallGaussianClass():
    def __init__(self):
        self.gauinfo = """# B3LYP/6-31+g* force
SCRF=(PCM,solvent=water)
iop(2/15=1,3/32=2,5/86=100,5/11=2)

curveture NEB Yuki Mitsuta

-1 1
"""
        self.atomlist = ["C","H","H","H","Cl","Br"]
        self.atomN = len(self.atomlist)
        self.gaufileNumber = 0
        if os.path.exists("./gaussiandir"):
            while True:
                filename = "./gaussiandir/gau%s.com"%self.gaufileNumber
                if not os.path.exists(filename):
                    break
                self.gaufileNumber += 1
    def getinitialpoint(self):
        _point = [ 0.000,  0.000,  0.000, #C
                   0.000,  1.036,  0.3355, #H
                   0.897, -0.518,  0.3355, #H
                  -0.897, -0.518,  0.3355, #H
                   0.000,  0.000, -1.816, #Cl
                   0.000,  0.000,  5.000]  #Br
        _point = np.array(_point)
        #_point = functions.imagetransration(_point)
        return _point
    def getfinalpoint(self):
        _point = [ 0.000,  0.000,  0.000,#C
                   0.000,  1.038, -0.3260,#H
                   0.899, -0.519, -0.3260,#H
                  -0.899, -0.519, -0.3260,#H
                   0.000,  0.000, -5.000,#Cl
                   0.000,  0.000,  1.9700]#Br
        _point = np.array(_point)
        #_point = functions.imagetransration(_point)
        return _point
    def f(self,x):
        filename = "gau%s.com"%self.gaufileNumber
        self.setupcom(filename, x)
        sp.call(["./callgau.sh",filename])
        logname = filename.replace("com","log")
        if not os.path.exists(logname):
            raise IOError("There is not %s; is Gussian work correctly?")
        for line in open(logname):
            if "SCF Done" in line:
                line = line.split()
                returnf = float(line[4])
                continue
            if "Error termination" in line:
                raise IOError("the Gaussian shows Error termination")
        if not os.path.exists("gaussiandir"):
            os.mkdir("gaussiandir")
        shutil.move(filename,"gaussiandir")
        shutil.move(logname,"gaussiandir")
        self.gaufileNumber += 1
        return returnf
    def g(self,x):
        lognames = []
        if type(x) == np.ndarray:
            filename = "gau%s.com"%self.gaufileNumber
            logname = filename.replace("com","log")
            lognames = [logname]
            self.setupcom(filename, x)
            sp.call(["./callgau.sh",filename])
            self.gaufileNumber += 1
        elif type(x) == list:
            for xpoint in x:
                filename = "gau%s.com"%self.gaufileNumber
                logname = filename.replace("com","log")
                lognames.append(logname)
                self.setupcom(filename, xpoint)
                self.gaufileNumber += 1
            sp.call(["./callgau.sh",filename])
        else:
            raise TypeError("in calculation of gradient, type of x is %s; not np.ndarray or list."%type(x))
        returngrads = []
        for logname in lognames:
            returngrad = self.importforce(logname)
            returngrads.append(returngrad)
        if not os.path.exists("gaussiandir"):
            os.mkdir("gaussiandir")
        for filename in glob.glob("gau*.com"):
            shutil.move(filename,"gaussiandir")
        for logname in glob.glob("gau*.log"):
            shutil.move(logname,"gaussiandir")
        for returngrad in returngrads:
            returngrad[0] = 0.0
            returngrad[1] = 0.0
            returngrad[2] = 0.0
            returngrad[12] = 0.0
            returngrad[13] = 0.0
            returngrad[15] = 0.0
            returngrad[16] = 0.0
        if type(x) == np.ndarray:
            return returngrads[0]
        else:
            return returngrads
    def setupcom(self, filename, x):
        #xdamp = [0.0,0.0,0.0]+list(x) # add the position of center (carbon)
        xdamp = x
        writeline = ""
        writeline += self.gauinfo
        for i in range(self.atomN):
            writeline += "{0}  {1[0]: 6.5f} {1[1]: 6.5f} {1[2]: 6.5f}\n".format(self.atomlist[i], xdamp[3*i:3*i+3])
        writeline += "\n\n"
        with open(filename, "w") as wf:
            wf.write(writeline)
    def importforce(self, logname):
        if not os.path.exists(logname):
            raise IOError("There is not %s; is Gussian work correctly?"%logname)
        returngrad = []
        forcesQ = False
        for line in open(logname):
            if "Forces (Hartrees/Bohr)" in line:
                forcesQ = True
                continue
            if forcesQ:
                if "Cartesian Forces" in line:
                    forcesQ = False
                    continue
                if "Number" in line:
                    continue
                if "------" in line:
                    continue
                line = line.split()
                returngrad.extend(line[2:])
            if "Error termination" in line:
                raise IOError("the Gaussian shows Error termination")
        #returngrad = returngrad[3:]
        returngrad = np.array(returngrad,dtype=float)
        returngrad *= -1.0 # change from Forces to Gradient
        returngrad *= 1.0/0.529177 # change from Bohr^-1 to Angstrom^-1
        return returngrad
def main():
    fclass = CallGaussianClass()
    NEBc = NEB.NEBclass()
    NEBc.initialpoint = fclass.getinitialpoint()
    NEBc.finalpoint = fclass.getfinalpoint()
    NEBc.dim = len(NEBc.initialpoint)
    restartQ = False
    if os.path.exists('images/'):
        fileN = len(glob.glob("images/image_*.csv"))
        iterNdamp = 0
        while iterNdamp < fileN:
            filename = "./images/image_%s.csv"%iterNdamp
            if os.path.exists(filename):
                NEBc.imagelist = []
                for line in open(filename):
                    NEBc.imagelist.append(np.array(line.split(","), dtype=float) - NEBc.initialpoint)
                NEBc.iterN = copy.copy(iterNdamp)
                restartQ = True
            iterNdamp += 1
    if not restartQ:
        NEBc.Nimage = 10
        NEBc.imagelist = [(NEBc.finalpoint - NEBc.initialpoint)*x/(NEBc.Nimage - 1) for x in range(NEBc.Nimage)]
        NEBc.update_tlist()
        #NEBc.imagelist = [functions.imagetransration(image) for image in NEBc.imagelist]
        #NEBc.Lpath = np.linalg.norm(NEBc.initialpoint - NEBc.finalpoint)
        #NEBc.tlist = [NEBc.Lpath*x/(NEBc.Nimage-1) for x in range(NEBc.Nimage)]
    else:
        NEBc.Nimage = len(NEBc.imagelist)
        NEBc.update_tlist()
    NEBc.elasticMethod = "CurvatureNEB"
    #NEBc.elasticMethod = "NEB"
    #NEBc.curvratethreshold = 5.0
    NEBc.minimizeMethod = "SDG"
    NEBc.stepsize = 0.5
    NEBc.tdeltaminimum = 1.0/30.0
    NEBc.tdeltamax = 1.0/10.0
    NEBc.VvertThreshold = 0.01 # Hartrees/Angstrom
    #NEBc.translationQ = True
    NEBc.translationQ = False
    NEBc.gradientParallelizeQ = True
    NEBc.NimageRestrected = False
    #NEBc.kappa = 1.0e1
    #NEBc.kappa = 10.0
    #NEBc.kappa = 1.0
    NEBc.kappa = 0.1
    #NEBc.kappa = 0.01
    #NEBc.kappa = 0.001
    #NEBc.kappa = 0.0001
    NEBc.curvatureRateMin = 0.0
    curveNEB.main(fclass.f, fclass.g, NEBc)
if __name__ == "__main__":
    main()


