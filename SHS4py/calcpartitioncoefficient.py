#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021/02/08 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import subprocess as sp

import numpy as np 

from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class PointClass():
    def __init__(self, line):
        _line      = line.split(",")
        self.name  = _line[0]
        self.point = np.array(_line[1:-2], dtype = float)
        self.z     = float(_line[-2])
        self.fe    = float(_line[-1])
def main():
    """
    
    """

    Temp    = 300.0           # tempeture (K)
    k_B     = 1.38065e-26      # Boltzmann constant (kJ / K)
    N_A     = 6.002214e23      # Avogadro constant (mol^-1)
    betainv = Temp * k_B * N_A # 1/beta (kJ / mol)
    beta    = 1.0 / betainv

    writeline = ""
    diffdic = {} 
    for z in range(31):
        tlist = []
        MSDlist = []
        for line in open("kSim_%s/diffusionplot.csv"%z):
            t, msd = line.split(",")
            if 10 < float(t):
                continue
            tlist.append(float(t))
            MSDlist.append(float(msd))
        diff = calcdiff(tlist, MSDlist)
        diff = diff / 2.0
        writeline += "%s, %s\n"%(z, diff)
        diffdic[z] = diff
    writelinePM = ""
    diffdicPM = {} 
    for z in range(-30, 31):
        tlist = []
        MSDlist = []
        if 0.0 <= z:
            diffpath ="kSim_%s/diffusionplotPlus.csv"%abs(z)
        else:
            diffpath ="kSim_%s/diffusionplotMinus.csv"%abs(z)
        for line in open(diffpath):
            t, msd = line.split(",")
            if 10 < float(t):
                continue
            tlist.append(float(t))
            MSDlist.append(float(msd))
        diff = calcdiff(tlist, MSDlist)
        #diff = diff / 2.0
        writelinePM += "%s, %s\n"%(z, diff)
        diffdicPM[z] = diff
    with open("./diffusion_z.csv", "w") as wf:
        wf.write(writeline)
    with open("./diffusion_zPM.csv", "w") as wf:
        wf.write(writelinePM)
    eqlist = []
    for line in open("./jobfiles_all/eqlist.csv"):
        if "#" in line:
            continue
        eqpoint = PointClass(line)
        eqlist.append(eqpoint)
    FEmin = min(eqpoint.fe for eqpoint in eqlist)
    for eqpoint in eqlist:
        eqpoint.fe -= FEmin
        eqpoint.P = np.exp(-beta * eqpoint.fe)
    PMFdic = {}
    FEref = 1.0e30
    for z in range(31):
        eqlist_z = [eqpoint for eqpoint in eqlist if int(eqpoint.z) == z]
        Ptotal = sum(eqpoint.P for eqpoint in eqlist_z) 
        PMFdic[z] = - betainv * np.log(Ptotal)
        if PMFdic[z] < FEref:
            FEref = PMFdic[z]
            z_min = z
        #PMFdic[z] = min(eqpoint.fe for eqpoint in eqlist_z)
    #z_min = 30
    #FEref = PMFdic[13]
    #FEref = PMFdic[30]
    #print('FEref = %s'%FEref)
    writeline = ""
    for z in range(31):
        PMFdic[z] -= FEref
        #print("%s, %s"%(z,PMFdic[z]))
        writeline += "%s, % 11.9f\n"%(z,PMFdic[z])
    with open("./PMF_z.csv", "w") as wf:
        wf.write(writeline)

    PartCoeff_inv = 0.0
    #for z in range(-z_min + 1, z_min):
    for z in range(-30,31):
        diff = diffdic[abs(z)] * 1.0e12 # 1 ps^-1 1.0e12 s^-1
        pmf = PMFdic[abs(z)]
        PartCoeff_inv += np.exp(beta*pmf)/diff
        #print('%s, %s'%(z, 1.0/PartCoeff_inv))
    PartCoeff = 1.0 / PartCoeff_inv * 1.0e-8 # 1 ang = 1.0e-8 cm
    print("Partiton Coefficient = %s cm/s"%PartCoeff)
    with open("./PartitionCoefficient.csv", "w") as wf:
        wf.write("%s\n"%PartCoeff)
    PartCoeff *= 1.0e7
    print("Partiton Coefficient = % 3.2f *1.0e-7 cm/s"%PartCoeff)

    MPtime = 0.0 # mean passage time
    intpart = 0.0
    for z in range(-30,31):
        diff = diffdic[abs(z)] * 1.0e12 # 1 ps^-1 = 1.0e12 s^-1
        pmf = PMFdic[abs(z)]
        #intpart += np.exp(-beta*pmf)
        #MPtime += np.exp(beta*pmf)/diff * intpart
        intpart += np.exp(-beta*pmf)/diff
        MPtime += np.exp(beta*pmf) * intpart
    print("Mean Passage Time = % 3.2f mus"%(MPtime * 1.0e6))
    with open("./Mean_Passage_Time.csv", "w") as wf:
        wf.write("%s\n"%MPtime)

    PartCoeff_inv = 0.0
    #for z in range(-z_min + 1, z_min):
    for z in range(-30, 31):
    #for z in range(-2, 3):
        diff = diffdicPM[z] * 1.0e12 # 1 ps^-1 1.0e12 s^-1
        pmf = PMFdic[abs(z)]
        PartCoeff_inv += np.exp(beta*pmf)/diff
        #print("%s; %s"%(z, 1.0/PartCoeff_inv*1.0e-8))
        print("%s; %s"%(z, np.exp(beta*pmf)*diff))
    PartCoeff = 1.0 / PartCoeff_inv * 1.0e-8 # 1 ang = 1.0e-8 cm
    print("Partiton Coefficient (PM)= %s cm/s"%PartCoeff)
    with open("./PartitionCoefficient_PM.csv", "w") as wf:
        wf.write("%s\n"%PartCoeff)
    PartCoeff *= 1.0e7
    print("Partiton Coefficient (PM) = % 3.2f *1.0e-7 cm/s"%PartCoeff)

    MPtime = 0.0 # mean passage time
    intpart = 0.0
    #for z in range(-z_min+1, z_min):
    #for z in range(-25, 25):
    for z in range(-30, 31):
    #for z in range(25, -25,-1):
        diff = diffdicPM[z] * 1.0e12 # 1 ps^-1 1.0e12 s^-1
        pmf = PMFdic[abs(z)]
        intpart += np.exp(-beta*pmf)
        MPtime += np.exp(beta*pmf)/diff * intpart
        #intpart += np.exp(beta*pmf)/diff
        #MPtime += np.exp(-beta*pmf) * intpart
    print("Mean Passage Time (PM) = % 3.2f mus"%(MPtime * 1.0e6))
    with open("./Mean_Passage_Time_PM.csv", "w") as wf:
        wf.write("%s\n"%MPtime)


def calcdiff(x, y):
    degree = 1
    x = np.array(x)
    y = np.array(y)
    #model = make_pipeline(PolynomialFeatures(degree,include_bias=False),LinearRegression(fit_intercept=False))
    model = make_pipeline(PolynomialFeatures(degree,include_bias=False),LinearRegression(fit_intercept=True))
    model.fit(x.reshape(-1,1),y)
    y_model=model.predict(x.reshape(-1,1))
    return model.steps[1][1].coef_[0]
if __name__ == "__main__":
    main()


