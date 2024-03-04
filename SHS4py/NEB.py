#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021/06/29 MitsutaYuki
#
# Distributed under terms of the MIT license.

import os, copy
import numpy as np

from scipy.optimize import minimize

from . import functions

class NEBclass():
    """
    class of nudged elastic band (NEB)

    """
    def __init__(self):
        """
        define initial constants
        these parameter must be prepaired in initial
        Nimage ; the number of images
        initialpoint; the initai point of images
        finalpoint; the final point of images
        imagelist; the position of images
        tlist; the list of t
        iterN; the numer of update images
        """
        self.Nimage = 0
        self.initialpoint = np.array([0.0,0.0])
        self.finalpoint = np.array([0.0,0.0])
        self.Nimage = 0
        self.imagelist = []
        self.dim = 0
        self.tlist = []
        self.iterN = 0
        self.tdeltaminimum = 0.05
        self.tdeltamax = 0.1
        self.minimizeMethod = "SDG"
        self.stepsize = 0.0005
        self.elasticMethod = "None"
        self.curvratethreshold = 100.0
        self.translationQ = False
        self.gradientParallelizeQ = False
        self.kappa = 0.1
        self.curvatureRateMin = 0.1
    def updateGrad(self,g,curvlist):
        if self.gradientParallelizeQ:
            self.Vgrad = g([self.initialpoint+x for x in self.imagelist[1:-1]])
        else:
            self.Vgrad = [g(self.initialpoint+x) for x in self.imagelist[1:-1]]
        self.Vgrad = [0.0]+self.Vgrad+[0.0]
        self.Vvert = [self.Vgrad[k] - np.dot(self.Vgrad[k], self.Eholo[k]) * self.Eholo[k] for k in range(self.Nimage)]
        if self.elasticMethod == "NEB":
            for k in range(1, self.Nimage-1):
                tdelta0 = self.tlist[k+1]-self.tlist[k]
                tdelta1 = self.tlist[k]-self.tlist[k-1]
                Vband   = self.kappa*(tdelta0-tdelta1)
                self.Vvert[k] -= self.Eholo[k]*Vband
        elif self.elasticMethod == "CurvatureNEB":
            curvlistdamp = [np.abs(curv) for curv in curvlist]
            #curvlistdamp = [max([self.curvatureRateMin, curv/max(curvlistdamp)]) for curv in curvlistdamp]
            curvlistdamp = [np.sqrt(curv/max(curvlistdamp)) for curv in curvlistdamp]
            for k in range(1,self.Nimage-1):
                tdelta0 = self.tlist[k+1]-self.tlist[k] -self.tdeltaminimum*self.Lpath
                tdelta0 *= (curvlistdamp[k+1]+curvlistdamp[k])*0.5
                tdelta1 = self.tlist[k]-self.tlist[k-1]-self.tdeltaminimum*self.Lpath
                tdelta1 *= (curvlistdamp[k-1]+curvlistdamp[k])*0.5
                #if k ==1 or k == self.Nimage-2:
                    #Vband   = curvlistdamp[k]*self.kappa*(tdelta0-tdelta1)
                #elif 0.0<=np.abs(curvlistdamp[k-1]-curvlistdamp[k+1])<0.01:
                #elif 0.0<=curvlistdamp[k]<0.01:
                    #Vband   = 0.01*self.kappa*(tdelta0-tdelta1)
                #else:
                #if True:
                    #Vband   = np.abs(curvlistdamp[k-1]-curvlistdamp[k+1])*self.kappa*(tdelta0-tdelta1)
                    #Vband   = self.kappa*(curvlistdamp[k+1]*tdelta0-curvlistdamp[k-1]*tdelta1)
                Vband   = self.kappa*(tdelta0-tdelta1)
                self.Vvert[k] -= self.Eholo[k]*Vband
    def updateImages(self):
        if self.minimizeMethod == "SDG":
            self.imagelist = [self.imagelist[k] - self.stepsize*self.Vvert[k] for k in range(self.Nimage)]
            #self.imagelist[0] = np.array([0.0 for _ in range(self.dim)])
            #self.imagelist[-1] = self.finalpoint-self.initialpoint
            if self.translationQ:
                self.imagelist = [functions.imagetransration(image) for image in self.imagelist]
        self.update_tlist()
    def update_tlist(self):
        self.tlist = [0.0]+[np.linalg.norm(self.imagelist[k-1]-self.imagelist[k]) for k in range(1,self.Nimage)]
        self.tlist = [sum(self.tlist[:k+1]) for k in range(self.Nimage)]
        self.Lpath = float(self.tlist[-1])
        #self.tlist /= self.tlist[-1]
    def exportImages(self):
        writeline = ""
        for p in self.imagelist:
            writeline += ", ".join(["%s"%x for x in p+self.initialpoint])
            writeline += "\n"
        if not os.path.exists("./images"):
            os.mkdir("./images")
        with open("./images/image_%s.csv"%self.iterN, "w") as wf:
            wf.write(writeline)
    def exportEholo(self):
        writeline = ""
        for p in self.Eholo:
            writeline += ", ".join(["%s"%x for x in p])
            writeline += "\n"
        if not os.path.exists("./Eholos"):
            os.mkdir("./Eholos")
        with open("./Eholos/Eholo_%s.csv"%self.iterN, "w") as wf:
            wf.write(writeline)
    def addImages(self):
        #print("add images")
        self.tlist = list(self.tlist)
        #print(self.tlist)
        addQ = False
        #while True:
        if True:
            #tdeltaave = np.mean([self.tlist[k]-self.tlist[k-1] for k in range(1,self.Nimage-1)])
            for k in range(1,self.Nimage-1):
                tdelta = self.tlist[k]-self.tlist[k-1]
                #print("tdelta",tdelta)
                #print("tdeltamax",self.tdeltamax*self.Lpath)
                if self.tdeltamax*self.Lpath < tdelta:
                    newimage = self.imagelist[k] + self.imagelist[k-1]
                    newimage *= 0.5
                    newt = self.tlist[k] + self.tlist[k-1]
                    newt *= 0.5
                    newtdelta = self.tlist[k+1]-newt
                    if newtdelta < self.tdeltaminimum*self.Lpath:
                        continue
                    newtdelta = newt - self.tlist[k-1]
                    if newtdelta < self.tdeltaminimum*self.Lpath:
                        continue
                    self.imagelist.insert(k,newimage)
                    self.tlist.insert(k, newt)
                    self.Nimage += 1
                    addQ = True
                    print("in addImages,Nimage = %s"%self.Nimage)
                    break
            #else:
                #break
        if addQ:
            self.update_tlist()
        return addQ
    def removeImages(self):
        #print("remove images")
        self.tlist = list(self.tlist)
        removeQ = False
        #while True:
        if True:
            for k in range(1,self.Nimage-1):
                v1 = self.imagelist[k+1] - self.imagelist[k]
                v2 = self.imagelist[k-1] - self.imagelist[k]
                if 0.0 < np.dot(v1,v2):
                    #tdelta2 = self.tlist[k+1]-self.tlist[k-1]
                    #if self.tdeltamax*self.Lpath<tdelta2:
                        #continue
                    self.imagelist.pop(k)
                    self.tlist.pop(k)
                    self.Nimage -= 1
                    removeQ = True
                    print("in removeImages,Nimage = %s"%self.Nimage)
                    break
                tdelta = self.tlist[k]-self.tlist[k-1]
                if tdelta < self.tdeltaminimum*self.Lpath:
                    tdelta2 = self.tlist[k+1]-self.tlist[k-1]
                    #if self.tdeltamax*self.Lpath<tdelta2:
                        #continue
                    self.imagelist.pop(k)
                    self.tlist.pop(k)
                    self.Nimage -= 1
                    removeQ = True
                    print("in removeImages,Nimage = %s"%self.Nimage)
                    break
            #else:
                #break
        if removeQ:
            self.update_tlist()
        return removeQ
    def replaceImages_angle(self):
        self.tlist = list(self.tlist)
        removeQ = False
        for k in range(1,self.Nimage-1):
            v1 = self.imagelist[k+1] - self.imagelist[k]
            v2 = self.imagelist[k-1] - self.imagelist[k]
            if 0.0 < np.dot(v1,v2):
                newimage = self.imagelist[k+1] + self.imagelist[k-1]
                newimage *= 0.5
                self.imagelist[k] = newimage
                removeQ = True
                print("in replaceImages_angle,Nimage = %s"%self.Nimage)
                break
        return removeQ
    def CurvatureString(self, curvlist, RGclass):
        #print("line(124); ",self.Nimage)
        changeQ = False
        #print(curvlist)
        curvlistmax = max(np.abs(curvlist))
        if type(self.tlist) == np.ndarray:
            self.tlist = list(self.tlist)
        for k in range(1,self.Nimage-1):
            v1 = self.imagelist[k+1] - self.imagelist[k]
            v2 = self.imagelist[k-1] - self.imagelist[k]
            if 0.0 < np.dot(v1,v2):
                tdelta = self.tlist[k+1] - self.tlist[k-1]
                #if self.tdeltamax*self.Lpath< tdelta:
                if False:
                    newimage = self.imagelist[k+1] + self.imagelist[k-1]
                    newimage *= 0.5
                    self.imagelist[k] = newimage
                else:
                    self.imagelist.pop(k)
                    self.tlist.pop(k)
                    self.Nimage -= 1
                changeQ = True
                print("line(178); ",self.Nimage)
                break
            if np.sqrt(np.abs(curvlist[k])/curvlistmax) < 1.0/self.curvratethreshold:
                if 1.0-1.0/self.curvratethreshold < np.sqrt(np.abs(curvlist[k])/curvlistmax):
                    continue
                if 1.0-1.0/self.curvratethreshold < np.sqrt(np.abs(curvlist[k])/curvlistmax):
                    continue
                tdelta = self.tlist[k+1] - self.tlist[k-1]
                if tdelta < self.tdeltamax*self.Lpath:
                    self.imagelist.pop(k)
                    self.tlist.pop(k)
                    self.Nimage -= 1
                    changeQ = True
                    print("line(191); ",self.Nimage)
                    break
                if np.sqrt(np.abs(curvlist[k-1])/curvlistmax) < 1.0/self.curvratethreshold:
                    tdelta = self.tlist[k] - self.tlist[k-1]
                    if tdelta < self.tdeltamax*self.Lpath*0.75:
                        imagedelta = self.imagelist[k+1] - self.imagelist[k-1]
                        newimagelist = self.imagelist[k]+imagedelta*0.1
                        self.imagelist[k] = newimagelist
                        changeQ = True
                        print("line(200); ",self.Nimage)
                        break
            elif 1.0-1.0/self.curvratethreshold < np.sqrt(np.abs(curvlist[k])/curvlistmax):
                tdelta = self.tlist[k+1] - self.tlist[k-1]
                if tdelta/3.0 < self.tdeltaminimum*self.Lpath< tdelta/3.0:
                    newtlist = [self.tlist[k-1]+tdelta/3.0,self.tlist[k-1]+tdelta/3.0*2.0]
                    self.tlist.pop(k)
                    self.tlist[k:k] = newtlist
                    #imagedelta = self.imagelist[k+1] - self.imagelist[k-1]
                    #newimagelist = [self.imagelist[k-1]+imagedelta/3.0, self.imagelist[k], self.imagelist[k-1]+imagedelta/3.0*2.0]
                    v0 = np.array([newtlist[0] for _ in range(self.dim)])
                    v1 = np.array([newtlist[1] for _ in range(self.dim)])
                    newimagelist = [RGclass.Jlist[k]@v0+RGclass.blist[k],RGclass.Jlist[k]@v1+RGclass.blist[k]]
                    self.imagelist.pop(k)
                    self.imagelist[k:k] = newimagelist
                    self.Nimage += 1
                    changeQ = True
                    print("line(217); ",self.Nimage)
                    break
                if 1.0-1.0/self.curvratethreshold < np.sqrt(np.abs(curvlist[k-1])/curvlistmax):
                    tdelta = self.tlist[k] - self.tlist[k-1]
                    if self.tdeltaminimum*self.Lpath*0.75 < tdelta:
                        imagedelta = self.imagelist[k+1] - self.imagelist[k-1]
                        newimagelist = self.imagelist[k]-imagedelta*0.15
                        self.imagelist[k] = newimagelist
                        changeQ = True
                        print("line(226); ",self.Nimage)
                        break
        if changeQ:
            self.update_tlist()
        return changeQ
    def elasticCurvature2(self, curvlist):
        #print("line(124); ",self.Nimage)
        changeQ = False
        #print(curvlist)
        curvlistrange = max(curvlist) - min(curvlist)
        if type(self.tlist) == np.ndarray:
            self.tlist = list(self.tlist)
        for k in range(1,self.Nimage-1):
            v1 = self.imagelist[k+1] - self.imagelist[k]
            v2 = self.imagelist[k-1] - self.imagelist[k]
            if 0.0 < np.dot(v1,v2):
                self.imagelist.pop(k)
                self.tlist.pop(k)
                self.Nimage -= 1
                changeQ = True
                print("line(210); ",self.Nimage)
                break
            deltabefore = curvlist[k-1]-curvlist[k]
            deltanext   = curvlist[k+1]-curvlist[k]
            if deltabefore/curvlistrange < 0.1 and deltanext/curvlistrange < 0.1:
                tdelta = self.tlist[k+1] - self.tlist[k-1]
                if tdelta < self.tdeltamax*self.Lpath:
                    #print(tdelta)
                    self.imagelist.pop(k)
                    self.tlist.pop(k)
                    self.Nimage -= 1
                    changeQ = True
                    print("line(222); ",self.Nimage)
                    break
            elif k+1 != self.Nimage-1 and self.curvratethreshold < np.abs(deltanext/deltabefore):
                tdelta = self.tlist[k+2] - self.tlist[k]
                if tdelta/3.0 < self.tdeltaminimum*self.Lpath:
                    continue
                newtlist = [self.tlist[k]+tdelta/3.0, self.tlist[k]+tdelta/3.0*2.0]
                self.tlist.pop(k+1)
                self.tlist[k+1:k+1] = newtlist
                imagedelta = self.imagelist[k+2] - self.imagelist[k]
                newimagelist = [self.imagelist[k]+imagedelta/3.0, self.imagelist[k]+imagedelta/3.0*2.0]
                self.imagelist.pop(k+1)
                self.imagelist[k+1:k+1] = newimagelist
                self.Nimage += 1
                changeQ = True
                print("line(237); ",self.Nimage)
                break
            elif k-1 != 0 and self.curvratethreshold <= np.abs(deltabefore/deltanext):
                tdelta = self.tlist[k] - self.tlist[k-2]
                if tdelta/3.0 < self.tdeltaminimum*self.Lpath:
                    continue
                newtlist = [self.tlist[k-2]+tdelta/3.0, self.tlist[k-2]+tdelta/3.0*2.0]
                self.tlist = list(self.tlist)
                self.tlist.pop(k-1)
                self.tlist[k-1:k-1] = newtlist
                imagedelta = self.imagelist[k] - self.imagelist[k-2]
                newimagelist = [self.imagelist[k-2]+imagedelta/3.0, self.imagelist[k-2]+imagedelta/3.0*2.0]
                self.imagelist.pop(k-1)
                self.imagelist[k-1:k-1] = newimagelist
                self.Nimage += 1
                changeQ = True
                print("line(253); ",self.Nimage)
                break
        if changeQ:
            self.update_tlist()
        return changeQ
