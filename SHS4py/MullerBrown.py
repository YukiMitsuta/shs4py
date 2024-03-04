#! /usr/bin/env python
#-*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021/06/29 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import subprocess as sp
import numpy as np

import curveNEB
from curveNEB import NEB

class MullerBrown():
    """
    setup Muller Brown function
    """
    def __init__(self):
        """
        define constants
        """
        self.A = [-200.0, -100.0, -170.0, 15.0]
        self.a = [  -1.0,   -1.0,   -6.5,  0.7]
        self.b = [   0.0,    0.0,   11.0,  0.6]
        self.c = [ -10.0,  -10.0,  -6.5,   0.7]
        self.x0 = [  1.0,    0.0,  -0.5,  -1.0]
        self.y0 = [  0.0,    0.5,   1.5,   1.0]
    def f(self,p):
        """
        return potential
        """
        x,y = p
        returnf = 0.0
        returnf += self.A[0]*np.exp( self.a[0]*((x - self.x0[0])**2) + self.b[0]*(x - self.x0[0])*(y - self.y0[0]) + self.c[0]*((y - self.y0[0])**2))
        returnf += self.A[1]*np.exp( self.a[1]*((x - self.x0[1])**2) + self.b[1]*(x - self.x0[1])*(y - self.y0[1]) + self.c[1]*((y - self.y0[1])**2))
        returnf += self.A[2]*np.exp( self.a[2]*((x - self.x0[2])**2) + self.b[2]*(x - self.x0[2])*(y - self.y0[2]) + self.c[2]*((y - self.y0[2])**2))
        returnf += self.A[3]*np.exp( self.a[3]*((x - self.x0[3])**2) + self.b[3]*(x - self.x0[3])*(y - self.y0[3]) + self.c[3]*((y - self.y0[3])**2))
        return returnf
    def g(self,p):
        """
        return gradient
        """
        x,y = p
        returngrad = [0.0, 0.0]
        returngrad[0] += self.A[0]*(2.0*self.a[0]*(x-self.x0[0])+self.b[0]*(y-self.y0[0])) *np.exp( self.a[0]*((x - self.x0[0])**2) + self.b[0]*(x - self.x0[0])*(y - self.y0[0]) + self.c[0]*((y - self.y0[0])**2))
        returngrad[0] += self.A[1]*(2.0*self.a[1]*(x-self.x0[1])+self.b[1]*(y-self.y0[1])) *np.exp( self.a[1]*((x - self.x0[1])**2) + self.b[1]*(x - self.x0[1])*(y - self.y0[1]) + self.c[1]*((y - self.y0[1])**2))
        returngrad[0] += self.A[2]*(2.0*self.a[2]*(x-self.x0[2])+self.b[2]*(y-self.y0[2])) *np.exp( self.a[2]*((x - self.x0[2])**2) + self.b[2]*(x - self.x0[2])*(y - self.y0[2]) + self.c[2]*((y - self.y0[2])**2))
        returngrad[0] += self.A[3]*(2.0*self.a[3]*(x-self.x0[3])+self.b[3]*(y-self.y0[3])) *np.exp( self.a[3]*((x - self.x0[3])**2) + self.b[3]*(x - self.x0[3])*(y - self.y0[3]) + self.c[3]*((y - self.y0[3])**2))

        returngrad[1] += self.A[0]*(self.b[0]*(x-self.x0[0])+2.0*self.c[0]*(y-self.y0[0])) *np.exp( self.a[0]*((x - self.x0[0])**2) + self.b[0]*(x - self.x0[0])*(y - self.y0[0]) + self.c[0]*((y - self.y0[0])**2))
        returngrad[1] += self.A[1]*(self.b[1]*(x-self.x0[1])+2.0*self.c[1]*(y-self.y0[1])) *np.exp( self.a[1]*((x - self.x0[1])**2) + self.b[1]*(x - self.x0[1])*(y - self.y0[1]) + self.c[1]*((y - self.y0[1])**2))
        returngrad[1] += self.A[2]*(self.b[2]*(x-self.x0[2])+2.0*self.c[2]*(y-self.y0[2])) *np.exp( self.a[2]*((x - self.x0[2])**2) + self.b[2]*(x - self.x0[2])*(y - self.y0[2]) + self.c[2]*((y - self.y0[2])**2))
        returngrad[1] += self.A[3]*(self.b[3]*(x-self.x0[3])+2.0*self.c[3]*(y-self.y0[3])) *np.exp( self.a[3]*((x - self.x0[3])**2) + self.b[3]*(x - self.x0[3])*(y - self.y0[3]) + self.c[3]*((y - self.y0[3])**2))

        return returngrad
        
def main():
    """
    
    """
    fclass = MullerBrown()
    #ftest(fclass)
    #gradtest(fclass)
    NEBc = NEB.NEBclass()
    NEBc.initialpoint = np.array([-0.558224, 1.44173])
    NEBc.finalpoint = np.array([0.623499, 0.0280378])
    NEBc.Nimage = 10
    NEBc.imagelist = [(NEBc.finalpoint - NEBc.initialpoint)*x/(NEBc.Nimage - 1) for x in range(NEBc.Nimage)]
    NEBc.Lpath = np.linalg.norm(NEBc.initialpoint - NEBc.finalpoint)
    NEBc.tlist = [x/NEBc.Nimage*NEBc.Lpath for x in range(NEBc.Nimage)]
    
    NEBc.dim = 2
    #NEBc.elasticMethod = "SoftRestriction"
    #NEBc.elasticMethod = "Curvature"
    #NEBc.elasticMethod = "NEB"
    NEBc.elasticMethod = "CurvatureNEB"
    #NEBc.elasticMethod = "CurvatureString"
    #NEBc.elasticMethod = "None"
    NEBc.curvratethreshold = 3.0
    NEBc.minimizeMethod = "SDG"
    #NEBc.minimizeMethod = "SDG_Curv"
    NEBc.stepsize = 0.00010
    NEBc.tdeltaminimum = 1.0/15.0
    NEBc.tdeltamax = 1.0/8.0
    NEBc.VvertThreshold = 1.0
    NEBc.kappa = 0.25
    #NEBc.kappa = 1.0e1
    #NEBc.kappa = 0.1
    NEBc = curveNEB.main(fclass.f, fclass.g, NEBc)
    #plotstr = ""
    #for k in range(NEBc.Nimage):
        #p = NEBc.initialpoint + NEBc.imagelist[k]
        #plotstr += "%s, %s\n"%(NEBc.tlist[k], fclass.f(p))
    #with open("reactionpath.csv", "w") as wf:
        #wf.write(plotstr)

def ftest(fclass):
    writeline = ""
    for x in range(-20,10):
        x = 0.1*x
        for y in range(-10,25):
            y = 0.1*y
            writeline += "%s,%s,%s\n"%(x,y,fclass.f([x,y]))
    with open("./MBtest_f.csv","w") as wf:
        wf.write(writeline)
def gradtest(fclass):
    writeline_gx = ""
    writeline_gy = ""
    for x in range(-20,10):
        x = 0.1*x
        for y in range(-10,25):
            y = 0.1*y
            writeline_gx += "%s,%s,%s\n"%(x,y,fclass.g([x,y])[0])
            writeline_gy += "%s,%s,%s\n"%(x,y,fclass.g([x,y])[1])
    #print(writeline)
    with open("./MBtest_gx.csv","w") as wf:
        wf.write(writeline_gx)
    with open("./MBtest_gy.csv","w") as wf:
        wf.write(writeline_gy)
if __name__ == "__main__":
    main()


