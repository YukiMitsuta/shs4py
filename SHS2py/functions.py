#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
the effective functions for SHS2py

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
"""
import os, glob, shutil, sys, re
import time, copy, inspect
import tarfile, zipfile, random
import subprocess      as sp
import numpy           as np
import multiprocessing as mp

#from . import const

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
    if qlist[0] < 0:
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
def periodicpoint(x, const):
    """
    periodicpoint: periodic calculation of x
    """
    bdamp = copy.copy(x)
    #return bdamp
    #periodicmax =   np.pi
    #periodicmin = - np.pi
    if const.periodicQ:
    #if True:
        for i in range(len(x)):
            if x[i] < const.periodicmin or const.periodicmax < x[i]:
                bdamp[i]  = (x[i] - const.periodicmax) % (const.periodicmin - const.periodicmax)
                bdamp[i] += const.periodicmax
    return bdamp
