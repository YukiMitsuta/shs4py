#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import subprocess as sp
import random
from minepy import MINE, pstats
#from pyHSICLasso import HSICLasso
import numpy as np
from mpi4py import MPI

def main():
    """
    
    """
    #mine = MINE()
    #hsic_lasso = HSICLasso()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    collist = []
    for colname in glob.glob("./run*/COL*"):
        for line in open(colname):
            if "#" in line:
                continue
            line = line.split()
            t = float(line[0])
            if t < 10000.0:
                continue
            elif 50000.0 < t:
                break
            collist.append(line)
        #break
    #MINEs, TICs =  pstats(collist)
    #print(TICs)
    for i in range(len(collist[0])-1):
        for j in range(i+1, len(collist[0])-1):
        #for j in range(i+1, i + 5):
            miclist = []
            ticlist = []
            for _ in range(10):
            #if True:
                colpart = random.sample(collist, 10000)
                #colpart = collist
                x = np.array([a[i+1] for a in colpart], dtype=float)
                y = np.array([a[j+1] for a in colpart], dtype=float)
                #xy = np.array([x,y])
                #mine = MINE()
                mine = MINE(est="mic_e")
                mine.compute_score(x,y)
                miclist.append(mine.mic())
                ticlist.append(mine.tic())
            #miclist = comm.gather(mine.mic(), root=0)
            #ticlist = comm.gather(mine.tic(), root=0)

            #hsic_lasso.input(xy, np.array([0,1]))
            #hsic_lasso.input(np.array([[1, 1, 1], [2, 2, 2]]), np.array([0, 1]))
            #hsic_lasso.regression(5)
            #hsic_lasso.classification(10)
            #print(hsic_lasso.dump())
            if rank == root:
                print("%s,%s, %s, %s"%(i,j,np.mean(miclist),np.mean(ticlist)), flush = True)
                with open("./minedata.csv", "a") as wf:
                    wf.write("%s, %s, %s, %s\n"%(i,j,np.mean(miclist),np.mean(ticlist)))


if __name__ == "__main__":
    main()

