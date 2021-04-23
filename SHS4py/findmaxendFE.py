#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2016 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import subprocess as sp

def main():
    """
    
    """
    endN = 0
    eqlist = []
    runningeqlist = []
    for line in open("./jobfiles_meta/eqlist.csv"):
        line = line.split(",")
        eqname = line[0]
        if os.path.exists("./jobfiles_meta/%s/end.txt"%eqname):
            eqlist.append(line)
        elif os.path.exists("./jobfiles_meta/%s/running.txt"%eqname):
            runningeqlist.append(line)
    minimumfe = min(float(a[-1]) for a in eqlist)
    #print("%s, % 5.3f"%(eqlist[-1][0], float(eqlist[-1][-1]) - minimumfe))
    print("endN = %s"%len(eqlist))
    print("*"*50)
    for eqpoint in runningeqlist:
        sphereApath = "./jobfiles_meta/%s/IOEsphereA.txt"%eqpoint[0]
        if os.path.exists(sphereApath):
            for line in open(sphereApath):
                pass
            print("%s, % 5.3f ; IOEsphereA = %s"%(eqpoint[0], float(eqpoint[-1]) - minimumfe, float(line)))
        else:
            print("%s, % 5.3f"%(eqpoint[0], float(eqpoint[-1]) - minimumfe))

if __name__ == "__main__":
    main()


