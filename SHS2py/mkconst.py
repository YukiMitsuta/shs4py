#! /usr/bin/env python3
# -*- coding: utf-8 -*- # vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
A module to make constants (global variables)
"""
import os, sys, fcntl
import numpy as np

def main(constC):
    _keylist = constC.__dict__.keys()
    if not "betainv" in _keylist:
        Temp           = 298.0            # tempeture (K)
        k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
        N_A            = 6.002214e23      # Avogadro constant (mol^-1)
        constC.betainv = Temp * k_B * N_A # 1/beta (kJ / mol)
        constC.beta    = 1.0 / constC.betainv

    if not "systemname" in _keylist:
        constC.systemname = "main"

    if not "pwdpath" in _keylist:
        constC.pwdpath    = os.getcwd()
    if not os.path.exists("%s/CONTROL"%constC.pwdpath):
        os.mkdir("%s/CONTROL"%constC.pwdpath)

    if not "lockfilepath" in _keylist:
        constC.lockfilepath = constC.pwdpath + '/CONTROL/LockFile'
    if not "lockfilepath_UIlist" in _keylist:
        constC.lockfilepath_UIlist = constC.pwdpath + '/CONTROL/LockFile_UIlist'

    if not os.path.exists(constC.lockfilepath):
        with open(constC.lockfilepath, "w") as wf:
            wf.write("")
    if not os.path.exists(constC.lockfilepath_UIlist):
        with open(constC.lockfilepath_UIlist, "w") as wf:
            wf.write("")

    if not "threshold" in _keylist:
        constC.threshold = 1.0
    if not "minimize_threshold" in _keylist:
        constC.minimize_threshold = 0.01
    if not "moveQ" in _keylist:
        constC.moveQ     = False
    if not "neiborfluct_threshold" in _keylist:
        constC.neiborfluct_threshold = 0.05 # (neighbor ADD)  * neiborfluct_threshold < (new ADD)

    if not "sameEQthreshold" in _keylist:
        constC.sameEQthreshold = 0.05

    if not "IOEsphereA_initial" in _keylist:
       constC. IOEsphereA_initial = 0.20
    if not "IOEsphereA_dist" in _keylist:
        constC.IOEsphereA_dist    = 0.05

    if not "deltas0" in _keylist:
        constC.deltas0 = 0.1
    if not "deltas" in _keylist:
        constC.deltas  = 0.05

    if not "lADDnQ" in _keylist:
        constC.lADDnQ           = False
    if not "IOEl" in _keylist:
        constC.IOEl             = 999
    constC.IOEl_forADDstart = constC.IOEl * 2
    constC.IOEl_forcollect  = constC.IOEl * 3

    os.environ["OMP_NUM_THREADS"] = "1"
    if not "wallmax" in _keylist:
        constC.wallmax         = np.array([ 1.0e30 for _ in range(100)])
    if not "wallmin" in _keylist:
        constC.wallmin         = np.array([ -1.0e30    for _ in range(100)])

    if not "Ddimer" in _keylist:
        constC.Ddimer          = 0.05
    if not "phitol" in _keylist:
        constC.phitol          = 0.08


    if not "calc_mpiQ" in _keylist:
        constC.calc_mpiQ = False
    if not "parallelMetaDQ" in _keylist:
        constC.parallelMetaDQ = False

    if not "digThreshold" in _keylist:
        constC.digThreshold = 5

    if not "WellTempairedQ" in _keylist:
        constC.WellTempairedQ = False
    if constC.WellTempairedQ:
        if not "WT_Biasfactor" in _keylist:
            constC.WT_Biasfactor  = 6
        if not "WT_Biasfactor_ffactor" in _keylist:
            constC.WT_Biasfactor_ffactor  = constC.WT_Biasfactor / (constC.WT_Biasfactor - 1.0)


    if not "initialpointN" in _keylist:
        constC.initialpointN = 100
    if not "gridQ" in _keylist:
        constC.gridQ = False
    if not "grid_min" in _keylist:
        constC.grid_min = 0.0
    if not "grid_max" in _keylist:
        constC.grid_max = 5.0
    if not "grid_bin" in _keylist:
        constC.grid_bin = 1000 

    if not "periodicQ" in _keylist:
        constC.periodicQ = False
    if not "periodicmax" in _keylist:
        constC.periodicmax = 1.0e30
    if not "periodicmin" in _keylist:
        constC.periodicmin = -1.0e30
    if not "calc_cupyQ" in _keylist:
        constC.calc_cupyQ = False
    if constC.calc_cupyQ:
        if not "cp" in _keylist:
            import cupy as cp
            constC.cp = cp 
    if not "GPUgatherQ" in _keylist:
        constC.GPUgatherQ = False

    if not "cythonQ" in _keylist:
        constC.cythonQ = False
    if constC.cythonQ:
        import pyximport  # for cython
        pyximport.install()
        try:
            from . import calcgau
        except ImportError:
            import calcgau
        constC.calcgau = calcgau
        include_dirs = [np.get_include()]
    if not "PBmetaDQ" in _keylist:
        constC.PBmetaDQ = False
    if not "exportADDpointsQ" in _keylist:
        constC.exportADDpointsQ = False

    if not "use_jacQ" in _keylist:
        constC.use_jacQ = False

    writestr  = ""
    writestr += "systemname            = %s\n"%constC.systemname
    writestr += "pwdpath               = %s\n"%constC.pwdpath
    writestr += "lockfilepath          = %s\n"%constC.lockfilepath
    writestr += "lockfilepath_UIlist   = %s\n"%constC.lockfilepath_UIlist
    writestr += "threshold             = %s\n"%constC.threshold
    writestr += "moveQ                 = %s\n"%constC.moveQ
    writestr += "threshold             = %s\n"%constC.threshold
    writestr += "neiborfluct_threshold = %s\n"%constC.neiborfluct_threshold
    writestr += "sameEQthreshold       = %s\n"%constC.sameEQthreshold
    writestr += "IOEsphereA_initial    = %s\n"%constC.IOEsphereA_initial
    writestr += "IOEsphereA_dist       = %s\n"%constC.IOEsphereA_dist
    writestr += "deltas0               = %s\n"%constC.deltas0
    writestr += "deltas                = %s\n"%constC.deltas
    writestr += "lADDnQ                = %s\n"%constC.lADDnQ
    if constC.lADDnQ:
        writestr += "IOEl                  = %s\n"%constC.IOEl
        writestr += "IOEl_forADDstart      = %s\n"%constC.IOEl_forADDstart
        writestr += "IOEl_forcollect       = %s\n"%constC.IOEl_forcollect
    writestr += "Ddimer                = %s\n"%constC.Ddimer
    writestr += "phitol                = %s\n"%constC.phitol
    writestr += "digThreshold          = %s\n"%constC.digThreshold
    writestr += "WellTempairedQ        = %s\n"%constC.WellTempairedQ
    if constC.WellTempairedQ:
        writestr += "WT_Biasfactor         = %s\n"%constC.WT_Biasfactor
        writestr += "WT_Biasfactor_ffactor = %s\n"%constC.WT_Biasfactor_ffactor
    writestr += "initialpointN         = %s\n"%constC.initialpointN
    writestr += "gridQ                 = %s\n"%constC.gridQ
    if constC.gridQ:
        writestr += "grid_min              = %s\n"%constC.grid_min
        writestr += "grid_max              = %s\n"%constC.grid_max
        writestr += "grid_bin              = %s\n"%constC.grid_bin
    writestr += "calc_mpiQ             = %s\n"%constC.calc_mpiQ
    writestr += "parallelMetaDQ        = %s\n"%constC.parallelMetaDQ
    writestr += "calc_cupyQ            = %s\n"%constC.calc_cupyQ
    writestr += "GPUgatherQ            = %s\n"%constC.GPUgatherQ
    writestr += "cythonQ               = %s\n"%constC.cythonQ
    writestr += "PBmetaDQ              = %s\n"%constC.PBmetaDQ
    writestr += "exportADDpointsQ      = %s\n"%constC.exportADDpointsQ

    if not os.path.exists("./jobfiles_meta"):
        os.mkdir("./jobfiles_meta")
    with open("./jobfiles_meta/constants.txt", "w") as wf:
        wf.write(writestr)
    return constC
