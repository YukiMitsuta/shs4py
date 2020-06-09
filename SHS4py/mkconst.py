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
        constC.Temp    = 298.0            # tempeture (K)
        k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
        N_A            = 6.002214e23      # Avogadro constant (mol^-1)
        constC.betainv = constC.Temp * k_B * N_A # 1/beta (kJ / mol)
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
        constC.sameEQthreshold = 0.1

    if not "IOEsphereA_initial" in _keylist:
       constC. IOEsphereA_initial = 0.20
    if not "IOEsphereA_dist" in _keylist:
        constC.IOEsphereA_dist    = 0.05

    if not "deltas0" in _keylist:
        constC.deltas0 = 0.1
    if not "deltas" in _keylist:
        constC.deltas  = 0.01

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
    if not "EQwallmax" in _keylist:
        constC.EQwallmax         = np.array([ 1.0e30 for _ in range(100)])
    if not "EQwallmin" in _keylist:
        constC.EQwallmin         = np.array([-1.0e30 for _ in range(100)])
    if not "abslist" in _keylist:
        constC.abslist           = np.array([ False  for _ in range(100)])

    if not "Ddimer" in _keylist:
        constC.Ddimer          = 0.05
        #constC.Ddimer          = 0.005
    if not "Ddimer_max" in _keylist:
        constC.Ddimer_max      = 0.10
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
    if not "grid_importQ" in _keylist:
        constC.grid_importQ = False
    if not "grid_min" in _keylist:
        constC.grid_min = 0.0
    if not "grid_max" in _keylist:
        constC.grid_max = 5.0
    if not "grid_bin" in _keylist:
        constC.grid_bin = 1000 

    if not "periodicQ" in _keylist:
        constC.periodicQ = False
    if not "periodicmax" in _keylist:
        #constC.periodicmax = [1.0e30 for _ in range(100)]
        constC.periodicmax = 1.0e30
    if not "periodicmin" in _keylist:
        #constC.periodicmin = [-1.0e30 for _ in range(100)]
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
        try:
            from . import calcVES
        except ImportError:
            import calcVES
        constC.calcgau = calcgau
        constC.calcVES = calcVES
        include_dirs = [np.get_include()]
    if not "PBmetaDQ" in _keylist:
        constC.PBmetaDQ = False
    if not "exportADDpointsQ" in _keylist:
        constC.exportADDpointsQ = False

    if not "use_jacQ" in _keylist:
        constC.use_jacQ = False

    if not "x0randomQ" in _keylist:
        constC.x0randomQ = False

    if not "coeffNmax" in _keylist:
        constC.coeffNmax  = 100
    #if not "coeffNskip" in _keylist:
        #constC.coeffNskip = 5

    if not "coeffPickN" in _keylist:
        constC.coeffPickN = False
    if not "coeffPickNlist" in _keylist:
        constC.coeffPickNlist = False
    if not "coeffabsmin" in _keylist:
        constC.coeffabsmin = 0.0

    if not "chkBifurcationQ" in _keylist:
        constC.chkBifurcationQ = False
    if not "chkinitialTSQ" in _keylist:
        constC.chkinitialTSQ = False
    if not "bifucationTH" in _keylist:
        constC.bifucationTH = np.pi / 8.0
    if not "bifucation_eigTH" in _keylist:
        constC.bifucation_eigTH = -0.1
    if not "s_bif0" in _keylist:
        constC.s_bif0 = np.pi / 8.0
    if not "s_bif" in _keylist:
        constC.s_bif = 0.01
    if not "abslist" in _keylist:
        constC.abslist = [False for _ in range(100)]

    constC.writestr  = ""
    constC.writestr += "systemname            = %s\n"%constC.systemname
    constC.writestr += "pwdpath               = %s\n"%constC.pwdpath
    constC.writestr += "Temp                  = %s\n"%constC.Temp
    constC.writestr += "lockfilepath          = %s\n"%constC.lockfilepath
    constC.writestr += "lockfilepath_UIlist   = %s\n"%constC.lockfilepath_UIlist
    constC.writestr += "threshold             = %s\n"%constC.threshold
    constC.writestr += "moveQ                 = %s\n"%constC.moveQ
    constC.writestr += "threshold             = %s\n"%constC.threshold
    constC.writestr += "neiborfluct_threshold = %s\n"%constC.neiborfluct_threshold
    constC.writestr += "sameEQthreshold       = %s\n"%constC.sameEQthreshold
    constC.writestr += "IOEsphereA_initial    = %s\n"%constC.IOEsphereA_initial
    constC.writestr += "IOEsphereA_dist       = %s\n"%constC.IOEsphereA_dist
    constC.writestr += "deltas0               = %s\n"%constC.deltas0
    constC.writestr += "deltas                = %s\n"%constC.deltas
    constC.writestr += "lADDnQ                = %s\n"%constC.lADDnQ
    if constC.lADDnQ:
        constC.writestr += "IOEl                  = %s\n"%constC.IOEl
        constC.writestr += "IOEl_forADDstart      = %s\n"%constC.IOEl_forADDstart
        constC.writestr += "IOEl_forcollect       = %s\n"%constC.IOEl_forcollect
    constC.writestr += "Ddimer                = %s\n"%constC.Ddimer
    constC.writestr += "phitol                = %s\n"%constC.phitol
    constC.writestr += "digThreshold          = %s\n"%constC.digThreshold
    constC.writestr += "WellTempairedQ        = %s\n"%constC.WellTempairedQ
    if constC.WellTempairedQ:
        constC.writestr += "WT_Biasfactor         = %s\n"%constC.WT_Biasfactor
        constC.writestr += "WT_Biasfactor_ffactor = %s\n"%constC.WT_Biasfactor_ffactor
    constC.writestr += "initialpointN         = %s\n"%constC.initialpointN
    constC.writestr += "gridQ                 = %s\n"%constC.gridQ
    if constC.gridQ:
        constC.writestr += "grid_min              = %s\n"%constC.grid_min
        constC.writestr += "grid_max              = %s\n"%constC.grid_max
        constC.writestr += "grid_bin              = %s\n"%constC.grid_bin
    constC.writestr += "calc_mpiQ             = %s\n"%constC.calc_mpiQ
    constC.writestr += "parallelMetaDQ        = %s\n"%constC.parallelMetaDQ
    constC.writestr += "calc_cupyQ            = %s\n"%constC.calc_cupyQ
    constC.writestr += "GPUgatherQ            = %s\n"%constC.GPUgatherQ
    constC.writestr += "cythonQ               = %s\n"%constC.cythonQ
    constC.writestr += "PBmetaDQ              = %s\n"%constC.PBmetaDQ
    constC.writestr += "exportADDpointsQ      = %s\n"%constC.exportADDpointsQ

    #if not os.path.exists("./jobfiles_meta"):
        #os.mkdir("./jobfiles_meta")
    #with open("./jobfiles_meta/constants.txt", "w") as wf:
        #wf.write(constC.writestr)
    return constC
