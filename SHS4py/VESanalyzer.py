#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

import os, glob, shutil, copy
from statistics import stdev
from . import functions
import numpy as np
from scipy import interpolate

class VESfuncC(object):
    def __init__(self, functionName, pldic, const):
        self.const = const
        self.functionName = functionName
        if self.const.calc_cupyQ:
            if functionName != "BF_FOURIER":
                if functionName != "BF_CHEBYSHEV":
                    if functionName != "BF_CHEBYSHEV_SYMMETRY":
                        print("Error; calc_cupy can applied only BF_FOURIER or BF_CHEBYSHEV")
                        exit()
        if functionName == "BF_FOURIER":
            self.readpldic_triF(pldic)
            self.fourier(pldic)
        elif functionName == "BF_COSINE":
            self.readpldic_triF(pldic)
            self.cosine(pldic)
        elif functionName == "BF_SINE":
            self.readpldic_triF(pldic)
            self.sine(pldic)
        elif functionName == "BF_CHEBYSHEV":
            self.readpldic_cheb(pldic)
            self.chebyshev(pldic)
        elif functionName == "BF_CHEBYSHEV_SYMMETRY":
            self.readpldic_cheb(pldic)
            self.chebyshev_symm(pldic)
        elif functionName == "BF_LEUGENDRE":
            self.readpldic_cheb(pldic)
            self.leugendre(pldic)
        else:
            print("ERROR; functionName %s is not prepaired"%functionName)
            exit()
    def readpldic_triF(self, pldic):
        minimum = pldic["MINIMUM"]
        if '-pi' in minimum:
            minimum = - np.pi
        elif 'pi' in minimum:
            minimum = np.pi
        else:
            minimum = float(minimum)
        maximum = pldic["MAXIMUM"]
        if '-pi' in maximum:
            maximum = - np.pi
        elif 'pi' in maximum:
            maximum = np.pi
        else:
            maximum = float(maximum)
        self.piPinv = 2.0 * np.pi / (maximum - minimum)
    def readpldic_cheb(self, pldic):
        minimum = pldic["MINIMUM"]
        if '-pi' in minimum:
            minimum = - np.pi
        elif 'pi' in minimum:
            minimum = np.pi
        else:
            minimum = float(minimum)
        maximum = pldic["MAXIMUM"]
        if '-pi' in maximum:
            maximum = - np.pi
        elif 'pi' in maximum:
            maximum = np.pi
        else:
            maximum = float(maximum)
        #print(minimum, flush = True)
        #print(maximum, flush = True)
        self.calc_t = lambda x: (x - (maximum + minimum) * 0.5) / (maximum - minimum) * 2.0
        self.tconst = 2.0 / (maximum - minimum)
        #print(self.tconst, flush = True)
        #print(self.calc_t(40.0), flush = True)
        #print(self.calc_t(0.0), flush = True)
    def fourier(self, pldic):
        self.f        = self.fourier_f
        self.grad     = self.fourier_grad
        self.gradgrad = self.fourier_gradgrad
    def fourier_f(self, x_i, k):
        if 10 < x_i:
            print("ERROR; x_i = %s"%x_i, flush = True) 
            exit()
        if self.const.cythonQ:
            return self.const.calcVES.fourier_f(x_i, k, self.piPinv)
        if k == 0:
            return 1.0
        else:
            #a = ((k // 2)+1) * self.piPinv
            a = ((k-1) // 2+1) * self.piPinv
            if k % 2 == 0:
                return np.sin(a * x_i)
            else:
                return np.cos(a * x_i)
    def fourier_grad(self, x_i, k):
        if self.const.cythonQ:
            return self.const.calcVES.fourier_grad(x_i, k, self.piPinv)
        if k == 0:
            return 0.0
        else:
            #a = ((k // 2)+1) * self.piPinv
            a = ((k-1) // 2+1) * self.piPinv
            if k % 2 == 0:
                return   a * np.cos(a * x_i)
            else:
                return - a * np.sin(a * x_i)
    def fourier_gradgrad(self, x_i, k):
        if self.const.cythonQ:
            return self.const.calcVES.fourier_gradgrad(x_i, k, self.piPinv)
        if k == 0:
            return 0.0
        else:
            #a = ((k // 2)+1) * self.piPinv
            a = ((k-1) // 2+1) * self.piPinv
            if k % 2 == 0:
                return - a * a * np.sin(a * x_i)
            else:
                return - a * a * np.cos(a * x_i)
    def cosine(self, pldic):
        self.f        = self.cosine_f
        self.grad     = self.cosine_grad
        self.gradgrad = self.cosine_gradgrad
    def cosine_f(self, x_i, k):
        #if self.const.cythonQ:
            #return self.const.calcVES.cosine_f(x_i, k, self.piPinv)
        if k == 0:
            return 1.0
        else:
            return np.cos(k * self.piPinv * x_i)
    def cosine_grad(self, x_i, k):
        #if self.const.cythonQ:
            #return self.const.calcVES.cosine_grad(x_i, k, self.piPinv)
        if k == 0:
            return 0.0
        else:
            a = k * self.piPinv
            return - a * np.sin(a * x_i)
    def cosine_gradgrad(self, x_i, k):
        #if self.const.cythonQ:
            #return self.const.calcVES.cosine_gradgrad(x_i, k, self.piPinv)
        if k == 0:
            return 0.0
        else:
            a = k * self.piPinv
            return - a * a * np.cos(a * x_i)
    def sine(self, pldic):
        self.f        = self.sine_f
        self.grad     = self.sine_grad
        self.gradgrad = self.sine_gradgrad
    def sine_f(self, x_i, k):
        #if self.const.cythonQ:
            #return self.const.calcVES.sine_f(x_i, k, self.piPinv)
        if k == 0:
            return 1.0
        else:
            return np.sin(k * self.piPinv * x_i)
    def sine_grad(self, x_i, k):
        #if self.const.cythonQ:
            #return self.const.calcVES.sine_grad(x_i, k, self.piPinv)
        if k == 0:
            return 0.0
        else:
            a = k * self.piPinv
            return a * np.cos(a * x_i)
    def sine_gradgrad(self, x_i, k):
        #if self.const.cythonQ:
            #return self.const.calcVES.sine_gradgrad(x_i, k, self.piPinv)
        if k == 0:
            return 0.0
        else:
            a = k * self.piPinv
            return - a * a * np.sin(a * x_i)
    def chebyshev(self, pldic):
        self.f        = self.chebyshev_f
        self.grad     = self.chebyshev_grad
        self.gradgrad = self.chebyshev_gradgrad
    def chebyshev_f(self, x_i, k):
        if k == 0:
            return 1.0
        t = self.calc_t(x_i)
        if k == 1:
            return t
        if self.const.cythonQ:
            return self.const.calcVES.chebyshev_f(t, k)
        series_f = [t, 1.0]
        #for i in range(2, k+1):
        for _ in range(k-1):
            #returnf = 2.0 * t * series_f[i-1] - series_f[i-2]
            #series_f.append(returnf)
            returnf = 2.0 * t * series_f[0] - series_f[1]
            series_f = [returnf, series_f[0]]
        #print("x_i, t, returnf = %s, %s, %s"%(x_i, t,returnf))
        return returnf
    def chebyshev_grad(self, x_i, k):
        if k == 0:
            return 0.0
        if k == 1:
            #return 1.0
            return self.tconst
        t = self.calc_t(x_i)
        if self.const.cythonQ:
            return self.const.calcVES.chebyshev_grad(t, k, self.tconst)
        series_f    = [t,   1.0]
        series_grad = [1.0, 0.0]
        #for i in range(2, k+1):
        for _ in range(k-1):
            returnf = 2.0 * t * series_f[0] - series_f[1]
            returngrad = 2.0 *     series_f[0] \
                       + 2.0 * t * series_grad[0] \
                                 - series_grad[1]
            series_f    = [returnf, series_f[0]]
            series_grad = [returngrad, series_grad[0]]
        return returngrad * self.tconst
    def chebyshev_gradgrad(self, x_i, k):
        if k == 0:
            return 0.0
        if k == 1:
            return 0.0
        t = self.calc_t(x_i)
        if self.const.cythonQ:
            return self.const.calcVES.chebyshev_gradgrad(t, k, self.tconst)
        series_f        = [t,   1.0]
        series_grad     = [1.0, 0.0]
        series_gradgrad = [0.0, 0.0]
        for _ in range(k-1):
            returnf = 2.0 * t * series_f[0] - series_f[1]
            returngrad = 2.0 *     series_f[0] \
                       + 2.0 * t * series_grad[0] \
                       - series_grad[1]
            returngradgrad = 4.0 *     series_grad[0] \
                           + 2.0 * t * series_gradgrad[0] \
                                     - series_gradgrad[1]
            series_f        = [returnf, series_f[0]]
            series_grad     = [returngrad, series_grad[0]]
            series_gradgrad = [returngradgrad, series_gradgrad[0]]
        return returngradgrad * self.tconst * self.tconst
    def chebyshev_symm(self, pldic):
        self.f        = self.chebyshev_symm_f
        self.grad     = self.chebyshev_symm_grad
        self.gradgrad = self.chebyshev_symm_gradgrad
    def chebyshev_symm_f(self, x_i, k):
        return self.chebyshev_f(x_i, k*2)
    def chebyshev_symm_grad(self, x_i, k):
        return self.chebyshev_grad(x_i, k*2)
    def chebyshev_symm_gradgrad(self, x_i, k):
        return self.chebyshev_gradgrad(x_i, k*2)
    def legendre(self, pldic):
        self.f        = self.legendre_f
        self.grad     = self.legendre_grad
        self.gradgrad = self.legendre_gradgrad
    def legendre_f(self, x_i, k):
        if k == 0:
            return 1.0
        t = self.calc_t(x_i)
        if k == 1:
            return t
        series_f = [t, 1.0]
        for i in range(k-1):
            a = (2.0 * (i - 1) + 1) / i
            b = (i - 1) / i
            returnf = a * t * series_f[0] \
                        - b * series_f[1]
            series_f = [returnf, series_f[0]]
        return returnf
    def legendre_grad(self, x_i, k):
        if k == 0:
            return 0.0
        t = self.calc_t(x_i)
        if k == 1:
            return self.tconst
        series_f    = [t,   1.0]
        series_grad = [1.0, 0.0]
        for i in range(k-1):
            a = (2.0 * (i - 1) + 1) / i
            b = (i - 1) / i
            returnf = a * t * series_f[0] \
                        - b * series_f[1]
            returngrad = a     * series_f[0] \
                       + a * t * series_grad[0] \
                           - b * series_grad[1]
            series_f    = [returnf, series_f[0]]
            series_grad = [returngrad, series_grad[0]]
        return returngrad * self.tconst
    def legendre_gradgrad(self, x_i, k):
        if k == 0:
            return 0.0
        t = self.calc_t(x_i)
        if k == 1:
            return 0.0
        series_f        = [t,   1.0]
        series_grad     = [1.0, 0.0]
        series_gradgrad = [0.0, 0.0]
        #for i in range(2, k+1):
        for _ in range(k-1):
            a = (2.0 * (i - 1) + 1) / i
            b = (i - 1) / i
            returnf = a * t * series_f[0] \
                        - b * series_f[1]
            returngrad = a     * series_f[0] \
                       + a * t * series_grad[0] \
                           - b * series_grad[1]
            returngradgrad = 2.0 * a * series_grad[0] \
                             + a * t * series_gradgrad[0] \
                                 - b * series_gradgrad[1]
            series_f        = [returnf, series_f[0]]
            series_grad     = [returngrad, series_grad[0]]
            series_gradgrad = [returngradgrad, series_gradgrad[0]]
        return returngradgrad * self.tconst * self.tconst

class VESpotential(object):
    def __init__(self, plumedpath, const, rank, root, size, comm):
        self.const = const
        self.rank  = rank
        self.root  = root
        self.size  = size
        self.comm  = comm
        if self.rank == self.root:
            print("start VESpotential")
        if not os.path.exists(plumedpath):
            if self.rank == self.root:
                print("ERROR; There is not %s"%plumedpath)
            exit()
        pldicC = functions.PlumedDatClass(plumedpath)
        OPTlist         = []
        PRINTlist       = []
        self.TDlist     = []
        self.VESlist    = []
        BFlist_unsorted = []
        for pldic in pldicC.pldiclist:
            for a in pldic["options"]:
                if "BF" in a:
                    BFlist_unsorted.append(pldic)
                elif "TD" in a:
                    self.TDlist.append(pldic)
                elif "VES" in a:
                    self.VESlist.append(pldic)
                elif "OPT" in a:
                    OPTlist.append(pldic)
                elif "PRINT" in a:
                    PRINTlist.append(pldic)
                else:
                    continue
                break
        if len(self.VESlist) == 0:
            if self.rank == self.root:
                print("ERROR; There is not VES_LINEAR_EXPANSION in %s"%plumedpath)
            exit()

        for VESdic in self.VESlist:
            for OPTdic in OPTlist:
                OPTbiasnames = OPTdic["BIAS"].split(",")
                if len(OPTbiasnames) == 1:
                    if OPTdic["BIAS"] == VESdic["LABEL"]:
                        VESdic["coeffpath"] = OPTdic["COEFFS_FILE"]
                        if not os.path.exists(VESdic["coeffpath"]):
                            if self.rank == self.root:
                                print("ERROR; There is not %s"%coeff_path)
                            exit()
                        break
                else:
                    findcoeffpathQ = False
                    for OPTbiasN, OPTbiasname in enumerate(OPTbiasnames):
                        if OPTbiasname == VESdic["LABEL"]:
                            VESdic["coeffpath"] = OPTdic["COEFFS_FILE"].replace(".",".c-%s."%OPTbiasN)
                            findcoeffpathQ = True
                            if not os.path.exists(VESdic["coeffpath"]):
                                if self.rank == self.root:
                                    print("ERROR; There is not %s"%VESdic["coeffpath"])
                                exit()
                            break
                    else:
                        if "COEFFS" in VESdic.keys():
                            if os.path.exists(VESdic["COEFFS"]):
                                findcoeffpathQ = True
                                VESdic["coeffpath"] = VESdic["COEFFS"]
                            else:
                                if self.rank == self.root:
                                    print("Caution! ; There is not %s."%VESdic["COEFFS"], flush = True)
                    if findcoeffpathQ:
                        break
            else:
                VESdic["coeffpath"] = "coeffs.data"
                #exit()
            #print("%s -> %s"%(VESdic["LABEL"], VESdic["coeffpath"]))
            VESdic["BFlist"] = []
            for func_label in VESdic["BASIS_FUNCTIONS"].split(','):
                for BFdic in BFlist_unsorted:
                    if BFdic["LABEL"] == func_label:
                        VESdic["BFlist"].append(BFdic)
                        break

            findTDQ = True
            try:
                TD_label =  VESdic["TARGET_DISTRIBUTION"]
            except KeyError:
                TDdic = {"options":"TD_UNIFORM"}
                VESdic["TDdic"] = TDdic
                findTDQ = False
            if findTDQ:
                for TDdic in self.TDlist:
                    if TDdic["LABEL"] == TD_label:
                        VESdic["TDdic"] = copy.copy(TDdic)
                        break
                else:
                    if self.rank == self.root:
                        print("ERROR; There is not TARGET_DISTRIBUTION that is labeled as %s"%TD_label)
                    exit()
        for PRINTdic in PRINTlist:
            arglist = PRINTdic["ARG"].split(",")
        argVES = []
        for VESdic in self.VESlist:
            argVES.extend(VESdic["ARG"].split(","))
        argVES = list(set(argVES))
        self.arg_not_use = set(arglist) - set(argVES)
        if len(self.arg_not_use) != 0:
            if self.rank == self.root:
                print("Caution! %s do not append VES potential; remove from SHS calculation"%self.arg_not_use)
        #for PRINTdic in PRINTlist:
            #ARGorder = {}
            #for i, printarg in enumerate(PRINTdic["ARG"].split(",")):
                #ARGorder[printarg] = copy.copy(i)
        ARGorder = {}
        i = 0
        writeline = ""
        for printarg in arglist:
            if printarg in argVES:
                ARGorder[printarg] = copy.copy(i)
                writeline += "%s, %s\n"%(i, printarg)
                i += 1
        if self.rank == self.root:
            with open("./usedarg.csv", "w") as wf:
                wf.write(writeline)
        self.dim = len(ARGorder)
        for VESdic in self.VESlist:
            VESdic["order"] = []
            #VESdic["BFfunctions"] = [ "Nan"       for _ in range(self.dim)]
            VESdic["BFfunctions"] = {}
            for argSortN, arg in enumerate(VESdic["ARG"].split(",")):
                VESdic["order"].append(ARGorder[arg])
                BFdic = VESdic["BFlist"][argSortN]
                #for BFdic in VESdic["BFlist"]:
                if True:
                    for a in BFdic["options"]:
                        if "BF" in a:
                            break
                    #print(a)
                    VESdic["BFfunctions"][ARGorder[arg]] = VESfuncC(a, BFdic, self.const)
            argN = len(VESdic["order"])

            if self.rank == self.root:
                print("import %s"%VESdic["coeffpath"], flush = True)
                if len(const.coeffPickNlist) == 0:
                    for line in open(VESdic["coeffpath"]):
                        if "#!" in line:
                            if "iteration" in line:
                                line  = line.split()
                                iterN = int(line[-1])
                                PickNlist = [iterN]
                else:
                    PickNlist = const.coeffPickNlist
                VESdic["coefflistlist"] = []
                coefflist = False
                for line in open(VESdic["coeffpath"]):
                    if "#!" in line:
                        if "FIELDS" in line:
                            if coefflist is not False:
                                if iterN in PickNlist:
                                    VESdic["coefflistlist"].append(coefflist)
                                    if const.coeffNmax < len(VESdic["coefflistlist"]):
                                        matchN = len(VESdic["coefflistlist"]) - const.coeffNmax
                                        VESdic["coefflistlist"] = VESdic["coefflistlist"][matchN:]
                            coefflist = []

                        elif "iteration" in line:
                            line  = line.split()
                            iterN = int(line[-1])
                        continue
                    if not iterN in PickNlist:
                        continue
                    line = line.replace('\n','').split()
                    if len(line) == 0:
                        continue
                    alpha = float(line[-3])
                    if abs(alpha) < const.coeffabsmin:
                        continue
                    #coeff     = [ "Nan"       for _ in range(self.dim)]
                    coeff   = {}
                    coeff_k = [int(line[i]) for i in range(argN)]
                    for i, argorder in enumerate(VESdic["order"]):
                        coeff[argorder] = copy.copy(coeff_k[i])
                    #VESdic["coefflist"].append(coeff + [float(line[-3])])
                    #coefflist.append(coeff + [alpha])
                    coeff["alpha"] = copy.copy(alpha)

                    coefflist.append(coeff)

                if iterN in PickNlist:
                    VESdic["coefflistlist"].append(coefflist)
                    if const.coeffNmax < len(VESdic["coefflistlist"]):
                        matchN = len(VESdic["coefflistlist"]) - const.coeffNmax
                        VESdic["coefflistlist"] = VESdic["coefflistlist"][matchN:]

                VESdic["coefflist"] = VESdic["coefflistlist"][-1]
                coefflist     = VESdic["coefflist"]
                coefflistlist = VESdic["coefflistlist"]
            else:
                coefflist     = None
                coefflistlist = None
            if self.const.calc_mpiQ:
                coefflist     = self.comm.bcast(coefflist, root=0)
                coefflistlist = self.comm.bcast(coefflistlist, root=0)
                VESdic["coefflist"] = copy.copy(coefflist)
                VESdic["coefflistlist"] = copy.copy(coefflistlist)
        if self.const.gridQ:
            mpicounter = -1
            for VESdic in self.VESlist:
                mpicounter += 1
                #if mpicounter % self.size != self.rank:
                    #continue
                importedQ = False
                if self.const.grid_importQ:
                    npzpath = VESdic["coeffpath"]+".z_f.npy"
                    if os.path.exists(npzpath):
                        if self.rank == self.root:
                            print("import grid %s"%npzpath, flush = True)
                        importedQ = True
                        z_f = np.load(npzpath)
                    npzpath = VESdic["coeffpath"]+".z_grad.npy"
                    if os.path.exists(npzpath):
                        z_grad = np.load(npzpath)
                    else:
                        importedQ = False
                    npzpath = VESdic["coeffpath"]+".x.npy"
                    if os.path.exists(npzpath):
                        x = np.load(npzpath)
                    else:
                        importedQ = False
                    npzpath = VESdic["coeffpath"]+".y.npy"
                    if os.path.exists(npzpath):
                        y = np.load(npzpath)
                    else:
                        importedQ = False
                if importedQ is False:
                    VESdic, x, y, z_f, z_grad = self.calcgrid(VESdic)
                    if self.rank == self.root:
                        npzpath = VESdic["coeffpath"]+".z_f.npy"
                        np.save(npzpath, z_f)
                        npzpath = VESdic["coeffpath"]+".z_grad.npy"
                        np.save(npzpath, z_grad)
                        npzpath = VESdic["coeffpath"]+".x.npy"
                        np.save(npzpath, x)
                        npzpath = VESdic["coeffpath"]+".y.npy"
                        np.save(npzpath, y)
                        writeline_f = ""
                        writeline_grad = ""
                        for i in range(len(x)):
                            x_i = x[i]
                            for j in range(len(y)):
                                y_j     = y[j]
                                f_ij    = z_f[i,j]
                                #grad_ij = z_grad[i,j]
                                writeline_f    += "{0: 10.9f}, {1: 10.9f}, {2: 10.9f}\n".format(x_i, y_j, f_ij)
                                #writeline_grad += "{0: 10.9f}, {1: 10.9f}, {2: 10.9f}\n".format(x_i, y_j, grad_ij)
                        with open(VESdic["coeffpath"]+"_fgrid.csv", "w") as wf:
                            wf.write(writeline_f)
                        #with open(VESdic["coeffpath"]+"_gradgrid.csv", "w") as wf:
                            #wf.write(writeline_grad)
                if mpicounter % self.size != self.rank:
                    continue
                VESdic["f_grid"]    =  interpolate.RectBivariateSpline(x, y, z_f)
                VESdic["grad_grid"] = [interpolate.RectBivariateSpline(x, y, z_grad[i]) for i in range(2)]
        if self.const.calc_cupyQ:
            if self.const.calc_mpiQ:
                print("Error: cupy and mpi4py must not be used at the same time!")
                exit()
            self.useFourierQ     = False
            self.useChebyshevQ   = False
            indexlist_fourier    = []
            indexlist_chebyshev  = []
            chevindexmax = 0
            indexevenQ = []
            alphalist  = []
            self.calctlist = [lambda x:x for _ in range(self.dim)]
            self.tconstlist = [1.0 for _ in range(self.dim)]
            self.FurierQs    = [False for _ in range(self.dim)]
            self.chebyshevQs = [False for _ in range(self.dim)]
            for VESdic in self.VESlist:
                for coeff in VESdic["coefflist"]:
                    indexcoeff_f = np.zeros(self.dim)
                    indexcoeff_c = np.zeros(self.dim, dtype=int)
                    evenQcoeff = [False for _ in range(self.dim)]
                    for i, argorder in enumerate(VESdic["order"]):
                        BFdic = VESdic["BFlist"][i]
                        if "BF_FOURIER" in BFdic["options"]:
                            self.useFourierQ = True
                            self.FurierQs[argorder] = True
                            idx = coeff[argorder]
                            if idx != 0 and idx%2==0:
                                idxevenQ = True
                            else:
                                idxevenQ = False
                            idxmodified = ((coeff[argorder]-1)//2+1)
                            indexcoeff_f[argorder] = idxmodified
                            evenQcoeff[argorder] = idxevenQ
                        elif "BF_CHEBYSHEV" in BFdic["options"]:
                            self.calctlist[argorder] = VESdic["BFfunctions"][argorder].calc_t
                            self.tconstlist[argorder] = VESdic["BFfunctions"][argorder].tconstlist
                            self.useChebyshevQ = True
                            self.chebyshevQs[argorder] = True
                            idx = int(coeff[argorder])
                            indexcoeff_c[argorder] = idx
                            if chevindexmax < idx:
                                chevindexmax = idx
                        elif "BF_CHEBYSHEV_SYMMETRY" in BFdic["options"]:
                            self.calctlist[argorder] = VESdic["BFfunctions"][argorder].calc_t
                            self.chebyshevQs[argorder] = True
                            self.useChebyshevQ = True
                            idx = int(coeff[argorder])
                            indexcoeff_c[argorder] = idx*2
                            if chevindexmax < idx*2:
                                chevindexmax = idx*2
                        else:
                            print("Error in VESpotential@VESanalyzer: BF cannot calculate")
                            exit()
                    indexlist_fourier.append(indexcoeff_f)
                    indexlist_chebyshev.append(indexcoeff_c)
                    indexevenQ.append(evenQcoeff)
                    alphalist.append(coeff["alpha"])
            if self.useFourierQ:
                self.index_fourier_CP = self.const.cp.array(indexlist_fourier, 
                                            dtype=self.const.cp.float32)
                self.indexN_fourier = len(indexlist_fourier)
                loaded_from_source = r'''
extern "C"{
__global__ void calctheta_f(const bool* evenQ, float* y, unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
      if (evenQ[tid]) {
        y[tid] = 1.570796327-y[tid];
      } 
    }
}
__global__ void calctheta_f_diffAll(const bool* evenQ, const bool* diffQ, float* y, unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) 
    {
      if (diffQ[tid]) {
        if (evenQ[tid]) {
          y[tid] = y[tid];
        } else {
          y[tid] = 1.570796327+y[tid];
        }
      } else {
        if (evenQ[tid]) {
          y[tid] = 1.570796327-y[tid];
        }
      }
    }
}
__global__ void calctheta_f_diff(const bool* evenQ, float* y, unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) 
    {
      if (evenQ[tid]) {
        y[tid] = y[tid];
      } else {
        y[tid] = 1.570796327+y[tid];
      }
    }
}
}'''
                module = self.const.cp.RawModule(code=loaded_from_source)
                self.ker_calctheta_f = module.get_function("calctheta_f")
                self.ker_calctheta_f_diff = module.get_function("calctheta_f_diff")
                self.ker_calctheta_f_diffAll = module.get_function("calctheta_f_diffAll")
            if self.useChebyshevQ:
                self.index_chebyshev_CP = self.const.cp.array(indexlist_chebyshev, 
                                            dtype=self.const.cp.float32)
                self.indexN_chevyshev = len(indexlist_chebyshev)
                #module = self.const.cp.RawModule(code=makeChebyshevKernel_sympy(chevindexmax))
                #module = self.const.cp.RawModule(code=makeChebyshevKernel())
                #self.ker_chevF = module.get_function("chevF")
                self.ker_chevF = self.const.cp.RawKernel(r'''
extern "C" __global__ 
void chevF(const float* ind, const float* x1, float* y) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind[tid] < 0.5) {
      y[tid] = 1.0;
    } else if (0.5 < ind[tid] && ind[tid] < 1.5) {
      y[tid] = x1[tid];
    } else {
      float f0=x1[tid];
      float f1=1.0;
      float fdamp;
      int k;
      for (k=0;k<ind[tid]-1; k++) {
        fdamp = 2.0*x1[tid]*f0-f1;
        f1=f0;
        f0=fdamp;
      }
      y[tid] = fdamp;
    }
}
''', "chevF")
                #self.ker_chevF_diff = module.get_function("chevF_diff")
                self.ker_chevF_diff = self.const.cp.RawKernel(r'''
extern "C" __global__ 
void chevF_diff(const float* ind, const bool* diffQ, const float* x1, float* y) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind[tid] < 0.5) {
      if (diffQ[tid]){
        y[tid] = 0.0;
      } else {
        y[tid] = 1.0;
      }
    } else if (0.5 < ind[tid] && ind[tid] < 1.5) {
      if (diffQ[tid]){
        y[tid] = 1.0;
      } else {
        y[tid] = x1[tid];
      }
    } else {
        float f0=x1[tid];
        float f1=1.0;
        float grad0=1.0;
        float grad1=0.0;
        float fdamp;
        float graddamp;
        int k;
        for (k=0;k<ind[tid]-1; k++) {
          fdamp = 2.0*x1[tid]*f0-f1;
          f1=f0;
          f0=fdamp;
          if (diffQ[tid]){
            graddamp = 2.0 * f0 + 2.0 * x1[tid] * grad0 - grad1;
            grad1 = grad0;
            grad0 = graddamp;
          } 
        }
        if (diffQ[tid]){
          y[tid] = fdamp;
        } else {
          y[tid] = graddamp;
        }
    }
}
''', "chevF_diff")
                self.index_chebyshev = indexlist_chebyshev
            if self.useFourierQ is False:
                if self.useChebyshevQ is False:
                    print("Error in VESpotential@VESanalyzer: cannot convert Base function to Cupy\n is this calcluation performed with BF_FOURIER or BF_CHEBYSHEV(_SYMMETORY) ?")
                    exit()
            self.indexevenQcp = self.const.cp.array(indexevenQ, dtype=self.const.cp.bool)
            self.alphacp = self.const.cp.array(alphalist, dtype=self.const.cp.float32)



#            for VESdic in self.VESlist:
#                indexlist2 = []
#                indexevenQ = []
#                alphalist = []
#                for coeff in VESdic["coefflist"]:
#                    indexcoeff = []
#                    evenQcoeff = []
#                    for i, argorder in enumerate(VESdic["order"]):
#                        idx = coeff[argorder]
#                        if idx != 0 and idx%2==0:
#                            idxevenQ = True
#                        else:
#                            idxevenQ = False
#                        idxmodified = ((coeff[argorder]-1)//2+1)
#                        indexcoeff.append(idxmodified)
#                        evenQcoeff.append(idxevenQ)
#                    indexlist2.append(indexcoeff)
#                    indexevenQ.append(evenQcoeff)
#                    alphalist.append(coeff["alpha"])
#                indexcp = self.const.cp.array(indexlist2, dtype=self.const.cp.float32)
#                indexevenQcp = self.const.cp.array(indexevenQ, dtype=self.const.cp.bool)
#                alphacp = self.const.cp.array(alphalist, dtype=self.const.cp.float32)
#                VESdic["indexCP"] = indexcp
#                VESdic["indexevenQcp"] = indexevenQcp
#                VESdic["alphacp"] = alphacp
#                VESdic["N"] = len(indexlist2)

        if self.rank == self.root:
            print("end VESpotential", flush = True)
            print("Dimension = %s"%self.dim, flush = True)
    def calcgrid(self, VESdic):
        if type(self.const.grid_min) is list:
            argorder = VESdic["order"][0]
            #x = np.arange(self.const.grid_min[argorder], self.const.grid_max[argorder],
                    #(self.const.grid_max[argorder] - self.const.grid_min[argorder])/self.const.grid_bin[argorder])
            x = np.linspace(self.const.grid_min[argorder], self.const.grid_max[argorder], self.const.grid_bin[argorder])
            argorder = VESdic["order"][1]
            #y = np.arange(self.const.grid_min[argorder], self.const.grid_max[argorder],
                    #(self.const.grid_max[argorder] - self.const.grid_min[argorder])/self.const.grid_bin[argorder])
            y = np.linspace(self.const.grid_min[argorder], self.const.grid_max[argorder], self.const.grid_bin[argorder])
        else:
            #x = np.arange(self.const.grid_min, self.const.grid_max,
                #(self.const.grid_max - self.const.grid_min)/self.const.grid_bin)
            x = np.linspace(self.const.grid_min, self.const.grid_max, self.const.grid_bin)
            #y = np.arange(self.const.grid_min, self.const.grid_max,
                #(self.const.grid_max - self.const.grid_min)/self.const.grid_bin)
            y = np.linspace(self.const.grid_min, self.const.grid_max, self.const.grid_bin)
        xxyy = np.meshgrid(x, y)
        xxyy = np.meshgrid(x, y)
        z_f = np.zeros((len(x),len(y)))
        #z_grad = [np.zeros((len(x),len(y))) for _ in range(self.dim)]
        z_grad = [np.zeros((len(x),len(y))) for _ in range(2)]
        if self.rank == self.root:
            print("make grid %s"%VESdic["coeffpath"], flush = True)
        self.TDdic = VESdic["TDdic"]
        mpicounter = -1
        for x_index in range(len(x)):
            for y_index in range(len(y)):
                if self.rank == self.root:
                    if mpicounter % 1000 == 0:
                        print(mpicounter, flush = True)
                mpicounter += 1
                if mpicounter % self.size != self.rank:
                    continue
                for coeff in VESdic["coefflist"]:
                    f_VES = float(coeff["alpha"])
                    for i, argorder in enumerate(VESdic["order"]):
                        if i == 0:
                            xdamp = x[x_index]
                        elif i == 1:
                            xdamp = y[y_index]
                        elif self.const.CVfixQ:
                            if i == 2:
                                xdamp = self.const.fixlist[-1][-1]
                        else:
                            print("ERROR: CVs of VES functions is larger than 2")
                            exit()
                        f_VES *= VESdic["BFfunctions"][argorder].f(xdamp, coeff[argorder])
                    z_f[x_index, y_index] -= f_VES
                    #for i_grad in VESdic["order"]:
                    #for i_grad in range(self.dim):
                        #if not i_grad in coeff:
                            #continue
                    for i_grad in range(2):
                        g_VES = float(coeff["alpha"])
                        for i, argorder in enumerate(VESdic["order"]):
                            if i == 0:
                                xdamp = x[x_index]
                            elif i == 1:
                                xdamp = y[y_index]
                            elif self.const.CVfixQ:
                                if i == 2:
                                    xdamp = self.const.fixlist[-1][-1]
                            #if argorder == i_grad:
                            if i == i_grad:
                                g_VES *= VESdic["BFfunctions"][argorder].grad(xdamp, coeff[argorder])
                            else:
                                g_VES *= VESdic["BFfunctions"][argorder].f(xdamp, coeff[argorder])
                        z_grad[i_grad][x_index, y_index] -= g_VES
                z_f[x_index, y_index] = self.f_TD(None, z_f[x_index, y_index])
                for i_grad in range(2):
                    z_grad[i_grad][x_index, y_index] = self.grad_TD(None, z_grad[i_grad][x_index, y_index])
        if self.const.calc_mpiQ:
            z_f_g = self.comm.gather(z_f, root=0)
            z_grad_g = self.comm.gather(z_grad, root=0)
            if self.rank == self.root:
                z_f = np.zeros((len(x),len(y)))
                for z_fdamp in z_f_g:
                    z_f += z_fdamp
                z_grad = [np.zeros((len(x),len(y))) for _ in range(2)]
                for z_graddamp in z_grad_g:
                    #for i_grad in range(self.dim):
                    for i_grad in range(2):
                        z_grad[i_grad] += z_graddamp[i_grad]
            else:
                z_f_return = None
                z_grad_return = None
            z_f_return = self.comm.bcast(z_f, root=0)
            z_grad_return = self.comm.bcast(z_grad, root=0)
        else:
            z_f_return = z_f
            z_grad_return = z_grad

        if self.rank == self.root:
            print("end grid %s"%VESdic["coeffpath"], flush = True)
        return VESdic, x, y, z_f_return, z_grad_return
    def f(self, x):
        if self.const.CVfixQ:
            _x = self.replaceX(x)
        else:
            _x = x
        if self.const.periodicQ:
            _x = functions.periodicpoint(_x, self.const)
        xdamp = copy.copy(_x)
        if type(self.const.abslist) is list:
            for i, absQ in enumerate(self.const.abslist):
                if absQ:
                    xdamp[i] = abs(xdamp[i])
        returnf = 0.0
        if self.const.gridQ:
            mpicounter = -1
            for VESdic in self.VESlist:
                mpicounter += 1
                if mpicounter % self.size != self.rank:
                    continue
                xdamp0 = xdamp[VESdic["order"][0]]
                xdamp1 = xdamp[VESdic["order"][1]]
                returnf += VESdic["f_grid"].ev(xdamp0, xdamp1)
            if self.const.calc_mpiQ:
                returnf_g = self.comm.gather(returnf, root=0)
                if self.rank == self.root:
                    returnf = sum(returnf_g)
                returnf = self.comm.bcast(returnf, root=0)
            return returnf
        if self.const.calc_cupyQ:
            if self.const.calc_mpiQ:
                print("Error: cupy and mpi4py must not be used at the same time!")
                exit()
            returnf = 0.0
            xdampCP = self.const.cp.array(
                                    [self.calctlist[argorder](xdamp[argorder])
                                     for argorder in range(self.dim)
                                    ]
                            )
            #print(xdampCP)
            if self.useFourierQ:
                y_f = self.const.cp.zeros((self.indexN_fourier, self.dim), dtype=self.const.cp.float32)
                y_f += self.index_fourier_CP
                #y_f = self.const.cp.array(self.index_fourier_CP)
                for argorder in range(self.dim):
                    y_f[:,argorder] *= xdampCP[argorder]
                #self.ker_calctheta_f((self.indexN_fourier,),(self.dim,),
                self.ker_calctheta_f((self.indexN_fourier,),(self.dim,),
                                (self.indexevenQcp, y_f, self.indexN_fourier*self.dim))
                y_f  = self.const.cp.cos(y_f)
                y_f  = self.const.cp.prod(y_f, axis=1)
            if self.useChebyshevQ:
                xdampCPrepeat = self.const.cp.array([xdampCP],dtype=self.const.cp.float32)
                xdampCPrepeat = self.const.cp.repeat(xdampCPrepeat,self.indexN_chevyshev, axis=0)
                y_c = self.const.cp.zeros((self.indexN_chevyshev, self.dim), dtype=self.const.cp.float32)
                #print( self.const.cp.sum(y_c))
                #for argorder in range(self.dim):
                    #y_c[:,argorder] += xdampCP[argorder]
                #xdampCP = self.const.cp.repeat(xdampCP,self.indexN_chevyshev,axis=0)
                self.ker_chevF((self.indexN_chevyshev,),(self.dim,),
                        (self.index_chebyshev_CP, xdampCPrepeat, y_c))
                #print("y_c; ",y_c[:1])
                #print("index;",self.index_chebyshev_CP[-10:])
                #print("xdamp;",xdampCPrepeat[-10:])
                #print("y_c; ",y_c[-10:])
                y_c  = self.const.cp.prod(y_c, axis=1)
                #print(y_c[-2000:-1000])
                #print(y_c[:1000])
#                y_c = []
#                #y_c = self.const.cp.zeros(self.indexN_chevyshev, dtype=self.const.cp.float32)
#                for i, indexlist in enumerate(self.index_chebyshev):
#                    pind = 1.0
#                    for argorder in range(self.dim):
#                        if self.chebyshevQs[argorder]:
#                            pind *= self.VESlist[0]["BFfunctions"][argorder].f(xdamp[argorder],indexlist[argorder])
#                    #print(pind)
#                    #y_c[i] += float(pind)
#                    y_c.append(float(pind))
#                y_c = self.const.cp.array(y_c, dtype=self.const.cp.float32)

            if self.useFourierQ and self.useChebyshevQ:
                y = y_f*y_c
            elif self.useFourierQ:
                y = y_f
            elif self.useChebyshevQ:
                y = y_c
            y *= self.alphacp
            y  = self.const.cp.sum(y)
            self.TDdic = self.VESlist[0]["TDdic"]
            returnf -= self.f_TD(xdamp, y)
            #print(xdamp)
            #print("returnf(cupy)",returnf)
            #exit()

            return float(returnf)

            returnf = 0.0
        mpicounter = -1
        for VESdic in self.VESlist:
            self.TDdic = VESdic["TDdic"]
            f_partVES = 0.0
            for coeff in VESdic["coefflist"]:
                mpicounter += 1
                if mpicounter % self.size != self.rank:
                    continue
                y = float(coeff["alpha"])
                for argorder in VESdic["order"]:
                    y *= VESdic["BFfunctions"][argorder].f(xdamp[argorder], coeff[argorder])
                f_partVES -= y
            returnf += self.f_TD(xdamp, f_partVES)
        if self.const.calc_mpiQ:
            returnf_g = self.comm.gather(returnf, root=0)
            if self.rank == self.root:
                returnf = sum(returnf_g)
            returnf = self.comm.bcast(returnf, root=0)
            #if self.rank == self.root:
                #print("returnf, returnf_grid = %s, %s"%(returnf,returnf_grid),flush=True)
            #exit()
        #print(x)
        #print(xdamp)
        #print(returnf)
        #print("returnf(cpu)",returnf)
        #exit()
        return returnf
    def f_stdev(self, x):
        returnflist = [0.0 for _ in range(len(self.VESlist[0]["coefflistlist"]))]

        for VESdic in self.VESlist:
            for coefflistN, coefflist in enumerate(VESdic["coefflistlist"]):
                returnf = 0.0
                self.TDdic = VESdic["TDdic"]
                for coeffN, coeff in enumerate(coefflist):
                    if coeffN % self.size != self.rank:
                        continue
                    #y = float(coeff[self.dim])
                    y = float(coeff["alpha"])
                    for i in range(self.dim):
                        #if coeff[i] == "Nan":
                            #continue
                        if not i in coeff:
                            continue
                        y *= VESdic["BFfunctions"][i].f(x[i], coeff[i])
                    returnf -= y
                if self.const.calc_mpiQ:
                    returnf_g = self.comm.gather(returnf, root=0)
                    if self.rank == self.root:
                        returnf = sum(returnf_g)
                    returnf = self.comm.bcast(returnf, root=0)
                returnflist[coefflistN] += self.f_TD(x, returnf)
        #return stdev(returnflist)
        return returnflist
    def grad(self, x):
        #print(x)
        if self.const.CVfixQ:
            _x = self.replaceX(x)
        else:
            _x = x
        if self.const.periodicQ:
            _x = functions.periodicpoint(_x, self.const)
        xdamp = copy.copy(_x)
        for i, absQ in enumerate(self.const.abslist):
            if absQ:
                xdamp[i] = abs(xdamp[i])
        returngrad = np.zeros(self.dim)
        if self.const.gridQ:
            mpicounter = -1
            for VESdic in self.VESlist:
                mpicounter += 1
                if mpicounter % self.size != self.rank:
                    continue
                xdamp0 = xdamp[VESdic["order"][0]]
                xdamp1 = xdamp[VESdic["order"][1]]

                for i_grad in range(2):
                    returngrad[VESdic["order"][i_grad]] += VESdic["grad_grid"][i_grad].ev(xdamp0,xdamp1)
            if self.const.calc_mpiQ:
                returngrad_g = self.comm.gather(returngrad, root=0)
                if self.rank == self.root:
                    returngrad = np.zeros(self.dim)
                    for returngraddamp in returngrad_g:
                        returngrad += returngraddamp
                returngrad = self.comm.bcast(returngrad, root=0)
            if self.const.CVfixQ:
                returngraddamp = np.zeros(len(x))
                i_index = 0
                for i, (fixQ, _) in enumerate(self.const.fixlist):
                    if fixQ:
                        continue
                    returngraddamp[i_index] = returngrad[i]
                    i_index += 1
                returngrad = returngraddamp
            return returngrad
        if self.const.calc_cupyQ:
            if self.const.calc_mpiQ:
                print("Error: cupy and mpi4py must not be used at the same time!")
                exit()
            returngrad = np.zeros(self.dim)
            #xdampCP      = self.const.cp.array(
                                    #[self.calctlist[argorder](xdamp[argorder])
                                     #for argorder in range(self.dim)
                                    #]
                            #)
            #xdampCP      = xdamp
            returngrad = self.grad_partial_cupyAll(xdamp)
            #returngrad = self.grad_TD(xdamp, returngrad)
            returngrad = np.array(returngrad)
        else:
        #if True:
            returngrad = np.zeros(self.dim)
            mpicounter = -1
            for VESdic in self.VESlist:
                self.TDdic = VESdic["TDdic"]
                grad_partVES = np.zeros(self.dim)
                for coeff in VESdic["coefflist"]:
                    mpicounter += 1
                    if mpicounter % self.size != self.rank:
                        continue
                    for i_grad in range(self.dim):
                        if not i_grad in coeff:
                            continue
                        #if coeff[i_grad] == "Nan":
                            #continue
                        #y = float(coeff[self.dim])
                        #y = coeff[self.dim]
                        y = float(coeff["alpha"])
                        #for i in range(self.dim):
                        for i in VESdic["order"]:
                            #if not i in coeff:
                                #continue
                            if i == i_grad:
                                y *= VESdic["BFfunctions"][i].grad(xdamp[i], coeff[i])
                            else:
                                y *= VESdic["BFfunctions"][i].f(xdamp[i], coeff[i])
                        grad_partVES[i_grad] -= y
                returngrad += self.grad_TD(xdamp, grad_partVES)
        if self.const.calc_mpiQ:
            returngrad_g = self.comm.gather(returngrad, root=0)
            if self.rank == self.root:
                returngrad = np.zeros(self.dim)
                for returngraddamp in returngrad_g:
                    returngrad += returngraddamp
            returngrad = self.comm.bcast(returngrad, root=0)
            #exit()
        #print("returngrad(cpu) ",returngrad)
        #exit()
        if self.const.CVfixQ:
            returngraddamp = np.zeros(len(x))
            i_index = 0
            for i, (fixQ, _) in enumerate(self.const.fixlist):
                if fixQ:
                    continue
                returngraddamp[i_index] = returngrad[i]
                i_index += 1
            returngrad = returngraddamp
        #if self.rank == self.root:
            #printline = ""
            #_d = returngrad - returngrad_grid
            #for _dpoint in _d:
                #printline += "% 5.4f, "%_dpoint
            #print("returngrad - returngrad_grid = %s"%printline, flush=True)
            #if 1.0 < np.linalg.norm(_d):
                #print(_x)

        #print("returngrad ",np.linalg.norm(returngrad))
        return returngrad
    def grad_partial_cupy(self, VESdic, xdampCP):
        returngrad = np.zeros(self.dim)
        argN         = len(VESdic["order"])
        indexcp      = VESdic["indexCP"]
        indexevenQcp = VESdic["indexevenQcp"]
        alphacp      = VESdic["alphacp"]
        for i_order, i_grad in enumerate(VESdic["order"]):
            grad_cp  = self.const.cp.zeros((VESdic["N"], argN), dtype=self.const.cp.float32)
            grad_cp += indexcp
            diffQ    = []
            for argorder in VESdic["order"]:
                if i_grad == argorder:
                    diffQ.append(True)
                else:
                    diffQ.append(False)
            diffQ   = self.const.cp.array([[diffQ]])
            diffQcp = self.const.cp.repeat(diffQ,VESdic["N"],axis=0)
            for i, argorder in enumerate(VESdic["order"]):
                grad_cp[:,i] *= xdampCP[argorder]
            self.const.ker_calctheta_f_diff((VESdic["N"],),(argN,),
                        (self.indexcp, self.indexevenQcp, diffQcp, grad_cp, VESdic["N"]*argN))
            grad_cp  = self.const.cp.cos(grad_cp)
            grad_cp  = self.const.cp.prod(grad_cp, axis=1)
            grad_cp *= alphacp
            grad_cp *= indexcp[:,i_order]
            grad_cp  = self.const.cp.sum(grad_cp)
            returngrad[i_grad] -= grad_cp
        return returngrad
    def grad_partial_cupyAll(self, xdamp):
        #xdampCP    = self.const.cp.array(xdamp)
        xdampCP = self.const.cp.array(
                                    [self.calctlist[argorder](xdamp[argorder])
                                     for argorder in range(self.dim)
                                    ]
                            )
        returngrad = np.zeros(self.dim)
        if self.useFourierQ:
            y_f = self.const.cp.zeros((self.indexN_fourier, self.dim), dtype=self.const.cp.float32)
            y_f += self.index_fourier_CP
            for argorder in range(self.dim):
                y_f[:,argorder] *= xdampCP[argorder]
            self.ker_calctheta_f((self.indexN_fourier,),(self.dim,),
                                (self.indexevenQcp, y_f, self.indexN_fourier*self.dim))
            y_f  = self.const.cp.cos(y_f)
        if self.useChebyshevQ:
            xdampCPrepeat = self.const.cp.array([xdampCP], dtype=self.const.cp.float32)
            xdampCPrepeat = self.const.cp.repeat(xdampCPrepeat,self.indexN_chevyshev, axis=0)
#            y_c = []
#            for i, indexlist in enumerate(self.index_chebyshev):
#                pind = 1.0
#                for argorder in range(self.dim):
#                    if self.chebyshevQs[argorder]:
#                        pind *= self.VESlist[0]["BFfunctions"][argorder].f(xdamp[argorder],indexlist[argorder])
#                y_c.append(float(pind))
#            y_c = self.const.cp.array(y_c, dtype=self.const.cp.float32)
            y_c = self.const.cp.zeros((self.indexN_chevyshev, self.dim), dtype=self.const.cp.float32)
            self.ker_chevF((self.indexN_chevyshev,),(self.dim,),
                        (self.index_chebyshev_CP, xdampCPrepeat, y_c))
            y_c  = self.const.cp.prod(y_c, axis=1)
        for i_grad in range(self.dim):
            #print("xdampcp",xdampCP)
            if self.useFourierQ:
                if self.FurierQs[i_grad]:
                    grad_f = self.const.cp.zeros((self.indexN_fourier, self.dim), dtype=self.const.cp.float32)
                    grad_f += self.index_fourier_CP
                    #grad_f = self.const.cp.array(self.index_fourier_CP)
                    for argorder in range(self.dim):
                        grad_f[:,argorder] *= xdampCP[argorder]
                    diffQ= [False for _ in range(self.dim)]
                    diffQ[i_grad] = True
                    diffQ = self.const.cp.array([diffQ])
                    diffQcp = self.const.cp.repeat(diffQ,self.indexN_fourier,axis=0)
                    #print(diffQcp)
                    #print("1144;i ",grad_f[:,i_grad])
                    #print("1145; ",self.indexevenQcp[:,i_grad])
                    self.ker_calctheta_f_diffAll((self.indexN_fourier,),(self.dim,),
                        (self.indexevenQcp, diffQcp, grad_f, self.indexN_fourier*self.dim))
                    #print("1148; ",grad_f[:,i_grad])
                    #grad_f2 = self.const.cp.cos(grad_f)
                    #print("1150; ",grad_f2[:,i_grad])
                    #grad_f2  = self.const.cp.prod(grad_f2, axis=1)
                    grad_f = self.const.cp.cos(grad_f)
                    grad_f  = self.const.cp.prod(grad_f, axis=1)


#                    grad_f_diff  = self.const.cp.zeros((self.indexN_fourier,2), 
#                                        dtype=self.const.cp.float32)
#                    grad_f_diff[:,0] += self.index_fourier_CP[:,i_grad]
#                    grad_f_diff *= xdampCP[i_grad]
#                    evenQcp_diff  = self.const.cp.zeros((self.indexN_fourier,2), 
#                                        dtype=self.const.cp.float32)
#                    evenQcp_diff[:,0] = self.indexevenQcp[:,i_grad]
#                    print("1160; ",grad_f_diff)
#                    print("1161: ",evenQcp_diff)
#                    self.ker_calctheta_f_diff((self.indexN_fourier,),(2,),
#                        (evenQcp_diff, grad_f_diff, self.indexN_fourier*2))
#                    grad_f_diff = grad_f_diff[:,0]
#                    print("1165; ",grad_f_diff)
#                    grad_f_diff = self.const.cp.cos(grad_f_diff)
#                    print("1167; ",grad_f_diff)
#                    #grad_f_diff *= self.index_fourier_CP[:,i_grad]
#                    grad_f = self.const.cp.zeros((self.indexN_fourier, self.dim), dtype=self.const.cp.float32)
#                    grad_f += y_f
#                    #grad_f = self.const.cp.array(y_f)
#                    grad_f[:,i_grad] = 1.0
#                    grad_f  = self.const.cp.prod(grad_f, axis=1)
#                    grad_f *= grad_f_diff
#                    #print(grad_f_diff)
#                    print("="*100)
#                    print("grad_f;  ",grad_f)
#                    print("grad_f2; ",grad_f2)
#                    print("="*100)
#                    #grad_f = grad_f2
                else:
                    #grad_f = self.const.cp.zeros((self.indexN_fourier, self.dim), dtype=self.const.cp.float32)
                    #grad_f += y_f
                    grad_f = self.const.cp.array(y_f)
                    grad_f  = self.const.cp.prod(grad_f, axis=1)
                #print("="*100)
                #print("i_grad;",i_grad)
                #print("grad_f;",grad_f)
                #print("="*100)
            if self.useChebyshevQ:
#                diffQ = []
#                for argorder in range(self.dim):
#                    if self.chebyshevQs[argorder]:
#                        if i_grad == argorder:
#                            diffQ.append(True)
#                        else:
#                            diffQ.append(False)
#                    else:
#                            diffQ.append(False)
#                diffQ = self.const.cp.array([[diffQ]])
#                diffQcp = self.const.cp.repeat(diffQ,self.indexN_fourier,axis=0)
##
#                grad_c = self.const.cp.zeros((self.indexN_chevyshev, self.dim), 
#                                    dtype=self.const.cp.float32)
#
#                for argorder in range(self.dim):
#                    grad_c[:,argorder] += xdampCP[argorder]
#                self.ker_chevF_diff((self.indexN_chevyshev,),(self.dim,),
#                                (self.index_chebyshev_CP, diffQcp, xdampCPrepeat, grad_c))
#                grad_c  = self.const.cp.prod(grad_c, axis=1)
#
##                #grad_c = self.const.cp.ones((self.indexN_chevyshev,), 
##                                    #dtype=self.const.cp.float32)
##


                #if self.chebyshevQs[i_grad]:
                if False:
                    grad_c = []
                    #grad_c = self.const.cp.zeros(self.indexN_chevyshev, dtype=self.const.cp.float32)
                    for i, indexlist in enumerate(self.index_chebyshev):
                        pind = 1.0
                        for argorder in range(self.dim):
                            if self.chebyshevQs[argorder]:
                                if i_grad == argorder:
                                    #print(self.VESlist[0]["BFfunctions"][argorder].grad(xdamp[argorder],indexlist[argorder]))
                                    pind *= self.VESlist[0]["BFfunctions"][argorder].grad(xdamp[argorder],indexlist[argorder])
                                else:
                                    pind *= self.VESlist[0]["BFfunctions"][argorder].f(xdamp[argorder],indexlist[argorder])
                        #grad_c[i] += float(pind)
                        grad_c.append(float(pind))
                    grad_c = self.const.cp.array(grad_c, dtype=self.const.cp.float32)
                    #print(grad_c)
                else:
                    grad_c = self.const.cp.array(y_c)
                #print("+"*100)
                #print("grad_f; ",grad_f)
                #print("grad_c; ",grad_c)
                #print("+"*100)
            if self.useFourierQ and self.useChebyshevQ:
                grad_cp = grad_f*grad_c
            elif self.useFourierQ:
                grad_cp = grad_f
            elif self.useChebyshevQ:
                grad_cp = grad_c
            grad_cp *= self.alphacp
            if self.FurierQs[i_grad]:
                grad_cp *= self.index_fourier_CP[:,i_grad]
            grad_cp  = self.const.cp.sum(grad_cp)
            returngrad[i_grad] -= grad_cp
        #print("returngrad(cupy) ",returngrad)
        #exit()
        return returngrad
    def hessian(self, x):
        if self.const.CVfixQ:
            _x = self.replaceX(x)
        else:
            _x = x
        if self.const.periodicQ:
            _x = functions.periodicpoint(_x, self.const)
        if self.const.periodicQ:
            _x = functions.periodicpoint(_x, self.const)
        xdamp = copy.copy(_x)
        for i, absQ in enumerate(self.const.abslist):
            if absQ:
                xdamp[i] = abs(xdamp[i])
        returnhess = np.zeros((self.dim, self.dim))
        for VESdic in self.VESlist:
            self.TDdic = VESdic["TDdic"]
            for coeffN, coeff in enumerate(VESdic["coefflist"]):
                if coeffN % self.size != self.rank:
                    continue
                for i_hess in VESdic["order"]:
                    y = float(coeff["alpha"])
                    for i in VESdic["order"]:
                        if i == i_hess:
                            y *= VESdic["BFfunctions"][i].gradgrad(xdamp[i], coeff[i])
                        else:
                            y *= VESdic["BFfunctions"][i].f(xdamp[i], coeff[i])
                    returnhess[i_hess, i_hess] -= y
        #returnhess_cupy = copy.copy(returnhess)
        if self.const.calc_cupyQ:
            #returnhess_cupy  += self.hess_partial_cupyAll(xdamp)
            returnhess  += self.hess_partial_cupyAll(xdamp)
        #print(returnhess_cupy)
        else:
        #if True:
            for VESdic in self.VESlist:
                self.TDdic = VESdic["TDdic"]
                for coeffN, coeff in enumerate(VESdic["coefflist"]):
                    if coeffN % self.size != self.rank:
                        continue
                    for i_hess in VESdic["order"]:
                        for j_hess in VESdic["order"]:
                            if i_hess <= j_hess:
                                continue
                            y = float(coeff["alpha"])
                            for i in VESdic["order"]:
                                if i == i_hess or i == j_hess:
                                    y *= VESdic["BFfunctions"][i].grad(xdamp[i], coeff[i])
                                else:
                                    y *= VESdic["BFfunctions"][i].f(xdamp[i], coeff[i])
                            returnhess[i_hess, j_hess] -= y
                            returnhess[j_hess, i_hess] -= y
        #print(returnhess_cupy)
        #print(returnhess)
        #hessdamp = returnhess - returnhess_cupy
        #for i_hess in range(self.dim):
            #for j_hess in range(self.dim):
                #print("(%s,%s)-> % 10.9f"%(i_hess,j_hess,hessdamp[i_hess,j_hess]))
        #exit()
        returnhess = self.hessian_TD(xdamp, returnhess)
        if self.const.calc_mpiQ:
            returnhess_g = self.comm.gather(returnhess, root=0)
            if self.rank == self.root:
                returnhess = np.zeros((self.dim, self.dim))
                for returnhessdamp in returnhess_g:
                    returnhess += returnhessdamp
            returnhess = self.comm.bcast(returnhess, root=0)
        if self.const.CVfixQ:
            returnhessdamp = np.zeros((len(x),len(x)))
            i_index = 0
            for i, (fixQ_i, _) in enumerate(self.const.fixlist):
                if fixQ_i:
                    continue
                j_index = 0
                for j, (fixQ_j, _) in enumerate(self.const.fixlist):
                    if fixQ_j:
                        continue
                    returnhessdamp[i_index, j_index] = returnhess[i,j]
                    j_index+= 1
                i_index += 1
            returnhess = returnhessdamp
        return returnhess
    def hess_partial_cupyAll(self, xdamp):
        xdampCP = self.const.cp.array(
                                    [self.calctlist[argorder](xdamp[argorder])
                                     for argorder in range(self.dim)
                                    ]
                            )
        #print("xdampCP;",xdampCP)
        if self.useFourierQ:
            y_f = self.const.cp.zeros((self.indexN_fourier, self.dim), dtype=self.const.cp.float32)
            y_f += self.index_fourier_CP
            for argorder in range(self.dim):
                y_f[:,argorder] *= xdampCP[argorder]
            self.ker_calctheta_f((self.indexN_fourier,),(self.dim,),
                                (self.indexevenQcp, y_f, self.indexN_fourier*self.dim))
            y_f  = self.const.cp.cos(y_f)
        if self.useChebyshevQ:
            xdampCPrepeat = self.const.cp.array([xdampCP], dtype=self.const.cp.float32)
            xdampCPrepeat = self.const.cp.repeat(xdampCPrepeat,self.indexN_chevyshev, axis=0)
            y_c = self.const.cp.zeros((self.indexN_chevyshev, self.dim), dtype=self.const.cp.float32)
            #self.ker_chevF((self.indexN_chevyshev,),(self.dim,),
                                #(self.index_chebyshev_CP, xdampCPreleat, y_c, self.indexN_chevyshev*self.dim))
            self.ker_chevF((self.indexN_chevyshev,),(self.dim,),
                        (self.index_chebyshev_CP, xdampCPrepeat, y_c))
            #print(self.index_chebyshev_CP[-2])
            #print(y_c[-1])
            y_c  = self.const.cp.prod(y_c, axis=1)
        returnhess = np.zeros((self.dim,self.dim))
        for i_hess in range(self.dim):
            for j_hess in range(self.dim):
                if i_hess <= j_hess:
                    continue
                if self.useFourierQ:
                    if self.FurierQs[i_hess] or self.FurierQs[j_hess]:
                        grad_f = self.const.cp.zeros((self.indexN_fourier, self.dim),
                                                            dtype=self.const.cp.float32)
                        grad_f += self.index_fourier_CP
                        for argorder in range(self.dim):
                            grad_f[:,argorder] *= xdampCP[argorder]
                        diffQ= [False for _ in range(self.dim)]
                        diffQ[i_hess] = True
                        diffQ[j_hess] = True
                        diffQ = self.const.cp.array([diffQ])
                        diffQcp = self.const.cp.repeat(diffQ,self.indexN_fourier,axis=0)
                        self.ker_calctheta_f_diffAll((self.indexN_fourier,),(self.dim,),
                            (self.indexevenQcp, diffQcp, grad_f, self.indexN_fourier*self.dim))
                        grad_f = self.const.cp.cos(grad_f)
                        grad_f  = self.const.cp.prod(grad_f, axis=1)
                    else:
                        grad_f = self.const.cp.array(y_f)
                        grad_f  = self.const.cp.prod(grad_f, axis=1)
                if self.useChebyshevQ:
                    if self.chebyshevQs[i_hess] or self.chebyshevQs[j_hess]:
                        if False:
                            grad_c = []
                        else:
                            grad_c = self.const.cp.array(y_c)
                    else:
                        grad_c = self.const.cp.array(y_c)
                if self.useFourierQ and self.useChebyshevQ:
                    grad_cp = grad_f*grad_c
                elif self.useFourierQ:
                    grad_cp = grad_f
                elif self.useChebyshevQ:
                    grad_cp = grad_c
                grad_cp *= self.alphacp
                if self.FurierQs[i_hess]:
                    grad_cp *= self.index_fourier_CP[:,i_hess]
                if self.FurierQs[j_hess]:
                    grad_cp *= self.index_fourier_CP[:,j_hess]
                grad_cp  = self.const.cp.sum(grad_cp)
                returnhess[i_hess,j_hess] -= grad_cp
                returnhess[j_hess,i_hess] -= grad_cp
        return returnhess
    def f_TD(self, x, returnf):
        if "TD_UNIFORM" in self.TDdic["options"]:
            return returnf
        elif "TD_WELLTEMPERED" in self.TDdic["options"]:
            factor = 1.0 / (1.0 - 1.0 / float(self.TDdic["BIASFACTOR"]))
            return returnf * factor
        else:
            if self.rank == self.root:
                for a in self.TDdic["options"]:
                    if "TD" in a:
                        print("ERROR; Target Distribution name %s is not prepaired"%a)
            exit()
    def grad_TD(self, x, returngrad):
        if "TD_UNIFORM" in self.TDdic["options"]:
            return returngrad
        elif "TD_WELLTEMPERED" in self.TDdic["options"]:
            factor = 1.0 / (1.0 - 1.0 / float(self.TDdic["BIASFACTOR"]))
            return returngrad * factor
        else:
            if self.rank == self.root:
                for a in self.TDdic["options"]:
                    if "TD" in a:
                        print("ERROR; Target Distribution name %s is not prepaired"%a)
            exit()
    def hessian_TD(self, x, returnhess):
        if "TD_UNIFORM" in self.TDdic["options"]:
            return returnhess
        elif "TD_WELLTEMPERED" in self.TDdic["options"]:
            factor = 1.0 / (1.0 - 1.0 / float(self.TDdic["BIASFACTOR"]))
            return returnhess * factor
        else:
            if self.rank == self.root:
                for a in self.TDdic["options"]:
                    if "TD" in a:
                        print("ERROR; Target Distribution name %s is not prepaired"%a)
            exit()
    def replaceX(self, x):
        _x = []
        x_index = 0
        for fixQ, fixX in self.const.fixlist:
            if fixQ:
                _x.append(fixX)
            else:
                _x.append(x[x_index])
                x_index += 1
        return np.array(_x)
def makeChebyshevKernel():
    """
    
    """
    loaded_from_source = r'''
extern "C"{
__global__ void chevF(const int* ind, const float* x1, float* y, \
                              unsigned int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //if (tid < N)
    //{
      if (ind[tid] == 0) {
        y[tid] = 1.0;
      } else if (ind[tid] == 1) {
        y[tid] = x1[tid];
      } else {
        float f0=x1[tid];
        float f1=1.0;
        //float fdamp = 2.0*x1[tid]*f0-f1;
        float fdamp = f0;
        int k;
        //for (k=0;k<ind[tid]; k++) {
        for (k=0;k<0; k++) {
          f1=f0;
          f0=fdamp+0.0;
          fdamp = 2.0*x1[tid]*f0-f1;
        }
        y[tid] = fdamp;
      }
    //}
}
__global__ void chevF_diff(const int* ind, const float* x1, const bool* diffQ, float* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
      if (ind[tid] == 0) {
        if (diffQ[tid]){
          y[tid] = 0.0;
        } else {
          y[tid] = 1.0;
        }
      } else if (ind[tid] == 1) {
        if (diffQ[tid]){
          y[tid] = 1.0;
        } else {
          y[tid] = x1[tid];
        }
      } else {
        float f0=x1[tid];
        float f1=1.0;
        float grad0=1.0;
        float grad1=0.0;
        float fdamp;
        float graddamp;
        int k;
        for (k=0;k<ind[tid]; k++) {
          fdamp = 2.0*x1[tid]*f0-f1;
          f0=f1;
          f1=fdamp;
          if (diffQ[tid]){
            graddamp = 2.0 * f0 + 2.0 * x1[tid] * grad0 - grad1;
            grad1 = grad0;
            grad0 = graddamp;
          } 
        }
        if (diffQ[tid]){
          y[tid] = fdamp;
        } else {
          y[tid] = graddamp;
        }
      }
    }
}
}'''
    return loaded_from_source
def makeChebyshevKernel_sympy(chevindexmax):
    """
    
    """
    import sympy 
    x = sympy.Symbol('x')
    Tlist = [1,x]
    for i in range(chevindexmax):
        T = 2*x*Tlist[-1] -Tlist[-2]
        Tlist.append(sympy.simplify(T))
        #Tlist.append(T)
    Tdiffs = []
    for T in Tlist:
        Tdiff = sympy.diff(T,x)
        Tdiff = sympy.simplify(Tdiff)
        Tdiffs.append(Tdiff)
    ifformat = """ else if (ind1[tid] == %s) {
          y[tid] = %s;
        }"""
    for i,T in enumerate(Tlist):
        T = str(T).replace("x","xxx[tid]")
        T = changestarstar(T)
        if i == 0:
            chevFstr = """        if (ind1[tid] == %s) {
          y[tid] = %s;
        }"""%(i,T)
            continue
        chevFstr += ifformat%(i,T)
    for i,Tdiff in enumerate(Tdiffs):
        Tdiff = str(Tdiff).replace("x","xxx[tid]")
        Tdiff = changestarstar(Tdiff)
        if i == 0:
            chevFdiffstr = """        if (ind1[tid] == %s) {
          y[tid] = %s;
        }"""%(i,Tdiff)
            continue
        chevFdiffstr += ifformat%(i,Tdiff)

    loaded_from_source = r'''
extern "C"{
__global__ void chevF(const int* ind1, const float* xxx, float* y, unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
%s
    }
}
__global__ void chevF_diff(const int* ind1, const float* xxx, const bool* diffQ, float* y, unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
      if (diffQ[tid]) {
%s
      } else {
%s
      }
    }
}
}'''%(chevFstr,chevFdiffstr, chevFstr)
    #print(loaded_from_source)
    #exit()
    return loaded_from_source
def changestarstar(T):
    T = T.split()
    returnT = ""
    for T_part in T:
        if "**" in T_part:
            T_flont,n = T_part.split("**")
            Tlist = T_flont.split("*")
            n = int(n)
            returnT += Tlist[0]
            for Tfrontpart in Tlist[1:-1]:
                returnT += "*" + Tfrontpart
            Tntime = "*%s"%(Tlist[-1])
            returnT += Tntime*n
        else:
            returnT += " %s "%T_part
    return returnT
