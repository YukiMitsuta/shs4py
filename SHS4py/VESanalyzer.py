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
            return 1.0
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
            return 1.0
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
            TD_label =  VESdic["TARGET_DISTRIBUTION"]
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
                    coeff     = {}
                    coeff_VES = [int(line[i]) for i in range(argN)]
                    for i, argorder in enumerate(VESdic["order"]):
                        coeff[argorder] = copy.copy(coeff_VES[i])
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
                VESdic["grad_grid"] = [interpolate.RectBivariateSpline(x, y, z_grad[i]) for i in range(self.dim)]
        if self.rank == self.root:
            print("end VESpotential", flush = True)
            print("Dimension = %s"%self.dim, flush = True)
    def calcgrid(self, VESdic):
        if type(self.const.grid_min) is list:
            argorder = VESdic["order"][0]
            x = np.arange(self.const.grid_min[argorder], self.const.grid_max[argorder],
                    (self.const.grid_max[argorder] - self.const.grid_min[argorder])/self.const.grid_bin[argorder])
            argorder = VESdic["order"][1]
            y = np.arange(self.const.grid_min[argorder], self.const.grid_max[argorder],
                    (self.const.grid_max[argorder] - self.const.grid_min[argorder])/self.const.grid_bin[argorder])
        else:
            x = np.arange(self.const.grid_min, self.const.grid_max,
                (self.const.grid_max - self.const.grid_min)/self.const.grid_bin)
            y = np.arange(self.const.grid_min, self.const.grid_max,
                (self.const.grid_max - self.const.grid_min)/self.const.grid_bin)
        xxyy = np.meshgrid(x, y)
        xxyy = np.meshgrid(x, y)
        z_f = np.zeros((len(x),len(y)))
        z_grad = [np.zeros((len(x),len(y))) for _ in range(self.dim)]
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
                        else:
                            print("ERROR: CVs of VES functions is larger than 2")
                            exit()
                        f_VES *= VESdic["BFfunctions"][argorder].f(xdamp, coeff[argorder])
                    z_f[x_index, y_index] -= f_VES
                    for i_grad in VESdic["order"]:
                        g_VES = float(coeff["alpha"])
                        for i, argorder in enumerate(VESdic["order"]):
                            if i == 0:
                                xdamp = x[x_index]
                            elif i == 1:
                                xdamp = y[y_index]
                            if argorder == i_grad:
                                g_VES *= VESdic["BFfunctions"][argorder].grad(xdamp, coeff[argorder])
                            else:
                                g_VES *= VESdic["BFfunctions"][argorder].f(xdamp, coeff[argorder])
                        z_grad[i_grad][x_index, y_index] -= g_VES
                z_f[x_index, y_index] = self.f_TD(None, z_f[x_index, y_index])
                for i_grad in range(self.dim):
                    z_grad[i_grad][x_index, y_index] = self.f_TD(None, z_grad[i_grad][x_index, y_index])
        if self.const.calc_mpiQ:
            z_f_g = self.comm.gather(z_f, root=0)
            z_grad_g = self.comm.gather(z_grad, root=0)
            if self.rank == self.root:
                z_f = np.zeros((len(x),len(y)))
                for z_fdamp in z_f_g:
                    z_f += z_fdamp
                z_grad = [np.zeros((len(x),len(y))) for _ in range(self.dim)]
                for z_graddamp in z_grad_g:
                    for i_grad in range(self.dim):
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
                #for coeff in VESdic["coefflist"]:
                    #ynum = 0
                    #for i in range(self.dim):
                        #if not i in coeff:
                            #continue
                        #if ynum == 0:
                            #xdamp0 = xdamp[i]
                        #elif ynum == 1:
                            #xdamp1 = xdamp[i]
                    #break
                #xdamp0 = xdamp[VESdic["CVindex"][0]]
                #xdamp1 = xdamp[VESdic["CVindex"][1]]
                xdamp0 = xdamp[VESdic["order"][0]]
                xdamp1 = xdamp[VESdic["order"][1]]
                #returnf += VESdic["f_grid"](xdamp)
                #print(VESdic["f_grid"](xdamp0, xdamp1))
                #print(VESdic["f_grid"].ev(xdamp0, xdamp1))

                returnf += VESdic["f_grid"].ev(xdamp0, xdamp1)
            if self.const.calc_mpiQ:
                returnf_g = self.comm.gather(returnf, root=0)
                if self.rank == self.root:
                    returnf = sum(returnf_g)
                returnf = self.comm.bcast(returnf, root=0)
            return returnf
            #returnf_grid = copy.copy(returnf)
            #returnf = 0.0
            #print("*"*40)
        mpicounter = -1
        for VESdic in self.VESlist:
            #mpicounter += 1
            #if mpicounter % self.size != self.rank:
                #continue
            self.TDdic = VESdic["TDdic"]
            f_partVES = 0.0
            for coeff in VESdic["coefflist"]:
                mpicounter += 1
                if mpicounter % self.size != self.rank:
                    continue
                #y = float(coeff[self.dim])
                #y = coeff[self.dim]
                y = float(coeff["alpha"])
                #for i in range(self.dim):
                    #if coeff[i] == "Nan":
                        #continue
                    #if not i in coeff:
                        #continue
                for argorder in VESdic["order"]:
                    #y *= VESdic["BFfunctions"][i].f(xdamp[i], coeff[i])
                    y *= VESdic["BFfunctions"][argorder].f(xdamp[argorder], coeff[argorder])
                f_partVES -= y
            #print(f_partVES)
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
                #for coeff in VESdic["coefflist"]:
                    #ynum = 0
                    #for i in range(self.dim):
                        #if not i in coeff:
                            #continue
                        #if ynum == 0:
                            #xdamp0 = xdamp[i]
                        #elif ynum == 1:
                            #xdamp1 = xdamp[i]
                    #break
                #xdamp0 = xdamp[VESdic["CVindex"][0]]
                #xdamp1 = xdamp[VESdic["CVindex"][1]]
                xdamp0 = xdamp[VESdic["order"][0]]
                xdamp1 = xdamp[VESdic["order"][1]]

                for i_grad in range(self.dim):
                    #returngrad[i_grad] += VESdic["f_grid"][i_grad](xdamp)
                    returngrad[i_grad] += VESdic["grad_grid"][i_grad].ev(xdamp0,xdamp1)
            if self.const.calc_mpiQ:
                returngrad_g = self.comm.gather(returngrad, root=0)
                if self.rank == self.root:
                    returngrad = np.zeros(self.dim)
                    for returngraddamp in returngrad_g:
                        returngrad += returngraddamp
                returngrad = self.comm.bcast(returngrad, root=0)

            #print("returngrad = %s"%returngrad)
            if self.const.CVfixQ:
                returngraddamp = np.zeros(len(x))
                i_index = 0
                for i, (fixQ, _) in enumerate(self.const.fixlist):
                    if fixQ:
                        continue
                    returngraddamp[i_index] = returngrad[i]
                    i_index += 1
                returngrad = returngraddamp
            #print(x, flush = True)
            #print(xdamp, flush = True)
            #print(returngrad, flush = True)
            return returngrad
            #returngrad_grid = copy.copy(returngrad)
            #returngrad = np.zeros(self.dim)
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
                    for i in range(self.dim):
                        #if coeff[i] == "Nan":
                            #continue
                        if not i in coeff:
                            continue
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
            #if self.rank == self.root:
                #print("returngrad, returngrad_grid = %s, %s"%(returngrad,returngrad_grid),flush=True)
            #exit()
        if self.const.CVfixQ:
            returngarddamp = np.zeros(len(x))
            i_index = 0
            for i, (fixQ, _) in enumerate(self.const.fixlist):
                if fixQ:
                    continue
                returngraddamp[i_index] = returngrad[i]
                i_index += 1
            returngrad = returngraddamp
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
            hess_partVES = np.zeros((self.dim, self.dim))
            for coeffN, coeff in enumerate(VESdic["coefflist"]):
                if coeffN % self.size != self.rank:
                    continue
                for i_hess in range(self.dim):
                    #if coeff[i_hess] == "Nan":
                        #continue
                    if not i_hess in coeff:
                        continue
                    for j_hess in range(i_hess, self.dim):
                        #if coeff[j_hess] == "Nan":
                            #continue
                        if not j_hess in coeff:
                            continue
                        #y = float(coeff[self.dim])
                        y = float(coeff["alpha"])
                        if i_hess == j_hess:
                            for i in range(self.dim):
                                #if coeff[i] == "Nan":
                                    #continue
                                if not i in coeff:
                                    continue
                                if i == i_hess:
                                    y *= VESdic["BFfunctions"][i].gradgrad(xdamp[i], coeff[i])
                                else:
                                    y *= VESdic["BFfunctions"][i].f(xdamp[i], coeff[i])
                            hess_partVES[i_hess, i_hess] -= y
                        else:
                            for i in range(self.dim):
                                #if coeff[i] == "Nan":
                                    #continue
                                if not i in coeff:
                                    continue
                                if i == i_hess or i == j_hess:
                                    y *= VESdic["BFfunctions"][i].grad(xdamp[i], coeff[i])
                                else:
                                    y *= VESdic["BFfunctions"][i].f(xdamp[i], coeff[i])
                            hess_partVES[i_hess, j_hess] -= y
                            hess_partVES[j_hess, i_hess] -= y
            returnhess += self.hessian_TD(xdamp, hess_partVES)
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
