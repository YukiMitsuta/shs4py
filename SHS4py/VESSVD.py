#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2023/02/15 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import numpy as np 
import itertools
#from scipy.linalg import eig

from . import VESanalyzer,mkconst

#import mpi4py

class tensor_class():
    def __init__(self, indices, index, term):
        self.indices = indices
        self.index = index
        self.term = term
    def chkindices(self,idx):
        for i in range(len(self.indices)):
            if self.indices[i]!=idx[i]:
                return False
        return True

class VESSVDclass(VESanalyzer.VESpotential):
    def __init__(self, plumedpath, const, rank, root, size, comm):
        super().__init__(plumedpath,const,rank,root,size,comm)
        self.Udic = {}
        self.sdic = {}
        self.Vdic = {}
    def calcSVD(self):
        for VESdic in self.VESlist:
            coeffsname = VESdic["coeffpath"]
            if self.rank == self.root:
                print("calc SVD of "+coeffsname)
            shape = []
            for line in open(coeffsname):
                if not "#!" in line:
                    continue
                if not "shape" in line:
                    continue
                line = line.split()
                shape.append(int(line[-1]))
            mat = np.zeros(shape)
            for line in open(coeffsname):
                if "#!" in line:
                    continue
                line = line.split()
                if len(line) == 0:
                    continue
                i = int(line[0])
                j = int(line[1])
                mat[i,j] = float(line[2])
            U, s, V = np.linalg.svd(mat)
            VESdic["SVD_U"] = U
            VESdic["SVD_s"] = s
            stot = sum(s)
            VESdic["SVD_stot"] = stot
            VESdic["SVD_V"] = V
            Sent = [x/stot for x in s]
            VESdic["Sent"] = -sum([x*np.log(x) for x in Sent])
            if self.rank == self.root:
                print("Entangled Entropy = %4.3f"%(VESdic["Sent"]))
                print(s)
            #print("0.0 < Sent < %4.3f"%np.log(len(s)))
            sdamp = 0.0
            VESdic["SVD_s_component"] = []
            writeline = ""
            for k in range(len(s)-1):
                sdamp += s[k]
                writeline += "%6.5f, "%(sdamp/stot)
                VESdic["SVD_s_component"].append(sdamp/stot)
            sdamp += s[-1]
            writeline += "%6.5f\n"%(sdamp/stot)
            csvpath = VESdic["coeffpath"]+"_sdiff.csv"
            with open(csvpath, "w") as wf:
                wf.write(writeline)
            dim_tau = len(s)
            dim_rho = len(U[:,0])
            dim_sig = len(V[0])
            Aten = np.zeros((dim_tau,dim_rho,dim_sig))
            for tau in range(dim_tau):
                for rho in range(dim_rho):
                    for sig in range(dim_sig):
                        Aten[tau,rho,sig] = U[rho,tau]*s[tau]*V[tau,sig]
            VESdic["SVD_A"] = Aten
            VESdic["dim_tau"] = dim_tau
            VESdic["dim_rho"] = dim_rho
            VESdic["dim_sig"] = dim_sig
            svdB = []
            print("start calculation of SVD_B")
            if "VES_TENSOR_EXPANSION" in VESdic["options"]:
                dimensions=VESdic["tensorlist"][-1]["tensor_k"][2:]
                dimensions = [x+1 for x in dimensions]
                dimensions = [dim_tau]+dimensions
                VESdic["dimensions"] = dimensions
                for tau in range(dim_tau):
                    svdB_tau = []
                    for tensor in VESdic["tensorlist"]:
                        rho = tensor["rho"]
                        sigma = tensor["sigma"]
                        indices = [tau]+tensor["tensor_k"][2:]
                        index = getVecIndex(indices,dimensions)
                        for B in svdB_tau:
                            #if B.index == index:
                            if B.indices == indices:
                            #if B.chkindices(indices):
                                B.term += Aten[tau,rho,sigma]*tensor["tensor"]
                                break
                        else:
                            svdB_tau.append(tensor_class(indices,index,Aten[tau,rho,sigma]*tensor["tensor"]))
                    svdB.extend(svdB_tau)
                    print(len(svdB))
            else:
                dimensions = [dim_tau,dim_rho,dim_sig]
                VESdic["dimensions"] = dimensions
                for tau in range(dim_tau):
                    for rho in range(dim_rho):
                        for sig in range(dim_sig):
                            indices = [tau,rho,sig]
                            index = getVecIndex(indices,dimensions)
                            svdB.append(tensor_class(indices,index,Aten[tau,rho,sig]))
            VESdic["SVD_B"] = svdB
            print("Length of svdB: %s"%len(svdB))

    def exportFgrid(self):
        grid_min = -np.pi
        grid_max = np.pi
        grid_bin = 100
        x = np.linspace(grid_min, grid_max, grid_bin)
        y = np.linspace(grid_min, grid_max, grid_bin)
        xxyy = np.meshgrid(x, y)
        for VESdic in self.VESlist:
            writeline = ""
            z_f = np.zeros((len(x),len(y)))
            for x_index in range(len(x)):
                for y_index in range(len(y)):
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
                    z_f[x_index, y_index] = self.f_TD(None, z_f[x_index, y_index])
            for x_index in range(len(x)):
                for y_index in range(len(y)):
                    writeline += "% 10.9f, % 10.9f, % 10.7f\n"%(x[x_index],y[y_index],z_f[x_index,y_index])
            csvpath = VESdic["coeffpath"]+"_grid.csv"
            with open(csvpath, "w") as wf:
                wf.write(writeline)
            print("%s is written"%csvpath)
    def exportFgrid_SVD(self,singular_threshold):
        grid_min = -np.pi
        grid_max = np.pi
        grid_bin = 100
        x = np.linspace(grid_min, grid_max, grid_bin)
        y = np.linspace(grid_min, grid_max, grid_bin)
        xxyy = np.meshgrid(x, y)
        for VESdic in self.VESlist:
            if self.rank == self.root:
                writeline = ""
            U = VESdic["SVD_U"]
            s = VESdic["SVD_s"]
            V = VESdic["SVD_V"]
            z_f = np.zeros((len(x),len(y)))
            stot = VESdic["SVD_stot"]
            for x_index in range(len(x)):
                for y_index in range(len(y)):
                    f_VES = 0.0
                    sdamp = 0.0
                    for k in range(len(s)):
                        sdamp += s[k]
                        for mpiN, coeff in enumerate(VESdic["coefflist"]):
                            if mpiN % self.size != self.rank:
                                continue
                            a = 1.0
                            for i, argorder in enumerate(VESdic["order"]):
                                if i == 0:
                                    xdamp = x[x_index]
                                    a *= U[coeff[argorder],k]
                                elif i == 1:
                                    xdamp = y[y_index]
                                    a *= V[k,coeff[argorder]]
                                a *= VESdic["BFfunctions"][argorder].f(xdamp, coeff[argorder])
                            a *= s[k]
                            f_VES += a
                        if singular_threshold < sdamp/stot:
                            if self.rank == self.root:
                                if x_index == 0 and y_index == 0:
                                    print("chi = %s"%k,flush = True)
                            break
                    z_f[x_index, y_index] -= f_VES
                    z_f[x_index, y_index] = self.f_TD(None, z_f[x_index, y_index])
                #if self.rank == self.root:
                    #print(x_index,flush=True)
                    #print("% 10.9f, % 10.9f, % 10.7f\n"%(x[x_index],y[y_index],z_f[x_index,y_index]))
            z_f_g = self.comm.gather(z_f, root=0)
            if self.rank == self.root:
                z_f = np.zeros((len(x),len(y)))
                for z_fdamp in z_f_g:
                    z_f += z_fdamp
                for x_index in range(len(x)):
                    for y_index in range(len(y)):
                        writeline += "% 10.9f, % 10.9f, % 10.7f\n"%(x[x_index],y[y_index],z_f[x_index,y_index])
            if self.rank == self.root:
                csvpath = VESdic["coeffpath"]+"_SVDgrid.csv"
                with open(csvpath, "w") as wf:
                    wf.write(writeline)
                print("%s is written"%csvpath)
#    def exportA(self):
#        for VESdic in self.VESlist:
#            dim_tau = VESdic["dim_tau"]
#            dim_rho = VESdic["dim_rho"]
#            dim_sig = VESdic["dim_sig"]
#            Aten    = VESdic["SVD_A"]
#            writeline = ""
#            #formatline = "%8d%8d%8d% 27.16f%8d\n"
#            formatline = "%8d%8d%8d% 30.16e%8d\n"
#            index = 0
#            for tau in range(dim_tau):
#                for rho in range(dim_rho):
#                    for sig in range(dim_sig):
#                        writeline += formatline%(tau,rho,sig,Aten[tau,rho,sig],index)
#                        index += 1
#
#            exportpath = VESdic["coeffpath"].replace("coeffs.","tensor.",1)
#            with open(exportpath, "w") as wf:
#                wf.write(writeline)
#            print("Export %s"%exportpath)
    def exportMixedTensor(self,mixlabel,singular_threshold):
        mixVESlist = []
        for VESdic in self.VESlist:
            if VESdic["LABEL"] in mixlabel:
                mixVESlist.append(VESdic)
        dim_tens = []
        arglist = []
        dim_args = []
        for VESdic in mixVESlist:
            print(VESdic["SVD_s_component"])
            dim_ten = len([x for x in VESdic["SVD_s_component"] if x < singular_threshold])
            #dim_ten = len([x for x in VESdic["SVD_s_component"]])
            dim_ten += 2
            dim_tens.append(dim_ten)
            args = VESdic["ARG"].split(",")
            if "VES_TENSOR_EXPANSION" in VESdic["options"]:
                arglist.extend(args[2:])
                dim_args.append(len(args)-2)
            else:
                arglist.extend(args)
                dim_args.append(len(args))
        print("dim_tens;",dim_tens)
        print("arglist;",arglist)
        mixVES0 = mixVESlist[0]
        mixVES0dims = mixVES0["dimensions"][1:] 
        dimensions = dim_tens + mixVES0dims
        mixVES1 = mixVESlist[1]
        mixVES1dims = mixVES1["dimensions"][1:] 
        dimensions += mixVES1dims
        print("dimensions;",dimensions)


        writelines = []
        for svdB0 in mixVES0["SVD_B"]:
            idx0 = svdB0.indices
            rho = idx0[0] + 1
            if dim_tens[0] <= rho:
                continue
            sigma = 0
            productlist = [range(x) for x in mixVES1dims]
            for idx1 in itertools.product(*productlist):
                #print("idx1",idx1)
                indices = [rho,sigma]+idx0[1:]+list(idx1)
                if sum(idx1) == 0:
                    term = svdB0.term
                else:
                    term = 0.0
                index = getVecIndex(indices,dimensions)
                writeline = indices + [term,index]
                writelines.append(writeline)
                #exit()
            #print(writelines)
            #exit()
        for svdB1 in mixVES1["SVD_B"]:
            idx1 = svdB1.indices
            rho = 0
            sigma = idx1[0] + 1
            if dim_tens[1] <= sigma:
                continue
            productlist = [range(x) for x in mixVES0dims]
            for idx0 in itertools.product(*productlist):
                indices = [rho,sigma]+list(idx0)+idx1[1:]
                if sum(idx0) == 0:
                    term = svdB1.term
                else:
                    term = 0.0
                index = getVecIndex(indices,dimensions)
                writeline = indices + [term,index]
                writelines.append(writeline)

        for svdB0,svdB1 in itertools.product(mixVES0["SVD_B"],mixVES1["SVD_B"]):
            indices = []
            idx0 = svdB0.indices
            idx1 = svdB1.indices
            rho = idx0[0] + 1
            if dim_tens[0] <= rho:
                continue
            indices.append(rho)
            sigma = idx1[0] + 1
            if dim_tens[1] <= sigma:
                continue
            indices.append(sigma)
            indices.extend(idx0[1:]+idx1[1:])
            term = svdB0.term * svdB1.term
            #print(indices)
            #print(dimensions)
            index = getVecIndex(indices,dimensions)
            writeline = indices + [term,index]
            #print(writeline)
            writelines.append(writeline)
            #if len(writelines) > 10:
                #break
            if rho == 1 and sigma == 1:
                indices = [0,0] + idx0[1:]+idx1[1:]
                index = getVecIndex(indices,dimensions)
                term = 1.0
                writeline = indices + [term,index]
                writelines.append(writeline)

                
        writelines.sort(key=lambda x:x[-1])

        writestr = ""
        for writeline in writelines:
            for x in writeline:
                if isinstance(x,float):
                    writestr += "% 30.16e"%x
                else:
                    writestr += "%8d"%x
            writestr +="\n"

        index_max = max([x[-1] for x in writelines])
        index_max += 1
        
        dim_rho =  dim_tens[0]
        dim_sig =  dim_tens[1]
        writehead = "#! FIELDS idx_rho idx_sigma "
        for argname in arglist:
            writehead += "idx_%s "%argname
        writehead += "b0.tensor index\n"
        writehead += "#! SET time 0.0\n"
        writehead += "#! SET iteration 0\n"
        writehead += "#! SET type LinearBasisSet\n"
        writehead += "#! SET ndimensions %s\n"%len(dimensions)
        writehead += "#! SET ntensor_total %s\n"%index_max
        writehead += "#! SET shape_rho   %s\n"%dim_rho
        writehead += "#! SET shape_sigma %s\n"%dim_sig
        for i,argname in enumerate(arglist):
            writehead += "#! SET shape_%s %s\n"%(argname,dimensions[i+2])
        exportpath = mixVES0["coeffpath"].replace("coeffs.","tensor.",1)
        with open(exportpath, "w") as wf:
            wf.write(writehead)
            wf.write(writestr)


        print("Export %s"%exportpath)
        productlist =[]
        productlist.append(range(dim_rho))
        productlist.append(range(dim_sig))
        exportpath = mixVES0["coeffpath"].replace("coeffs.","coeffs_initial.",1)
        index = 0
        formatline = "%8d%8d% 30.16e% 30.16e%8d\n"
        writeline = ""
        productlist.reverse()
        for sig,rho in itertools.product(*productlist):
            if rho == 0 and sig == 0:
                a = 0.0
            elif rho == 0:
                a = 1.0
            elif sig == 0:
                a = 1.0
            else:
                a = 0.0
            writeline += formatline%(rho,sig,a,a,index)
            index += 1
        writehead  = "#! FIELDS idx_rho idx_sigma b0.coeffs b0.aux_coeffs index\n"
        writehead += "#! SET time 0.0\n"
        writehead += "#! SET iteration 0\n"
        writehead += "#! SET type LinearBasisSet\n"
        writehead += "#! SET ndimensions 2\n"
        writehead += "#! SET ncoeffs_total %s\n"%index
        writehead += "#! SET shape_rho   %s\n"%dim_rho
        writehead += "#! SET shape_sigma %s\n"%dim_sig
        with open(exportpath, "w") as wf:
            wf.write(writehead)
            wf.write(writeline)
        print("Export %s"%exportpath)


class constClass():
    pass
def main():
    """
    
    """
    plumedpath = "./plumedVES.dat"
    constC = constClass()
    constC = mkconst.main(constC)
    constC.calc_mpiQ = True
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    coeffsnames = glob.glob("coeffs/coeffs.c-*.data")
    VESclass = VESSVDclass(plumedpath,constC,rank,root,size,comm)
    for coeffsname in coeffsnames:
        print(coeffsname)

    exportnormalgrid(VESclass,rank,root)
    VESclass.calcSVD()
    exit()
    mixlabel = ["b0","b1"]
    singular_threshold = 1.00
    VESclass.exportMixedTensor(mixlabel,singular_threshold)
    for singular_threshold in [1.00,0.99,0.95,0.90,0.85,0.80,0.75,0.50]:
    #for singular_threshold in [0.70,0.65,0.60,0.55]:
        VESclass.exportFgrid_SVD(singular_threshold)
        if rank == root:
            print(singular_threshold)
            os.chdir("coeffs")
            dirname = "SVD%3.2f"%singular_threshold
            os.mkdir(dirname)
            print(dirname)
            for csvname in glob.glob("*.csv"):
                shutil.move(csvname,dirname)
            os.chdir("../")
def exportnormalgrid(VESclass,rank,root):
    VESclass.exportFgrid()
    if rank == root:
        os.chdir("coeffs")
        dirname = "normal"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        print(dirname)
        for csvname in glob.glob("*.csv"):
            shutil.move(csvname,dirname)
        os.chdir("../")
def getVecIndex(indices,dimensions):
    ndimensions=len(dimensions)
    index = indices[ndimensions-1]
    #print(ndimensions-1,index)
    for i in range(ndimensions-1,0,-1):
        index = index*dimensions[i-1]+indices[i-1]
        #print(i-1,dimensions[i-1],indices[i-1],index)
    #print(indices)
    #print(dimensions)
    #print(index)
    return index

if __name__ == "__main__":
    main()


