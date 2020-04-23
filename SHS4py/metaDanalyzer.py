#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

"""
A module to analyze results of metadynamics

Available class:
    GaussianC    : This class is performed as gaussian function.
    Metad_result : This class is the result of metad calculation
Available functions:
    main : main part of the analysis of metadynamics
"""
import numpy as np
import copy, os


class GaussianC(object):
    """
    This class is performed as gaussian function.

    Available functions:
        periodic_dist : the distance between a and b with periodic
        calcdist      : the distance between x and self.s
        f             : the potential of A Gauss function on x
        grad          : the gradient of A Gauss function on x
        hessian       : the hessian of A Gauss function on x
    """
    def __init__(self, const, dim = None, s = None, sigmainv = None, h = None):
        self.dim         = dim
        self.s           = s
        self.sigmainv    = sigmainv
        self.h           = h
        self.const       = const
        self.periodicQ   = self.const.periodicQ
        self.periodicmax = self.const.periodicmax
        self.periodicmin = self.const.periodicmin
        if self.const.calc_mpiQ:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.root = 0
        else:
            self.rank = 0
            self.root = 0
            self.size = 1
    def periodic_dist(self, a, b):
        if self.periodicQ:
            if b < self.periodicmin + a or self.periodicmax + a < b:
                bdamp  = (b - self.periodicmax - a) % (self.periodicmin - self.periodicmax)
                bdamp += self.periodicmax + a
            else:
                bdamp = b
        else:
            bdamp = b
        return a - bdamp
    def calcdist(self, x):
        d = []
        dis = 0.0
        for i in range(self.dim):
            if self.const.periodicQ:
                dpoint = self.periodic_dist(x[i], self.s[i]) * self.sigmainv[i]
            else:
                dpoint = (x[i] - self.s[i] ) * self.sigmainv[i]
            dis   += dpoint * dpoint
            d.append(dpoint)
        return True, d, dis
    def f(self, x):
        if self.const.cythonQ:
            #returnf = self.const.calcgau.f_periodic(
            returnf = self.const.calcgau.f(
                    x, self.s, self.sigmainv, self.periodicmax, self.periodicmin, 
                    self.dim, self.h, self.const.periodicQ)
            return returnf
        else:
            getdistQ, d, tot = self.calcdist(x)
            if getdistQ is False:
                return 0.0
            returnf = - self.h * np.exp(- 0.5 * tot)
            return returnf 
    def grad(self, x):
        if self.const.cythonQ:
            returngrad = self.const.calcgau.grad(x, self.s, self.sigmainv, self.periodicmax, self.periodicmin, self.dim, self.h)
            return returngrad 
        else:
            returngrad = np.zeros(self.dim)
            getdistQ, d, tot = self.calcdist(x)
            if getdistQ is False:
                return  returngrad
            _f = - self.h * np.exp(- 0.5 * tot)
            for i in range(self.dim):
                returngrad[i] = - d[i] * _f * self.sigmainv[i]
            return returngrad
    def hessian(self, x):
        if self.const.cythonQ:
            returnhess = self.const.calcgau.hessian(x, self.s, self.sigmainv, self.periodicmax, self.periodicmin, self.dim, self.h)
            if self.const.WellTempairedQ:
                returnhess = returnhess * self.const.WT_Biasfactor_ffactor
            return returnhess
        else:
            returnhess = np.zeros((self.dim, self.dim))
            getdistQ, d, tot = self.calcdist(x)
            if getdistQ is False:
                return returnhess
            _f =  - self.h * np.exp(- 0.5 * tot)
            for i in range(self.dim):
                for j in range(i, self.dim):
                    if  i == j:
                        returnhess[i,i] = _f * (d[i] * d[i] - 1.0) * self.sigmainv[i] * self.sigmainv[i]
                    else:
                        returnhess[i,j] = _f * d[i] * d[j] * self.sigmainv[i] * self.sigmainv[j]
                        returnhess[j,i] = copy.copy(returnhess[i,j])
            if self.const.WellTempairedQ:
                returnhess = returnhess * self.const.WT_Biasfactor_ffactor
            return returnhess
### end class GaussianC ###
class Metad_result(object):
    """
    This class is the result of metad calculation

    hillpath : the path of an output of metadynamics (by using Plumed)

    Available functions:
        f       : the free energy of metadynamics on x. (f(x) = -W_F(x)) 
        gard    : the gradient of f on x.
        hessian : the hessian of f on x.
    """
    def __init__(self, hillpath, const, maxtime = 1.0e30, t_delta = 0.0):
        self.const       = const
        if self.const.calc_mpiQ:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.root = 0
        else:
            self.rank = 0
            self.root = 0
            self.size = 1
        self.hillCs = []
        if self.const.PBmetaDQ:
            #hillpaths = glob.glob(hillpath)
            self.hillCslist = []
            for path in hillpath:
                if self.rank == self.root:
                    print("import %s"%path)
                hillcounter = 0
                hillCs = []
                for line in open(path):
                    line = line.split()
                    if line[0] == "#!":
                        if "FIELDS" in line:
                            dim = int((len(line) - 5) / 2)
                            if self.rank == self.root:
                                print("dimension = %s"%dim)
                        continue
                    if hillcounter % self.size == self.rank:
                        t = float(line[0])
                        if maxtime + 1 < t:
                            break
                    hillC          = GaussianC(const)
                    hillC.dim      = dim
                    hillC.s        = np.array(line[1:dim + 1], dtype = float)
                    _sigma         = np.array(line[dim + 1: dim * 2 + 1], dtype = float)
                    hillC.sigmainv = np.array([1.0 / si for si in _sigma])
                    hillC.h        = float(line[-2])
                    hillCs.append(hillC)
                    hillcounter += 1
                self.hillCslist.append(hillCs)
            self.dim = len(self.hillCslist)
        else:
            if self.rank == self.root:
                print("import %s"%hillpath)
            hillcounter = 0
            linecounter = 0
            for line in open(hillpath):
                line = line.split()
                if line[0] == "#!":
                    if "FIELDS" in line:
                        self.dim = int((len(line) - 5) / 2)
                        if self.rank == self.root:
                            print("dimension = %s"%self.dim)
                    continue
                linecounter += 1
                if maxtime < linecounter * t_delta:
                    break
                if hillcounter % self.size == self.rank:
                    #t = float(line[0])
                    #if maxtime + 1 < t:
                        #break
                    hillC          = GaussianC(const)
                    hillC.dim      = self.dim
                    hillC.s        = np.array(line[1:self.dim + 1], dtype = float)
                    _sigma         = np.array(line[self.dim + 1: self.dim * 2 + 1], dtype = float)
                    hillC.sigmainv = np.array([1.0 / si for si in _sigma])
                    hillC.h        = float(line[-2])
                    self.hillCs.append(hillC)
                #hillcounter += 1
                if self.const.calc_cupyQ:
                    if self.const.parallelMetaDQ:
                        hillcounter += 1
                    else:
                        if self.rank != self.root:
                            hillcounter += 1
                else:
                    hillcounter += 1

        if self.rank == self.root:
            print("calculation time   : % 10.1f ps"%float(line[0]))
            #if maxtime < t:
            if maxtime < linecounter * t_delta:
                print("collect data until Total of MetaD time = % 10.1f ps"%maxtime)
            print("The numbe of hills : %s"%(len(self.hillCs) * self.size))
        if self.const.gridQ:
            if self.rank == self.root:
                print("Try grid calculation")
            self.import_grid()
        if self.const.calc_cupyQ:
            if self.const.parallelMetaDQ:
                allocateCupyQ = True
            else:
                if self.rank == self.root:
                    allocateCupyQ = True
                else:
                    allocateCupyQ = False
            if allocateCupyQ:
                self.h_cupy          = self.const.cp.array([hillC.h for hillC in self.hillCs])
                self.slistall        = self.const.cp.array(np.array([hillC.s for hillC in self.hillCs]).transpose())
                self.sigmainvlistall = self.const.cp.array(np.array([hillC.sigmainv for hillC in self.hillCs]).transpose())
                @self.const.cp.fuse(kernel_name = "f_kernel")
                def f_kernel(dis_sq, h):
                    return - h * self.const.cp.exp(- 0.5 * dis_sq)
                if self.const.periodicQ:
                    @self.const.cp.fuse(kernel_name = "dist_forf_kernel0")
                    def dist_forf_kernel0(x_i, s_i, sigmainv_i, periodicmin, periodicmax):
                        sdamp  = (s_i - periodicmax - x_i) % (periodicmin - periodicmax)
                        sdamp += periodicmax + x_i
                        dpoint = (x_i - sdamp) * sigmainv_i
                        return dpoint * dpoint
                    @self.const.cp.fuse(kernel_name = "dist_forf_kernel")
                    def dist_forf_kernel(x_i, s_i, sigmainv_i, dis_before, periodicmin, periodicmax):
                        sdamp  = (s_i - periodicmax - x_i) % (periodicmin - periodicmax)
                        sdamp += periodicmax + x_i
                        dpoint = (x_i - sdamp) * sigmainv_i
                        return dis_before + dpoint * dpoint
                    @self.const.cp.fuse(kernel_name = "dist_forgrad_kernel0")
                    def dist_forgrad_kernel0(x_i, s_i, sigmainv_i, periodicmin, periodicmax):
                        sdamp  = (s_i - periodicmax - x_i) % (periodicmin - periodicmax)
                        sdamp += periodicmax + x_i
                        dpoint = (x_i - sdamp) * sigmainv_i
                        return dpoint * sigmainv_i, dpoint * dpoint
                    @self.const.cp.fuse(kernel_name = "dist_forgrad_kernel")
                    def dist_forgrad_kernel(x_i, s_i, sigmainv_i, dis_before, periodicmin, periodicmax):
                        sdamp  = (s_i - periodicmax - x_i) % (periodicmin - periodicmax)
                        sdamp += periodicmax + x_i
                        dpoint = (x_i - sdamp) * sigmainv_i
                        return dpoint * sigmainv_i, dis_before + dpoint * dpoint
                else:
                    @self.const.cp.fuse(kernel_name = "dist_forf_kernel0")
                    def dist_forf_kernel0(x_i, s_i, sigmainv_i):
                        dpoint = (x_i - s_i) * sigmainv_i
                        return dpoint * dpoint
                    @self.const.cp.fuse(kernel_name = "dist_forf_kernel")
                    def dist_forf_kernel(x_i, s_i, sigmainv_i, dis_before):
                        dpoint = (x_i - s_i) * sigmainv_i
                        return dis_before + dpoint * dpoint
                    @self.const.cp.fuse(kernel_name = "dist_forgrad_kernel0")
                    def dist_forgrad_kernel0(x_i, s_i, sigmainv_i):
                        dpoint = (x_i - s_i) * sigmainv_i
                        return dpoint * sigmainv_i, dpoint * dpoint
                    @self.const.cp.fuse(kernel_name = "dist_forgrad_kernel")
                    def dist_forgrad_kernel(x_i, s_i, sigmainv_i, dis_before):
                        dpoint = (x_i - s_i) * sigmainv_i
                        return dpoint * sigmainv_i, dis_before + dpoint * dpoint
                self.f_kernel             = f_kernel
                self.dist_forf_kernel     = dist_forf_kernel
                self.dist_forf_kernel0    = dist_forf_kernel0
                self.dist_forgrad_kernel0 = dist_forgrad_kernel0
                self.dist_forgrad_kernel  = dist_forgrad_kernel
    def f(self, x, ADDQ = False):
        if self.const.gridQ:
            returnf = self.f_grid(x)
        elif self.const.calc_cupyQ:
            returnf = self.f_cupy(x, ADDQ)
        elif self.const.calc_mpiQ:
            x = self.comm.bcast(x, root=0)
            if self.const.PBmetaDQ:
                V_klist = []
                for i, hillCs in enumerate(self.hillCslist):
                    V_k = 0.0
                    for hillC in self.hillCs:
                        V_k += hillC.f(x[i])
                    V_klist = [V_kdamp]
                    V_klistg = self.comm.gather(V_klist, root=0)
                    if self.rank == self.root:
                        V_k_damp = 0.0
                        for V_hill in V_klistg:
                            V_k_damp += V_hill[0]
                        V_k = float(V_k_damp)
                    else:
                        V_k = None
                    V_k = self.comm.bcast(V_k, root=0)
                    V_klist.append(V_k)
                if self.rank == self.root:
                    returnf = 0.0
                    for V_k in V_klist:
                        returnf += np.exp(- self.const.beta * V_k)
                    returnf = self.const.betainv * np.log(returnf)
                else:
                    returnf = None
                returnf = self.comm.bcast(returnf, root=0)
            else:
                returnfdamp = 0.0
                for hillC in self.hillCs:
                    returnfdamp += hillC.f(x)
                returnflist = [returnfdamp]
                returnflistg = self.comm.gather(returnflist, root=0)
                if self.rank == self.root:
                    returnf_damp = 0.0
                    for f_hill in returnflistg:
                        returnf_damp += f_hill[0]
                    returnf = float(returnf_damp)
                else:
                    returnf = None
                returnf = self.comm.bcast(returnf, root=0)
        else:
            returnf = 0.0
            for hillC in self.hillCs:
                returnf += hillC.f(x)
        if self.const.WellTempairedQ:
            returnf = returnf * self.const.WT_Biasfactor_ffactor 
        return returnf
    def f_grid(self, x):
        gridNlist = []
        for x_i in x:
            gridNlist.append( (x_i - self.const.grid_min) // self.xdelta )
        x_min = np.array([self.const.grid_min + self.xdelta * gridN for gridN in gridNlist])
        gridcounter = 0
        f_xminlist = [False, False]
        for x_before, f_before in self.f_gridlist:
            if gridcounter % self.size == self.rank:
                if np.allclose(x_before, x_min):
                    f_xminlist = [self.rank, f_before]
            gridcounter += 1
        f_xminlistg = self.comm.gather(f_xminlist, root=0)
        if self.rank == self.root:
            f_xmin = []
            for xrank, f_before in f_xminlistg:
                if not xrank is False:
                    f_xmin = [f_before]
                    break
        else:
            f_xmin = None
        f_xmin = self.comm.bcast(f_xmin, root=0)
        if len(f_xmin) == 0:
            if self.const.calc_mpiQ:
                gridNlist = self.comm.bcast(gridNlist, root=0)
                f_xmindamp = 0.0
                for hillC in self.hillCs:
                    f_xmindamp += hillC.f(x_min)
                #if not isinstance(f_xmindamp, float):
                    #print("%s: %s"%(self.rank, f_xmindamp))
                f_xminlistfcacd  = [self.rank, f_xmindamp]
                f_xminlistg = None
                f_xminlistg = self.comm.gather(f_xminlistfcacd, root=0)
                if self.rank == self.root:
                    f_xminroot = 0.0
                    for calcrank, f_hill in f_xminlistg:
                        if not isinstance(f_hill, float):
                            print("=== f_xminlistg ===")
                            print(f_xminlistg)
                            print("x_min = %s"%x_min)
                            print("=== f_xminlistg ===")
                        f_xminroot += f_hill
                    print("f_xminroot = %s"%f_xminroot)
                    #f_xmin = float(f_xminroot)
                    f_xmin = f_xminroot
                else:
                    f_xmin = None
                f_xmin = self.comm.bcast(f_xmin, root=0)
                if not isinstance(f_xmin, float):
                    #print(f_xmin)
                    print("%s: %s"%(self.rank, f_xmindamp), flush = True)
                    exit()
            else:
                f_xmin = 0.0
                for hillC in self.hillCs:
                    f_xmin += hillC.f(x_min)
            if self.rank == self.root:
                writeline = ""
                for pointx in x_min:
                    writeline += "% 7.5f,"%pointx
                writeline += "%s\n"%f_xmin
                with open("%s/jobfiles_meta/f_grid.csv"%self.const.pwdpath, "a") as wf:
                    wf.write(writeline)
            self.f_gridlist.append([x_min, f_xmin])
        else:
            f_xmin = f_xmin[0]
        #print("x_min  = %s"%x_min)
        #print("f_xmin = %s"%f_xmin)
        returnf = copy.copy(f_xmin)
        gridlist_inter = []
        flist_inter    = []
        for i in range(len(x)):
            if self.rank == self.root:
                xdelta_array     = np.zeros(len(x))
                xdelta_array[i] += self.xdelta
                x_max = x_min + xdelta_array
            else:
                x_max  = None
            x_max = self.comm.bcast(x_max, root=0)
            gridcounter = 0
            f_xmaxlist = [False, False]
            for x_before, f_before in self.f_gridlist:
                if gridcounter % self.size == self.rank:
                    if np.allclose(x_before, x_max):
                        f_xmaxlist = [self.rank, f_before]
                gridcounter += 1
            f_xmaxlistg = self.comm.gather(f_xmaxlist, root=0)
            if self.rank == self.root:
                f_xmax = []
                for xrank, f_before in f_xmaxlistg:
                    if not xrank is False:
                        f_xmax = [f_before]
                        break
            else:
                f_xmax = None
            f_xmax = self.comm.bcast(f_xmax, root=0)

            if len(f_xmax) == 0:
                if self.const.calc_mpiQ:
                    x_max = self.comm.bcast(x_max, root=0)
                    #gridNlist = self.comm.bcast(gridNlist, root=0)
                    f_xmaxdamp = 0.0
                    for hillC in self.hillCs:
                         f_xmaxdamp += hillC.f(x_max)
                    if not isinstance(f_xmaxdamp, float):
                    #if True:
                        print("in %s"%self.rank)
                        print("ERROR: f_xmaxdamp is not float: %s"%f_xmaxdamp)
                    f_xmaxlistaca  = [self.rank, f_xmaxdamp]
                    f_xmaxlistg    = None
                    f_xmaxlistg    = self.comm.gather(f_xmaxlistaca, root=0)
                    if self.rank == self.root:
                        f_xmaxroot = 0.0
                        for calcrank, f_hill in f_xmaxlistg:
                            if not isinstance(f_hill, float):
                                print("=== f_xmaxlistg ===")
                                print(f_xmaxlistg)
                                print("x_max = %s"%x_max)
                                print("=== f_xmaxlistg ===")
                            f_xmaxroot += f_hill
                        print("f_xmaxroot = %s"%f_xmaxroot)
                        f_xmax = float(f_xmaxroot)
                    else:
                        f_xmax = None
                    f_xmax = self.comm.bcast(f_xmax, root=0)
                else:
                    f_xmax = 0.0
                    for hillC in self.hillCs:
                        f_xmax += hillC.f(x_max)
                if self.rank == self.root:
                    writeline = ""
                    for pointx in x_max:
                        writeline += "% 7.5f,"%pointx
                    writeline += "%s\n"%f_xmax
                    with open("%s/jobfiles_meta/f_grid.csv"%self.const.pwdpath, "a") as wf:
                        wf.write(writeline)
                self.f_gridlist.append([x_max, f_xmax])
            else:
                f_xmax = f_xmax[0]
            returnf += (x_min[i] - x[i]) / self.const.grid_bin * (f_xmin - f_xmax)
        return returnf
    def f_cupy(self, x, ADDQ):
        if self.rank == self.root:
            if ADDQ:
            #if False:
                _hlist = self.h_part
                _slist = self.slist_part
                _sigmainvlist = self.sigmainvlist_part
            else:
                _hlist = self.h_cupy
                _slist = self.slistall
                _sigmainvlist = self.sigmainvlistall
            calcdpointQ = True
            if self.const.periodicQ:
                dis_sqlist = self.dist_forf_kernel0(x[0], _slist[0], _sigmainvlist[0], 
                                self.const.periodicmin, self.const.periodicmax)
            else:
                dis_sqlist = self.dist_forf_kernel0(x[0], _slist[0], _sigmainvlist[0])
            for i in range(1, len(x)):
                if self.const.periodicQ:
                    dis_sqlist = self.dist_forf_kernel(x[i], _slist[i], _sigmainvlist[i], 
                                dis_sqlist, self.const.periodicmin, self.const.periodicmax)
                else:
                    dis_sqlist = self.dist_forf_kernel(x[i], _slist[i], _sigmainvlist[i], dis_sqlist)
            returnf = self.const.cp.sum(self.f_kernel(dis_sqlist, _hlist))
            #returnfdamp = self.const.cp.sum(self.f_kernel(dis_sqlist, _hlist))
            #returnf = float(returnfdamp)
        else:
            returnf = None
        if self.const.calc_mpiQ:
            returnf = self.comm.bcast(returnf, root=0)
        return returnf
    def cupy_float_test(self, returnfdamp):
        return float(returnfdamp)
    def f_convergence(self, x):
        returnf = 0.0
        writeline = ""
        #writeline += "# tergetpoint = %s\n"%x
        for i, hillC in enumerate(self.hillCs):
            returnf += hillC.f(x)
            if i % 1000 == 0:
                f_conv = returnf * self.const.WT_Biasfactor_ffactor 
                print(f_conv)
                writeline += "%s, %s\n"%(i, f_conv)
        writeline.rstrip("\n")
        with open("./convtest.csv", "w") as wf:
            wf.write(writeline) 
    def fError_corrylation(self):
        """
        J Phys Chem B. 2015 Jan 22;119(3):736-42. doi: 10.1021/jp504920s.
        the second term of Eq (12)
        """
        from scipy.integrate import nquad
        import random
        import itertools
        #sigmadelta = 5.0 / self.hillCs[0].sigmainv
        sigmadelta = 10.0 / self.hillCs[0].sigmainv
        #sigmadelta = 20.0 / self.hillCs[0].sigmainv
        #sigmadelta = 0.0 / self.hillCs[0].sigmainv
        if self.rank == self.root:
            lim = []
            for j in range(self.dim):
                #lim.append([min(hillC.s[j] for hillC in self.hillCs[:i]) - sigmadelta[j],
                        #max(hillC.s[j] for hillC in self.hillCs[:i]) + sigmadelta[j]])
                lim.append([min(hillC.s[j] for hillC in self.hillCs) - sigmadelta[j],
                        max(hillC.s[j] for hillC in self.hillCs) + sigmadelta[j]])
        else:
            lim = None
        if self.const.calc_mpiQ:
            lim    = self.comm.bcast(lim,    root=0)

        fsum = 0.0
        term = 1.0
        hillN = len(self.hillCs)

        #meshpointMin = np.array([x[0] for x  in lim]) + sigmadelta * 0.5
        meshpointMin = np.array([x[0] for x  in lim])
        meshpoints = []
        meshpointsdamp = []
        lenbefore = len(meshpoints)
        for hillC in self.hillCs:
            meshnest = []
            for j in range(self.dim):
                i_mesh = 0
                while True:
                    p_j = meshpointMin[j] + (i_mesh + 1) * sigmadelta[j]
                    if hillC.s[j] < p_j:
                        break
                    i_mesh += 1
                meshnest.append(i_mesh)
            meshpointsdamp.append(tuple(meshnest))
            #if meshnest in meshpoints:
                #continue
        meshpointsdamp = list(set(meshpointsdamp))
        #print('len(meshpointsdamp) = %s'%len(meshpointsdamp))
        meshpoints = []
        for meshnest in meshpointsdamp:
            for iterpoints in itertools.product([-1,0,1], repeat=self.dim):
            #for iterpoints in itertools.product([0], repeat=self.dim):
                #if 3 < sum([abs(x) for x in iterpoints]):
                    #continue
                meshpoints.append(tuple([meshnest[j] + iterpoints[j] for j in range(self.dim)]))
            meshpoints = list(set(meshpoints))
            #if lenbefore != len(meshpoints):
                #print('len(meshpoints) = %s'%len(meshpoints))
            lenbefore = len(meshpoints)
        meshpoints = set(meshpoints)
        meshpoints = [[meshnest, 0.0] for meshnest in meshpoints]
        print(' # len(meshpoints) = %s'%len(meshpoints))
        term = 1.0
        for j in range(self.dim):
            term *= sigmadelta[j]
        fsum  = 0.0
        for meshN, meshpoint in enumerate(meshpoints):
            p_nest, fsum_mesh = meshpoint
            p = [meshpointMin[j] + sigmadelta[j] * p_nest[j] for j in range(self.dim)]
            covlist = []
            for i in range(5000000):
                #print(i + 1 % 10)
                point = []
                for j in range(len(self.hillCs[0].s)):
                    point.append(p[j] + sigmadelta[j] * random.random())
                meshpoints[meshN][1] += float(self.fError(point))
                if (i + 1)  % 10 == 0:
                    covlist.append(meshpoints[meshN][1] / (i + 1))
                    #print(covlist)
                if (i + 1)  == 100:
                    if covlist[-1] < 1.0e-5 / len(meshpoints):
                        break
                if (i + 1)  % 1000 == 0:
                    if covlist[-1] < 1.0e-5 / len(meshpoints):
                        break
                    else:
                        covlist = np.array(covlist)
                        #print(covlist)
                        print("%s, %s, %s"%(meshN, i, np.var(covlist)))
                        if np.var(covlist) < 1.0e-5 / len(meshpoints):
                            break
                        covlist = []
            meshpoints[meshN][1] /= (i + 1)
            fsum = sum(fsum_mesh for _, fsum_mesh in meshpoints)
            if meshN % 1000 == 0:
                with open("./MCconv.csv", "a") as wf:
                    #wf.write('%s, %s, %s\n'%(hillN, (i+1) * len(meshpoints), fsum * term))
                    wf.write('%s, %s, %s\n'%(hillN, meshN, fsum * term))
        #if True:
            #for meshN, meshpoint in enumerate(meshpoints):
                #lim = [[meshpointMin[j] + sigmadelta[j] * p_nest[j],
                        #meshpointMin[j] + sigmadelta[j] * (p_nest[j] + 1)]
                        #for j in range(self.dim)]
                ##fsum += nquad(self.fError, lim, opts=Options)
                #fsum += nquad(self.fError, lim)[0]
                #print("fsum(%s) = %s)"%(meshN, fsum))

#            for meshN, meshpoint in enumerate(meshpoints):
#                p_nest, fsum_mesh = meshpoint
#                p = [meshpointMin[j] + sigmadelta[j] * p_nest[j] for j in range(self.dim)]
#                point = []
#                for j in range(len(self.hillCs[0].s)):
#                    point.append(p[j] + sigmadelta[j] * random.random())
#                #meshpoints[meshN][1] += self.fError(point)
#                fsum += self.fError(point)
            #if i + 1  % 10 == 0:
            #if True:
                #fsum = sum(fsum_mesh for _, fsum_mesh in meshpoints)
                #with open("./MCconv.csv", "a") as wf:
                    #wf.write('%s, %s, %s\n'%(hillN, (i+1) * len(meshpoints), fsum * term))
                    #wf.write('%s, %s, %s\n'%(hillN, (i+1) * len(meshpoints), fsum * term / (i + 1)))
            #if (i+1) * len(meshpoints) > 10000000:
                #break
        #fsum = sum(fsum_mesh for _, fsum_mesh in meshpoints)
        term = np.log(fsum * term ) * self.const.betainv
        return term

#        for j in range(len(self.hillCs[0].s)):
#            term *= lim[j][1] - lim[j][0]
#        #for i in range(10000000):
#        for i in range(5000000):
#            point = []
#            for j in range(len(self.hillCs[0].s)):
#                point.append(random.uniform(lim[j][0],lim[j][1]))
#            fsum += self.fError(point)
#            if i % 10000 == 0:
#                with open("./MCconv.csv", "a") as wf:
#                    wf.write('%s, %s, %s\n'%(hillN, i, fsum * term / (i + 1)))
#        term = np.log(fsum * term / (i + 1)) * self.const.betainv
        #term = np.log(fsum * term / (i + 1)) + self.const.WT_Biasfactor_ffactor * self.const.beta

        #Options = []
        #for _ in range(len(self.hillCs[0].s)):
            #Options.append({'epsabs': 1.0e-1, 'epsrel': 1.0e-1})
        #term = nquad(self.fError, lim, opts=Options)
        #term = np.log(term[0]) + self.const.WT_Biasfactor_ffactor * self.const.beta
#        return term


#        for i in range(10000, len(self.hillCs), 10000):
#            if self.rank == self.root:
#                lim = []
#                for j in range(len(self.hillCs[0].s)):
#                    lim.append([min(hillC.s[j] for hillC in self.hillCs[:i]) - sigmadelta[j],
#                            max(hillC.s[j] for hillC in self.hillCs[:i]) + sigmadelta[j]])
#            else:
#                lim = None
#            if self.const.calc_mpiQ:
#                lim    = self.comm.bcast(lim,    root=0)
#            if self.rank == self.root:
#                self.i_untill = i
#            else:
#                self.i_untill = i % self.size
#            Options = []
#            for _ in range(len(self.hillCs[0].s)):
#                Options.append({'epsabs': 1.0e-3, 'epsrel': 1.0e-3})
#            term = nquad(self.fError, lim, opts=Options)
#            #term = np.log(term) 
#            term = np.log(term[0]) + self.const.WT_Biasfactor_ffactor * self.const.beta
#            if self.rank == self.root:
#                print('%s ; %s'%(i, term))
#            with open("./fError.csv", "w") as wf:
#                wf.write('%s, %s\n'%(i, term))
    def fError(self, xdamp):
        x = np.array(xdamp)
        #returnf = - self.f(x) * self.const.WT_Biasfactor_ffactor * self.const.beta
        returnf = - self.f(x) * self.const.beta
        #returnf = self.f(x) * self.const.beta
        #print(returnf)
        returnf = np.exp(returnf) - 1.0
        #returnf = np.exp(returnf)
        #returnf =  np.exp(- self.f(x)) - 1.0
        #if 0.0001 < returnf:
           #print(returnf, flush = True)
        return returnf
#        returnf = 0.0
#        for i, hillC in enumerate(self.hillCs):
#            if self.rank == self.root:
#                if i % self.size != self.rank:
#                    continue
#            returnf += hillC.f(x)
#            if i == self.i_untill:
#                if self.const.calc_mpiQ:
#                    returnfg = self.comm.gather(returnf, root=0)
#                    if self.rank == self.root:
#                        returnf  = 0
#                        for returnfdamp in returnfg:
#                            returnf += returnfdamp
#                        f_conv = np.exp(- returnf) - 1.0
#                        if 0.00001 < f_conv:
#                            print(f_conv, flush = True)
#                    else:
#                        f_conv = None
#                    f_conv = self.comm.bcast(f_conv, root=0)
#                else:
#                    f_conv = np.exp(- returnf)
#                #f_conv = - returnf * self.const.WT_Biasfactor_ffactor * self.const.beta
#                #print(f_conv)
#                #f_conv = np.exp(f_conv)
#                #print('%s, %s'%(x, f_conv))
#                #f_conv = np.exp(- returnf)
#                return f_conv
    def grad(self, x, debagQ = False, ADDQ = False):
        if self.const.calc_cupyQ:
            returngrad = self.grad_cupy(x, ADDQ)
        elif self.const.gridQ:
            returngrad = self.grad_grid(x)
        elif self.const.calc_mpiQ:
            if self.const.PBmetaDQ:
                V_klist    = []
                grad_klist = []
                for i, hillCs in enumerate(self.hillCslist):
                    V_k    = 0.0
                    grad_k = np.zeros(hillC.dim)
                    for hillC in self.hillCs:
                        V_k    += hillC.f(x[i])
                        grad_k += hillC.grad(x[i])
                    V_klist = [V_kdamp]
                    V_klistg = self.comm.gather(V_klist, root=0)
                    grad_klist = [grad_kdamp]
                    grad_klistg = self.comm.gather(grad_klist, root=0)
                    if self.rank == self.root:
                        V_k_damp = 0.0
                        for V_hill in V_klistg:
                            V_k_damp += V_hill[0]
                        V_k = float(V_k_damp)
                        grad_k_damp = 0.0
                        for grad_hill in grad_klistg:
                            grad_k_damp += grad_hill[0]
                        grad_k = float(grad_k_damp)
                    else:
                        V_k = None
                        grad_k = None
                    V_k    = self.comm.bcast(V_k,    root=0)
                    grad_k = self.comm.bcast(grad_k, root=0)
                    V_klist.append(V_k)
                    grad_klist.append(grad_k)
                returngrad = np.zeros(len(V_klist))
                Pinv      = 1.0 / sum([np.exp(-self.const.beta * V_k) 
                                         for V_k in V_klist])
                for i in range(len(V_klist)):
                    returngrad[i] = grad_klist[i] * np.exp(
                                    - self.const.beta * V_klist[i]) * Pinv
            else:
                returngrad = np.zeros(self.dim)
                for i, hillC in enumerate(self.hillCs):
                    returngrad += hillC.grad(x)
                returngradlist = [returngrad]
                returngradlist = self.comm.gather(returngradlist, root=0)
                if self.rank == self.root:
                    returngrad_damp = np.zeros(self.dim)
                    for grad_hill in returngradlist:
                        returngrad_damp += grad_hill[0]
                    returngrad = returngrad_damp
                returngrad = self.comm.bcast(returngrad, root=0)
        else:
            returngrad = np.zeros(self.dim)
            for hillC in self.hillCs:
                returngrad += hillC.grad(x)
        if self.const.WellTempairedQ:
            returngrad = returngrad * self.const.WT_Biasfactor_ffactor
        return returngrad
    def grad_grid(self, x):
        gridNlist = []
        for x_i in x:
            gridNlist.append( (x_i - self.const.grid_min) // self.xdelta )
        x_min = np.array([self.const.grid_min + self.xdelta * gridN 
                             for gridN in gridNlist])
        gridcounter   = 0
        grad_xminlist = [False, False]
        for x_before, grad_before in self.grad_gridlist:
            if gridcounter % self.size == self.rank:
                if np.allclose(x_before, x_min):
                    grad_xminlist = [self.rank, grad_before]
            gridcounter += 1
        grad_xminlistg = self.comm.gather(grad_xminlist, root=0)
        if self.rank == self.root:
            grad_xmin = []
            for xrank, grad_before in grad_xminlistg:
                if not xrank is False:
                    grad_xmin = [grad_before]
                    break
        else:
            grad_xmin = None
        grad_xmin = self.comm.bcast(grad_xmin, root=0)
        if len(grad_xmin) == 0:
            if self.const.calc_mpiQ:
                returngrad = np.zeros(self.dim)
                for hillC in self.hillCs:
                    returngrad += hillC.grad(x_min)
                returngradlist = [returngrad]
                returngradlist = self.comm.gather(returngradlist, root=0)
                if self.rank == self.root:
                    returngrad_damp = np.zeros(self.dim)
                    for grad_hill in returngradlist:
                        returngrad_damp += grad_hill[0]
                    grad_xmin = returngrad_damp
                else:
                    grad_xmin = None
                grad_xmin = self.comm.bcast(grad_xmin, root=0)
            else:
                grad_xmin = np.zeros(self.dim)
                for hillC in self.hillCs:
                    grad_xmin += hillC.grad(x_min)
            if self.rank == self.root:
                writeline = ""
                for pointx in x_min:
                    writeline += "% 7.5f,"%pointx
                for pointgrad in grad_xmin:
                    writeline += "%s,"%pointgrad
                writeline = writeline.rstrip(",") + "\n"
                with open("%s/jobfiles_meta/grad_grid.csv"%self.const.pwdpath, "a") as wf:
                    wf.write(writeline)
            self.grad_gridlist.append([x_min, grad_xmin])
        else:
            grad_xmin = grad_xmin[0]
        returngrad = copy.copy(grad_xmin)
        for i in range(len(x)):
            xdelta_array     = np.zeros(len(x))
            xdelta_array[i] += self.xdelta
            x_max = x_min + xdelta_array
            gridcounter = 0
            grad_xmaxlist = [False, False]
            for x_before, grad_before in self.grad_gridlist:
                if gridcounter % self.size == self.rank:
                    if np.allclose(x_before, x_min):
                        grad_xmaxlist = [self.rank, grad_before]
                gridcounter += 1
            grad_xmaxlistg = self.comm.gather(grad_xmaxlist, root=0)
            if self.rank == self.root:
                grad_xmax = []
                for xrank, grad_before in grad_xmaxlistg:
                    if not xrank is False:
                        grad_xmax = [grad_before]
                        break
            else:
                grad_xmax = None
            grad_xmax = self.comm.bcast(grad_xmax, root=0)

            if len(grad_xmax) == 0:
                grad_xmax = np.zeros(self.dim)
                for hillC in self.hillCs:
                    grad_xmax += hillC.grad(x_max)
                writeline = ""
                for pointx in x_max:
                    writeline += "% 7.5f,"%pointx
                for pointgrad in grad_xmax:
                    writeline += "%s,"%pointgrad
                writeline = writeline.rstrip(",") + "\n"
                with open("%s/jobfiles_meta/grad_grid.csv"%self.const.pwdpath, "a") as wf:
                    wf.write(writeline)
                self.grad_gridlist.append([x_max, grad_xmax])
            else:
                grad_xmax = grad_xmax[0]
            returngrad += (x_min[i] - x[i]) / self.const.grid_bin * (grad_xmin - grad_xmax)
        return returngrad
    def grad_cupy(self, x, ADDQ):
        if self.rank == self.root:
            if ADDQ:
                _hlist = self.h_part
                _slist = self.slist_part
                _sigmainvlist = self.sigmainvlist_part
                hillClength = self.hillClength
            else:
                _hlist = self.h_cupy
                _slist = self.slistall
                _sigmainvlist = self.sigmainvlistall
                hillClength = len(self.hillCs)
            dlist_cupy = self.const.cp.zeros((len(x), hillClength))
            if self.const.periodicQ:
                #dpoint_i, dis_sqlist = self.dist_forgrad_kernel0(x[0], self.slistall[0], self.sigmainvlistall[0], self.const.periodicmin, self.const.periodicmax)
                dpoint_i, dis_sqlist = self.dist_forgrad_kernel0(x[0], _slist[0], _sigmainvlist[0], self.const.periodicmin, self.const.periodicmax)
            else:
                dpoint_i, dis_sqlist = self.dist_forgrad_kernel0(x[0], _slist[0], _sigmainvlist[0])

            dlist_cupy[0] = dpoint_i
            for i in range(1, len(x)):
                if self.const.periodicQ:
                    dpoint_i, dis_sqlist = self.dist_forgrad_kernel(
                        x[i], _slist[i], _sigmainvlist[i], dis_sqlist, self.const.periodicmin, self.const.periodicmax)
                else:
                    dpoint_i, dis_sqlist = self.dist_forgrad_kernel(
                        x[i], _slist[i], _sigmainvlist[i], dis_sqlist)
                dlist_cupy[i] = dpoint_i
            dlist_cupy    =   dlist_cupy.transpose()
            fresult_cupy  =   self.f_kernel(dis_sqlist, _hlist)
            gradlist_cupy =   self.const.cp.dot(fresult_cupy, dlist_cupy)
            returngrad    = - self.const.cp.asnumpy(gradlist_cupy)
        else:
            returngrad = None
        if self.const.calc_mpiQ:
            returngrad = self.comm.bcast(returngrad, root=0)
        return returngrad
    def hessian(self, x):
        if self.const.calc_mpiQ:
            if self.const.PBmetaDQ:
                V_klist    = []
                grad_klist = []
                gradgrad_klist = []
                for i, hillCs in enumerate(self.hillCslist):
                    V_k    = 0.0
                    grad_k = np.zeros(hillC.dim)
                    gradgrad_k = np.zeros(hillC.dim)
                    for hillC in self.hillCs:
                        V_k    += hillC.f(x[i])
                        grad_k += hillC.grad(x[i])
                        gradgrad_k += hillC.hessian(x[i])
                    V_klist = [V_kdamp]
                    V_klistg = self.comm.gather(V_klist, root=0)
                    grad_klist = [grad_kdamp]
                    grad_klistg = self.comm.gather(grad_klist, root=0)
                    gradgrad_klist = [gradgrad_kdamp]
                    gradgrad_klistg = self.comm.gather(gradgrad_klist, root=0)
                    if self.rank == self.root:
                        V_k_damp = 0.0
                        for V_hill in V_klistg:
                            V_k_damp += V_hill[0]
                        V_k = float(V_k_damp)
                        grad_k_damp = 0.0
                        for grad_hill in grad_klistg:
                            grad_k_damp += grad_hill[0]
                        grad_k = float(grad_k_damp)
                        gradgrad_k_damp = 0.0
                        for gradgrad_hill in gradgrad_klistg:
                            gradgrad_k_damp += gradgrad_hill[0]
                        gradgrad_k = float(gradgrad_k_damp)
                    else:
                        V_k        = None
                        grad_k     = None
                        gradgrad_k = None
                    V_k    = self.comm.bcast(V_k,    root=0)
                    grad_k = self.comm.bcast(grad_k, root=0)
                    gradgrad_k = self.comm.bcast(gradgrad_k, root=0)
                    V_klist.append(V_k)
                    grad_klist.append(grad_k)
                    gradgrad_klist.append(gradgrad_k)
                returnhessian = np.zeros((self.dim, self.dim))
                Pinv      = 1.0 / sum([np.exp(-self.const.beta * V_k) 
                                         for V_k in V_klist])
                Plist     = [np.exp(-self.const.beta * V_k) * Pinv for V_k in V_klist]
                for i in range(len(V_klist)):
                    for j in range(len(V_klist)):
                        if i == j:
                            returnhessian[i,i]  = gradgrad_klist[i] * Plist[i]
                            returnhessian[i,i] += self.const.beta * grad_k[i] * Plist[i] * (Plist[i] - 1.0) 
                        else:
                            returnhessian[i,j]  = self.const.beta * gradgrad_klist[i] * Plist[i] * Plist[j]
            else:
                returnhessian = np.zeros((self.dim, self.dim))
                if not self.const.parallelMetaDQ:
                    if self.rank == self.root:
                        for i, hillC in enumerate(self.hillCs):
                            if i % self.size == self.rank:
                                returnhessian += hillC.hessian(x)
                    else:
                        for hillC in self.hillCs:
                            returnhessian += hillC.hessian(x)
                else:
                    for hillC in self.hillCs:
                        returnhessian += hillC.hessian(x)
                returnhessianlist = [returnhessian]
                returnhessianlist = self.comm.gather(returnhessianlist, root=0)
                if self.rank == self.root:
                    returnhessian_damp = 0.0
                    for hessian_hill in returnhessianlist:
                        returnhessian_damp += hessian_hill[0]
                    returnhessian = returnhessian_damp
                returnhessian = self.comm.bcast(returnhessian, root=0)
        else:
            returnhessian = np.zeros((self.dim, self.dim))
            for hillC in self.hillCs:
                returnhessian += hillC.hessian(x)
        return returnhessian
    def import_grid(self):
        self.f_gridlist    = []
        self.grad_gridlist = []
        if os.path.exists("./jobfiles_meta/f_grid.csv"):
            for line in open("./jobfiles_meta/f_grid.csv"):
                line = line.replace("\n","").split(",")
                line = np.array(line, dtype = float)
                self.f_gridlist.append([line[:-1], line[-1]])
        if os.path.exists("./jobfiles_meta/grad_grid.csv"):
            for line in open("./jobfiles_meta/grad_grid.csv"):
                line  = line.replace("\n","").split(",")
                _x    = np.array(line[:self.dim], dtype = float)
                _grad = np.array(line[self.dim:], dtype = float)
                self.grad_gridlist.append([_x, _grad])
        self.xdelta = self.const.grid_max - self.const.grid_min
        self.xdelta = self.xdelta / self.const.grid_bin
### end class GaussianC ###
def main():
    metaD = Metad_result("../walker_5ns/HILLS")
    if self.rank == self.root:
        initialpointlist = [np.random.rand(metaD.dim) * 2.0 * np.pi - np.pi  for _ in range(100)]
    else:
        initialpointlist = None
    if const.calc_mpiQ:
        initialpointlist = self.comm.bcast(initialpointlist, root=0)

    SHSearch(metaD.f, metaD.grad, metaD.hessian,
            importinitialpointQ = False, initialpoints = initialpointlist, 
            SHSrank = self.rank, SHSroot = self.root, SHScomm = self.comm)
if __name__ == "__main__":
    main()
