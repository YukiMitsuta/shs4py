#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2021/06/29 MitsutaYuki
#
# Distributed under terms of the MIT license.

import os, copy
import numpy as np

from . import functions

class RGeoClass():
    """
    class of Riemann Geometory
    functions;
        setinitialJlist: setup the iniitial list of Jacobi matrixs
        updateJinvlist:  update the list of inverse Jacobi matrixs by using the
                         list of Jacobi matrixs
        TotalJvsJinv:    sum of　inner product of a vector in Jacobi matrix and
                         a vector in inverse Jcobi matrix
        gamma:           return connection coefficients

    variables;
        self.dim: dimension of CVs
        self.Jlist: list of Jacobi matrixs
        self.Jinvlist: list of inverse Jacobi matrixs

    """
    def __init__(self, NEBc):
        self.dim = NEBc.dim
        self.setinitialJlist(NEBc)
        self.updateJinvlist(NEBc)
    def setinitialJlist(self,NEBc):
        """
        setup the iniitial list of Jacobi matrixs
        class:
            NEBc: class of NEB
        """
        self.Jlist = []
        self.blist = []
        for k in range(NEBc.Nimage):
            p_image = NEBc.imagelist[k]
            if k < NEBc.Nimage-1:
                v = NEBc.imagelist[k+1] - NEBc.imagelist[k]
            else:
                v = NEBc.imagelist[k] - NEBc.imagelist[k-1]
            self.Jlist.append(functions.rotateMat(v)/ np.sqrt(NEBc.dim))
            v_xi = np.array([NEBc.tlist[k] for _ in range(NEBc.dim)])
            self.blist.append(NEBc.imagelist[k]-self.Jlist[k]@v_xi)

    def updateJlist(self,NEBc):
        """
        update the list of Jacobi matrixs by using
        the list of inverse Jacobi matrixs
        class:
            NEBc: class of NEB
        """
        self.Jlist = []
        for i_image in range(NEBc.Nimage):
            if all(sum(self.Jinvlist[i_image]) == 0.0):
                self.Jlist.append(np.zeros((self.dim,self.dim)))
            else:
                self.Jlist.append(np.linalg.inv(self.Jinvlist[i_image]))
    def updateJinvlist(self,NEBc):
        """
        update the list of inverse Jacobi matrixs by using
        the list of Jacobi matrixs
        class:
            NEBc: class of NEB
        """
        self.Jinvlist = []
        for i_image in range(NEBc.Nimage):
            if all(sum(self.Jlist[i_image]) == 0.0):
                self.Jinvlist.append(np.zeros((self.dim,self.dim)))
            else:
                self.Jinvlist.append(np.linalg.inv(self.Jlist[i_image]))
    def TotalJvsJinv(self, NEBc, k,lam,kinv,lam_inv):
        """
        sum of　inner product of a vector in Jacobi matrix and
        a vector in inverse Jcobi matrix
        class;
            NEBc: class of NEB
        variables;
            k: index of J in real space
            lam: index of J in reaction path space
            kinv: index of Jinv in real space
            lam_inv: index of Jinv in reaction path space
        """
        #if k == 0 or len(self.Jlist) <= k+1:
            #return 0
        #if kinv == 0 or len(self.Jlist) <= kinv+1:
            #return 0
        returnf = [self.Jlist[k][i,lam]*self.Jinvlist[kinv][lam_inv,i] for i in range(self.dim)]
        return sum(returnf)
    def gamma(self, NEBc, k, nu, lam, mu):
        """
        return connection coefficients
        (gamma^nu_lam_mu)_k
        """
        if k == 0 or NEBc.Nimage <= k+1:
            return 0
        returnf = 0.0
        returnf -= functions.fdelta(lam,nu)
        returnf -= functions.fdelta(mu,nu)
        returnf += self.TotalJvsJinv(NEBc, k+1,lam,k,nu)+self.TotalJvsJinv(NEBc,k+1,mu,k,nu)
        returnf /= NEBc.tlist[k+1]-NEBc.tlist[k]
        return returnf
    def gamma_dev(self, NEBc, k, kappa, nu, lam, mu):
        """
        return　ｐartial differential of deconnection coefficients
        del xi_kappa*(gamma^nu_lam_mu)_k/del xi_kappa
        """
        returnf = 0
        if k == 0 or NEBc.Nimage <= k+1:
            return 0
        if kappa != nu and kappa != lam and kappa != mu:# pattern1
            returnf = 0
        elif kappa == nu and kappa != lam and kappa != mu:# pattern2
            returnf += self.TotalJvsJinv(NEBc,k+1,lam,k,kappa)
            returnf += self.TotalJvsJinv(NEBc,k+1,mu,k,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k+1,lam,k-1,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k+1,mu,k-1,kappa)
            returnf += self.TotalJvsJinv(NEBc,k,lam,k-1,kappa)
            returnf += self.TotalJvsJinv(NEBc,k,mu,k-1,kappa)
        elif kappa != nu and kappa == lam and kappa != mu:# pattern3
            returnf += self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
            returnf += self.TotalJvsJinv(NEBc,k-1,kappa,k,nu)
            #returnf -= functions.fdelta(mu,nu)
        elif kappa != nu and kappa != lam and kappa == mu:# pattern4
            returnf += self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
            returnf += self.TotalJvsJinv(NEBc,k-1,kappa,k,nu)
            #returnf -= functions.fdelta(lam,nu)
        elif kappa == nu and kappa == lam and kappa != mu:# pattern5
            returnf += self.TotalJvsJinv(NEBc,k+1,kappa,k,kappa)
            returnf += self.TotalJvsJinv(NEBc,k+1,mu,k,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k,kappa,k-1,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k+1,mu,k-1,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k,mu,k-1,kappa)
        elif kappa == nu and kappa != lam and kappa == mu:# pattern6
            returnf += self.TotalJvsJinv(NEBc,k+1,kappa,k,kappa)
            returnf += self.TotalJvsJinv(NEBc,k+1,lam,k,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k,kappa,k-1,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k+1,lam,k-1,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k,lam,k-1,kappa)
        elif kappa != nu and kappa == lam and kappa == mu:# pattern7
            returnf += self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
            returnf -= self.TotalJvsJinv(NEBc,k-1,kappa,k,nu)
            returnf *= 2.0
        elif kappa == nu and kappa == lam and kappa == mu:# pattern8
            returnf += self.TotalJvsJinv(NEBc,k+1,kappa,k,kappa)
            returnf -= self.TotalJvsJinv(NEBc,k,kappa,k-1,kappa)
            returnf *= 2.0
        else:
            raise ValueError("gamma_dev cannot detect the dev")
        returnf /= (NEBc.tlist[k+1]-NEBc.tlist[k])*(NEBc.tlist[k]-NEBc.tlist[k-1])
        return returnf
    def Rtensor(self, NEBc, k, nu, kappa, lam, mu):
        """
        return Riemann tensor
        (R^nu_kappa_lam_mu)_k
        """
        #print("gamma_dev1(%s,%s,%s,%s) = %s"%(kappa,nu,lam,mu,self.gamma_dev(NEBc,k,kappa,nu,lam,mu)))
        returnf  = self.gamma_dev(NEBc,k,kappa,nu,lam,mu)
        #print("gamma_dev2",self.gamma_dev(NEBc,k,lam,nu,kappa,mu))
        returnf -= self.gamma_dev(NEBc,k,lam,nu,kappa,mu)
        returnf += sum(self.gamma(NEBc,k,i,lam,mu)*self.gamma(NEBc,k,nu,kappa,i) for i in range(self.dim))
        returnf -= sum(self.gamma(NEBc,k,i,kappa,mu)*self.gamma(NEBc,k,nu,lam,i) for i in range(self.dim))
        return returnf
    def TotalJvsJinv_index(self, NEBc, k,lam,kinv,lam_inv, invQ):
        returnvec = np.array([np.zeros([self.dim,self.dim]) for _ in range(NEBc.Nimage)])
        #if k == 0 or len(self.Jlist) <= k+1:
            #pass
        #elif kinv == 0 or len(self.Jlist) <= kinv+1:
            #pass
        #else:
        if True:
            for i in range(self.dim):
                if invQ:
                    returnvec[kinv][lam_inv,i] +=self.Jlist[k][i,lam]
                else:
                    returnvec[k][i,lam] +=self.Jinvlist[kinv][lam_inv,i]
        return returnvec
    def calcgammatimesgamma_index(self,NEBc,k,nu,kappa,lam,mu,invQ):
        returnvec = np.array([np.zeros([self.dim,self.dim]) for _ in range(NEBc.Nimage)])
        returnf = 0.0
        a = 1.0/(NEBc.tlist[k+1]-NEBc.tlist[k])**2
        for eta in range(NEBc.dim):
            sigma2 = functions.fdelta(lam,eta)+functions.fdelta(mu,eta)
            sigma1 = functions.fdelta(kappa,nu)+functions.fdelta(eta,nu)
            returnf += a*sigma1*sigma2
            for i in range(NEBc.dim):
                if invQ:
                    returnvec[k][eta,i] += 0.5*a*self.Jlist[k+1][i,lam]*self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
                    returnvec[k][nu,i]  += 0.5*a*self.Jlist[k+1][i,eta]*self.TotalJvsJinv(NEBc,k+1,lam,k,eta)
                    returnvec[k][nu,i]  += 0.5*a*self.Jlist[k+1][i,kappa]*self.TotalJvsJinv(NEBc,k+1,mu,k,eta)
                    returnvec[k][eta,i] += 0.5*a*self.Jlist[k+1][i,mu]*self.TotalJvsJinv(NEBc,k+1,eta,k,nu)
                    returnvec[k][nu,i]  += 0.5*a*self.Jlist[k+1][i,kappa]*self.TotalJvsJinv(NEBc,k+1,lam,k,eta)
                    returnvec[k][eta,i] += 0.5*a*self.Jlist[k+1][i,lam]*self.TotalJvsJinv(NEBc,k+1,eta,k,nu)
                    returnvec[k][eta,i] += 0.5*a*self.Jlist[k+1][i,mu]*self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
                    returnvec[k][nu,i]  += 0.5*a*self.Jlist[k+1][i,eta]*self.TotalJvsJinv(NEBc,k+1,mu,k,eta)
                    if sigma1 != 0.0:
                        returnvec[k][eta,i] -= a*self.Jlist[k+1][i,lam]*sigma1
                        returnvec[k][eta,i] -= a*self.Jlist[k+1][i,mu]*sigma1
                    if sigma2 != 0.0:
                        returnvec[k][nu,i] -= a*self.Jlist[k+1][i,kappa]*sigma2
                        returnvec[k][nu,i] -= a*self.Jlist[k+1][i,eta]*sigma2
                else:
                    returnvec[k+1][i,lam]   += 0.5*a*self.Jinvlist[k][eta,i]*self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
                    returnvec[k+1][i,eta]   += 0.5*a*self.Jinvlist[k][nu,i]*self.TotalJvsJinv(NEBc,k+1,lam,k,eta)
                    returnvec[k+1][i,kappa] += 0.5*a*self.Jinvlist[k][nu,i]*self.TotalJvsJinv(NEBc,k+1,mu,k,eta)
                    returnvec[k+1][i,mu]    += 0.5*a*self.Jinvlist[k][eta,i]*self.TotalJvsJinv(NEBc,k+1,eta,k,nu)
                    returnvec[k+1][i,kappa] += 0.5*a*self.Jinvlist[k][nu,i]*self.TotalJvsJinv(NEBc,k+1,lam,k,eta)
                    returnvec[k+1][i,lam]   += 0.5*a*self.Jinvlist[k][eta,i]*self.TotalJvsJinv(NEBc,k+1,eta,k,nu)
                    returnvec[k+1][i,mu]    += 0.5*a*self.Jinvlist[k][eta,i]*self.TotalJvsJinv(NEBc,k+1,kappa,k,nu)
                    returnvec[k+1][i,eta]   += 0.5*a*self.Jinvlist[k][nu,i]*self.TotalJvsJinv(NEBc,k+1,mu,k,eta)
                    if sigma1 != 0.0:
                        returnvec[k+1][i,lam] -= a*self.Jinvlist[k][eta,i]*sigma1
                        returnvec[k+1][i,mu] -= a*self.Jinvlist[k][eta,i]*sigma1
                    if sigma2 != 0.0:
                        returnvec[k+1][i,kappa] -= a*self.Jinvlist[k][nu,i]*sigma2
                        returnvec[k+1][i,eta] -= a*self.Jinvlist[k][nu,i]*sigma2
        return returnvec,returnf
    def gamma_dev_index(self, NEBc, k, kappa, nu, lam, mu, invQ):
        returnvec = np.array([np.zeros([self.dim,self.dim]) for _ in range(NEBc.Nimage)])
        returnf = 0
        #if k == 0 or NEBc.Nimage <= k+1:
            #pass
        if kappa != nu and kappa != lam and kappa != mu:
            pass
        elif kappa == nu and kappa != lam and kappa != mu:# pattern2
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,lam,k,kappa, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,mu,k,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k+1,lam,k-1,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k+1,mu,k-1,kappa, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k,lam,k-1,kappa, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k,mu,k-1,kappa, invQ)
        elif kappa != nu and kappa == lam and kappa != mu:# pattern3
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,kappa,k,nu, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k-1,kappa,k,nu, invQ)
            returnf -=functions.fdelta(mu,nu)
        elif kappa != nu and kappa != lam and kappa == mu:# pattern4
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,kappa,k,nu, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k-1,kappa,k,nu, invQ)
            returnf -= functions.fdelta(lam,nu)
        elif kappa == nu and kappa == lam and kappa != mu:# pattern5
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,kappa,k,kappa, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,mu,k,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k,kappa,k-1,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k+1,mu,k-1,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k,mu,k-1,kappa, invQ)
        elif kappa == nu and kappa != lam and kappa == mu:# pattern6
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,kappa,k,kappa, invQ)
            returnvec += self.TotalJvsJinv_index(NEBc,k+1,lam,k,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k,kappa,k-1,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k+1,lam,k-1,kappa, invQ)
            returnvec -= self.TotalJvsJinv_index(NEBc,k,lam,k-1,kappa, invQ)
        elif kappa != nu and kappa == lam and kappa == mu:# pattern7
            returnvec += 2.0*self.TotalJvsJinv_index(NEBc,k+1,kappa,k,nu, invQ)
            returnvec -= 2.0*self.TotalJvsJinv_index(NEBc,k-1,kappa,k,nu, invQ)
        elif kappa == nu and kappa == lam and kappa == mu:# pattern8
            returnvec += 2.0*self.TotalJvsJinv_index(NEBc,k+1,kappa,k,kappa, invQ)
            returnvec -= 2.0*self.TotalJvsJinv_index(NEBc,k,kappa,k-1,kappa, invQ)
        else:
            raise ValueError("Rtensor_index cannot detect the dev")

        returnvec /= (NEBc.tlist[k+1]-NEBc.tlist[k])*(NEBc.tlist[k]-NEBc.tlist[k-1])
        returnf /= (NEBc.tlist[k+1]-NEBc.tlist[k])*(NEBc.tlist[k]-NEBc.tlist[k-1])
        return returnvec,returnf
    def Rtensor_index(self, NEBc, k, nu, kappa, lam, mu, invQ):
        returnvec,returnf  = self.gamma_dev_index(NEBc,k,kappa,nu,lam,mu, invQ)
        returnvecdamp,returnfdamp = self.gamma_dev_index(NEBc,k,lam,nu,kappa,mu, invQ)
        returnvec -= returnvecdamp
        returnf   -= returnfdamp
        if True:
            returnvecdamp, returnfdamp = self.calcgammatimesgamma_index(NEBc,k,nu,kappa,lam,mu, invQ)
            returnvec += returnvecdamp
            returnf   += returnfdamp
            returnvecdamp, returnfdamp = self.calcgammatimesgamma_index(NEBc,k,nu,lam,kappa,mu, invQ)# kappa <-> lam
            returnvec -= returnvecdamp
            returnf   -= returnfdamp
        else:
            returnf   += sum(self.gamma(NEBc,k,i,lam,mu)*self.gamma(NEBc,k,nu,kappa,i) for i in range(self.dim))
            returnf   -= sum(self.gamma(NEBc,k,i,kappa,mu)*self.gamma(NEBc,k,nu,lam,i) for i in range(self.dim))
        #else:
            #returnvec += self.calcgammatimesgamma_index(NEBc,k,nu,kappa,lam,mu, invQ)
        if 1000 < returnf:
            print("k,nu,kappa,lam,mu,returnf= %s, %s, %s, %s, %s, %s"%(k,nu,kappa,lam,mu,returnf))
            print("J[%s]    = %s"%(k-1, self.Jlist[k-1]))
            print("Jinv[%s] = %s"%(k-1, self.Jinvlist[k-1]))
            print("J[%s]    = %s"%(k, self.Jlist[k]))
            print("Jinv[%s] = %s"%(k, self.Jinvlist[k]))
            print("J[%s]    = %s"%(k+1, self.Jlist[k+1]))
            print("Jinv[%s] = %s"%(k+1, self.Jinvlist[k+1]))
            print(sum(self.gamma(NEBc,k,i,lam,mu)*self.gamma(NEBc,k,nu,kappa,i) for i in range(self.dim)))
            print(sum(self.gamma(NEBc,k,i,kappa,mu)*self.gamma(NEBc,k,nu,lam,i) for i in range(self.dim)))
        returnvec = np.ravel(returnvec)
        return returnvec, returnf
    def getRtenslist(self,NEBc):
        geovec = [1.0/np.sqrt(NEBc.dim) for _ in range(NEBc.dim)]
        #geovec = [1.0/np.sqrt(NEBc.dim),- 1.0/np.sqrt(NEBc.dim)]
        Rtenslist = []
        for k in range(1,NEBc.Nimage-1):
            for kappa in range(NEBc.dim):
                for lam in range(NEBc.dim):
                    for mu in range(NEBc.dim):
                        Rtens = 0.0
                        for nu in range(NEBc.dim):
                            Rtens += self.Rtensor(NEBc,k,kappa,nu,lam,mu) *geovec[nu]
                        if 0.0 <= Rtens:
                        #if True:
                            Rtenslist.append([k,kappa,lam,mu,Rtens])
        #print("length of Rtenslist; ", len(Rtenslist))
        return Rtenslist
    def RicciTensor(self, NEBc, k, kappa, mu):
        returnf = 0.0
        for nu in range(NEBc.dim):
            returnf += self.Rtensor(NEBc, k, nu, kappa, nu, mu)
        return returnf
    def scalarCurv(self, NEBc, k):
        #print("k=",k)
        returnf = 0.0
        for kappa in range(NEBc.dim):
            for mu in range(NEBc.dim):
                g = 0.0
                for i in range(NEBc.dim):
                    g += self.Jinvlist[k][kappa, i]*self.Jinvlist[k][mu,i]
                #print("g;",g)
                #print("R; ",self.RicciTensor(NEBc,k,kappa,mu))
                returnf += g * self.RicciTensor(NEBc,k,kappa,mu)
        #exit()
        return returnf
    def velvec(self, NEBc, k):
        returnvec = np.zeros(NEBc.dim)
        for i in range(NEBc.dim):
            for mu in range(NEBc.dim):
                returnvec[i] += self.Jlist[k][i,mu]
        _norm = np.linalg.norm(returnvec)
        if _norm == 0.0:
            return returnvec
        else:
            return returnvec/_norm
    def optimizeJlist(self, NEBc, Rtenslist, optklist, invQ):
        # This constraint is performed to J times J inverse is identity matrix
        constraintdim = NEBc.dim*NEBc.dim*NEBc.Nimage
        constraintvec = []
        constraintMat = []
        matindex = 0
        for k in range(NEBc.Nimage):
            for mu in range(NEBc.dim):
                mat = np.array([np.zeros([NEBc.dim,NEBc.dim]) for _ in range(NEBc.Nimage)])
                for i in range(NEBc.dim):
                    if invQ:
                        mat[k][mu,i] += self.Jlist[k][i,mu]
                    else:
                        mat[k][i,mu] += self.Jinvlist[k][mu,i]
                mat = np.ravel(mat)
                constraintMat.append(mat)
                constraintvec.append(1.0)
        constraintMat=np.array(constraintMat)
        constraintvec=np.array(constraintvec)
        equationvec = []
        equationMat = []
        matindex = 0
        if invQ:
            Jvec = np.ravel(self.Jinvlist)
        else:
            Jvec = np.ravel(self.Jlist)
        equation_rankbefore = 0
        geovec = [1.0/np.sqrt(NEBc.dim) for _ in range(NEBc.dim)]
        #geovec = [1.0/np.sqrt(NEBc.dim),- 1.0/np.sqrt(NEBc.dim)]
        for k,kappa,lam,mu, Rtens in Rtenslist:
            mat = np.array([np.zeros([NEBc.dim*NEBc.dim]) for _ in range(NEBc.Nimage)])
            mat = np.ravel(mat)
            m = 0.0
            for nu in range(NEBc.dim):
                matdamp, mdamp = self.Rtensor_index(NEBc, k, kappa, nu, lam, mu, invQ)
                mat += matdamp * geovec[nu]
                m += mdamp * geovec[nu]
            equationMat.append(mat)
            if k in optklist:
                equationvec.append(float(m)-Rtens*0.9)
            else:
                equationvec.append(float(m)-Rtens)
            #Rtensdamp = np.dot(mat,Jvec)+m
            #equation_rank = np.linalg.matrix_rank(np.array(equationMat))
            #if equation_rankbefore == equation_rank:
                #equationMat.pop()
                #equationvec.pop()
            #else:
                #equation_rankbefore = equation_rank

        equationvec = np.array(equationvec)
        #print("equationvec; ",equationvec)
        #equationMat = np.array([x[NEBc.dim*NEBc.dim:] for x in equationMat])
        equationMat = np.array(equationMat)
        #normchk=np.linalg.norm(equationMat@Jvec-equationvec)
        #if 0.1 < normchk:
            #print("Error normchk = %s"%normchk)
        #print("equationMat@Jvec-equationvec; ", equationMat@Jvec-equationvec)
            #exit()
        #print("equationMat@Jvec; ", equationMat@Jvec)
        #print("equationMat@Jvec-equationvec; ", equationMat@Jvec-equationvec)
        print("Rtens; ",sum([x[-1] for x in Rtenslist]))
        #print("Rtens; ",[x[-1] for x in Rtenslist if x[0] in optklist])
        #print("Rtens; ",[[x[0],x[-1]] for x in Rtenslist])

        #Jvecdelta = Jvec[NEBc.dim*NEBc.dim:]
        Jvecdelta = Jvec
        #Rtens_mat = equationMat@Jvecdelta+equationvec
        #print("equationMat@Jvec+equationvec; ", Rtens_mat)
        #print("diff; ",np.linalg.norm(
        #Rtens_mat-np.array([x[-1] for x in Rtenslist]))
        #)
        #Jvecdelta = functions.convexOpt(equationMat,equationvec,constraintMat,constraintvec,Jvecdelta)
        Jvecdelta = functions.minimizeOpt(equationMat,equationvec,constraintMat,constraintvec,Jvecdelta)
        #Jvecdelta = [0.0 for _ in range(NEBc.dim*NEBc.dim)] + list(Jvecdelta)
        Jvec += Jvecdelta
        for k in range(NEBc.Nimage):
            for i in range(NEBc.dim):
                for j in range(NEBc.dim):
                    if invQ:
                        self.Jinvlist[k][i,j] = Jvec[NEBc.dim*NEBc.dim*k+NEBc.dim*i+j]
                    else:
                        self.Jlist[k][i,j] = Jvec[NEBc.dim*NEBc.dim*k+NEBc.dim*i+j]
    def optimizeJlist_neighborpoint(self, NEBc):
        for k in range(NEBc.Nimage):
            equationvec = []
            equationmat = []
            for i in range(NEBc.dim):
                mat, m = self.mkequation(NEBc, k, k, i)
                equationmat.append(mat)
                equationvec.append(m)
                if 1 <= k+1 <= NEBc.Nimage-1:
                    mat, m = self.mkequation(NEBc, k+1, k, i)
                    equationmat.append(mat)
                    equationvec.append(m)
                if 1 <= k-1 <= NEBc.Nimage-1:
                    mat, m = self.mkequation(NEBc, k-1, k, i)
                    equationmat.append(mat)
                    equationvec.append(m)
            #print("len(mat);", len(mat))
            #print("len(equationmat);", len(equationmat))
            equationmat = np.array(equationmat)
            equationvec = np.array(equationvec)
            #print("rank(equationmat);", np.linalg.matrix_rank(equationmat))
            Jvec = np.array(list(np.ravel(self.Jlist[k]))+list(self.blist[k]))
            Jvecdelta = functions.minimizeOpt(equationmat, equationvec, Jvec)
            returnvec = Jvec + Jvecdelta
            for i in range(NEBc.dim):
                self.blist[k][i] = returnvec[NEBc.dim*NEBc.dim+i]
                for j in range(NEBc.dim):
                    self.Jlist[k][i,j] = returnvec[NEBc.dim*i+j]
            #exit()
    def mkequation(self, NEBc,k_xi,k_x, i):
        mat   = np.zeros([NEBc.dim,NEBc.dim])
        blist = np.zeros(NEBc.dim)
        blist[i] = 1.0
        for mu in range(NEBc.dim):
            mat[i,mu] = NEBc.tlist[k_xi]
        mat = np.ravel(mat)
        returnmat = np.array(list(mat)+list(blist))
        returnm = NEBc.imagelist[k_xi][i]
        return returnmat, returnm
