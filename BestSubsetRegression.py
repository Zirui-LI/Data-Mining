import numpy as np
from scipy import linalg
import os

def solve_sym(xtx, xty):
        L = linalg.cholesky(xtx)
        return linalg.lapack.dpotrs(L, xty)[0]

def turnbits_rec(p):
    if(p==1):
        return np.array([[True, False],[True, True]])
    else:
        tmp1 = np.c_[ turnbits_rec(p - 1), 
                     np.array([False]*(2**(p - 1))).reshape((2**(p - 1), 1))]
        tmp2 = np.c_[ turnbits_rec(p-1), 
                     np.array([True]*(2**(p - 1))).reshape((2**(p - 1), 1))]
        return np.r_[tmp1, tmp2]

class BestSubsetRegress(object):
    """BestSubsetRegression

    A best subset regression based on Cp, AIC and cross validation.
    
    Attributes:
        x: A ndarray data of dependent variable.
        y: A ndarray data of independent variable.
        intercept: A boolean indicating if intercept is included in data.
        isCp: A boolean indicating if Cp is applied.
        isAIC: A boolean indicating if AIC is applied.
        isCV: A boolean indicating if cross validation is applied.
    """
    def __init__(self, x=0,  y=0, intercept=True, isCp=True, isAIC=True, isCV=True):
        self.n, self.p = x.shape
        if intercept:
            self.x = np.c_[np.ones((self.n, 1)), x]
            self.ind_var = turnbits_rec(self.p)
        else:
            self.x = x
            self.ind_var = turnbits_rec(self.p)[1:, 1:]
            
        self.y = y
        self.xtx = np.dot(self.x.T, self.x)
        self.xty = np.dot(self.x.T, self.y)
        
        self.isCp = isCp
        self.isAIC = isAIC
        self.isCV = isCV
        self.intercept = intercept

        self.b = [solve_sym(self.xtx[ind][:, ind], self.xty[ind]) 
                  for ind in self.ind_var]
        
    
    def Cp_AIC(self):
        mse = [np.sum(np.dot(self.xtx[ind][:, ind], b_)*b_) 
               for ind, b_ in zip(self.ind_var, self.b)]
        rss = np.dot(self.y, self.y) - mse
        d = np.sum(self.ind_var, axis=1)
        self.Cp = rss + 2*d*rss[-1]/(self.n - self.p - 1) 
        self.AIC = self.n*np.log(rss) + 2*d
        
        
    def cvreg(self):
        K = 10
        indices = np.array_split(np.random.permutation(np.arange(0, self.n)), K)
    
        def cvk(ind, index):
            txx = self.xtx[ind][:, ind] - np.dot((self.x[index][:, ind]).T, 
                                                 self.x[index][:, ind])
            txy = self.xty[ind] - np.dot((self.x[index][:, ind]).T, 
                                         self.y[index])
            tcoe = solve_sym(txx, txy)
            return np.sum(
                (self.y[index] - np.dot(self.x[index][:, ind], tcoe))**2)
        self.cverr = np.sum(np.array(
            [[cvk(ind, index) for index in indices] 
             for ind in self.ind_var]), axis = 1)/self.n
        
    
    def bestsubsetregress(self):
        self.names = np.array(['x' + str(i) for i in range(1, self.p+1)])
        
        if self.isCp or self.isAIC:
            BestSubsetRegress.Cp_AIC(self)
            
        if self.isCp:
            min_id_Cp = np.argmin(self.Cp)
            if self.intercept:
                sub_names_Cp = np.insert(self.names[self.ind_var[min_id_Cp][1:]], 0, 'mu')
            else:
                sub_names_Cp = self.names[self.ind_var[min_id_Cp]]
            sub_beta_Cp = self.b[min_id_Cp]
            print('The best subset and coeffecients under Cp:\n')
            print(dict(zip(sub_names_Cp, sub_beta_Cp)), '\n')
            
        if self.isAIC:
            min_id_AIC = np.argmin(self.AIC)
            if self.intercept:
                sub_names_AIC = np.insert(self.names[self.ind_var[min_id_AIC][1:]], 0, 'mu')
            else:
                sub_names_AIC = self.names[self.ind_var[min_id_AIC]]
            sub_beta_AIC = self.b[min_id_AIC]
            print('The best subset and coeffecients under AIC:\n')
            print(dict(zip(sub_names_AIC, sub_beta_AIC)), '\n')
            
        if self.isCV:   
            BestSubsetRegress.cvreg(self)
            min_id_CV = np.argmin(self.cverr)
            if self.intercept:
                sub_names_CV = np.insert(self.names[self.ind_var[min_id_CV][1:]], 0, 'mu')
            else:
                sub_names_CV = self.names[self.ind_var[min_id_CV]]
            sub_beta_CV = self.b[min_id_CV]
            print('The best subset and coeffecients under CV:\n')
            print(dict(zip(sub_names_CV, sub_beta_CV)), '\n')
