import numpy as np 
from tqdm import tqdm
from time import time, sleep

from scipy.optimize import curve_fit, least_squares, minimize
from scipy import optimize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import r2_score

class FitFunction:
    def __init__(self, func, fit_params=None, cv=None, method='Nelder-Mead'):
        self.func = func
        self.fit_params = fit_params
        self.cv = cv
        self.method = method
        
        if self.fit_params is None:
            print('fit_params is not set.')
            print('Please enter a dictionary with parameter names and initial values.')
            
        self.fitted_params_ = {}
    
    def residual_func(self, theta, x_train, y_train):
        return ((self.func(x_train, theta)-y_train)**2).sum()
    
    def _optimize(self, X, y):
        initial = np.array([self.fit_params[kk] for kk in self.fit_params.keys()])
        soln = minimize(self.residual_func, initial, args=(X, y), method=self.method)
        return soln
    
    def fit(self, X, y, cv=None):
        if cv is not None: self.cv = cv
        if self.cv is None:
            soln = self._optimize(X, y)
            best_theta = soln.x
            print(best_theta)
            for bt,kk in zip(best_theta,self.fit_params.keys()):
                self.fitted_params_[kk] = bt
        else:
            scrs = []
            sols = []
            fitted_params0 = {}
            for kk in self.fit_params.keys():
                fitted_params0[kk] = []
            kf = ShuffleSplit(n_splits=self.cv)
            for train_index, test_index in tqdm(kf.split(X)):
                # print('Size of dataset | train:{}, test:{}'.format(len(train_index), len(test_index)))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                sol = self._optimize(X_train, y_train)
                best_theta = sol.x
                if np.all(np.isfinite(best_theta)):
                    for bt,kk in zip(best_theta,self.fit_params.keys()):
                        fitted_params0[kk].append(bt)
                        self.fitted_params_[kk] = bt
                    scr = self.score(X_test, y_test)
                    # print(self.fitted_params_, scr)
                    sols.append(sol)
                    scrs.append(scr)
            sleep(1)

            self.fitted_params_std = {}
            for kk in self.fit_params.keys():
                self.fitted_params_[kk] = np.array(fitted_params0[kk])[np.array(scrs)>0].mean()
                self.fitted_params_std[kk] = np.array(fitted_params0[kk])[np.array(scrs)>0].std()
            self.cv_scores_ = np.array(scrs)
            
    def predict(self, X):
        best_theta = np.array([self.fitted_params_[kk] for kk in self.fitted_params_.keys()])
        y_model = self.func(X, best_theta)
        return y_model
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2