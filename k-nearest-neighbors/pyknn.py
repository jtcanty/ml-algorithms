import numpy as np
import matplotlib.pyplot as plt

import readline
import rpy2
from rpy2.robjects import r, pandas2ri

import warnings
warnings.filterwarnings('ignore')

class KNN:
    
    def __init__(self, dataset, n_iter, tol = 1e-5):
        self.dataset = dataset
        self.n_iter = n_iter
        self.tol = tol
        
        
    def data(self): 
        df = pandas2ri.ri2py(r[self.dataset])  
        X = df.iloc[:,0:2].values
        name = df.columns
        
        return X, name
    
    
    def plot(self, height, width):
        fig, axs = plt.subplots(height, width, figsize = (20,10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.25)
        
        try:
            axs = axs.ravel()
        
        except AttributeError:
            pass
        
        return fig, axs
    
        
    def knn(self, K):
        
        # Initialize params
        [X, name] = self.data()
        [n, d] = np.shape(X)
        mu = np.random.rand(K,d)
        
        # Setup storage arrays
        norm = np.zeros((n, d))
        label = np.zeros((n, 1))
        obj = np.zeros((self.n_iter, 1))
        
        # Standardarize data
        X_mean = np.mean(X, 0)[None,:]
        X_std = np.std(X, 0)[None,:]
        X_n = (X - X_mean)/X_std
        
        # Setup plots
        fig1, axs1 = self.plot(2, int(self.n_iter/2))
        
        # Main algorithm loop
        for i in range(0, self.n_iter):
                
            # Expectation step
            for k in range(0, K):      
                dist = X_n - mu[k,:]
                norm[:,k] = np.sum(dist**2, 1)
                
            # Recalculate labels
            label = np.argmin(norm, 1)
            
            # update the means
            for k in range(0, K):
                idx = np.where(label == k)
                n_idx = np.shape(idx)[1]
                mu_sum = np.sum(X_n[idx], 0)
                mu[k,:] = mu_sum/n_idx
                
            # Compute the objective function
            for k in range(0, K):
                dist = (X_n - mu[k,:])
                obj[i] = obj[i] + np.sum(dist[np.where(label == k)]**2)
        
            # Check for convergence
            if np.abs(obj[i] - obj[i-1]) <= self.tol:
                obj = obj[:i]
                break
                
            # Generate plot of clusters
            col = np.full((1, n),'None')
            cmap = ['red','blue']

            for k in range(0, K):
                col[0,np.where(label == k)] = cmap[k]
        
            axs1[i].scatter(X_n[:,0], X_n[:,1], c=col[0,:], alpha=0.2)
            axs1[i].scatter(mu[:,0], mu[:,1], marker='x', s=150, c = cmap, linewidth=3)
            axs1[i].set_title('Iteration:' + str(0+i))
            axs1[i].set_xlabel(name[0])
            axs1[i].set_ylabel(name[1]) 
            
        # Generate plot of objective function
        fig2, axs2 = self.plot(1, 1)
        
        axs2.plot(np.arange(0,len(obj)),obj, marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=2)
        axs2.set_xlabel('Iteration')
        axs2.set_ylabel('J')
        plt.show()
    