import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import scipy.special as sp
# import scipy as sp
import pandas as pd
from operator import attrgetter
from IPython.display import Image
np.set_printoptions(precision=8)

import pml
import time
import pickle
import operator

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
# https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links

def randgen(pr, N=1): 
    L = len(pr)
    return np.random.choice(range(L), size=N, replace=True, p=pr/np.sum(pr))

def log_sum_exp(l, axis=0):
    l_star = np.max(l, axis=axis, keepdims=True)
    return l_star + np.log(np.sum(np.exp(l - l_star),axis=axis,keepdims=True)) 

def safe_log_sum_exp(x, axis=0):
    return log_sum_exp(x,axis)

def normalize_exp(log_P, axis=0):
    a = np.max(log_P, keepdims=True, axis=axis)
    P = normalize(np.exp(log_P - a), axis=axis)
    return P

def normalize(A, axis=0):
    Z = np.sum(A, axis=axis, keepdims=True)
    idx = np.where(Z == 0)
    Z[idx] = 1
    return A/Z

def load_array(filename):
    X = np.loadtxt(filename)
    dim = int(X[0]);
    size = []
    for i in range(dim):
        size.append(int(X[i+1]));    
    X = np.reshape(X[dim+1:], size, order='F')
    return X;
        
def save_array(filename, X, format = '%.6f'):
    with open(filename, 'w') as f:
        dim = len(X.shape)
        f.write('%d\n' % dim)
        for i in range(dim):
            f.write('%d\n' % X.shape[i])
        temp = X.reshape(np.product(X.shape), order='F')
        for num in temp:
            f.write(str(num)+"\n")
        # np.savetxt(f, temp, fmt = format)
        
def plot_hist(data,xlines,title="",xlabel="",ylabel="",label_='changepoints'):
    (K,T) = data.shape
    fig = plt.figure(figsize=(30,4))
    ax = fig.gca()
    y,x = np.mgrid[slice(0, K+1, 1),slice(0,T+1,1)]
    im = ax.pcolormesh(x, y, data, cmap=cm.gray)
    fig.colorbar(im)
    ax.hold(True)
    plt1 = ax.vlines(np.arange(0,T), 0, xlines*K, colors='r', linestyles='-',label=label_,linewidth='3')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[plt1])
    fig.canvas.draw()

def plot_matrix(X, title='Title', xlabel='xlabel', ylabel='ylabel', figsize=None):
    if figsize is None:
        plt.figure(figsize=(25,6))
    else:
        plt.figure(figsize=figsize)
    plt.imshow(X, interpolation='none', vmax=np.max(X), vmin=0, aspect='auto')
    plt.colorbar()
    plt.set_cmap('gray_r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)