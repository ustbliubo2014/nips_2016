# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: fast_kmeans.py
@time: 2016/12/15 18:52
@contact: ustb_liubo@qq.com
@annotation: fast_kmeans
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


'''
assumption free k-mcmc (afk_mc2)
reference: nips 2016 Fast and Provably Good Seedings for k-Means
https://las.inf.ethz.ch/files/bachem16fast.pdf
finding inits for keyword vecs (kejso.com)
'''

import random
import numpy as np

def d2(c1, vec):
    #get distance
    return np.linalg.norm(c1, vec)**2

def cal_proposal_distribution(vecs, c1, n):
    dist = []
    for vec in vecs:
        dist.append(d2(c1, vec))
    sum_dist = sum(dist)
    q = map(lambda a: 0.5*(a+.0) / sum_dist + (1+.0)/(2*n), dist)
    return q

def construct_sample_list(q):
    return np.repeat(range(len(q)), map(lambda a: a*1000, q))

def d2_c(x, C):
    # C = {c1, c2, ...}
    # return min(d2(x, ct))
    tmp = map(lambda a: d2(x, a), C)
    return min(tmp)

def assumption_free_kmcmc(vecs, k, chainlen):
    '''
    :param vecs: data set
    :param k: k centroids
    :param chainlen: chain length
    :return: k inits
    '''
    #preprocessing calculate proposal distribution 'q'
    c1 = random.sample(vecs, 1)
    C = [c1,]
    q = cal_proposal_distribution(vecs, c1, len(vecs))
    sample_list = construct_sample_list(q) # index
    #find c2...ck
    for i in range(1,k):
        xi = random.sample(sample_list,1)
        x = vecs[xi]
        dx = d2_c(x, C)
        for j in range(1, chainlen):
            yi = random.sample(sample_list,1)
            y = vecs[yi]
            dy = d2_c(y, C)
            if (dy*q[xi])/(dx*q[yi]) > np.random.uniform():
                x = y
                xi = yi
                dx = dy
        C.append(x)
    return C

