from __future__ import print_function
from gradient_checks import eval_numerical_gradient
# from past.builtins import xrange
import numpy as np
from random import randrange

EPSILON_UPDATE_FACTOR = 0.5


def jacobianMV(g, x, v):
    gradient_at_x = g(x)
    return np.dot(gradient_at_x.T, v)


def epsilon_update(epsilon):
    return EPSILON_UPDATE_FACTOR*epsilon


def jacobian_test(f,g, x, epsilon0, num_iter=30, delta=0.1):
    d = np.random.randn(*x.shape)
    total_err1 = 0
    total_err2 = 0
    flag = True
    epsilon = epsilon0
    prev_value1 = np.linalg.norm(f(x+epsilon*d)-f(x))
    prev_value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-jacobianMV(f, x, epsilon*d))
    for i in range(0, num_iter):
        epsilon = epsilon_update(epsilon)
        value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
        value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-jacobianMV(f, x, epsilon*d))
        total_err1 = total_err1+value1
        total_err2 = total_err2+value2
        print (["Jacobian test iteration ",i,"  :  ",value1/prev_value1, value2/prev_value2])
        if np.abs(value2/prev_value2 - EPSILON_UPDATE_FACTOR*EPSILON_UPDATE_FACTOR) > delta or np.abs(value1/prev_value1 - EPSILON_UPDATE_FACTOR) > delta:
            flag = False
        prev_value1= value1
        prev_value2= value2
    return total_err1,total_err2, flag

