from __future__ import print_function
from gradient_checks import eval_numerical_gradient
# from past.builtins import xrange
import numpy as np
from random import randrange

EPSILON_UPDATE_FACTOR = 0.5

def jacobianMV(f, x, v):
    gradient_at_x = eval_numerical_gradient(f, x)
    return np.dot(gradient_at_x.T, v)

def epsilon_update(epsilon):
    return EPSILON_UPDATE_FACTOR*epsilon

def jacobian_test(f, x, epsilon0, num_iter = 30, delta=0.1):
    d = np.random.randn(*x.shape)
    total_err = 0
    flag = True
    epsilon = epsilon0
    #prev_value1 = np.linalg.norm(f(x+epsilon*d)-f(x))
    prev_value = np.linalg.norm(f(x+epsilon*d)-f(x)-jacobianMV(f, x, epsilon*d))
    for i in range(0, num_iter):
        epsilon = epsilon_update(epsilon)
        #value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
        value = np.linalg.norm(f(x + epsilon * d) - f(x) - jacobianMV(f, x, epsilon * d))
        total_err = total_err+value
        print (["Jcobian iteration ",i,"  :  ", value/prev_value])
        if value/prev_value > ((EPSILON_UPDATE_FACTOR*EPSILON_UPDATE_FACTOR)+delta):
            flag = False
        prev_value= value
    return total_err, flag

