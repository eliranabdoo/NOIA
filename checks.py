from __future__ import print_function
from gradient_checks import eval_numerical_gradient
# from past.builtins import xrange
import numpy as np
from random import randrange

EPSILON_UPDATE_FACTOR = 0.5


def epsilon_update(epsilon):
    return EPSILON_UPDATE_FACTOR*epsilon


def gradient_test(f, f_gradient, x, epsilon0, num_iter=30, delta=0.1):
    d = np.random.randn(*x.shape) * (2 / np.sqrt(np.size(x)))
    total_err1 = 0
    total_err2 = 0
    flag = True
    epsilon = epsilon0
    grad = f_gradient(x) # grad by x requires trnapose here.... TODO
    prev_value1 = np.linalg.norm(f(x+epsilon*d)-f(x))
    prev_value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-epsilon*np.dot(epsilon*d,grad))
    for i in range(0, num_iter):
        epsilon = epsilon_update(epsilon)
        value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
        value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-epsilon*np.dot(epsilon*d, grad))
        total_err1 = total_err1+value1
        total_err2 = total_err2+value2
        print (["Gradient test iteration ", i, "  :  ", value1/prev_value1, value2/prev_value2])
        if np.abs(value2/prev_value2 - EPSILON_UPDATE_FACTOR*EPSILON_UPDATE_FACTOR) > delta or np.abs(value1/prev_value1 - EPSILON_UPDATE_FACTOR) > delta:
            flag = False
        prev_value1 = value1
        prev_value2 = value2
    return total_err1,total_err2, flag


def jacobian_test(f, f_jacobianmv, x, epsilon0, num_iter=30, delta=0.1):
    d = np.random.randn(*x.shape) * (2 / np.sqrt(np.size(x)))
    total_err1 = 0
    total_err2 = 0
    passed_all_tests = True
    epsilon = epsilon0
    f = f(x)
    prev_value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
    prev_value2 = np.linalg.norm(f(x + epsilon * d) - f(x) - f_jacobianmv(x, epsilon * d))
    for i in range(0, num_iter):
        epsilon = epsilon_update(epsilon)
        value1 = np.linalg.norm(f(x + epsilon * d) - f(x), ord=2)
        value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-f_jacobianmv(x, epsilon*d), ord=2)
        total_err1 = total_err1+value1
        total_err2 = total_err2+value2
        conv_rate_1 = value1 / prev_value1
        conv_rate_2 = value2 / prev_value2
        print(["Jacobian test iteration ", i, "  :  ", conv_rate_1, conv_rate_2])
        if np.abs(conv_rate_2 - EPSILON_UPDATE_FACTOR*EPSILON_UPDATE_FACTOR) > delta or np.abs(conv_rate_1 - EPSILON_UPDATE_FACTOR) > delta:
            passed_all_tests = False

        prev_value1 = value1
        prev_value2 = value2
    return total_err1, total_err2, passed_all_tests

