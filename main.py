import numpy as np
import tables
import sys,os
import scipy.io
from softmax import LonelySoftmaxWithReg, train_with_sgd
import matplotlib.pyplot as plt
from softmax import FunctionsBoxes
from softmax import ResLayer
from softmax import ResNetwork
from softmax import accuracy
from gradient_checks import grad_check_sparse
import itertools
from checks import jacobian_test, gradient_test


def generate_all_combinations(grid):
    return itertools.product(*list(grid.values()))


def load_data(path):
    f = scipy.io.loadmat(path)
    t_data = f.get('Yt').T
    t_labels = np.argmax(f.get('Ct'), axis=0)
    v_data = f.get('Yv').T
    v_labels = np.argmax(f.get('Cv'), axis=0)
    num_labels = f.get('Cv').shape[0]

    return t_data, t_labels, v_data, v_labels, num_labels


def test():
    PATH = os.getcwd() + '\datasets\SwissRollData.mat'
    t_data, t_labels, v_data, v_labels, num_labels = load_data(PATH)

    # Normalize the data
    mean_image = np.mean(t_data, axis=0)
    t_data -= mean_image
    v_data -= mean_image
    t_data /= np.std(t_data, axis=0)
    v_data /= np.std(v_data, axis=0)

    test_data = t_data[0:50].T
    layer = ResLayer(test_data.shape)
    m = layer.calc_value(test_data)
    dx, dw1, dw2, db = layer.calc_grad(test_data)


def main():

    #run_tests()

    #while True:
    #    pass

    PATH = os.getcwd()+'\datasets\GMMDATA.mat'
    t_data, t_labels, v_data, v_labels, num_labels = load_data(PATH)

    # Normalize the data
    mean_image = np.mean(t_data, axis=0)
    t_data -= mean_image
    v_data -= mean_image
    t_data /= np.std(t_data, axis=0)
    v_data /= np.std(v_data, axis=0)

    #print(np.std(t_data, axis=0), np.mean(t_data, axis=0))

    hyperparams_grid = {
        "max_iter": [50],
        "batch_size": [500],
        "learning_rate": [0.1],
        "decay_rate": [0.1],
        "convergence_criteria": [0.01],
        "gamma": [0.7],
        "reg_param": [0]
    }

    max_acc = 0
    cur_acc = 0
    best_params = {}

    for hp_comb in generate_all_combinations(hyperparams_grid):
        hyperparams = {key: hp_comb[i] for key, i in zip(list(hyperparams_grid.keys()), range(len(hyperparams_grid.keys())))}

        cur_acc = run_unit(t_data, t_labels, v_data, v_labels, num_labels, **hyperparams)
        if cur_acc > max_acc:
            best_params = hyperparams
            max_acc = cur_acc

    print("Maximal accuracy of %d on validation set, achieved with : %s" % (max_acc, str(best_params)))

def run_tests():

    demo_layer = ResLayer(dim=10)
    x = np.random.rand(10, 1)*7

    ## Test layer jacobian w.r.t X ##
    layer_val = lambda x: demo_layer.calc_forward_pass(x)
    layer_jacobian_vec_x = lambda x, v: demo_layer.calc_backward_pass(x, v)[-1]
    jacobian_test(layer_val, layer_jacobian_vec_x, x, 10, 20, 0.1)
    ## Test layer jacobian w.r.t Params ##


    #######################################################

    demo_softmax = LonelySoftmaxWithReg(dim=10, num_labels=20, reg_param=0)
    x = np.random.rand(1, 10)
    y = np.array([10])
    w = demo_softmax.get_params_as_matrix()

    ## Test softmax gradient w.r.t X ##
    sm_x = lambda x: demo_softmax.calc_forward_pass(x, y)
    sm_gradient_x = lambda x: demo_softmax.grad_by_x(x, y)
   # gradient_test(sm_x, sm_gradient_x, x, 10, 20, 0.1)
    ## Test softmax gradient w.r.t Params ##
    sm_w = lambda w: FunctionsBoxes.softmax_loss_and_gradient_regularized(w, demo_softmax.add_bias_dimension(x), y, 0.1)[0]
    sm_gradient_params = lambda w: FunctionsBoxes.softmax_loss_and_gradient_regularized(w, demo_softmax.add_bias_dimension(x), y, 0.1)[1]
    gradient_test(sm_w, sm_gradient_params, w, 0.5, 20, 0.1)

    #######################################################
    loss, grad = demo_softmax.calc_loss_and_grad_for_batch(x, y)
    grad_err = grad_check_sparse(sm_w, demo_softmax.get_params_as_matrix(), grad, 10)




def run_unit(t_data, t_labels, v_data, v_labels, num_labels, **hyperparams):

    print("Running unit with: " + str(hyperparams))
    # Hyper parameters
    MAX_ITER = hyperparams['max_iter']
    BATCH_SIZE = hyperparams['batch_size']
    LEARNING_RATE = hyperparams['learning_rate']
    DECAY_RATE = hyperparams['decay_rate']
    CONVERGENCE_CRITERIA = hyperparams['convergence_criteria']
    GAMMA = hyperparams['gamma']
    REG_PARAM = hyperparams['reg_param']


    sm = LonelySoftmaxWithReg(dim=t_data.shape[1], num_labels=num_labels, reg_param=REG_PARAM)


    # Perform numerical vs analytical gradient check

    data = v_data[0].reshape(1, v_data[0].shape[0])
    labels = np.array([v_labels[0]])
    loss, grad = sm.calc_loss_and_grad_for_batch(data, labels)
    w=sm.get_params_as_matrix()
    print(['w shape:',w.shape])
    f = lambda x: \
        FunctionsBoxes.calc_value_and_grad(x,v_labels,w.T,REG_PARAM, calc_value=True, calc_grad=False)[0]
    g = lambda x: \
        FunctionsBoxes.calc_value_and_grad(x, v_labels, w.T, REG_PARAM, calc_value=False, calc_grad=True)[1]

    gradient_err = gradient_test(f, g, v_data, epsilon0=10, num_iter=20, delta=0.5)
    print(["gradient total error=", gradient_err])

    f_jacobianmv = lambda x,v: \
        np.dot(FunctionsBoxes.calc_grad_by_x(w.T, x, v_labels),v)
 #f, f_jacobianmv, x, epsilon0, num_iter=30, delta=0.1)
    #jacobian_err = jacobian_test(f, f_jacobianmv,v_data, epsilon0=0.5, num_iter=20, delta=0.5)
    #print(["Jacobian total error=", jacobian_err])



    #grad_err = grad_check_sparse(f, sm.get_params_as_matrix(), grad, 10)
    #assert grad_err < 0.1
    """
    jacobian_err = jacobian_test(f,g,sm.get_params_as_matrix(),epsilon0=0.5,num_iter=20,delta=0.5)
    print(["Jacobian total error=",jacobian_err])

    gradient_err = gradient_test(f,g, sm.get_params_as_matrix(), epsilon0=0.5, num_iter=20, delta=0.5)
    print(["gradient total error=", jacobian_err])
    """









    sm = ResNetwork(3, t_data.shape[1], REG_PARAM, num_labels)

    #predictions = sm.predict(v_data)

    loss_history, accuracy_history = train_with_sgd(sm, t_data=t_data, t_labels=t_labels, convergence_criteria=CONVERGENCE_CRITERIA,
                                                    decay_rate=DECAY_RATE,
                                                    batch_size=BATCH_SIZE,
                                                    max_iter=MAX_ITER,
                                                    learning_rate=LEARNING_RATE,
                                                    gamma=GAMMA,
                                                    v_data=v_data,
                                                    v_labels=v_labels)

    #print(loss_history)
    iterations = list(range(0, len(accuracy_history['test_set'])))
    test_accs = accuracy_history['test_set']
    validation_accs = accuracy_history['validation_set']

    plt.subplot(2, 1, 1)
    plt.plot(iterations, test_accs, 'o-')
    plt.title('Accuracies')
    plt.ylabel('test accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(iterations, validation_accs, '.-')
    plt.xlabel('iteration')
    plt.ylabel('validation accuracy')

    plt.show()

    return validation_accs[-1]


if __name__ == "__main__":
    main()
    #test()
