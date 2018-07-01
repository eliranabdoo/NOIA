import numpy as np
import tables
import sys,os
import scipy.io
from softmax import LonelySoftmaxWithReg, train_with_sgd
from softmax import FunctionsBoxes
from gradient_checks import grad_check_sparse

def accuracy(predictions, labels):
  return (100.0 * np.sum(predictions ==labels)
          / predictions.shape[0])


def load_data(path):
    f = scipy.io.loadmat(path)
    t_data = f.get('Yt').T
    t_labels = np.argmax(f.get('Ct'), axis=0)
    v_data = f.get('Yv').T
    v_labels = np.argmax(f.get('Cv'), axis=0)
    num_labels = f.get('Cv').shape[0]

    return t_data, t_labels, v_data, v_labels, num_labels


def main():
    PATH = os.getcwd()+'\datasets\GMMDATA.mat'
    t_data, t_labels, v_data, v_labels, num_labels = load_data(PATH)

    # Normalize the data
    mean_image = np.mean(t_data, axis=0)
    t_data -= mean_image
    v_data -= mean_image
    t_data /= np.std(t_data, axis=0)
    v_data /= np.std(v_data, axis=0)

    print(np.std(t_data, axis=0), np.mean(t_data, axis=0))

    # Hyper parameters
    REG_PARAM = 2e1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 100
    CONVERGENCE_CRITERIA = 0.1
    DECAY_RATE = 0.1
    MAX_ITER = 1000

    sm = LonelySoftmaxWithReg(dim=t_data.shape[1], num_labels=num_labels, reg_param=REG_PARAM)



    loss_history = train_with_sgd(sm, X=t_data, y=t_labels, convergence_criteria=CONVERGENCE_CRITERIA,
                                  decay_rate=DECAY_RATE,
                                  batch_size=BATCH_SIZE,
                                  max_iter=MAX_ITER,
                                  learning_rate=LEARNING_RATE)
    print(loss_history)

    # Perform numerical vs analytical gradient check
    loss, grad = sm.calc_loss_and_grad_for_batch(v_data, v_labels)
    f = lambda w: \
    FunctionsBoxes.softmax_loss_and_gradient_regularized(w, sm.add_bias_dimension(v_data), v_labels, REG_PARAM)[0]
    grad_err = grad_check_sparse(f, sm.get_params_as_matrix(), grad, 10)
    assert grad_err < 0.1

    print(accuracy(sm.predict(v_data), v_labels))
    print(accuracy(sm.predict(t_data), t_labels))

if __name__ == "__main__":
    main()







