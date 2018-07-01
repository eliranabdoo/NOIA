import numpy as np


class LossFunction:
    def calc_loss_and_grad_for_batch(self, X, y):
        pass

    def get_params_as_matrix(self):
        pass

    def update_params(self):
        pass


class LonelySoftmaxWithReg(LossFunction):
    def __init__(self, dim=None, num_labels=None, reg_param=None):
        self.W = np.random.randn(dim, num_labels)*np.sqrt(2/(dim+1))
        self.b = np.zeros([1, num_labels], dtype=np.double)
        self.reg = reg_param

    def calc_loss_and_grad_for_batch(self, X, y):
        return FunctionsBoxes.softmax_loss_and_gradient_regularized(self.get_params_as_matrix(),self.add_bias_dimension(X), y, self.reg)

    def get_params_as_matrix(self):
        return np.vstack((self.W, self.b))

    def add_bias_dimension(self, X):
        return np.column_stack((X, np.ones(X.shape[0])))

    def predict(self, X):
        return np.argmax(self.add_bias_dimension(X).dot(self.get_params_as_matrix()), axis=1)

    def update_params(self, params):
        self.b = params[-1]
        self.W = params[0:-1]


class FunctionsBoxes:

    @staticmethod
    def softmax_loss_and_gradient_regularized(W, X, y, reg):
        """
        Softmax loss function, with quadratic regularization

        Inputs and outputs are the same as softmax_loss_naive.
        """
        loss = 0.0

        num_classes = W.shape[1]
        num_train = X.shape[0]
        dW = np.zeros_like(W)

        scores = X.dot(W)
        scores_exp = np.exp(scores)

        numerical_stab_factors = np.max(scores, axis=1)
        normalized_scores = np.exp(scores.T - numerical_stab_factors.T).T
        scores_sums = np.sum(normalized_scores, axis=1)
        total_scores_mat = (normalized_scores.T / scores_sums.T).T
        labels_mat = np.zeros_like(scores)
        labels_mat[np.arange(0, num_train), y] = 1
        dW += (X.T).dot(total_scores_mat - labels_mat)
        dW /= num_train
        dW += (2 * reg) * W

        class_scores = normalized_scores[np.arange(len(scores_exp)), y.T]
        loss = np.sum(np.log(scores_sums) + np.log(np.ones(num_train) / class_scores))
        loss /= num_train
        loss += reg * np.sum(W * W)




        """numerical_stab_factors = np.exp(-1.0 * np.max(scores, axis=1))
        scores_sums = np.sum(scores_exp, axis=1)

        a_tag = (numerical_stab_factors * scores_exp.T)
        b_tag = (scores_sums * numerical_stab_factors)
        #  print (a_tag[1], b_tag[1], (a_tag/b_tag.T)[1])
        class_scores = scores_exp[np.arange(len(scores_exp)), y.T]
        a = np.multiply(scores_sums, numerical_stab_factors.T)
        b = (np.ones(num_train)) / (np.multiply(numerical_stab_factors, class_scores))
        total_score_mat = (a_tag / b_tag).T
        labels_mat = np.zeros_like(scores)
        labels_mat[np.arange(0, num_train), y] = 1
        dW += (X.T).dot(total_score_mat - labels_mat)
        dW /= num_train
        loss = np.sum(np.log(a) + np.log(b))
        loss /= num_train
        loss += reg * np.sum(W * W)
        dW += (2 * reg) * W"""

        return loss, dW


def train_with_sgd(loss_function, X, y, max_iter, learning_rate, decay_rate,
        batch_size, convergence_criteria):
    """
    We assume that the function can receive dynamic data size
    :param loss_function:
    :param X: The data set (ideally should be loaded to RAM on demand)
    :param y: The corresponding labels
    :param max_iter:
    :param learning_rate:
    :param decay_rate:
    :param batch_size:
    :return:
    """
    loss_history = []
    cur_learning_rate = learning_rate
    num_train, dim = X.shape
    num_of_batches = int(np.ceil(num_train / batch_size))
    cur_loss = 0.0
    for i in range(0, max_iter):
        x_batch = None
        y_batch = None

        cur_learning_rate = update_learning_rate(cur_learning_rate, decay_rate, i)

        assert len(X) == len(y)
        p = np.random.permutation(num_train)
        X = X[p]
        y = y[p]
        for j in range(0, num_of_batches):
            x_batch = X[j*batch_size:(j+1)*batch_size]
            y_batch = y[j*batch_size:(j+1)*batch_size]

            cur_loss, grad = loss_function.calc_loss_and_grad_for_batch(x_batch, y_batch)
            prev_params = loss_function.get_params_as_matrix()

            # Parameter update, change mto momentum\adaGrad
            updated_params = prev_params-learning_rate*grad

            loss_function.update_params(updated_params)
            loss_history.append(cur_loss)

        if np.abs(loss_history[-1]-loss_history[-2]) < convergence_criteria:
            break

    return loss_history


def update_learning_rate(learning_rate, decay_rate, iteration):
    if iteration % 100 == 0:
        return learning_rate*decay_rate
    return learning_rate