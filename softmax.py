import numpy as np


def accuracy(predictions, labels):
    return (100.0 * np.sum(predictions ==labels)
          / predictions.shape[0])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def update_learning_rate_simple(learning_rate, decay_rate, iteration):
    if iteration < 100:
        return 0.01
    else:
        return 0.001


def update_learning_rate_step(initial_learning_rate, interval, iteration, drop_rate):
    return initial_learning_rate * np.power(drop_rate,
                                     np.floor((1 + iteration) / interval))


class LossFunction:

    def calc_value_and_grad(self,X,y,calc_value=True,calc_grad=True):
        pass

    def get_params_as_matrix(self):
        pass

    def update_params(self, M):
        pass

    def assemble(self, *args):
        """ assume the params should be concatenated row wise """
        return np.vstack(args)

    def disassemble_params(self, P):
        pass


class ResLayer(LossFunction):
    def __init__(self, input_shape=None):
        self.N = input_shape[0]  # dimensionality
        self.W1 = np.random.randn(self.N, self.N)*np.sqrt(2/self.N)  # NxN mat
        self.W2 = np.random.randn(self.N, self.N)*np.sqrt(2/self.N)  # NxN mat
        self.b = np.zeros([self.N], dtype=np.double)  # Nx1  col vec

    def calc_forward_pass(self, X):
        """If X is a batch, should be of size dimXnum_of_samples"""
        return X + self.W2.dot((self.W1.dot(X).T + self.b.T).T)

    def calc_backward_pass(self, X, v):
        """If X is a batch, should be of size dimXnum_of_samples, but for now on we assume X is a single value
        Returns the jacobian w.r.t to the parameter transpose dot v,  and the updated v"""
        assert X.shape[0] == self.self.N
        W1_d_x_p_b = np.add(self.W1.dot(X).T, self.b).T
        sig = sigmoid(W1_d_x_p_b, False)
        sig_derivative = np.multiply(sig, 1 - sig)
        dy_db = np.multiply(self.W2, sig_derivative)
        dy_db_t_v = (dy_db).T.dot(v)
        dy_dw2_t_v = v.dot(sig.T)
        dy_dw1_t_v = dy_db_t_v.dot(X.T)
        new_v = (np.eye(self.N, self.N) + dy_db.dot(self.W1)).T.dot(v)
        return dy_dw1_t_v, dy_db_t_v, dy_dw2_t_v, new_v

    def update_params(self, P):
        self.W1 = P[0:self.N]
        self.b = P[self.N]
        self.W2 = P[self.N+1:self.N*2]

    def get_params_as_matrix(self):
        return self.assemble(self.W1.T, self.b.T, self.W2.T)


class ResNetwork(LossFunction):
    def __init__(self, L, input_shape, reg_param, num_labels):
        self.input_shape = input_shape
        self.L = L
        self.res_layers = [ResLayer(input_shape) for i in range(0, self.L)]
        self.softmax = LonelySoftmaxWithReg(dim=input_shape[1], num_labels=num_labels, reg_param=reg_param)

    def get_params_as_matrix(self,):
        """ Gets the whole network's parameter matrix """
        params_list = []
        for layer in self.res_layers:
            params_list.append(layer.get_params_as_matrix())
        params_list.append(self.softmax.get_params_as_matrix())
        return self.assemble(*params_list)

    def update_params(self, P):
        """ Gets the whole matrix from SGD, and need to update each of its layers properly"""
        N = self.input_shape[1]
        layer_params_num_rows = 2*N + 1
        for i in range(0,self.L):
            self.res_layers[i].update_params(P[i*layer_params_num_rows:(i+1)*layer_params_num_rows])
        self.softmax.update_params(P[-N-2:])

    def calc_value_and_grad(self,X,y,calc_value=True,calc_grad=True):
        """Returns the gradients with only respect to the parameters (without x)"""
        gradient = None
        if calc_grad:
            sum_of_losses = 0
            sum_of_gradients = None
            num_of_samples = X.shape[0]
            for i in range(0, num_of_samples):
                sample = X[i]
                label = y[i]
                x_history = self.forward_pass(sample,label)
                sum_of_losses += x_history[-1]
                cur_gradient = self.backward_pass(sample, label, x_history)
                if sum_of_gradients is None:
                    sum_of_gradients = cur_gradient
                else:
                    sum_of_gradients += cur_gradient
            gradient = sum_of_gradients / num_of_samples
        loss = None
        if calc_value:
            loss = sum_of_losses / num_of_samples
        return loss, gradient

    def predict(self,X):
        last_layer_batch=[]
        for i in range(X.shape[0]):
            x_history=self.forward_pass(X[i],y = None)
            last_x = x_history[-1]
            last_layer_batch.append(last_x)
        return np.argmax(self.add_bias_dimension(last_layer_batch).dot(self.softmax.get_params_as_matrix()), axis=1)

    def forward_pass(self, X, y): #X is a sample
        """Returns the network's value at X,y"""
        x = X
        x_history = []
        x_history.append(x)
        for i in range(0,self.L):
            x = self.res_layers[i].calc_forward_pass(x_history[-1])
            x_history.append(x)
        if y is not None:
            loss = self.softmax.calc_forward_pass(x, y)
            x_history.append(loss)

        return x_history

    def backward_pass(self, y, x_history): #X is a sample
        """Returns the network's gradient at x_history[0], y"""
        result = []
        softmax_input = x_history[-2]
        softmax_input = softmax_input.reshape(softmax_input.shape[0], 1).T  # TODO fix this shit
        __ , softmax_gradient = self.softmax.calc_loss_and_grad_for_batch(softmax_input, y)
        v = self.softmax.grad_by_x(softmax_input, y)  # No transpose is needed as it seems
        result = [softmax_gradient] + result # perhaps transpose is needed

        for i in range(1, self.L+1):
            current_layer = self.res_layers[self.L-i]
            input = x_history[-2-i]
            backward_pass_res = current_layer.calc_backward_pass(input, v)
            v = backward_pass_res[-1]
            cur_gradient = self.assemble(backward_pass_res[0:-1])
            result = [cur_gradient] + result

        return self.assemble(result)


class LonelySoftmaxWithReg(LossFunction):
    def __init__(self, dim=None, num_labels=None, reg_param=None):
        self.W = np.random.randn(dim, num_labels)*np.sqrt(2/(dim+1))
        self.b = np.zeros([1, num_labels], dtype=np.double)
        self.reg = reg_param

    def calc_loss_and_grad_for_batch(self, X, y):
        return FunctionsBoxes.softmax_loss_and_gradient_regularized(self.get_params_as_matrix(), self.add_bias_dimension(X), y, self.reg)

    def get_params_as_matrix(self):
        return np.vstack((self.W, self.b))

    def add_bias_dimension(self, X):
        return np.column_stack((X, np.ones(X.shape[0])))

    def grad_by_x(self, X, y):
        exp_total_sum = np.exp(np.dot(X.T,self.W[1]))
        for i in range(1,self.W.shape[0]):
            exp_total_sum+=np.exp(np.dot(X.T,self.W[i]))
        exp_total_sum = 1./exp_total_sum
        diag= np.diag(exp_total_sum)
        inner_sum =np.dot(diag,np.exp(np.dot(X.T,self.W[0])))-y[0]
        for i in range(1,self.W.shape[0]):
            inner_sum = np.dot(diag, np.exp(np.dot(X.T, self.W[i]))) - y[i]
        result = np.dot(self.W,inner_sum)
        return result

    def predict(self, X):
        return np.argmax(self.add_bias_dimension(X).dot(self.get_params_as_matrix()), axis=1)

    def update_params(self, params):
        self.b = params[-1]
        self.W = params[0:-1]

    def calc_forward_pass(self,X):
        return FunctionsBoxes.calc_forward_pass(self.get_params_as_matrix(),self.add_bias_dimension(X), y, self.reg)

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

        return loss, dW

    def calc_forward_pass(self,W,X,y,reg):
        """
        Softmax loss function, with quadratic regularization

        Output is the calculated loss
        """
        loss = 0.0
        num_classes = W.shape[1]
        num_train = X.shape[0]
        scores = X.dot(W)
        scores_exp = np.exp(scores)
        numerical_stab_factors = np.max(scores, axis=1)
        normalized_scores = np.exp(scores.T - numerical_stab_factors.T).T
        scores_sums = np.sum(normalized_scores, axis=1)
        class_scores = normalized_scores[np.arange(len(scores_exp)), y.T]
        loss = np.sum(np.log(scores_sums) + np.log(np.ones(num_train) / class_scores))
        loss /= num_train
        loss += reg * np.sum(W * W)
        return loss


def train_with_sgd(loss_function, t_data, t_labels, max_iter, learning_rate, decay_rate,
                   batch_size, convergence_criteria, gamma ,v_data, v_labels):
    """
    We assume that the function can receive dynamic data size
    :param loss_function:
    :param t_data: The data set (ideally should be loaded to RAM on demand)
    :param t_labels: The corresponding labels
    :param max_iter:
    :param learning_rate:
    :param decay_rate:
    :param batch_size:
    :return:
    """
    m = np.zeros(loss_function.get_params_as_matrix().shape, dtype=np.double)
    loss_history = []
    accuracy_history = {"test_set": [],
                        "validation_set": []
                        }
    cur_learning_rate = learning_rate
    num_train, dim = t_data.shape
    num_of_batches = int(np.ceil(num_train / batch_size))
    cur_loss = 0.0
    for i in range(0, max_iter):
        x_batch = None
        y_batch = None

        cur_learning_rate = update_learning_rate_step(learning_rate, 10, i, decay_rate)

        assert len(t_data) == len(t_labels)
        p = np.random.permutation(num_train)
        t_data = t_data[p]
        t_labels = t_labels[p]
        for j in range(0, num_of_batches):
            x_batch = t_data[j * batch_size:(j + 1) * batch_size]
            y_batch = t_labels[j * batch_size:(j + 1) * batch_size]

            #cur_loss, grad = loss_function.calc_loss_and_grad_for_batch(x_batch, y_batch)  # TODO this is for the softmax, we need to change the softmax's implementation to support the new func
            cur_loss, grad = loss_function.calc_value_and_grad(x_batch,y_batch, calc_value=True, calc_grad=True)

            prev_params = loss_function.get_params_as_matrix()

            # Parameter update, change mto momentum\adaGrad

            #updated_params = prev_params-learning_rate*grad

            # Momentum update
            m = gamma * m + cur_learning_rate*grad
            updated_params = prev_params - m

            loss_function.update_params(updated_params)
            loss_history.append(cur_loss)

        test_set_accuracy = accuracy(loss_function.predict(t_data), t_labels)
        validation_set_accuracy = accuracy(loss_function.predict(v_data), v_labels)

        print("After %d epochs, train set accuracy is %d" % (i+1, test_set_accuracy))
        print("After %d epochs, validation set accuracy is %d" % (i+1, validation_set_accuracy))

        accuracy_history['test_set'].append(test_set_accuracy)
        accuracy_history['validation_set'].append(validation_set_accuracy)

        #if np.abs(loss_history[-1]-loss_history[-2]) < convergence_criteria:
        #    break

    return loss_history, accuracy_history
