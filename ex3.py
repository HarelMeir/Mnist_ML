import numpy as np
import sys
import matplotlib.pyplot as plt


"""
sigmoid func.
"""
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


"""
loss func.
"""
def loss_func(train_x, train_y):
    sum_loss = np.sum(np.multiply(train_x, np.log(train_y)))
    loss = -(1 / train_x.shape[1]) * sum_loss
    return loss


"""
softmaxs
"""
def softmax(z):
    s = np.exp(z) / np.sum(np.exp(z), axis=0)
    return s

"""
function for one_hot encoding.
"""
def one_hot(train_y,train_x_size):
    train_y = train_y.reshape(1, train_x_size)
    new_train_y = np.eye(10)[train_y.astype('int32')]
    new_train_y.T.reshape(10, train_x_size)
    return new_train_y



"""
function to normlize train_x and test_x.
"""
def normalize_input(train_x):
    length = len(train_x)
    for i in range(length):
        train_x[i] = train_x[i] / 255

"""
Method to suffle the training sets.
"""
def shuffle(train_x, train_y):
    permutation = np.random.permutation(train_x.shape[1])
    x = train_x[:, permutation]
    y = train_y[:, permutation]
    return x, y


"""
fprop method - for moving forward.
"""
def fprop(X, params):

    z1 = np.matmul(params["W1"], X) + params["b1"]

    h1 = sigmoid(z1)

    z2 = np.matmul(params["W2"], h1) + params["b2"]

    h2 = sigmoid(z2)

    z3 = np.matmul(params["W3"], h2) + params["b3"]

    h3 = softmax(z3)

    return {"Z1": z1, "H1":h1, "Z2":z2, "H2": h2, "Z3":z3, "H3":h3}


"""
bport - for backtrack.
"""
def bprop(X, Y, parameters, cache, m_batch):
    d_z_3 = cache["H3"] - Y
    d_w_3 = (1. / m_batch) * np.matmul(d_z_3, cache["H2"].T)
    d_b_3 = (1. / m_batch) * np.sum(d_z_3, axis=1, keepdims=True)

    d_h_2 = np.matmul(parameters["W3"].T, d_z_3)
    d_z_2 = d_h_2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))
    d_w_2 = (1. / m_batch) * np.matmul(d_z_2, cache["H1"].T)
    d_b_2 = (1. / m_batch) * np.sum(d_z_2, axis=1, keepdims=True)

    d_h_1 = np.matmul(parameters["W2"].T, d_z_2)
    d_z_1 = d_h_1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    d_w_1 = (1. / m_batch) * np.matmul(d_z_1, X.T)
    d_b_1 = (1. / m_batch) * np.sum(d_z_1, axis=1, keepdims=True)

    return {"dW1": d_w_1, "db1": d_b_1, "dW2": d_w_2, "db2": d_b_2, "dW3": d_w_3, "db3": d_b_3}








def main():
    # loads the data.
    train_x_path, train_y_path, test_x_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # load data.
    train_x = np.loadtxt(train_x_path)
    train_y = np.loadtxt(train_y_path)
    test_x = np.loadtxt(test_x_path)

    # getting lines
    rows = train_x.shape[0]
    m_test = test_x.shape[0]
    # getting the data ready.
    normalize_input(train_x)
    normalize_input(test_x)

    train_y = train_y.reshape(1, rows)
    y_new_train = np.eye(10)[train_y.astype('int32')]
    y_new_train = y_new_train.T.reshape(10, rows)


    # doing  transpose.
    train_x = train_x.T
    test_x = test_x.T

    # vars for sizes.
    size_x = train_x.shape[0]
    size_h1 = 512
    size_h2 = 64

    # initialize 3 vectors randomly,using 2 hidden layers.
    w1 = np.random.randn(size_h1, size_x) * np.sqrt(1. / size_x)
    w2 = np.random.randn(size_h2, size_h1) * np.sqrt(1. / size_h1)
    w3 = np.random.randn(10, size_h2) * np.sqrt(1. / size_h2)

    # initialize biases, and transforming to calculable form.
    b1 = np.zeros((size_h1, 1)) * np.sqrt(1. / size_x)
    b2 = np.zeros((size_h2, 1)) * np.sqrt(1. / size_h1)
    b3 = np.zeros((10, 1)) * np.sqrt(1. / size_h2)

    # initializing a map with w's and biases.
    params = {"W1": w1, "W2": w2, "W3": w3, "b1": b1, "b2": b2, "b3": b3}

    # vars to update the diff in each epoch.
    diff_w_1 = np.zeros(params["W1"].shape)
    diff_w_2 = np.zeros(params["W2"].shape)
    diff_w_3 = np.zeros(params["W3"].shape)

    diff_b_1 = np.zeros(params["b1"].shape)
    diff_b_2 = np.zeros(params["b2"].shape)
    diff_b_3 = np.zeros(params["b3"].shape)

    # parameters for the algorithm.
    batch_size = 64
    l_r = 4
    eta = 0.9
    num_of_batches = rows // batch_size
    epochs = 20

    # training the model
    for i in range(epochs):
        # shuffeling the data.
        new_train_x, new_train_y = shuffle(train_x, y_new_train)
        # using min batch.
        for j in range(num_of_batches):
            begin = j * batch_size
            end = min(begin + batch_size, train_x.shape[1] - 1)
            x_train = new_train_x[:, begin:end]
            y_train = new_train_y[:, begin:end]
            m_batch = end - begin

            fprop_cache = fprop(x_train, params)
            bprop_cache = bprop(x_train, y_train, params, fprop_cache, m_batch)

            diff_w_1 = (eta * diff_w_1 + (1. - eta) * bprop_cache["dW1"])
            diff_w_2 = (eta * diff_w_2 + (1. - eta) * bprop_cache["dW2"])
            diff_w_3 = (eta * diff_w_3 + (1. - eta) * bprop_cache["dW3"])

            diff_b_1 = (eta * diff_b_1 + (1. - eta) * bprop_cache["db1"])
            diff_b_2 = (eta * diff_b_2 + (1. - eta) * bprop_cache["db2"])
            diff_b_3 = (eta * diff_b_3 + (1. - eta) * bprop_cache["db3"])

            params["W1"] = params["W1"] - l_r * diff_w_1
            params["b1"] = params["b1"] - l_r * diff_b_1
            params["W2"] = params["W2"] - l_r * diff_w_2
            params["b2"] = params["b2"] - l_r * diff_b_2
            params["W3"] = params["W3"] - l_r * diff_w_3
            params["b3"] = params["b3"] - l_r * diff_b_3

    fprop_cache = fprop(test_x, params)
    predictions = np.argmax(fprop_cache["H3"], axis=0)

    # saving predictions to test_predictions.csv
    np.savetxt("test_y", predictions, delimiter=",", fmt="%d")




if __name__ == "__main__":
    main()
