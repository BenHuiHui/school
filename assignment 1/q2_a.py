import csv
import os
import numpy as np

dims_list = [
    [14, 4],
    [14, 100, 40, 4],
    [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 14, 4],
    [14, 28, 28, 28, 28, 28, 28, 4],
]


relative_path = "Question2_123"
train_x_filename = "x_train.csv"
train_y_filename = "y_train.csv"

initial_low = -0.2
initial_high = 0.2

lda = 0.001
max_epoch = 40
momentum = 0.2


def read_input(name):
    lst = list()
    with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
            lst.append(np.asarray(row).astype(np.float32))
    return lst


def read_y(name):
    input_y = read_input(name)

    y_list = []

    for i in range(len(input_y)):
        if input_y[i] == 0:
            y_list.append([1, 0, 0, 0])
        elif input_y[i] == 1:
            y_list.append([0, 1, 0, 0])
        elif input_y[i] == 2:
            y_list.append([0, 0, 1, 0])
        elif input_y[i] == 3:
            y_list.append([0, 0, 0, 1])

    return y_list


def initialize(dimension):
    weights = []
    bias = []

    for i in range(len(dimension)-1):
        weights.append(np.random.uniform(initial_low, initial_high, (dimension[i], dimension[i+1])))
        bias.append(np.random.uniform(initial_low, initial_high, dimension[i+1]))

    return weights, bias


def calculate_output(curr_weights, curr_bias, curr_x):
    # Offset input layer that does not have z.
    z_list = [np.array([0])]
    output_list = [curr_x]

    for i in range(len(curr_weights)):
        z = np.dot(np.asarray(output_list[i]).astype(np.float32), np.asarray(curr_weights[i]).astype(np.float32)) \
            + np.asarray(curr_bias[i]).astype(np.float32)

        z_list.append(z)

        output = z

        # Relu.
        # Q: is Relu needed at last layer?
        if i != len(curr_weights) - 1:
            output = np.maximum(z, 0)

        output_list.append(output)

    return output_list, z_list


def calculate_softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def calculate_error(softmax, y):
    return -np.dot(y, np.transpose(np.log(softmax)))


def relu_gradient(value):
    if value > 0:
        return 1

    return 0


def pre_calculate_gradients(dimensions, z_list, y_list, soft_max, current_weights):
    gradients = [0] * (len(dimensions)-1)

    gradient = []
    for i in range(dimensions[len(dimensions)-1]):
        g = (soft_max[i] - y_list[i]) # * relu_gradient(z_list[len(dimensions)-1][i])
        gradient.append(g)

    gradients.append(gradient)

    for i in range(len(dimensions)-2, 0, -1):
        layer_gradient = []

        for j in range(dimensions[i]):
            g = 0
            for k in range(dimensions[i+1]):
                g = g + gradients[i+1][k] * relu_gradient(z_list[i][j]) * np.float32(current_weights[i][j][k])

            layer_gradient.append(g)

        gradients[i] = layer_gradient

    return gradients


def calculate_weights_gradient(outputs, gradients, dims):
    weights_gradients = []

    for i in range(len(dims)-1):
        weights_layer_gradients = []

        for j in range(dims[i]):
            weights_layer_gradient = []
            for k in range(dims[i+1]):
                weights_layer_gradient.append(gradients[i+1][k] * outputs[i][j])

            weights_layer_gradients.append(weights_layer_gradient)

        weights_gradients.append(weights_layer_gradients)

    return weights_gradients


def update(x, y, w, b, d, dw, db):
    o, z = calculate_output(w, b, x)
    s = calculate_softmax(o[len(o) - 1])
    error = calculate_error(s, y)
    g_b = pre_calculate_gradients(d, z, y, s, w)
    g_w = calculate_weights_gradient(o, g_b, d)
    for i in range(len(d) - 1):
        if len(dw) > 0:
            g_w[i] -= np.asarray(dw[i]) * momentum
            g_b[i] -= np.asarray(db[i]) * momentum

        w[i] = w[i] - lda * np.asarray(g_w[i])
        b[i] = b[i] - lda * np.asarray(g_b[i+1])

    dw = g_w
    db = g_b

    return w, b, error, dw, db


def batch(x, y, w, b, dw, db, start, end, dimension, max_epoch):
    for epoch in range(max_epoch):
        err_total = 0
        for i in range(start, end, 1):
            w, b, err, dw, db = update(x[i], y[i], w, b, dimension, dw, db)
            err_total += err

        err = err_total / (len(x)/6)
        print epoch, err

    return w, b, dw, db


def do_work(dimension, max_epoch):
    x = read_input(os.path.join(relative_path, train_x_filename))
    y = read_y(os.path.join(relative_path, train_y_filename))

    w, b = initialize(dimension)
    dw = []
    db = []

    batch_size = len(x) / 6

    for i in range(0, len(x)-batch_size, batch_size):
        w, b, dw, db = batch(x, y, w, b, dw, db, i, i+batch_size, dimension, max_epoch)
    # w, b, dw, db = batch(x, y, w, b, dw, db, 0, 0 + batch_size, dimension, max_epoch)


do_work(dims_list[1], max_epoch)
