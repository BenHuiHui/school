import csv
import os
import numpy as np

dims_list = [
    [14, 100, 40, 4],
    [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 14, 4],
    [14, 28, 28, 28, 28, 28, 28, 4],
]


relative_path = "Question2_123"
train_x_filename = "x_train.csv"
train_y_filename = "y_train.csv"
test_x_filename = "x_test.csv"
test_y_filename = "y_test.csv"

initial_low = -0.3
initial_high = 0.3

lda = 0.1
max_epoch = 100
momentum = 0.4
minimum_softmax = np.exp(-100)
batch_size = 6


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
        if i != len(curr_weights) - 1:
            output = np.maximum(z, 0)

        output_list.append(output)

    return output_list, z_list


def calculate_softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def calculate_error(softmax, y):
    # Offset value that is too small to be presented by float32.
    softmax = np.maximum(softmax, minimum_softmax)
    return -np.dot(y, np.transpose(np.log(softmax)))


def relu_gradient(value):
    if value > 0:
        return 1

    return 0


def pre_calculate_gradients(dimensions, z_list, y_list, soft_max, current_weights):
    gradients = [0] * (len(dimensions)-1)

    gradient = np.asarray(soft_max) - np.asarray(y_list)
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


def update(x, y, w, b, dim, dw, db, start, end):
    error_total = 0
    correct_total = 0

    acc_dw = []
    acc_db = []
    # Offset first column
    acc_db.append(np.zeros(1))

    for i in range(len(dim)-1):
        acc_dw.append(np.zeros((dim[i], dim[i+1])))
        acc_db.append(np.zeros(dim[i+1]))

    for i in range(start, end, 1):
        o, z = calculate_output(w, b, x[i])
        s = calculate_softmax(o[len(o) - 1])
        error = calculate_error(s, y[i])
        error_total += error

        max_index = np.argmax(s)
        correct = y[i][max_index]
        correct_total += correct

        g_b = pre_calculate_gradients(dim, z, y[i], s, w)
        g_w = calculate_weights_gradient(o, g_b, dim)

        for j in range(len(dim) - 1):
            acc_dw[j] += g_w[j]
            acc_db[j+1] += g_b[j+1]

    for i in range(len(dim) - 1):
        dw[i] = - lda * np.asarray(acc_dw[i]) / (end - start) + np.asarray(dw[i]) * momentum
        db[i+1] = - lda * np.asarray(acc_db[i+1]) / (end - start) + np.asarray(db[i+1]) * momentum
        w[i] = w[i] + dw[i]
        b[i] = b[i] + db[i+1]

    return w, b, error_total, dw, db, correct_total


def batch_error(x, y, w, b):
    o, z = calculate_output(w, b, x)
    s = calculate_softmax(o[len(o) - 1])
    e = calculate_error(s, y)
    max_index = np.argmax(s)
    correct = y[max_index]
    return e, correct


def batch(x, y, w, b, dw, db, start, end, dimension, max_epoch, x_test, y_test):
    for epoch in range(max_epoch):
        err_total = 0
        total_train_correct = 0
        for i in range(start, end, batch_size):
            w, b, err, dw, db, correct = update(x, y, w, b, dimension, dw, db, start, start+batch_size)
            err_total += err
            total_train_correct += correct

        err_train = err_total / (end - start)

        # Calculate test error.
        error_test_total = 0
        total_test_correct = 0
        for i in range(len(x_test)):
            err, correct = batch_error(x_test[i], y_test[i], w, b)
            error_test_total += err
            total_test_correct += correct

        err_test = error_test_total / len(x_test)

        print err_train, total_train_correct, (end - start), err_test, total_test_correct, len(x_test)

    return w, b, dw, db


def do_work(dimension, max_epoch):
    x = read_input(os.path.join(relative_path, train_x_filename))
    y = read_y(os.path.join(relative_path, train_y_filename))
    x_test = read_input(os.path.join(relative_path, test_x_filename))
    y_test = read_y(os.path.join(relative_path, test_y_filename))

    w, b = initialize(dimension)
    dw = []
    db = []
    db.append(np.zeros(0))
    for i in range(len(dimension)-1):
        dw.append(np.zeros((dimension[i], dimension[i + 1])))
        db.append(np.zeros(dimension[i + 1]))

    batch(x, y, w, b, dw, db, 0, len(x), dimension, max_epoch, x_test, y_test)


if __name__ == '__main__':
    do_work(dims_list[1], max_epoch)
