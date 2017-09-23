import csv
import os
import numpy as np

relative_path = "c"
weights_filenames = ["w-100-40-4.csv", "w-14*28-4.csv", "w-28*6-4.csv"]
bias_filenames = ["b-100-40-4.csv", "b-14*28-4.csv", "b-28*6-4.csv"]
output_bias_filenames = ["db-100-40-4.csv", "db-14-28-4.csv", "db-28-6-4.csv"]
output_weights_filenames = ["dw-100-40-4.csv", "dw-14-28-4.csv", "dw-28-6-4.csv"]

dims_list = [
    [14, 100, 40, 4],
    [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 14, 4],
    [14, 28, 28, 28, 28, 28, 28, 4],
]
x = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
y = [0, 0, 0, 1]


def read_input(name):
    lst = list()
    with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
            lst.append(np.asarray(row[1:]).astype(np.float32))
    return lst


def read_weights(name, dimensions):
    input_weights = read_input(name)

    current_row = 0
    weights_list = []

    for i in range(len(dimensions) - 1):
        dim_prev = dimensions[i]
        dim_next = dimensions[i + 1]

        assert len(input_weights[current_row]) == dim_next

        weight_for_current_dim = []

        for d in range(dim_prev):
            weight_for_current_dim.append(input_weights[current_row])
            current_row = current_row + 1

        weights_list.append(weight_for_current_dim)

    return weights_list


def read_bias(name, dimensions):
    input_bias = read_input(name)

    assert len(dimensions) == len(input_bias) + 1

    bias_list = []

    for i in range(len(dimensions) - 1):
        assert len(input_bias[i]) == dimensions[i+1]

        bias_list.append(input_bias[i])

    return bias_list


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


def write_to_file(filename, content, nested):
    with open(filename, 'wb') as resultFile:
        wr = csv.writer(resultFile)
        if nested:
            for rows in content:
                wr.writerows(rows)
        else:
            wr.writerows(content)


def do_work(weights_filename, bias_filename, output_weights_filename, output_bias_filename, dims):
    weights = read_weights(os.path.join(relative_path, weights_filename), dims)
    bias = read_bias(os.path.join(relative_path, bias_filename), dims)

    out, z = calculate_output(weights, bias, x)
    soft = calculate_softmax(out[len(out) - 1])

    g_b = pre_calculate_gradients(dims, z, y, soft, weights)

    g_w = calculate_weights_gradient(out, g_b, dims)

    write_to_file(output_weights_filename, g_w, 1)
    write_to_file(output_bias_filename, g_b[1:], 0)


if __name__ == '__main__':
    for index in range(len(dims_list)):

        do_work(weights_filenames[index], bias_filenames[index],
                output_weights_filenames[index], output_bias_filenames[index],
                dims_list[index])
