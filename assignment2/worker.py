import numpy as np
import code_base.layers


def q3():
    x_shape = (2, 3, 6, 6)
    w_shape = (3, 3, 4, 4)
    x = np.loadtxt('./input_files/conv_forward_in_x.csv', delimiter=',')
    x = x.reshape(x_shape)
    w = np.loadtxt('./input_files/conv_forward_in_w.csv', delimiter=',')
    w = w.reshape(w_shape)
    b = np.loadtxt('./input_files/conv_forward_in_b.csv', delimiter=',')

    conv_param = {'stride': 2, 'pad': 2}
    out, _ = code_base.layers.conv_forward(x, w, b, conv_param)
    np.savetxt('./output_files/conv_forward_out.csv', out.ravel(), delimiter=',')


q3()
