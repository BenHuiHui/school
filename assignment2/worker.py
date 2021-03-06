import numpy as np
import code_base.layers
import code_base.gradient_check
from code_base.layers import dropout_forward, dropout_backward


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


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


def q5():
    x_shape = (4, 3, 5, 5)
    w_shape = (2, 3, 3, 3)
    dout_shape = (4, 2, 5, 5)
    x = np.loadtxt('./input_files/conv_backward_in_x.csv')
    x = x.reshape(x_shape)
    w = np.loadtxt('./input_files/conv_backward_in_w.csv')
    w = w.reshape(w_shape)
    b = np.loadtxt('./input_files/conv_backward_in_b.csv')
    dout = np.loadtxt('./input_files/conv_backward_in_dout.csv')
    dout = dout.reshape(dout_shape)

    conv_param = {'stride': 1, 'pad': 2}

    dx_num = code_base.gradient_check.eval_numerical_gradient_array(lambda x: code_base.layers.conv_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = code_base.gradient_check.eval_numerical_gradient_array(lambda w: code_base.layers.conv_forward(x, w, b, conv_param)[0], w, dout)
    db_num = code_base.gradient_check.eval_numerical_gradient_array(lambda b: code_base.layers.conv_forward(x, w, b, conv_param)[0], b, dout)

    out, cache = code_base.layers.conv_forward(x, w, b, conv_param)
    dx, dw, db = code_base.layers.conv_backward(dout, cache)

    np.savetxt('./output_files/conv_backward_out_dx.csv', dx.ravel())
    np.savetxt('./output_files/conv_backward_out_dw.csv', dw.ravel())
    np.savetxt('./output_files/conv_backward_out_db.csv', db.ravel())

    # Your errors should be less than 1e-8'
    print('Testing conv_backward function')
    print('dx error: ', rel_error(dx, dx_num))
    print('dw error: ', rel_error(dw, dw_num))
    print('db error: ', rel_error(db, db_num))


def q7():
    x_shape = (3, 3, 8, 8)
    pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
    x = np.loadtxt('./input_files/maxpool_forward_in_x.csv')
    x = x.reshape(x_shape)

    out, _ = code_base.layers.max_pool_forward(x, pool_param)
    np.savetxt('./output_files/maxpool_forward_out.csv', out.ravel())

def q8():
    x_shape = (3, 2, 10, 10)
    dout_shape = (3, 2, 5, 5)
    x = np.loadtxt('./input_files/maxpool_backward_in_x.csv')
    x = x.reshape(x_shape)
    dout = np.loadtxt('./input_files/maxpool_backward_in_dout.csv')
    dout = dout.reshape(dout_shape)
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    dx_num = code_base.gradient_check.eval_numerical_gradient_array(
        lambda x: code_base.layers.max_pool_forward(x, pool_param)[0], x, dout)

    out, cache = code_base.layers.max_pool_forward(x, pool_param)
    dx = code_base.layers.max_pool_backward(dout, cache)

    np.savetxt('./output_files/maxpool_backward_out.csv', dx.ravel())

    print('Testing conv_backward function')
    print('dx error: ', rel_error(dx, dx_num))


def q18():
    x = np.loadtxt('./input_files/dropout_forward_in_x.csv')
    # Larger p means more dropout
    p = 0.3
    out_train, _ = dropout_forward(x, {'mode': 'train', 'p': p})
    out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

    np.savetxt('./output_files/dropout_forward_out_train.csv', out_train)
    np.savetxt('./output_files/dropout_forward_out_test.csv', out_test)


def q19():
    dout = np.loadtxt('./input_files/dropout_backward_in_dout.csv')
    x = np.loadtxt('./input_files/dropout_backward_in_x.csv')
    dropout_param = {'mode': 'train', 'p': 0.8}
    out, cache = dropout_forward(x, dropout_param)
    dx_train = dropout_backward(dout, cache)
    np.savetxt('./output_files/dropout_backward_out_train.csv', dx_train)

    dropout_param = {'mode': 'test', 'p': 0.8}
    out, cache = dropout_forward(x, dropout_param)
    dx_test = dropout_backward(dout, cache)
    np.savetxt('./output_files/dropout_backward_out_test.csv', dx_test)


# q3()
# q5()
# q7()
# q8()
# q18()
q19()
