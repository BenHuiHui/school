import numpy as np
import layers
from code_base.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward, conv_relu_forward, conv_relu_backward
from code_base.gradient_check import eval_numerical_gradient_array
from code_base.classifiers import cnn


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def verify_convo_forward():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)
    conv_param = {'stride': 2, 'pad': 2}
    out, _ = layers.conv_forward(x, w, b, conv_param)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                               [-0.18387192, -0.2109216 ]],
                              [[ 0.21027089,  0.21661097],
                               [ 0.22847626,  0.23004637]],
                              [[ 0.50813986,  0.54309974],
                               [ 0.64082444,  0.67101435]]],


                             [[[-0.98053589, -1.03143541],
                               [-1.19128892, -1.24695841]],
                              [[ 0.69108355,  0.66880383],
                               [ 0.59480972,  0.56776003]],
                              [[ 2.36270298,  2.36904306],
                               [ 2.38090835,  2.38247847]]]])
    # Compare your output to ours; difference should be around 2e-8
    print 'Testing conv_forward'
    print 'difference: ', rel_error(out, correct_out)


def verify_max_pooling_forward():
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

    out, _ = layers.max_pool_forward(x, pool_param)

    correct_out = np.array([[[[-0.26315789, -0.24842105],
                              [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632],
                              [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158],
                              [0.03157895, 0.04631579]]],
                            [[[0.09052632, 0.10526316],
                              [0.14947368, 0.16421053]],
                             [[0.20842105, 0.22315789],
                              [0.26736842, 0.28210526]],
                             [[0.32631579, 0.34105263],
                              [0.38526316, 0.4]]]])

    # Compare your output with ours. Difference should be around 1e-8.
    print('Testing max_pool_forward function:')
    print('difference: ', rel_error(out, correct_out))


def verify_sandwich_layer_conv_relu_pool():
    np.random.seed(231)
    x = np.random.randn(2, 3, 16, 16)
    w = np.random.randn(3, 3, 3, 3)
    b = np.random.randn(3, )
    dout = np.random.randn(2, 3, 8, 8)
    conv_param = {'stride': 1, 'pad': 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
    dx, dw, db = conv_relu_pool_backward(dout, cache)

    dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x,
                                           dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w,
                                           dout)
    db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b,
                                           dout)

    print('Testing conv_relu_pool')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


def verify_sandwich_layer_conv_relu():
    np.random.seed(231)
    x = np.random.randn(2, 3, 8, 8)
    w = np.random.randn(3, 3, 3, 3)
    b = np.random.randn(3, )
    dout = np.random.randn(2, 3, 8, 8)
    conv_param = {'stride': 1, 'pad': 2}

    out, cache = conv_relu_forward(x, w, b, conv_param)
    dx, dw, db = conv_relu_backward(dout, cache)

    dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)

    print('Testing conv_relu:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


def verify_initial_loss():
    model = cnn.ThreeLayerConvNet()

    N = 50
    X = np.random.randn(N, 3, 32, 32)
    y = np.random.randint(10, size=N)

    loss, grads = model.loss(X, y)
    print('Initial loss (no regularization): ', loss)

    model.reg = 0.5
    loss, grads = model.loss(X, y)
    print('Initial loss (with regularization): ', loss)


# verify_max_pooling_forward()
# verify_sandwich_layer_conv_relu_pool()
# verify_sandwich_layer_conv_relu()
verify_initial_loss()
