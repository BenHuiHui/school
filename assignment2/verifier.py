import numpy as np
import code_base.layers
from code_base.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward, conv_relu_forward, conv_relu_backward
from code_base.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from code_base.classifiers import cnn
import matplotlib.pyplot as plt
from code_base.solver import Solver
from code_base.data_utils import get_CIFAR2_data
from code_base.vis_utils import visualize_grid
import os


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
    out, _ = code_base.layers.conv_forward(x, w, b, conv_param)
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

    out, _ = code_base.layers.max_pool_forward(x, pool_param)

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


def verify_gradient():
    num_inputs = 2
    input_dim = (3, 16, 16)
    reg = 0.0
    num_classes = 10
    np.random.seed(231)
    X = np.random.randn(num_inputs, *input_dim)
    y = np.random.randint(num_classes, size=num_inputs)

    model = cnn.ThreeLayerConvNet(num_filters=3, filter_size=3,
                              input_dim=input_dim, hidden_dim=7,
                              dtype=np.float64)
    loss, grads = model.loss(X, y)
    for param_name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
        e = rel_error(param_grad_num, grads[param_name])
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


def verify_model_on_small_data():
    data = get_CIFAR2_data()
    for k, v in data.items():
        print('% s: ' % k, v.shape)

    np.random.seed(231)

    num_train = 100
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    model = cnn.ThreeLayerConvNet(weight_scale=1e-2)

    solver = Solver(model, small_data,
                    num_epochs=15, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=1)
    solver.train()

    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


def verify_on_full_data_without_dropout():
    model = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=500, reg=0, dropout=0)
    data = get_CIFAR2_data()

    solver = Solver(model, data,
                    num_epochs=10, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# verify_max_pooling_forward()
# verify_sandwich_layer_conv_relu_pool()
# verify_sandwich_layer_conv_relu()
# verify_initial_loss()
# verify_gradient()
# verify_model_on_small_data()
verify_on_full_data_without_dropout()
