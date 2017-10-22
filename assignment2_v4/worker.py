import numpy as np
from code_base.classifiers.cnn import ThreeLayerConvNet
from code_base.solver import Solver
from code_base.data_utils import get_CIFAR2_data
from code_base.layers import conv_forward


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def verify_conv_forward():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    conv_param = {'stride': 2, 'pad': 2}
    out, _ = conv_forward(x, w, b, conv_param)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216]],
                             [[0.21027089, 0.21661097],
                              [0.22847626, 0.23004637]],
                             [[0.50813986, 0.54309974],
                              [0.64082444, 0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[0.69108355, 0.66880383],
                              [0.59480972, 0.56776003]],
                             [[2.36270298, 2.36904306],
                              [2.38090835, 2.38247847]]]])

    # Compare your output to ours; difference should be around 2e-8
    print('Testing conv_forward')
    print('difference: ', rel_error(out, correct_out))


def verify_on_full_data():
    model = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=500, reg=0.01, dropout=0.4)
    data = get_CIFAR2_data()

    solver = Solver(model, data,
                    num_epochs=10, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()


#verify_conv_forward()
verify_on_full_data()
